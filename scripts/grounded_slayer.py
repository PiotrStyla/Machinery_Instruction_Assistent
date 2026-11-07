"""Opt-in, unverified Slayer answers; the source-only router remains the default."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from sft_slayer import apply_adapter, format_sft_prompt, upcast_frozen_blocks, upcast_frozen_head
from slayer_runtime import load_model
from evaluate_machine_expert import completion


def build_instruction(question, passages):
    sources = "\n\n".join(f"[{p['id']}] {p['section']}\n{p['text']}" for p in passages)
    return (
        "Odpowiedz krótko po polsku, wyłącznie na podstawie dokumentacji. "
        "Nie dodawaj procedur ani własnych założeń. Zakończ odpowiedź identyfikatorem "
        "źródła w nawiasach kwadratowych. Gdy brak informacji, napisz: Brak informacji w źródle.\n"
        f"Dokumentacja LinuxCNC 2.9.4:\n{sources}\n\nPytanie: {question}"
    )


def make_prompt(question, passages):
    return format_sft_prompt(build_instruction(question, passages))


def validate_grounded_adapter(model_dir, adapter):
    payload = torch.load(adapter, map_location="cpu", weights_only=True)
    config_bytes = (Path(model_dir) / "config.json").read_bytes()
    fingerprint = hashlib.sha256(config_bytes).hexdigest()
    if payload.get("base_config_sha256") != fingerprint:
        raise ValueError("adapter belongs to a different or unidentified base model")
    if payload.get("fp32_head") is not True:
        raise ValueError("grounded experiment requires the recorded frozen FP32 output-head policy")
    config = json.loads(config_bytes)
    expected_blocks = list(range(config["n_layer"] - 2, config["n_layer"]))
    if payload.get("fp32_blocks") != expected_blocks:
        raise ValueError("grounded adapter frozen-suffix precision does not match this experiment")
    return payload


def load_grounded_model(model_dir, adapter=None):
    if adapter:
        validate_grounded_adapter(model_dir, adapter)
    model, cfg = load_model(str(model_dir), "bfloat16", "cpu")
    if adapter:
        apply_adapter(model, str(adapter))
    else:
        upcast_frozen_head(model)
        upcast_frozen_blocks(model, tuple(range(cfg["n_layer"] - 2, cfg["n_layer"])))
    tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    return model, tokenizer, cfg


def generate_grounded(model, tokenizer, cfg, question, passages, max_input_tokens=1024, max_new_tokens=64):
    if max_input_tokens < 1 or max_new_tokens < 1:
        raise ValueError("token budgets must be positive")
    if not passages:
        raise ValueError("no source passages; experimental generation refused")
    prompt = make_prompt(question, passages)
    count = len(tokenizer.encode(prompt).ids)
    if count > max_input_tokens or count + max_new_tokens > cfg["block"]:
        raise ValueError(f"complete source context needs {count} input tokens; evidence is not truncated")
    started = time.perf_counter()
    answer, generated = completion(model, tokenizer, prompt, max_new_tokens)
    return {
        "status": "generated_unverified", "answer": answer,
        "input_tokens": count, "generated_tokens": generated,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "source_entailment": "not_automatically_verified",
        "precision": "BF16 hidden states/backbone; frozen FP32 head and last-two-block linears; FP32 LoRA",
        "notice": "Eksperymentalna odpowiedź Slayera, niezatwierdzona instrukcja naprawy lub uruchomienia maszyny.",
    }
