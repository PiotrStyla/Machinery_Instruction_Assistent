"""Low-memory, cache-aware inference for SlayerLab/Slayer1B-1.05.

The model is a custom dense GPT implementation. This module deliberately keeps
its reference architecture and adds only inference-time loading and KV caching.
"""
from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer


DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _trainer_module(model_dir: str):
    source = os.path.join(model_dir, "train_gpt_ref_r23.py")
    spec = importlib.util.spec_from_file_location("slayer_trainer", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_meta(model_dir: str, cfg: dict):
    trainer = _trainer_module(model_dir)
    c = SimpleNamespace(**cfg)
    with torch.device("meta"):
        model = trainer.GPT(c.vocab, c.n_layer, c.n_embd, c.n_head, c.block, c)
    return model


def _assign_parameter(module: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf in parent._parameters:
        parent._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
    elif leaf in parent._buffers:
        parent._buffers[leaf] = tensor
    else:
        raise KeyError(f"unknown state-dict entry: {name}")


def load_model(model_dir: str, dtype: str = "bfloat16", device: str = "auto"):
    """Load weights without first allocating a duplicate FP32 model.

    Safetensors are converted and assigned one tensor at a time to a
    meta-device model. Only the current source tensor and its converted copy
    are live during loading, which keeps peak RSS close to the target dtype.
    """
    if dtype not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}; choose one of {sorted(DTYPES)}")
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = _build_meta(model_dir, cfg)
    weights_path = os.path.join(model_dir, "model.safetensors")
    names = set()
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            _assign_parameter(model, name, handle.get_tensor(name).to(dtype=DTYPES[dtype]))
            names.add(name)
    for destination, source in cfg.get("tied_weights", {}).items():
        if source not in names:
            _assign_parameter(model, source, getattr(model, destination.split(".")[0]).weight)
    if hasattr(model, "head") and hasattr(model, "tok"):
        model.head.weight = model.tok.weight
    model = model.to(resolve_device(device)).eval()
    return model, cfg

def load_int8_model(model_dir: str, device: str = "cpu"):
    """Load a CPU dynamic-INT8 copy for speed/memory experiments."""
    target = resolve_device(device)
    if target.type != "cpu":
        raise ValueError("dynamic INT8 inference is supported only on CPU")
    model, cfg = load_model(model_dir, "float32", "cpu")
    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    return model, cfg


def _rope_interleaved(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Apply the interleaved RoPE used by Slayer1B-1.05."""
    head_dim = x.size(-1)
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim))
    angles = positions.to(torch.float32)[:, None] * inv[None, :]
    cos = angles.cos().to(x.dtype)[None, None, :, :]
    sin = angles.sin().to(x.dtype)[None, None, :, :]
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def _embed(model, ids: torch.Tensor) -> torch.Tensor:
    x = model.tok(ids)
    if getattr(model, "emb_dim", 0):
        x = model.emb_up(x)
    return x


def _prepare_qkv(model, block, x: torch.Tensor, positions: torch.Tensor, v0: Optional[torch.Tensor]):
    h = block.ln1(x)
    q, k, v = block.qkv(h).split([block.d, block.kvd, block.kvd], dim=-1)
    head_dim = x.size(-1) // block.nh
    q = q.view(x.size(0), x.size(1), block.nh, head_dim).transpose(1, 2)
    k = k.view(x.size(0), x.size(1), block.nkv, head_dim).transpose(1, 2)
    v = v.view(x.size(0), x.size(1), block.nkv, head_dim).transpose(1, 2)
    if getattr(model.cfg, "qk_norm", False):
        q = block.q_norm(q)
        k = block.k_norm(k)
    if getattr(model.cfg, "pos", "rope") == "rope":
        q = _rope_interleaved(q, positions, float(model.cfg.rope_theta))
        k = _rope_interleaved(k, positions, float(model.cfg.rope_theta))
    if getattr(model.cfg, "value_residual", False):
        if block.is_first:
            v0 = v
        else:
            v = v + block.vr_lambda * v0
    if block.nkv != block.nh:
        repeat = block.nh // block.nkv
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return q, k, v, v0


@dataclass
class SlayerKVCache:
    model: torch.nn.Module
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    position: int

    @classmethod
    @torch.inference_mode()
    def prefill(cls, model: torch.nn.Module, token_ids: torch.Tensor) -> tuple[torch.Tensor, "SlayerKVCache"]:
        cfg = model.cfg
        if getattr(cfg, "loop", 1) != 1:
            raise NotImplementedError("KV cache currently supports loop=1 only")
        if getattr(cfg, "mixer_layout", None) not in (None, "A" * cfg.n_layer):
            raise NotImplementedError("KV cache currently supports attention-only Slayer models")
        if getattr(cfg, "imha_t", 0):
            raise NotImplementedError("KV cache currently excludes IMHA")
        x = _embed(model, token_ids)
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        keys, values = [], []
        v0 = None
        for block in model.blocks:
            q, k, v, v0 = _prepare_qkv(model, block, x, positions, v0)
            keys.append(k.contiguous())
            values.append(v.contiguous())
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(token_ids.size(0), token_ids.size(1), block.d)
            x = x + block.proj(y)
            x = x + block.mlp(block.ln2(x))
        logits = model._head(model.lnf(x[:, -1:]))
        return logits, cls(model, keys, values, token_ids.size(1))

    @torch.inference_mode()
    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.size(1) != 1:
            raise ValueError("decode expects exactly one token per batch row")
        model = self.model
        positions = torch.tensor([self.position], device=token_ids.device)
        x = _embed(model, token_ids)
        v0 = None
        for index, block in enumerate(model.blocks):
            q, k, v, v0 = _prepare_qkv(model, block, x, positions, v0)
            k = torch.cat((self.keys[index], k), dim=2)
            v = torch.cat((self.values[index], v), dim=2)
            self.keys[index] = k
            self.values[index] = v
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            y = y.transpose(1, 2).contiguous().view(token_ids.size(0), 1, block.d)
            x = x + block.proj(y)
            x = x + block.mlp(block.ln2(x))
        self.position += 1
        return model._head(model.lnf(x))


def encode(
    tokenizer: Tokenizer,
    text: str,
    device: torch.device,
    max_context_tokens: Optional[int] = None,
) -> torch.Tensor:
    ids = tokenizer.encode(text).ids
    if max_context_tokens is not None:
        if max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        ids = ids[-max_context_tokens:]
    if not ids:
        raise ValueError("prompt tokenization produced no tokens")
    return torch.tensor([ids], dtype=torch.long, device=device)


def sample(logits: torch.Tensor, temperature: float = 0.0, top_k: int = 0) -> torch.Tensor:
    scores = logits[:, -1, :].float()
    if temperature <= 0:
        return scores.argmax(dim=-1, keepdim=True)
    if top_k > 0:
        values, indices = torch.topk(scores, min(top_k, scores.size(-1)), dim=-1)
        probs = torch.softmax(values / temperature, dim=-1)
        return indices.gather(-1, torch.multinomial(probs, 1))
    return torch.multinomial(torch.softmax(scores / temperature, dim=-1), 1)


def generate(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 0,
    max_context_tokens: Optional[int] = None,
):
    device = next(model.parameters()).device
    ids = encode(tokenizer, prompt, device, max_context_tokens)
    logits, cache = SlayerKVCache.prefill(model, ids)
    generated = list(ids[0].tolist())
    for _ in range(max_new_tokens):
        next_id = sample(logits, temperature, top_k)
        token = int(next_id.item())
        generated.append(token)
        if token == 0:
            break
        logits = cache.decode(next_id)
    return tokenizer.decode(generated, skip_special_tokens=False)
