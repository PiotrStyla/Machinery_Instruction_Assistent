"""Local CPU clients for Graphiti: E5 embeddings and real Qwen JSON generation."""
from __future__ import annotations

import os
os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["EMBEDDING_DIM"] = "384"

import asyncio
import json
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize

EMBED_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def cached_snapshot(repo):
    return snapshot_download(repo_id=repo, local_files_only=True)


class LocalE5(EmbedderClient):
    def __init__(self):
        self.config = EmbedderConfig(embedding_dim=384)
        self.path = cached_snapshot(EMBED_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.path, local_files_only=True).eval().to("cpu")
        self.lock = threading.Lock()

    @torch.inference_mode()
    def _encode(self, texts):
        # Graphiti's create() interface does not distinguish query from stored fact.
        # Use E5's symmetric similarity convention (query prefix on both sides).
        with self.lock:
            batch = self.tokenizer(["query: " + text for text in texts], padding=True, return_tensors="pt")
            if batch["input_ids"].size(1) > 512:
                raise ValueError("E5 input exceeds 512 tokens; evidence is not silently truncated")
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            return F.normalize(vectors, p=2, dim=1).tolist()

    async def create(self, input_data):
        text = input_data if isinstance(input_data, str) else list(input_data)[0]
        if not isinstance(text, str):
            raise TypeError("Local E5 expects text, not pre-tokenized IDs")
        return (await asyncio.to_thread(self._encode, [text]))[0]

    async def create_batch(self, input_data_list):
        if not input_data_list:
            return []
        return await asyncio.to_thread(self._encode, input_data_list)


class LocalCosineReranker(CrossEncoderClient):
    """Actual embedding-similarity ranker; default Graphiti RRF search does not call it."""
    def __init__(self, embedder):
        self.embedder = embedder

    async def rank(self, query, passages):
        if not passages:
            return []
        vectors = torch.tensor(await self.embedder.create_batch([query, *passages]))
        scores = (vectors[1:] @ vectors[0]).tolist()
        return sorted(zip(passages, scores), key=lambda item: item[1], reverse=True)


class LocalQwen(LLMClient):
    def __init__(self, log_path="reports/machine_expert_graphiti_llm.jsonl", max_calls=20):
        super().__init__(LLMConfig(model=LLM_MODEL, small_model=LLM_MODEL, temperature=0, max_tokens=384))
        self.path = cached_snapshot(LLM_MODEL)
        self.tokenizer = None
        self.local_model = None
        self.lock = threading.Lock()
        self.calls = 0
        self.max_calls = max_calls
        self.log_path = Path(log_path)

    def _run(self, messages, response_model, max_tokens):
        with self.lock:
            if self.calls >= self.max_calls:
                raise ValueError(f"local extraction experiment reached {self.max_calls} model calls")
            self.calls += 1
            if self.local_model is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    self.path, local_files_only=True, dtype=torch.bfloat16).eval().to("cpu")
            chat = [{"role": m.role, "content": m.content} for m in messages]
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if inputs["input_ids"].size(1) > 4096:
                raise ValueError("Qwen extraction input exceeds the explicit 4096-token experiment budget")
            started = time.perf_counter()
            with torch.inference_mode():
                output = self.local_model.generate(**inputs, max_new_tokens=min(max_tokens,384), do_sample=False,
                                                   pad_token_id=self.tokenizer.eos_token_id)
            generated = output[0, inputs["input_ids"].size(1):]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            record = {"call": self.calls, "model": LLM_MODEL, "local_snapshot": self.path,
                      "input_tokens": inputs["input_ids"].size(1), "generated_tokens": len(generated),
                      "elapsed_seconds": round(time.perf_counter()-started,3), "raw": text,
                      "schema": response_model.__name__ if response_model else None}
            # Accept ordinary Markdown JSON fences, never repair or synthesize model facts.
            candidate = text
            if candidate.startswith("```json\n") and candidate.endswith("```"):
                candidate = candidate[8:-3].strip()
            try:
                parsed = json.loads(candidate)
                if not isinstance(parsed, dict):
                    raise ValueError("model returned a non-object JSON value")
                if response_model is not None:
                    parsed = response_model.model_validate(parsed).model_dump()
            except (ValueError, TypeError) as exc:
                record.update(status="invalid_model_output", error=str(exc))
                self._write(record)
                raise ValueError(f"Local Qwen failed structured extraction; see {self.log_path}: {exc}") from exc
            record["status"] = "schema_valid_not_fact_verified"
            self._write(record)
            return parsed

    def _write(self, record):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False)+"\n")
        print(f"local_qwen_call={record['call']} status={record['status']} tokens={record['generated_tokens']}", flush=True)

    async def _generate_response(self, messages, response_model=None, max_tokens=384,
                                 model_size=ModelSize.medium, *, model=None):
        if model is not None and model != LLM_MODEL:
            raise ValueError(f"only cached local model supported: {LLM_MODEL}")
        return await asyncio.to_thread(self._run, messages, response_model, max_tokens)
