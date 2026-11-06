from __future__ import annotations

import argparse
import os
import time

import psutil
import torch
from tokenizers import Tokenizer

from slayer_runtime import DTYPES, SlayerKVCache, encode, load_int8_model, load_model


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Slayer1B cached decoding and parity")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="To jest krótki test modelu Slayer.")
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quantization", choices=("none", "int8"), default="none")
    parser.add_argument("--max-context-tokens", type=int)
    args = parser.parse_args()

    if args.quantization == "int8":
        model, _ = load_int8_model(args.model_dir, args.device)
    else:
        model, _ = load_model(args.model_dir, args.dtype, args.device)
    tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    ids = encode(tokenizer, args.prompt, next(model.parameters()).device, args.max_context_tokens)
    print(f"rss_after_load_mb={rss_mb():.1f}")

    with torch.inference_mode():
        t0 = time.perf_counter()
        full_logits = model(ids)[0][:, -1:]
        if ids.device.type == "cuda":
            torch.cuda.synchronize()
        full_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cached_logits, cache = SlayerKVCache.prefill(model, ids)
        if ids.device.type == "cuda":
            torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000
        max_abs = (full_logits.float() - cached_logits.float()).abs().max().item()

        token = ids[:, -1:]
        t0 = time.perf_counter()
        for _ in range(args.new_tokens):
            cached_logits = cache.decode(token)
            token = cached_logits[:, -1:].float().argmax(dim=-1, keepdim=True)
        if token.device.type == "cuda":
            torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) * 1000

    print(f"prompt_tokens={ids.size(1)}")
    print(f"reference_full_forward_ms={full_ms:.2f}")
    print(f"cached_prefill_ms={prefill_ms:.2f}")
    print(f"cache_parity_max_abs={max_abs:.6g}")
    print(f"decode_tokens={args.new_tokens}")
    print(f"decode_ms_total={decode_ms:.2f}")
    print(f"decode_tokens_per_second={args.new_tokens / (decode_ms / 1000):.3f}")
    print(f"rss_after_benchmark_mb={rss_mb():.1f}")


if __name__ == "__main__":
    main()
