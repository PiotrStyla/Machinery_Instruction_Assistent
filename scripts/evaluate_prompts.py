from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tokenizers import Tokenizer

from slayer_runtime import DTYPES, generate, load_model
from sft_slayer import apply_adapter


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed continuation prompts deterministically")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompts", default="eval/prompts.txt")
    parser.add_argument("--output", default="reports/slayer1b-1.06-greedy.jsonl")
    parser.add_argument("--adapter")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    prompts = [line.strip() for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
    model, cfg = load_model(args.model_dir, args.dtype, args.device)
    adapter = apply_adapter(model, args.adapter) if args.adapter else None
    tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts):
            started = time.perf_counter()
            text = generate(model, tokenizer, prompt, args.max_new_tokens, temperature=0.0, top_k=0)
            elapsed = time.perf_counter() - started
            record = {
                "index": index,
                "prompt": prompt,
                "text": text,
                "prompt_tokens": len(tokenizer.encode(prompt).ids),
                "elapsed_seconds": round(elapsed, 3),
                "model": cfg.get("step"),
                "dtype": args.dtype,
                "adapter": args.adapter,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index + 1}/{len(prompts)}] {prompt} ({elapsed:.1f}s)", flush=True)

    print(f"saved={output} adapter={adapter is not None}")


if __name__ == "__main__":
    main()
