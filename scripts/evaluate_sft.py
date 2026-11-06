from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tokenizers import Tokenizer

from slayer_runtime import DTYPES, generate, load_model
from sft_slayer import apply_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Slayer SFT on held-out instruction records")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    model, cfg = load_model(args.model_dir, args.dtype, args.device)
    if args.adapter:
        apply_adapter(model, args.adapter)
    tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            prompt = f"Użytkownik: {row['instruction']}\nAsystent:"
            started = time.perf_counter()
            generated = generate(model, tokenizer, prompt, args.max_new_tokens, temperature=0.0, top_k=0)
            elapsed = time.perf_counter() - started
            answer = generated[len(prompt):].lstrip() if generated.startswith(prompt) else generated
            record = {
                "index": index,
                "instruction": row["instruction"],
                "reference": row["response"],
                "generated": answer,
                "elapsed_seconds": round(elapsed, 3),
                "model": cfg.get("step"),
                "adapter": args.adapter,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index + 1}/{len(rows)}] {row['instruction']} ({elapsed:.1f}s)", flush=True)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
