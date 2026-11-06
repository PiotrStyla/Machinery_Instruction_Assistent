from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from slayer_runtime import DTYPES, load_model


def main():
    parser = argparse.ArgumentParser(description="Measure next-token perplexity on plain text")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="reports/slayer1b-1.06-perplexity.json")
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, cfg = load_model(args.model_dir, args.dtype, args.device)
    tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    text = Path(args.text).read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text).ids[: args.max_tokens + 1]
    if len(token_ids) < 2:
        raise SystemExit("evaluation text produced fewer than two tokens")

    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(token_ids) - 1, args.block):
            chunk = token_ids[start : start + args.block + 1]
            if len(chunk) < 2:
                break
            ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            logits = model(ids)[0]
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_tokens += targets.numel()
            print(f"tokens={total_tokens}/{len(token_ids) - 1}", flush=True)

    mean_nll = total_nll / total_tokens
    result = {
        "model": cfg.get("step"),
        "model_dir": args.model_dir,
        "dtype": args.dtype,
        "text": args.text,
        "tokens": total_tokens,
        "mean_nll_nats": mean_nll,
        "perplexity": math.exp(mean_nll),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
