"""ChatML generation for Slayer 1B instruct checkpoints on CPU or GPU.

The runtime in `slayer_runtime.py` speaks raw continuation (prompt in, text out,
no template). Instruction-tuned Slayer checkpoints are trained on ChatML with
`<|im_start|>` = 1, `<|im_end|>` = 2 and BOS `<|endoftext|>` = 0, so they need
the template plus a stop on 1/2 and a repetition penalty (the model card:
plain greedy loops in ~8% of answers, T=0.3/top-p=0.9/penalty 1.1 fixes it).

Usage:
    python scripts/slayer_chat.py --model-dir models/Slayer1B-1.09-Instruct \\
        --quantization int8 --prompt "Kim jesteś?" --max-new-tokens 96

    python scripts/slayer_chat.py --model-dir models/Slayer1B-1.09-Instruct \\
        --prompts-file eval/prompts.txt --out reports/slayer1b-1.09-instruct-chat.jsonl

    python scripts/slayer_chat.py --model-dir ... --no-cache      # pełny forward na krok
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from slayer_runtime import (
    DTYPES,
    SlayerKVCache,
    load_int8_model,
    load_model,
)

IM_START = 1
IM_END = 2
EOT = 0
STOP_IDS = (EOT, IM_START, IM_END)


def build_prompt(system: str, user: str) -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _penalize(scores: torch.Tensor, generated: list[int], penalty: float) -> torch.Tensor:
    if penalty == 1.0 or not generated:
        return scores
    index = torch.tensor(sorted(set(generated)), device=scores.device)
    values = scores.index_select(0, index)
    values = torch.where(values > 0, values / penalty, values * penalty)
    return scores.index_copy(0, index, values)


def _pick(scores: torch.Tensor, temperature: float, top_p: float, generator) -> int:
    if temperature <= 0:
        return int(torch.argmax(scores))
    probs = torch.softmax(scores / temperature, -1)
    sorted_probs, order = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, -1)
    keep = (cumulative - sorted_probs) < top_p
    order, weights = order[keep], sorted_probs[keep]
    if order.numel() == 0:
        return int(torch.argmax(scores))
    return int(order[torch.multinomial(weights / weights.sum(), 1, generator=generator)])


def _stepper(model, ids: torch.Tensor, use_cache: bool):
    """-> (next_logits [V], step(token_id) -> logits [V])."""
    if use_cache:
        try:
            logits, cache = SlayerKVCache.prefill(model, ids)

            def step(token_id: int) -> torch.Tensor:
                return cache.decode(torch.tensor([[token_id]], device=ids.device))[0, -1]

            return logits[0, -1], step
        except (NotImplementedError, AttributeError, RuntimeError):
            pass

    state = {"ids": ids}

    def step_full(token_id: int) -> torch.Tensor:
        token = torch.tensor([[token_id]], device=ids.device)
        state["ids"] = torch.cat([state["ids"], token], dim=1)
        return model(state["ids"])[0][0, -1]

    return model(ids)[0][0, -1], step_full


@torch.inference_mode()
def chat_once(
    model,
    tokenizer: Tokenizer,
    system: str,
    user: str,
    max_new_tokens: int = 96,
    temperature: float = 0.3,
    top_p: float = 0.9,
    penalty: float = 1.1,
    seed: int = 20261126,
    use_cache: bool = True,
) -> dict:
    device = next(model.parameters()).device
    block = getattr(model.cfg, "block", 4096)
    prompt = build_prompt(system, user)
    ids_list = [EOT] + tokenizer.encode(prompt).ids
    budget = max(8, block - max_new_tokens)
    if len(ids_list) > budget:
        ids_list = ids_list[:1] + ids_list[-(budget - 1):]      # BOS zostaje
    ids = torch.tensor([ids_list], dtype=torch.long, device=device)

    generator = torch.Generator(device=device).manual_seed(seed)
    started = time.perf_counter()
    scores, step = _stepper(model, ids, use_cache)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        token = _pick(_penalize(scores, generated, penalty), temperature, top_p, generator)
        if token in STOP_IDS:
            break
        generated.append(token)
        scores = step(token)
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(generated).split("<|im_end|>")[0].strip()
    return {
        "answer": text,
        "prompt_tokens": int(ids.size(1)),
        "generated_tokens": len(generated),
        "seconds": round(elapsed, 2),
        "tokens_per_second": round(len(generated) / elapsed, 2) if elapsed > 0 else 0.0,
        "stopped_on_eot": len(generated) < max_new_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatML generation for Slayer instruct checkpoints")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompts-file", default="")
    parser.add_argument("--system", default="Jesteś zwięzłym asystentem. Odpowiadasz po polsku.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quantization", choices=("none", "int8"), default="none")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.quantization == "int8":
        model, _ = load_int8_model(args.model_dir, args.device)
    else:
        model, _ = load_model(args.model_dir, args.dtype, args.device)
    tokenizer = Tokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))

    prompts: list[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        prompts.extend(
            line.strip() for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not prompts:
        raise SystemExit("podaj --prompt albo --prompts-file")

    rows = []
    for index, prompt in enumerate(prompts, 1):
        result = chat_once(
            model, tokenizer, args.system, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            penalty=args.penalty,
            seed=args.seed + index,
            use_cache=not args.no_cache,
        )
        rows.append({"prompt": prompt, **result})
        print(
            f"[{index}/{len(prompts)}] {result['seconds']}s "
            f"{result['tokens_per_second']} tok/s | {result['answer'][:120]!r}",
            flush=True,
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"model_dir": args.model_dir, "quantization": args.quantization,
                        "temperature": args.temperature, "top_p": args.top_p,
                        "penalty": args.penalty, "rows": rows},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"zapisano: {args.out}")


if __name__ == "__main__":
    main()
