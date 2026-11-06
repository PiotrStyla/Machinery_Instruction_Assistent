from __future__ import annotations

import argparse

from tokenizers import Tokenizer

from slayer_runtime import DTYPES, generate, load_int8_model, load_model
from sft_slayer import apply_adapter


def main():
    parser = argparse.ArgumentParser(description="Generate text with Slayer1B using a KV cache")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quantization", choices=("none", "int8"), default="none")
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    if args.quantization == "int8":
        if args.adapter:
            raise ValueError("INT8 runner does not support LoRA adapters")
        model, _ = load_int8_model(args.model_dir, args.device)
    else:
        model, _ = load_model(args.model_dir, args.dtype, args.device)
    if args.adapter:
        apply_adapter(model, args.adapter)
    tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    prompt = f"Użytkownik: {args.prompt}\nAsystent:" if args.adapter else args.prompt
    generated = generate(
        model,
        tokenizer,
        prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.max_context_tokens,
    )
    print(generated[len(prompt):].lstrip() if args.adapter and generated.startswith(prompt) else generated)


if __name__ == "__main__":
    main()
