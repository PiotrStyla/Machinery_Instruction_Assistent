from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Run the identical Slayer benchmark for multiple releases")
    parser.add_argument("--releases", nargs="+", default=["1.05", "1.06"])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="Polska gospodarka rozwija się, ponieważ")
    args = parser.parse_args()

    benchmark = ROOT / "scripts" / "benchmark_slayer.py"
    for release in args.releases:
        model_dir = ROOT / "models" / f"Slayer1B-{release}"
        if not (model_dir / "model.safetensors").exists():
            raise SystemExit(f"missing weights: {model_dir}")
        command = [
            sys.executable,
            str(benchmark),
            "--model-dir",
            str(model_dir),
            "--dtype",
            args.dtype,
            "--device",
            args.device,
            "--new-tokens",
            str(args.new_tokens),
            "--prompt",
            args.prompt,
        ]
        print(f"=== Slayer1B-{release} ===", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
