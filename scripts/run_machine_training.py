"""Sequential CPU LoRA experiments for Slayer 1.05/1.06; preserve all measurements, never assume improvement."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from training_measurements import append_event

ROOT = Path(__file__).resolve().parents[1]


def run_python(script, arguments):
    command = [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def evaluate_stage(args, release, folder, model_dir, metrics, split, phase, step, adapter=None, epoch=None):
    stem = f"{phase}-{epoch:03d}-{split}" if epoch is not None else f"{phase}-{split}"
    output = folder / f"{stem}.jsonl"
    arguments = ["--model-dir", model_dir, "--cases", args.data / f"{split}_cases.jsonl",
                 "--output", output, "--metrics-output", metrics, "--release", release,
                 "--split", split, "--phase", phase, "--global-step", step,
                 "--max-seq-len", args.max_seq_len, "--max-new-tokens", args.max_new_tokens,
                 "--threads", args.threads]
    if adapter:
        arguments += ["--adapter", adapter]
    append_event(metrics, "phase", release=release, phase=f"{phase}_{split}", global_step=step)
    run_python("evaluate_machine_sft.py", arguments)
    return json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))


def run_release(args, release, dataset):
    folder = args.output_root / release
    folder.mkdir(parents=True, exist_ok=True)
    metrics = folder / "metrics.jsonl"
    if metrics.exists():
        raise FileExistsError(f"metrics already exist; select a new output root: {metrics}")
    model_dir = args.models_root / f"Slayer1B-{release}"
    started = time.perf_counter()
    append_event(metrics, "experiment_start", release=release, epochs=args.epochs,
                 train_examples=dataset["train_examples"], valid_examples=dataset["valid_examples"],
                 test_examples=dataset["test_examples"], model_dir=str(model_dir),
                 selection="lowest validation response loss among trained epochs; test not used for selection")
    try:
        baseline_valid = evaluate_stage(args, release, folder, model_dir, metrics, "valid", "baseline", 0)
        baseline_test = evaluate_stage(args, release, folder, model_dir, metrics, "test", "baseline", 0)
        previous = None
        candidates = []
        for epoch in range(1, args.epochs + 1):
            adapter = args.adapters_root / release / f"epoch-{epoch:03d}.pt"
            if adapter.exists():
                raise FileExistsError(f"checkpoint exists; choose a new adapters root: {adapter}")
            append_event(metrics, "phase", release=release, phase="training", epoch=epoch,
                         global_step=(epoch - 1) * dataset["train_examples"])
            arguments = ["--model-dir", model_dir, "--train", args.data / "train.jsonl",
                         "--valid", args.data / "valid.jsonl", "--output", adapter,
                         "--epochs", 1, "--rank", 4, "--alpha", 8, "--last-blocks", 2,
                         "--learning-rate", args.learning_rate, "--max-seq-len", args.max_seq_len,
                         "--seed", args.seed + epoch - 1, "--threads", args.threads,
                         "--fp32-head", "--fp32-suffix", "--metrics-output", metrics,
                         "--append-metrics", "--skip-validation"]
            if previous:
                arguments += ["--resume", previous]
            run_python("sft_slayer.py", arguments)
            step = epoch * dataset["train_examples"]
            validation = evaluate_stage(args, release, folder, model_dir, metrics, "valid", "epoch", step,
                                        adapter=adapter, epoch=epoch)
            candidates.append(dict(epoch=epoch, global_step=step, adapter=str(adapter), validation=validation))
            previous = adapter
        best = min(candidates, key=lambda item: item["validation"]["mean_loss"])
        final_test = evaluate_stage(args, release, folder, model_dir, metrics, "test", "final", best["global_step"],
                                    adapter=best["adapter"])
        summary = dict(release=release, baseline_valid=baseline_valid, baseline_test=baseline_test,
                       trained_epochs=candidates, selected=best, final_test=final_test,
                       strict_match_change=final_test["exact_match_rate"] - baseline_test["exact_match_rate"],
                       elapsed_seconds=round(time.perf_counter() - started, 3),
                       warning="Loss/reference match/citation IDs are separate diagnostics, not proof of semantic correctness.")
        (folder / "summary.json").write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        append_event(metrics, "experiment_end", release=release, global_step=args.epochs * dataset["train_examples"],
                     selected_epoch=best["epoch"], best_adapter=best["adapter"],
                     elapsed_seconds=summary["elapsed_seconds"])
        return summary
    except BaseException as error:
        append_event(metrics, "experiment_failed", release=release, error=f"{type(error).__name__}: {error}",
                     elapsed_seconds=round(time.perf_counter() - started, 3))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/machine_expert/sft"))
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/machine_training"))
    parser.add_argument("--adapters-root", type=Path, default=Path("adapters/machine_expert"))
    parser.add_argument("--releases", nargs="+", choices=("1.05", "1.06"), default=["1.05", "1.06"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    if min(args.epochs, args.max_seq_len, args.max_new_tokens, args.threads) < 1 or args.learning_rate <= 0:
        parser.error("positive epochs, budgets, threads and learning rate required")
    for field in ("data", "models_root", "output_root", "adapters_root"):
        setattr(args, field, getattr(args, field).resolve())
    dataset = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in dataset["sha256"].items():
        if hashlib.sha256((args.data / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"dataset checksum mismatch: {filename}")
    for release in args.releases:
        model = args.models_root / f"Slayer1B-{release}"
        for filename in ("config.json", "tokenizer.json", "model.safetensors", "train_gpt_ref_r23.py"):
            if not (model / filename).is_file():
                raise FileNotFoundError(model / filename)
        if (args.output_root / release / "metrics.jsonl").exists():
            raise FileExistsError("existing run must not be overwritten; choose a new output root")
        if any((args.adapters_root / release / f"epoch-{epoch:03d}.pt").exists() for epoch in range(1, args.epochs + 1)):
            raise FileExistsError("existing adapters must not be overwritten; choose a new adapters root")
    args.output_root.mkdir(parents=True, exist_ok=True)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update(dataset=dataset, execution="CPU, sequential releases", precision="frozen FP32 head/suffix linears; remaining backbone and hidden states BF16; FP32 LoRA")
    with (args.output_root / "run.json").open("x", encoding="utf-8") as handle:
        json.dump(configuration, handle, ensure_ascii=False, indent=2)
    results = [run_release(args, release, dataset) for release in args.releases]
    (args.output_root / "summary.json").write_bytes((json.dumps(results, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print("Requested release experiments completed:", args.output_root, flush=True)


if __name__ == "__main__":
    main()
