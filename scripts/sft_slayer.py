from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from slayer_runtime import DTYPES, load_model


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.normal_(self.lora_a, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        update = F.linear(F.linear(x.float(), self.lora_a), self.lora_b)
        return base_output + (update * self.scale).to(base_output.dtype)


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    targets: tuple[str, ...],
    block_indices: tuple[int, ...] | None = None,
) -> int:
    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name.rsplit(".", 1)[-1] not in targets:
            continue
        if block_indices is not None and name.startswith("blocks."):
            block_index = int(name.split(".", 2)[1])
            if block_index not in block_indices:
                continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, rank, alpha))
        count += 1
    if count == 0:
        raise RuntimeError(f"no linear layers matched targets={targets} blocks={block_indices}")
    return count


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if "lora_" in name}


def save_adapter(
    model: nn.Module,
    path: Path,
    model_dir: str,
    rank: int,
    alpha: float,
    targets: tuple[str, ...],
    block_indices: tuple[int, ...] | None,
    optimizer: torch.optim.Optimizer | None = None,
    global_step: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_dir": model_dir,
        "rank": rank,
        "alpha": alpha,
        "targets": list(targets),
        "blocks": list(block_indices) if block_indices is not None else None,
        "global_step": global_step,
        "state": adapter_state(model),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_adapter(model: nn.Module, path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state"]
    missing = []
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            if name not in state:
                missing.append(name)
            else:
                parameter.data.copy_(state[name].to(parameter.device, parameter.dtype))
    if missing:
        raise RuntimeError(f"adapter is missing parameters: {missing[:3]}")
    return payload

def apply_adapter(model: nn.Module, path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    targets = tuple(payload["targets"])
    blocks = payload.get("blocks")
    block_indices = tuple(blocks) if blocks is not None else None
    inject_lora(model, int(payload["rank"]), float(payload["alpha"]), targets, block_indices)
    load_adapter(model, path)
    model.eval()
    return payload


def _example(tokenizer: Tokenizer, instruction: str, response: str, max_seq_len: int, device: torch.device):
    prompt = f"Użytkownik: {instruction}\nAsystent:"
    full_ids = tokenizer.encode(prompt + " " + response).ids
    prompt_len = len(tokenizer.encode(prompt).ids)
    full_ids = full_ids[:max_seq_len]
    if len(full_ids) < 2 or prompt_len >= len(full_ids):
        raise ValueError("example has no response tokens after truncation")
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    labels = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    labels[:, : max(0, prompt_len - 1)] = -100
    return input_ids, labels


def _read_jsonl(path: str) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(set(row) != {"instruction", "response"} for row in rows):
        raise ValueError(f"invalid SFT JSONL: {path}")
    return rows


def _loss(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _, loss = model(input_ids, targets=labels)
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"non-finite training loss: {loss}")
    return loss

def _rss_mb() -> float:
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a low-rank SFT adapter for Slayer1B")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--last-blocks", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--metrics-output")
    parser.add_argument("--resume")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--append-metrics", action="store_true")
    args = parser.parse_args()
    if args.rank < 1 or args.epochs < 1 or args.max_steps < 0:
        raise ValueError("rank and epochs must be positive; max-steps must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = _read_jsonl(args.train)
    valid_rows = _read_jsonl(args.valid)
    model, cfg = load_model(args.model_dir, args.dtype, args.device)
    tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    device = next(model.parameters()).device
    targets = ("qkv", "proj", "down")
    if args.last_blocks < 0 or args.last_blocks > cfg["n_layer"]:
        raise ValueError(f"last-blocks must be between 0 and {cfg['n_layer']}")
    block_indices = (
        tuple(range(cfg["n_layer"] - args.last_blocks, cfg["n_layer"])) if args.last_blocks else None
    )
    matched = inject_lora(model, args.rank, args.alpha, targets, block_indices)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    resume_payload = None
    if args.resume:
        resume_payload = load_adapter(model, args.resume)
        if resume_payload.get("optimizer_state"):
            optimizer.load_state_dict(resume_payload["optimizer_state"])
    started = time.perf_counter()
    metrics_output = Path(args.metrics_output) if args.metrics_output else Path(args.output).with_suffix(".training.jsonl")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_handle = metrics_output.open("a" if args.append_metrics else "w", encoding="utf-8")
    metrics_handle.write(
        json.dumps(
            {
                "event": "run_start",
                "timestamp": time.time(),
                "model": cfg.get("step"),
                "model_dir": args.model_dir,
                "train_examples": len(train_rows),
                "valid_examples": len(valid_rows),
                "rank": args.rank,
                "alpha": args.alpha,
                "learning_rate": args.learning_rate,
                "last_blocks": list(block_indices) if block_indices is not None else None,
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "resume": args.resume,
                "global_step_start": resume_payload.get("global_step", 0) if resume_payload else 0,
                "rss_mb": _rss_mb(),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    metrics_handle.flush()
    history = []

    global_step = resume_payload.get("global_step", 0) if resume_payload else 0
    for epoch in range(args.epochs):
        order = list(range(len(train_rows)))
        random.shuffle(order)
        running = 0.0
        steps_done = 0
        for step, index in enumerate(order, start=1):
            row = train_rows[index]
            input_ids, labels = _example(tokenizer, row["instruction"], row["response"], args.max_seq_len, device)
            loss = _loss(model, input_ids, labels)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip).item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            steps_done += 1
            running += float(loss.detach().item())
            metrics_handle.write(
                json.dumps(
                    {
                        "event": "step",
                        "timestamp": time.time(),
                        "epoch": epoch + 1,
                        "epoch_step": step,
                        "global_step": global_step,
                        "loss": float(loss.detach().item()),
                        "running_loss": running / steps_done,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm,
                        "sequence_tokens": input_ids.size(1),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "rss_mb": _rss_mb(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            metrics_handle.flush()
            if step == 1 or step == len(order) or step % 8 == 0:
                print(f"epoch={epoch + 1}/{args.epochs} step={step}/{len(order)} loss={running / step:.4f}", flush=True)
            if args.max_steps and global_step >= args.max_steps:
                break
        history.append({"epoch": epoch + 1, "steps": steps_done, "train_loss": running / steps_done})
        metrics_handle.write(
            json.dumps(
                {
                    "event": "epoch",
                    "timestamp": time.time(),
                    "epoch": epoch + 1,
                    "steps": steps_done,
                    "train_loss": running / steps_done,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "rss_mb": _rss_mb(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        metrics_handle.flush()
        if args.max_steps and global_step >= args.max_steps:
            break

    model.eval()
    if args.skip_validation:
        valid_loss = None
        metrics_handle.write(
            json.dumps(
                {
                    "event": "validation_skipped",
                    "timestamp": time.time(),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "rss_mb": _rss_mb(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    else:
        with torch.inference_mode():
            valid_losses = []
            for row in valid_rows:
                input_ids, labels = _example(tokenizer, row["instruction"], row["response"], args.max_seq_len, device)
                valid_losses.append(float(_loss(model, input_ids, labels).item()))
        valid_loss = sum(valid_losses) / len(valid_losses)
        metrics_handle.write(
            json.dumps(
                {
                    "event": "validation",
                    "timestamp": time.time(),
                    "valid_loss": valid_loss,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "rss_mb": _rss_mb(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    metrics_handle.write(
        json.dumps(
            {
                "event": "run_end",
                "timestamp": time.time(),
                "global_step": global_step,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "rss_mb": _rss_mb(),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    metrics_handle.flush()
    metrics_handle.close()
    output = Path(args.output)
    save_adapter(
        model,
        output,
        args.model_dir,
        args.rank,
        args.alpha,
        targets,
        block_indices,
        optimizer,
        global_step,
    )
    metrics = {
        "model": cfg.get("step"),
        "model_dir": args.model_dir,
        "train_examples": len(train_rows),
        "valid_examples": len(valid_rows),
        "matched_linear_layers": matched,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "global_step": global_step,
        "history": history,
        "resume": args.resume,
        "skip_validation": args.skip_validation,
        "valid_loss": valid_loss,
        "metrics_output": str(metrics_output),
    }
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
