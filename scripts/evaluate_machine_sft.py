"""Measure a real Slayer completion and strict reference match, not semantic truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from grounded_slayer import generate_grounded, load_grounded_model
from machine_expert import read_rows
from machine_router import EvidenceRouter
from prepare_machine_sft import supervised_row
from sft_slayer import _example, _loss
from training_measurements import answer_scores, append_event


def evaluate(args):
    torch.set_num_threads(args.threads)
    corpus = EvidenceRouter(corpus_path=args.corpus).corpus
    rows = read_rows(args.cases)
    if not rows:
        raise ValueError("evaluation cases must not be empty")
    model, tokenizer, cfg = load_grounded_model(args.model_dir, args.adapter)
    for row in rows:
        target = supervised_row(row, corpus)
        _example(tokenizer, target["instruction"], target["response"], args.max_seq_len, torch.device("cpu"))
        if len(tokenizer.encode(" " + target["response"]).ids) > args.max_new_tokens:
            raise ValueError(f"reference answer cannot fit generation budget: {row['id']}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    with output.open("x", encoding="utf-8", newline="\n") as handle, torch.inference_mode():
        for index, row in enumerate(rows, 1):
            passages = [corpus[key] for key in row["source_ids"]]
            target = supervised_row(row, corpus)
            inputs, labels = _example(tokenizer, target["instruction"], target["response"], args.max_seq_len,
                                      next(model.parameters()).device)
            loss = float(_loss(model, inputs, labels).item())
            generated = generate_grounded(model, tokenizer, cfg, row["question"], passages,
                                          args.max_seq_len, args.max_new_tokens)
            record = dict(row, reference=target["response"], response_loss=loss,
                          **generated, **answer_scores(generated["answer"], target["response"], row["source_ids"]))
            record.update(evidence_mode="oracle_complete_source_sections", source_urls=[p["url"] for p in passages])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
            if args.metrics_output:
                append_event(args.metrics_output, "evaluation_progress", release=args.release, split=args.split,
                             phase=args.phase, global_step=args.global_step, done=index, total=len(rows))
            print(f"{args.release} {args.phase}/{args.split} {index}/{len(rows)} {row['id']}: exact={record['exact_match']} citation={record['correct_citation_ids']}", flush=True)
    summary = {
        "release": args.release, "split": args.split, "phase": args.phase, "global_step": args.global_step,
        "examples": len(records), "mean_loss": statistics.mean(row["response_loss"] for row in records),
        "exact_match_rate": sum(row["exact_match"] for row in records) / len(records),
        "citation_rate": sum(row["correct_citation_ids"] for row in records) / len(records),
        "median_generation_seconds": statistics.median(row["elapsed_seconds"] for row in records),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model_dir": args.model_dir, "adapter": args.adapter,
        "cases_sha256": hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(),
        "source_corpus_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
        "evidence_mode": "oracle_complete_source_sections",
        "quality_warning": "Exact full-answer match and citation IDs are not a semantic truth or safety verifier.",
    }
    output.with_suffix(".summary.json").write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    if args.metrics_output:
        append_event(args.metrics_output, "evaluation", **summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--corpus", default="data/machine_expert/chunks.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output")
    parser.add_argument("--release", required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--phase", choices=("baseline", "epoch", "final"), required=True)
    parser.add_argument("--global-step", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.max_seq_len, args.max_new_tokens, args.threads) < 1 or args.global_step < 0:
        parser.error("positive budgets/threads and nonnegative step required")
    evaluate(args)


if __name__ == "__main__":
    main()
