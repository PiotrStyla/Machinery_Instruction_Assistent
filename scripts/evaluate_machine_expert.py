"""Measure raw Slayer answers with oracle versus retrieved evidence. Not a repair assistant."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from machine_expert import ABSTENTION, NOTICE, read_rows, search
from slayer_runtime import SlayerKVCache, encode, load_model, sample


def prepare_prompt(question, passages, tokenizer, max_input_tokens, max_new_tokens, model_context):
    sources = "\n\n".join(f"[{p['id']}] {p['section']}\n{p['text']}" for p in passages)
    prompt = (
        "Odpowiedz krótko po polsku, wyłącznie na podstawie dokumentacji poniżej. "
        "Nie dodawaj procedur ani własnych założeń. Gdy brak informacji, napisz: Brak informacji w źródle.\n"
        f"Dokumentacja LinuxCNC 2.9.4 (oryginał angielski):\n{sources}\n\n"
        f"Pytanie: {question}\nOdpowiedź:"
    )
    tokens = len(tokenizer.encode(prompt).ids)
    if tokens > max_input_tokens or tokens + max_new_tokens > model_context:
        raise ValueError(f"complete evidence exceeds context budget: {tokens} input tokens")
    return prompt, tokens


@torch.inference_mode()
def completion(model, tokenizer, prompt, max_new_tokens):
    ids = encode(tokenizer, prompt, next(model.parameters()).device)
    logits, cache = SlayerKVCache.prefill(model, ids)
    generated = []
    for index in range(max_new_tokens):
        token = sample(logits)
        value = int(token.item())
        if value == 0:
            break
        generated.append(value)
        if index + 1 < max_new_tokens:
            logits = cache.decode(token)
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--corpus", default="data/machine_expert/chunks.jsonl")
    parser.add_argument("--questions", default="data/machine_expert/questions.jsonl")
    parser.add_argument("--db", default="data/machine_expert/index.sqlite")
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", nargs="+", choices=("oracle", "retrieved"), default=["oracle", "retrieved"])
    parser.add_argument("--limit", type=int, default=0, help="0 means all questions")
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    if args.limit < 0 or args.max_input_tokens < 1 or args.max_new_tokens < 1:
        parser.error("limit must be nonnegative and token budgets positive")
    corpus = {row["id"]: row for row in read_rows(args.corpus)}
    questions = read_rows(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    corpus_hash = hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()
    model, tokenizer, cfg = None, None, None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in questions:
            for mode in args.modes:
                passages = ([corpus[key] for key in row["expected_ids"]] if mode == "oracle"
                            else search(args.db, row["question"], top_k=1))
                record = {"question_id": row["id"], "question": row["question"], "mode": mode,
                          "model_dir": args.model_dir, "corpus_sha256": corpus_hash,
                          "reference_answer": row["reference_answer"], "answerable": row["answerable"],
                          "expected_ids": row["expected_ids"], "retrieved_ids": [p["id"] for p in passages],
                          "sources": [{key: p[key] for key in ("id", "section", "url", "revision", "source_path", "line_start", "line_end")}
                                      for p in passages], "notice": NOTICE}
                started = time.perf_counter()
                if not passages:
                    record.update(status="no_evidence_guard", answer=ABSTENTION, generated_tokens=0)
                else:
                    if model is None:
                        model, cfg = load_model(args.model_dir, "bfloat16", "cpu")
                        tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
                    try:
                        prompt, count = prepare_prompt(row["question"], passages, tokenizer, args.max_input_tokens,
                                                       args.max_new_tokens, cfg["block"])
                    except ValueError as exc:
                        record.update(status="context_budget_exceeded", error=str(exc))
                    else:
                        answer, count_generated = completion(model, tokenizer, prompt, args.max_new_tokens)
                        record.update(status="generated_unverified", answer=answer, input_tokens=count,
                                      generated_tokens=count_generated, source_entailment="not_automatically_assessed")
                record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"{row['id']} {mode}: {record['status']} ({record['elapsed_seconds']}s)", flush=True)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
