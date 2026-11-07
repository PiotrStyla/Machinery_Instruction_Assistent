"""Build a small, explicit source-grounded SFT experiment; splits are question-disjoint, not fact-disjoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from grounded_slayer import build_instruction
from machine_expert import normalize, read_rows
from machine_router import EvidenceRouter


# AI-assistant-authored paraphrases with exact source spans; no model labelling is invoked by the builder.
VALID = [
    ("v01", "q13", "Jaką jednostkę czasu ma parametr P polecenia G4?", "The P number is the time in seconds that all axes will remain unmoving."),
    ("v02", "q09", "Która płaszczyzna jest domyślna po wyborze G17?", "'G17' - XY (default)"),
    ("v03", "q05", "W jakich jednostkach długości pracuje G20?", "'G20' - to use inches for length units."),
    ("v04", "q04", "Czy działanie G53 utrzymuje się modalnie w kolejnych liniach?", "'G53' is not modal and\nmust be programmed on each line."),
    ("v05", "q18", "Jaki numer układu odpowiada wyborowi G54?", "'G54'   - select coordinate system 1"),
    ("v06", "q21", "Co dokumentacja mówi o dokładności toru w G61?", "Moves will slow or stop as needed to reach every programmed point."),
    ("v07", "q22", "Jak zachowuje się G61.1 na końcach segmentów ruchu?", "movement will stop at the end of each\n  programmed segment."),
    ("v08", "q07", "Skąd liczone jest przesunięcie osi przy aktywnym G91?", "numbers usually represent increments from the current coordinate."),
    ("v09", "q24", "Czy G91.1 przywraca domyślne przyrostowe znaczenie IJK?", "`G91.1` Returns I, J & K to their default behavior."),
]
TEST = [
    ("t01", "q15", "Jakie wartości lub brak parametru P są błędem dla postoju G4?", "the P number is negative or not specified."),
    ("t02", "q12", "Czy po wybraniu płaszczyzny UV, WU lub VW można opisywać łuki?", "The UV, WU and VW planes do not support arcs."),
    ("t03", "q06", "Jaka jednostka długości obowiązuje po wyborze G21?", "'G21' - to use millimeters for length units."),
    ("t04", "q16", "Czy brak aktywnego G0 lub G1 powoduje błąd użycia G53?", "G53 is used without G0 or G1 being active"),
    ("t05", "q19", "Jaki numer układu współrzędnych przypisano do G59.3?", "'G59.3' - select coordinate system 9"),
    ("t06", "q21", "Czy w G61 dwa kolejne dokładnie współliniowe ruchy muszą się zatrzymać?", "two sequential moves are exactly co-linear movement will not stop."),
    ("t07", "q22", "Czy dokładne zatrzymanie G61.1 dotyczy każdego zaprogramowanego segmentu?", "movement will stop at the end of each\n  programmed segment."),
    ("t08", "q08", "Do jakiego układu odnosi się zwykle pozycja osi w trybie G90?", "the currently active coordinate system."),
    ("t09", "q23", "Co tryb G90.1 oznacza dla przesunięć I, J oraz K?", "'G90.1' - absolute distance mode for I, J & K offsets."),
]


def supervised_row(case, corpus):
    passages = [corpus[key] for key in case["source_ids"]]
    citation = " ".join(f"[{key}]" for key in case["source_ids"])
    return {"instruction": build_instruction(case["question"], passages),
            "response": case["reference_answer"].strip() + " " + citation}


def build_cases(questions, corpus):
    by_id = {row["id"]: row for row in questions}
    train = [dict(id=row["id"], question=row["question"], reference_answer=row["reference_answer"],
                  source_ids=row["expected_ids"]) for row in questions if row["answerable"]]
    def heldout(definitions):
        cases = []
        for identifier, reference_id, question, quote in definitions:
            reference = by_id[reference_id]
            if not any(quote in corpus[key]["text"] for key in reference["expected_ids"]):
                raise ValueError(f"source quote not found: {identifier}")
            answer = reference["reference_answer"]
            if identifier == "t06":
                answer = "Nie. Dwa kolejne dokładnie współliniowe ruchy w G61 nie muszą się zatrzymać."
            cases.append(dict(id=identifier, question=question, reference_answer=answer,
                              source_ids=reference["expected_ids"], source_quote=quote))
        return cases
    valid, test = heldout(VALID), heldout(TEST)
    question_sets = [{normalize(row["question"]).strip() for row in split} for split in (train, valid, test)]
    if any(question_sets[a] & question_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("question leakage between splits")
    return train, valid, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/machine_expert/chunks.jsonl")
    parser.add_argument("--questions", default="data/machine_expert/questions.jsonl")
    parser.add_argument("--output", default="data/machine_expert/sft")
    args = parser.parse_args()
    corpus = EvidenceRouter(corpus_path=args.corpus).corpus
    train, valid, test = build_cases(read_rows(args.questions), corpus)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        for filename, records in ((f"{name}_cases.jsonl", rows),
                                  (f"{name}.jsonl", [supervised_row(row, corpus) for row in rows])):
            raw = ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)).encode("utf-8")
            target = root / filename
            if target.exists() and target.read_bytes() != raw:
                raise FileExistsError(f"dataset differs; choose a new output directory: {target}")
            target.write_bytes(raw)
            hashes[filename] = hashlib.sha256(raw).hexdigest()
    manifest = {
        "source_corpus_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
        "source_revision": next(iter(corpus.values()))["revision"],
        "train_examples": len(train), "valid_examples": len(valid), "test_examples": len(test),
        "split_unit": "distinct manually authored questions; source sections and many facts are shared",
        "unseen_documents": False, "label_generation_at_runtime": False,
        "label_provenance": "Existing project references plus AI-assistant-authored paraphrases/references; source-span checks only, no independent human review in this workflow.",
        "evaluation_warning": "Strict full-answer reference match is not a semantic truth verifier or industrial benchmark.",
        "sha256": hashes,
    }
    (root / "manifest.json").write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
