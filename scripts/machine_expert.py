"""Offline evidence search, not machine diagnosis or permission to execute G-code."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import unicodedata
from contextlib import closing
from pathlib import Path

NOTICE = "Informacja dokumentacyjna, nie instrukcja uruchomienia lub naprawy maszyny. Wymagane zatwierdzone procedury i wykwalifikowany personel."
ABSTENTION = "Nie znaleziono odpowiedzi w wybranym korpusie LinuxCNC 2.9.4. Nie należy zgadywać procedury."
# Explicit bilingual glossary, independent of benchmark questions. No translation model/API.
GLOSSARY = {
    "cal": "inches units", "milimetr": "millimeters units", "jednost": "units",
    "absolut": "absolute distance", "bezwzgled": "absolute distance", "przyrost": "incremental distance",
    "wzgled": "incremental distance", "wspolrzed": "coordinate", "uklad": "coordinate system",
    "maszyn": "machine", "modal": "modal", "plaszczyz": "plane", "luk": "arc",
    "srodek": "offsets arc", "posto": "dwell", "pauz": "dwell", "czeka": "dwell",
    "zatrzym": "stop", "odcink": "segment", "tor": "path", "sekund": "seconds",
    "czas": "time", "wrzecion": "spindle", "chlodz": "coolant", "korekc": "compensation",
    "kompens": "compensation", "promien": "cutter compensation", "obrot": "rotation",
}
CODE_RE = re.compile(r"(?<![\w.])(?:[GM]\s*\d+(?:\.\d+)?|[EA]-?\d+)(?!\w|\.\d)", re.I)
GROUPS = {
    "g4": ["G4"], "g17-g19.1": ["G17", "G18", "G19", "G17.1", "G18.1", "G19.1"],
    "g20-g21": ["G20", "G21"], "g53": ["G53"],
    "g54-g59.3": ["G54", "G55", "G56", "G57", "G58", "G59", "G59.1", "G59.2", "G59.3"],
    "g61": ["G61"], "g61.1": ["G61.1"], "g90-g91": ["G90", "G91"],
    "g90.1-g91.1": ["G90.1", "G91.1"],
}


def normalize(text):
    text = text.lower().replace("ł", "l")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_index(corpus_path, db_path):
    rows = read_rows(corpus_path)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty corpus or duplicate section IDs")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as db, db:
        db.execute("DROP TABLE IF EXISTS passages")
        db.execute("DROP TABLE IF EXISTS lookup")
        db.execute("CREATE TABLE passages (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        db.execute("CREATE VIRTUAL TABLE lookup USING fts5(id UNINDEXED, title, text)")
        for row in rows:
            db.execute("INSERT INTO passages VALUES (?,?)", (row["id"], json.dumps(row, ensure_ascii=False)))
            db.execute("INSERT INTO lookup VALUES (?,?,?)", (row["id"], row["section"], row["text"]))
    return len(rows)


def search(db_path, query, top_k=3):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"build the offline index first: {db_path}")
    # These named controllers are explicitly outside this LinuxCNC corpus.
    normalized = normalize(query)
    if re.search(r"\b(haas|fanuc|siemens|heidenhain)\b", normalized):
        return []
    codes = {re.sub(r"\s+", "", code.upper()) for code in CODE_RE.findall(query)}
    with closing(sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)) as db:
        if codes:
            rows = [json.loads(row[0]) for row in db.execute("SELECT payload FROM passages")]
            # Related codes may be discussed inside a section (e.g. G0/G1 in G53).
            # An unknown alarm never falls back to loosely matching normal words.
            known = {re.sub(r"\s+", "", code.upper()) for row in rows
                     for code in CODE_RE.findall(row["text"])}
            if not codes <= known:
                return []
            matched = [row for row in rows if codes.intersection(GROUPS.get(row["id"], []))]
            matched.sort(key=lambda row: (-len(codes.intersection(GROUPS.get(row["id"], []))), row["id"]))
            return [dict(row, score=float(len(codes.intersection(GROUPS.get(row["id"], [])))))
                    for row in matched[:top_k]]
        words = set(re.findall(r"[a-z]{3,}", normalized))
        expanded = set()
        for word in words:
            for prefix, values in GLOSSARY.items():
                if word.startswith(prefix):
                    expanded.update(values.split())
        # Keep English query support without letting arbitrary Polish tokens manufacture evidence.
        terms = words | expanded
        if not terms:
            return []
        expression = " OR ".join('"' + term + '"' for term in sorted(terms))
        found = db.execute(
            "SELECT passages.payload, bm25(lookup,0,3,1) FROM lookup JOIN passages ON passages.id=lookup.id "
            "WHERE lookup MATCH ? ORDER BY bm25(lookup,0,3,1) LIMIT ?", (expression, top_k)).fetchall()
        return [dict(json.loads(payload), score=-score) for payload, score in found]


def benchmark(db_path, questions_path, output):
    results = []
    for question in read_rows(questions_path):
        started = time.perf_counter()
        hits = search(db_path, question["question"], 3)
        ids = [hit["id"] for hit in hits]
        expected = set(question["expected_ids"])
        results.append(dict(question, retrieved_ids=ids,
                            hit_at_1=bool(expected.intersection(ids[:1])),
                            hit_at_3=bool(expected.intersection(ids)),
                            abstained=not hits, milliseconds=round(1000*(time.perf_counter()-started),3)))
    supported = [r for r in results if r["answerable"]]
    unknown = [r for r in results if not r["answerable"]]
    report = {"questions": len(results), "answerable": len(supported), "unanswerable": len(unknown),
              "hit_rate_at_1": sum(r["hit_at_1"] for r in supported)/len(supported) if supported else None,
              "hit_rate_at_3": sum(r["hit_at_3"] for r in supported)/len(supported) if supported else None,
              "unknown_abstention_rate": sum(r["abstained"] for r in unknown)/len(unknown) if unknown else None,
              "note": "Section hit rates, not answer correctness. Hand-authored narrow-scope benchmark; no general safety guarantee.",
              "results": results}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return {k:v for k,v in report.items() if k != "results"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument("--corpus", default="data/machine_expert/chunks.jsonl")
    index.add_argument("--db", default="data/machine_expert/index.sqlite")
    find = sub.add_parser("search")
    find.add_argument("--db", default="data/machine_expert/index.sqlite")
    find.add_argument("--question", required=True)
    find.add_argument("--top-k", type=int, default=3)
    find.add_argument("--json", action="store_true")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--db", default="data/machine_expert/index.sqlite")
    bench.add_argument("--questions", default="data/machine_expert/questions.jsonl")
    bench.add_argument("--output", default="reports/machine_expert_retrieval.json")
    args = parser.parse_args()
    if args.command == "index":
        print(json.dumps({"indexed": build_index(args.corpus, args.db), "db": args.db}))
    elif args.command == "benchmark":
        print(json.dumps(benchmark(args.db, args.questions, args.output), ensure_ascii=False, indent=2))
    else:
        hits = search(args.db, args.question, args.top_k)
        if args.json:
            print(json.dumps({"hits": hits, "notice": NOTICE, "abstention": ABSTENTION if not hits else None}, ensure_ascii=False, indent=2))
        else:
            print(NOTICE)
            print("Candidate evidence; not a model-verified answer." if hits else ABSTENTION)
            for hit in hits:
                print(f"\n[{hit['id']}] {hit['section']}\n{hit['url']}\n{hit['text']}")


if __name__ == "__main__":
    main()
