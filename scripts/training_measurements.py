"""Dependency-free training event IO and deliberately strict answer diagnostics."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path


def normalized_answer(text):
    return " ".join(text.casefold().split())


def answer_scores(answer, reference, source_ids):
    cited = {part.casefold() for part in re.findall(r"\[([^\[\]]+)\]", answer)}
    expected = {key.casefold() for key in source_ids}
    return {"exact_match": normalized_answer(answer) == normalized_answer(reference),
            "correct_citation_ids": cited == expected}


def append_event(path, event, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(event=event, timestamp=time.time(), **fields), ensure_ascii=False) + "\n")


def read_events(path, require_event=True):
    path = Path(path)
    if not path.exists():
        return [], []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    events, errors = [], []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or (require_event and "event" not in row):
                raise ValueError("expected a JSON object with the required fields")
        except (ValueError, UnicodeDecodeError) as error:
            if index == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
                continue  # A concurrent writer has not committed this final line yet.
            errors.append(f"{path.name}:{index + 1}: {type(error).__name__}")
        else:
            events.append(row)
    return events, errors
