from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


API = "https://datasets-server.huggingface.co/rows"


def main():
    parser = argparse.ArgumentParser(description="Fetch a reproducible Polish evaluation-text proxy")
    parser.add_argument("--output", default="eval/corpus/wiki-pl-20231101-proxy.txt")
    parser.add_argument("--manifest", default="eval/corpus/wiki-pl-20231101-proxy.json")
    parser.add_argument("--offset", type=int, default=100)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=120000)
    args = parser.parse_args()

    params = {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.pl",
        "split": "train",
        "offset": args.offset,
        "length": args.length,
    }
    url = f"{API}?{urlencode(params)}"
    with urlopen(url, timeout=60) as response:
        payload = json.load(response)

    sections = []
    rows = []
    remaining = args.max_chars
    for item in payload["rows"]:
        row = item["row"]
        text = f"{row['title']}\n\n{row['text'].strip()}"
        if remaining <= 0:
            break
        text = text[:remaining]
        sections.append(text)
        rows.append({"row_idx": item["row_idx"], "id": row["id"], "title": row["title"], "url": row["url"]})
        remaining -= len(text)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    manifest = {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.pl",
        "split": "train",
        "offset": args.offset,
        "requested_rows": args.length,
        "rows": rows,
        "source_api": url,
        "chars": sum(len(section) for section in sections),
        "caveat": "External evaluation proxy; disjointness from Slayer training data is not cryptographically proven.",
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
