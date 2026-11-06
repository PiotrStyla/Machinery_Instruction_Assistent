"""Fetch a licensed, pinned LinuxCNC excerpt corpus; runtime search stays offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

REVISION = "b8ccac8f835c0610233fdf799255c63f2b9738ac"  # LinuxCNC v2.9.4
FILES = ("gcode/g-code.adoc", "common/overleaf.adoc", "common/gpld-copyright.adoc")
SECTIONS = ("g4", "g17-g19.1", "g20-g21", "g53", "g54-g59.3", "g61", "g61.1", "g90-g91", "g90.1-g91.1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/machine_expert")
    args = parser.parse_args()
    output = Path(args.output)
    sources = output / "original"
    sources.mkdir(parents=True, exist_ok=True)
    manifest = {"project": "LinuxCNC", "version": "2.9.4", "revision": REVISION,
                "license": "GFDL-1.1-or-later; no invariant sections or cover texts",
                "fetched_at": datetime.now(timezone.utc).isoformat(), "files": []}
    raw_docs = {}
    for name in FILES:
        url = f"https://raw.githubusercontent.com/LinuxCNC/linuxcnc/{REVISION}/docs/src/{name}"
        with urlopen(Request(url, headers={"User-Agent": "SlayerMachineExpert/0.1"}), timeout=60) as response:
            raw = response.read()
        path = sources / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        raw_docs[name] = raw.decode("utf-8")
        manifest["files"].append({"path": f"original/{name}", "url": url,
                                  "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    lines = raw_docs["gcode/g-code.adoc"].splitlines()
    chunks = []
    for anchor in SECTIONS:
        start = lines.index(f"[[gcode:{anchor}]]")
        end = next((i for i in range(start + 2, len(lines))
                    if lines[i].startswith("== ")), len(lines))
        # The following section's anchor is not evidence for this section.
        while end > start and (not lines[end - 1].strip() or lines[end - 1].startswith("[[")):
            end -= 1
        title = re.sub(r"\(\(\(.*", "", lines[start + 1][3:]).strip()
        chunks.append({"id": anchor, "title": "LinuxCNC 2.9.4 G-Codes", "section": title,
                       "text": "\n".join(lines[start:end]), "revision": REVISION,
                       "source_path": "docs/src/gcode/g-code.adoc", "line_start": start + 1, "line_end": end,
                       "url": f"https://github.com/LinuxCNC/linuxcnc/blob/{REVISION}/docs/src/gcode/g-code.adoc#L{start+1}-L{end}"})
    (output / "chunks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks), encoding="utf-8")
    manifest["corpus_sha256"] = hashlib.sha256((output / "chunks.jsonl").read_bytes()).hexdigest()
    manifest["extraction"] = "Untranslated complete selected G-code sections with original line numbers; no images or include expansion. Modified compilation by Slayer Machine Expert; originals and license retained."
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"chunks": len(chunks), "revision": REVISION, "output": str(output)}))


if __name__ == "__main__":
    main()
