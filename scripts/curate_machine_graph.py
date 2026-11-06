"""Compile explicitly human-curated LinuxCNC relations with exact evidence checks.

This is structured graph input, not autonomous LLM extraction or benchmark labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from machine_expert import read_rows


def curated_facts():
    facts = []

    def add(section, source, relation, target, fact, quote):
        facts.append(dict(chunk_id=section, source=source, relation=relation, target=target,
                          fact=fact, evidence_quote=quote, extraction="human_curated"))

    for code, unit in (("G20", "inches"), ("G21", "millimeters")):
        add("g20-g21", code, "USES_LENGTH_UNIT", unit,
            f"{code} uses {unit} for length units.", f"'{code}' - to use {unit} for length units.")
    for code, plane in (("G17", "XY"), ("G18", "ZX"), ("G19", "YZ"),
                        ("G17.1", "UV"), ("G18.1", "WU"), ("G19.1", "VW")):
        add("g17-g19.1", code, "SELECTS_PLANE", plane,
            f"{code} selects the {plane} plane" + (", which is the default plane." if code == "G17" else "."),
            f"'{code}' - {plane}" + (" (default)" if code == "G17" else ""))
    for plane in ("UV", "WU", "VW"):
        add("g17-g19.1", plane, "DOES_NOT_SUPPORT", "arcs",
            f"The {plane} plane does not support arcs.", "The UV, WU and VW planes do not support arcs.")
    add("g4", "G4", "HAS_PARAMETER", "P seconds", "G4 P specifies a dwell time in seconds, including fractions.",
        "* 'P' - seconds to dwell (floating point)")
    add("g4", "G4", "LEAVES_UNCHANGED", "spindle coolant IO",
        "G4 does not affect spindle, coolant or any I/O.", "G4 does not affect spindle, coolant and any I/O.")
    add("g4", "G4", "INVALID_PARAMETER", "negative or missing P",
        "For G4 a negative or missing P number is an error.", "the P number is negative or not specified.")
    add("g53", "G53", "USES_FRAME", "machine coordinate system",
        "G53 moves in the machine coordinate system.", "G53 Move in Machine Coordinates")
    add("g53", "G53", "HAS_MODALITY", "non-modal",
        "G53 is not modal and must be programmed on each line.", "'G53' is not modal and\nmust be programmed on each line.")
    add("g53", "G53", "REQUIRES_ACTIVE", "G0 or G1",
        "G53 requires G0 or G1 to be active (either is sufficient).", "G53 is used without G0 or G1 being active")
    add("g53", "G53", "INCOMPATIBLE_WITH", "cutter compensation",
        "G53 is an error while cutter compensation is on.", "G53 is used while cutter compensation is on.")
    for code, number in zip(("G54", "G55", "G56", "G57", "G58", "G59", "G59.1", "G59.2", "G59.3"), range(1, 10)):
        add("g54-g59.3", code, "SELECTS_FRAME", f"coordinate system {number}",
            f"{code} selects coordinate system {number}.", f"select coordinate system {number}")
        add("g54-g59.3", code, "INCOMPATIBLE_WITH", "cutter compensation",
            f"Selecting coordinate system {code} while cutter compensation is on is an error.",
            "selecting a coordinate system is used while cutter compensation is on.")
    add("g61", "G61", "USES_PATH_MODE", "exact path",
        "G61 reaches every programmed point, slowing or stopping as needed; exactly co-linear moves do not stop.",
        "Moves will slow or stop as needed to reach every programmed point.")
    add("g61.1", "G61.1", "USES_PATH_MODE", "exact stop",
        "G61.1 stops at the end of each programmed segment.", "movement will stop at the end of each\n  programmed segment.")
    add("g90-g91", "G90", "USES_DISTANCE_MODE", "absolute positions",
        "G90 axis numbers usually represent positions in the active coordinate system, with documented exceptions.",
        "'G90' - absolute distance mode")
    add("g90-g91", "G91", "USES_DISTANCE_MODE", "incremental positions",
        "G91 axis numbers usually represent increments from the current coordinate.",
        "numbers usually represent increments from the current coordinate.")
    add("g90.1-g91.1", "G90.1", "USES_ARC_OFFSET_MODE", "absolute I J K offsets",
        "G90.1 uses absolute distance mode for I, J and K offsets; required pairs depend on the plane.",
        "'G90.1' - absolute distance mode for I, J & K offsets.")
    add("g90.1-g91.1", "G91.1", "USES_ARC_OFFSET_MODE", "incremental I J K offsets",
        "G91.1 uses incremental I, J and K offsets and restores their default behavior.",
        "'G91.1' - incremental distance mode for I, J & K offsets.")
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/machine_expert/chunks.jsonl")
    parser.add_argument("--output", default="data/machine_expert/graph_facts.jsonl")
    args = parser.parse_args()
    corpus = {row["id"]: row for row in read_rows(args.corpus)}
    facts = curated_facts()
    for index, fact in enumerate(facts, 1):
        chunk = corpus[fact["chunk_id"]]
        if fact["evidence_quote"] not in chunk["text"]:
            raise ValueError(f"source evidence missing for {fact['source']} {fact['relation']}: {fact['evidence_quote']!r}")
        fact.update(id=f"fact-{index:03}", revision=chunk["revision"], url=chunk["url"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(fact, ensure_ascii=False)+"\n" for fact in facts), encoding="utf-8")
    print(json.dumps({"facts": len(facts), "extraction": "human_curated", "sections": len({f['chunk_id'] for f in facts})}))


if __name__ == "__main__":
    main()
