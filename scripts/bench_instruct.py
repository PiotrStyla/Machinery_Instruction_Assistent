"""Deterministic instruction/format benchmark for Slayer instruct checkpoints.

Every check is machine-verifiable (parse, regex, count); no judge model and no
gold answers, so the number does not depend on anyone's taste. Suites mirror the
claims in the model card, which lets us verify them on our own runtime:

  format  — JSON only, one sentence, exactly three numbered items, word limit,
            English answer, no verbatim echo of the question;
  tool    — `<tool_call>` with valid JSON naming an offered tool, plus two
            no-tool cases where calling anything is a failure;
  loops   — greedy vs T=0.3/penalty 1.1: repetition rate and turn closure.

Usage:
    python scripts/bench_instruct.py --model-dir models/Slayer1B-1.09-Instruct \\
        --out reports/instruct-bench-1.09.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from slayer_chat import chat_once
from slayer_runtime import DTYPES, load_model

SYSTEM_PLAIN = "Jesteś zwięzłym asystentem. Odpowiadasz po polsku."

TOOLS = {
    "get_time": {
        "description": "Current time in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    "calculator": {
        "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
    "search_docs": {
        "description": "Search the CNC documentation for a term.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
}

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
POLISH_DIACRITICS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def tools_system() -> str:
    listing = "\n".join(json.dumps({"name": name, **spec}, ensure_ascii=False) for name, spec in TOOLS.items())
    return (
        "Jesteś asystentem dokumentacji technicznej. Odpowiadasz po polsku.\n\n"
        "# Tools\n\nYou can call functions to help with the user's request. Available functions:\n"
        f"<tools>\n{listing}\n</tools>\n\n"
        "To call a function, reply with one or more blocks:\n"
        '<tool_call>\n{"name": "<function name>", "arguments": {<arguments as JSON>}}\n</tool_call>'
    )


# ---------- checks ----------

def check_json(answer: str) -> bool:
    text = answer.strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and {"kod", "opis"} <= set(payload)


def check_one_sentence(answer: str) -> bool:
    text = answer.strip()
    if not text or "\n" in text:
        return False
    if re.match(r"^\s*(?:[-*•]|\d+[.)])\s", text):
        return False
    return len(re.findall(r"[.!?](?:\s|$)", text)) == 1


def check_three_items(answer: str) -> bool:
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    numbered = [line for line in lines if re.match(r"^[123][.)]\s+\S", line)]
    return len(lines) == 3 and len(numbered) == 3


def check_word_limit(answer: str, limit: int = 20) -> bool:
    words = [w for w in re.split(r"\s+", answer.strip()) if w]
    return 0 < len(words) <= limit


def check_english(answer: str) -> bool:
    return bool(re.search(r"\b(the|is|are|and)\b", answer.lower())) and not (set(answer) & POLISH_DIACRITICS)


def check_no_echo(answer: str, question: str) -> bool:
    core = re.sub(r"[^\wąćęłńóśźż ]+", " ", question.lower())
    core = " ".join(core.split())
    return core[:40] not in " ".join(answer.lower().split())


def parse_tool_call(answer: str) -> tuple[str, dict] | None:
    match = TOOL_CALL_RE.search(answer)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    name, arguments = payload.get("name"), payload.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return name, arguments


def repetition_ratio(answer: str, n: int = 3) -> float:
    words = answer.lower().split()
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


# ---------- cases ----------

FORMAT_CASES = [
    {
        "name": "json_only",
        "system": SYSTEM_PLAIN,
        "user": 'Zwróć wyłącznie obiekt JSON z polami "kod" i "opis" dla komendy G90 CNC, '
                "bez żadnego tekstu poza JSON.",
        "check": check_json,
    },
    {
        "name": "one_sentence",
        "system": SYSTEM_PLAIN,
        "user": "Jednym zdaniem: co to jest G53?",
        "check": check_one_sentence,
    },
    {
        "name": "three_numbered_items",
        "system": SYSTEM_PLAIN,
        "user": "Wypisz dokładnie trzy parametry skrawania. Każdy w nowej linii, "
                "z numeracją 1., 2., 3. Bez wstępu i podsumowania.",
        "check": check_three_items,
    },
    {
        "name": "word_limit",
        "system": SYSTEM_PLAIN,
        "user": "W maksymalnie 20 słowach wyjaśnij, co robi M3 w sterowaniu CNC.",
        "check": check_word_limit,
    },
    {
        "name": "english",
        "system": SYSTEM_PLAIN,
        "user": "Answer in English: what does G54 mean in CNC?",
        "check": check_english,
    },
    {
        "name": "no_echo",
        "system": SYSTEM_PLAIN,
        "user": "Nie powtarzaj pytania. Odpowiedz: jaka jest różnica między G90 a G91?",
        "check": None,   # sprawdzane osobno z pytaniem
    },
]

TOOL_CASES = [
    {"name": "tool_time", "user": "Która jest godzina w Krakowie?", "expect": "get_time"},
    {"name": "tool_calc", "user": "Ile to jest 17 * 23?", "expect": "calculator"},
    {"name": "tool_docs", "user": "Co dokumentacja mówi o komendzie G54?", "expect": "search_docs"},
    {"name": "tool_args_city", "user": "Sprawdź czas w Warszawie.", "expect": "get_time"},
    {"name": "no_tool_thanks", "user": "Dziękuję, to wszystko.", "expect": None},
    {"name": "no_tool_smalltalk", "user": "Napisz krótkie powitanie po polsku.", "expect": None},
]

LOOP_CASES = [
    {"name": "loop_wiki", "system": SYSTEM_PLAIN, "user": "Napisz akapit o toczeniu na CNC."},
    {"name": "loop_docs", "system": SYSTEM_PLAIN, "user": "Opisz ustawianie punktu zerowego przedmiotu."},
    {"name": "loop_open", "system": "Jesteś asystentem.", "user": "Kontynuuj: Sterowniki CNC dzielimy na"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic instruction benchmark for Slayer instruct checkpoints")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--loop-tokens", type=int, default=192,
                        help="budżet odpowiedzi dla zestawu 'loops'; mniejszy ucina odpowiedź w połowie zdania")
    parser.add_argument("--only", nargs="+", default=["format", "tool", "loops"],
                        choices=["format", "tool", "loops"])
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    model, _ = load_model(args.model_dir, args.dtype, args.device)
    tokenizer = Tokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))

    def ask(system: str, user: str, max_tokens: int, temperature: float = 0.3, penalty: float = 1.1) -> dict:
        return chat_once(
            model, tokenizer, system, user,
            max_new_tokens=max_tokens, temperature=temperature,
            top_p=0.9, penalty=penalty, seed=args.seed, use_cache=True,
        )

    rows: list[dict] = []

    # --- format
    format_pass = 0
    for case in (FORMAT_CASES if "format" in args.only else []):
        result = ask(case["system"], case["user"], args.max_new_tokens)
        answer = result["answer"]
        ok = check_no_echo(answer, case["user"]) if case["check"] is None else case["check"](answer)
        format_pass += bool(ok)
        rows.append({"suite": "format", "case": case["name"], "pass": bool(ok),
                     "answer": answer[:200], **{k: result[k] for k in ("seconds", "tokens_per_second", "stopped_on_eot")}})
        print(f"[format] {case['name']:<22} {'OK ' if ok else 'FAIL'} {answer[:80]!r}", flush=True)

    # --- tools
    tool_pass = 0
    for case in (TOOL_CASES if "tool" in args.only else []):
        result = ask(tools_system(), case["user"], max(args.max_new_tokens, 96))
        answer = result["answer"]
        call = parse_tool_call(answer)
        if case["expect"] is None:
            ok = call is None
        else:
            ok = call is not None and call[0] == case["expect"]
        tool_pass += bool(ok)
        rows.append({"suite": "tool", "case": case["name"], "pass": bool(ok),
                     "called": call[0] if call else None,
                     "answer": answer[:200], **{k: result[k] for k in ("seconds", "tokens_per_second", "stopped_on_eot")}})
        print(f"[tool]   {case['name']:<22} {'OK ' if ok else 'FAIL'} call={call[0] if call else None}", flush=True)

    # --- loops: greedy vs sampled
    loop_summary = {}
    for mode, temperature, penalty in ((("greedy", 0.0, 1.0), ("sampled", 0.3, 1.1)) if "loops" in args.only else ()):
        ratios, closed, speeds = [], 0, []
        for case in LOOP_CASES:
            result = ask(case["system"], case["user"], args.loop_tokens,
                         temperature=temperature, penalty=penalty)
            ratio = repetition_ratio(result["answer"])
            ratios.append(ratio)
            closed += bool(result["stopped_on_eot"])
            speeds.append(result["tokens_per_second"])
            rows.append({"suite": f"loops-{mode}", "case": case["name"],
                         "repetition_ratio": round(ratio, 3),
                         "closed": bool(result["stopped_on_eot"]),
                         "answer": result["answer"][:200],
                         **{k: result[k] for k in ("seconds", "tokens_per_second", "generated_tokens")}})
        loop_summary[mode] = {
            "mean_repetition_ratio": round(sum(ratios) / len(ratios), 3),
            "closed": f"{closed}/{len(LOOP_CASES)}",
            "mean_tokens_per_second": round(sum(speeds) / len(speeds), 2),
        }
        print(f"[loops]  {mode:<8} powtórzenia={loop_summary[mode]['mean_repetition_ratio']} "
              f"domknięte={loop_summary[mode]['closed']} {loop_summary[mode]['mean_tokens_per_second']} tok/s", flush=True)

    summary = {
        "model_dir": args.model_dir,
        "dtype": args.dtype,
        "only": args.only,
        "loop_tokens": args.loop_tokens,
        "loops": loop_summary,
    }
    if "format" in args.only:
        summary["format_pass"] = f"{format_pass}/{len(FORMAT_CASES)}"
        summary["format_score"] = round(100 * format_pass / len(FORMAT_CASES), 1)
    if "tool" in args.only:
        summary["tool_pass"] = f"{tool_pass}/{len(TOOL_CASES)}"
        summary["tool_score"] = round(100 * tool_pass / len(TOOL_CASES), 1)

    print("\n| zestaw | wynik |")
    print("|---|---|")
    if "format" in args.only:
        print(f"| format ({len(FORMAT_CASES)} sprawdzeń) | {summary['format_pass']} = {summary['format_score']}% |")
    if "tool" in args.only:
        print(f"| tool-call ({len(TOOL_CASES)} sprawdzeń) | {summary['tool_pass']} = {summary['tool_score']}% |")
    for mode, data in loop_summary.items():
        print(f"| pętle {mode} | powtórzenia {data['mean_repetition_ratio']}, domknięte {data['closed']}, "
              f"{data['mean_tokens_per_second']} tok/s |")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nzapisano: {args.out}")


if __name__ == "__main__":
    main()
