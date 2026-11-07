"""Local, dependency-free training dashboard. No synthetic points or assumed improvement."""
from __future__ import annotations

import argparse
import html
import math
import os
import tempfile
import time
from pathlib import Path

from training_measurements import read_events

RELEASES = ("1.05", "1.06")


def metric_series(events, metric, split=None):
    fields = {"train_loss": ("step", "loss"), "valid_loss": ("evaluation", "mean_loss"),
              "exact_match": ("evaluation", "exact_match_rate"), "citation": ("evaluation", "citation_rate")}
    kind, field = fields[metric]
    result = []
    for row in events:
        if row.get("event") != kind or (split is not None and row.get("split") != split) or field not in row:
            continue
        step, value = row.get("global_step"), row[field]
        if not isinstance(step, (int, float)) or not isinstance(value, (int, float)) or not math.isfinite(step) or not math.isfinite(value):
            raise ValueError(f"non-numeric/non-finite {metric} measurement")
        if metric in ("exact_match", "citation"):
            if not 0 <= value <= 1:
                raise ValueError("rate outside 0..1")
            value *= 100
        result.append((float(step), float(value)))
    return sorted(result)


def experiment_state(events):
    starts = [i for i, row in enumerate(events) if row.get("event") == "experiment_start"]
    if not starts:
        return dict(status="waiting", step=0, total=0, phase="", progress=None)
    scope = events[starts[-1]:]
    total = scope[0].get("epochs", 0) * scope[0].get("train_examples", 0)
    terminals = [row for row in scope if row.get("event") in ("experiment_end", "experiment_failed")]
    status = ("complete" if terminals[-1]["event"] == "experiment_end" else "failed") if terminals else "running"
    step = max([row.get("global_step", 0) for row in scope if row.get("event") in ("step", "experiment_end")] or [0])
    phases = [row for row in scope if row.get("event") == "phase"]
    progress_rows = [row for row in scope if row.get("event") == "evaluation_progress"]
    return dict(status=status, step=step, total=total, phase=phases[-1].get("phase", "") if phases else "",
                progress=progress_rows[-1] if progress_rows else None)


def plot_svg(series, percent=False, planned_steps=0):
    width, height = 560, 250
    left, right, top, bottom = 58, 18, 18, 43
    pw, ph = width - left - right, height - top - bottom
    points = [point for values in series.values() for point in values]
    if not points:
        return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Brak pomiarów">'
                f'<text x="{width/2}" y="{height/2}" text-anchor="middle">'
                'Brak pomiarów — nie zastępujemy ich zerami</text></svg>')
    xmax = max([point[0] for point in points] + [float(planned_steps), 1.0])
    ymin = min([point[1] for point in points] + [0.0]) if not percent else 0.0
    ymax = 100.0 if percent else max([point[1] for point in points] + [0.1]) * 1.12
    if ymax <= ymin:
        ymax = ymin + 1
    def xy(step, value):
        return left + pw * step / xmax, top + ph * (1 - (value - ymin) / (ymax - ymin))
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Pomiary według kroku treningu">']
    for index in range(5):
        value = ymin + (ymax - ymin) * index / 4
        y = xy(0, value)[1]
        label = f"{value:.0f}%" if percent else f"{value:.2f}"
        svg += [f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" class="grid"/>',
                f'<text x="{left-9}" y="{y+4:.2f}" text-anchor="end">{label}</text>']
    for index in range(5):
        step = xmax * index / 4
        x = xy(step, 0)[0]
        svg.append(f'<text x="{x:.2f}" y="{height-22}" text-anchor="middle">{step:g}</text>')
    svg.append(f'<text x="{left+pw/2}" y="{height-3}" text-anchor="middle">Krok aktualizacji LoRA</text>')
    for index, release in enumerate(RELEASES):
        values = series.get(release, [])
        color = f"series-{index}"
        coordinates = " ".join(f"{xy(x,y)[0]:.2f},{xy(x,y)[1]:.2f}" for x, y in values)
        if len(values) > 1:
            svg.append(f'<polyline points="{coordinates}" class="trace {color}"/>')
        for step, value in values:
            x, y = xy(step, value)
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" class="point {color}"><title>Slayer {release}; krok {step:g}; {value:.4f}</title></circle>')
    return "".join(svg) + "</svg>"


def answer_table(root, release):
    before, errors_a = read_events(root / release / "baseline-test.jsonl", require_event=False)
    after, errors_b = read_events(root / release / "final-test.jsonl", require_event=False)
    by_id = {row["id"]: row for row in after}
    rows = []
    for row in before:
        final = by_id.get(row["id"])
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in
                    (row["question"], row.get("answer", ""), final.get("answer", "") if final else "Jeszcze nie zmierzono")) + "</tr>")
    if not rows:
        return "", errors_a + errors_b
    return (f'<details><summary>Slayer {release}: rzeczywiste odpowiedzi testowe przed i po</summary>'
            '<div class="table-scroll"><table><thead><tr><th>Pytanie</th><th>Przed</th><th>Po, wybrany adapter</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div></details>'), errors_a + errors_b


def render_dashboard(root, refresh=None):
    root = Path(root)
    all_events, errors, statuses = {}, [], []
    names = {"waiting": "Oczekiwanie", "running": "W toku", "complete": "Zakończony", "failed": "Błąd / przerwany"}
    planned = 0
    for release in RELEASES:
        events, issues = read_events(root / release / "metrics.jsonl")
        all_events[release] = events
        errors.extend(f"{release}: {issue}" for issue in issues)
        state = experiment_state(events)
        planned = max(planned, state["total"])
        phase = state["phase"].replace("_", " ")
        progress = state["progress"]
        if progress and state["phase"] == f"{progress['phase']}_{progress['split']}":
            phase += f" ({progress['done']}/{progress['total']})"
        status = names[state["status"]]
        log_link = (f'<a href="{release}/metrics.jsonl">Dziennik JSONL</a>'
                    if (root / release / "metrics.jsonl").exists() else "Jeszcze brak dziennika")
        statuses.append(f'<tr><th>Slayer {release}</th><td>{status}</td><td>{state["step"]} / {state["total"] or "—"}</td><td>{html.escape(phase)}</td>'
                        f'<td>{log_link}</td></tr>')
    panels = []
    for title, metric, split, percent, note in (
        ("Strata podczas uczenia", "train_loss", None, False, "Każdy punkt to rzeczywisty krok. Próbki mają różną trudność; niżej nie oznacza automatycznie lepszej odpowiedzi."),
        ("Strata na walidacji", "valid_loss", "valid", False, "Te same pytania walidacyjne przed treningiem i po epokach. Służą do wyboru adaptera, nie są testem końcowym."),
        ("Ścisła zgodność odpowiedzi: walidacja", "exact_match", "valid", True, "Porównanie całej odpowiedzi z wzorcem, po normalizacji wielkości liter i białych znaków. Poprawne parafrazy mogą dostać 0%."),
        ("Ścisła zgodność odpowiedzi: test", "exact_match", "test", True, "Tylko pomiar przed treningiem i po nim. Test nie wybiera checkpointu. To nie jest ogólna miara prawdziwości."),
        ("Identyfikatory cytowanych źródeł: test", "citation", "test", True, "Poprawny identyfikator źródła nie potwierdza poprawności treści odpowiedzi."),
    ):
        series = {}
        for release, events in all_events.items():
            try:
                series[release] = metric_series(events, metric, split)
            except ValueError as error:
                errors.append(f"{release}: {error}")
                series[release] = []
        panels.append(f'<section class="plot"><h2>{title}</h2>{plot_svg(series, percent, planned)}<p>{note}</p></section>')
    answers = []
    for release in RELEASES:
        table, issues = answer_table(root, release)
        answers.append(table)
        errors.extend(f"{release}: {issue}" for issue in issues)
    alert = ('<aside class="error"><strong>Błąd danych pomiarowych</strong><ul>' + "".join(f'<li>{html.escape(error)}</li>' for error in errors) + '</ul></aside>') if errors else ""
    meta = f'<meta http-equiv="refresh" content="{refresh:g}">' if refresh else ""
    return '''<!doctype html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">''' + meta + '''
<title>Slayer1B — trening na dokumentacji</title><style>
:root{color-scheme:light dark;--fg:#20313a;--muted:#55636b;--border:#cad3d9;--surface:#fafcfd;--accent:#245b80;--error:#a5202a;--c1:#1465bb;--c2:#b5571a}
@media(prefers-color-scheme:dark){:root{--fg:#e8edf0;--muted:#b2bfc8;--border:#465460;--surface:#192228;--accent:#9ec5e0;--error:#ffafb4;--c1:#86baff;--c2:#ffbc85}}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--fg);font:15px/1.5 "Segoe UI",sans-serif}main{max-width:1220px;margin:auto;padding:28px 24px 48px}h1{font-size:30px;line-height:1.2;margin:0 0 12px;font-weight:650}h2{font-size:19px;font-weight:600;margin:0 0 12px}p{max-width:82ch;margin:8px 0;color:var(--muted)}a{color:var(--accent)}a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}.lead{margin-bottom:20px}.legend{display:flex;gap:24px;margin:18px 0}.legend span:before{content:"";display:inline-block;width:22px;height:3px;margin-right:8px;vertical-align:middle;background:currentColor}.series-0,.legend .first{color:var(--c1)}.series-1,.legend .second{color:var(--c2)}.panels{display:grid;grid-template-columns:1fr 1fr;gap:28px 30px;margin-top:28px}.plot{min-width:0}.plot p{font-size:13px}.plot svg{width:100%;height:auto;display:block}svg text{fill:var(--muted);font:14px "Segoe UI",sans-serif}.grid{stroke:var(--border);stroke-width:1}.trace{fill:none;stroke:currentColor;stroke-width:2}.point{fill:currentColor;stroke:var(--surface);stroke-width:1}.series-1.trace{stroke-dasharray:5 3}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}th{font-weight:600}td{font-variant-numeric:tabular-nums}.table-scroll{overflow-x:auto}.table-scroll td{min-width:230px;white-space:pre-wrap}details{margin:20px 0}summary{cursor:pointer;font-weight:600}.error{border-left:4px solid var(--error);padding:12px 18px;color:var(--error);margin:18px 0}.fine{font-size:13px;margin-top:24px}
@media(max-width:900px){.panels{grid-template-columns:1fr}main{padding:20px 12px}h1{font-size:25px}}
</style></head><body><main><h1>Slayer1B. Trening na dokumentacji</h1>
<p class="lead">Porównujemy 1.05 i 1.06, nie zakładamy poprawy z góry. Loss, zgodność ze wzorcem i cytowanie to różne pomiary. Odpowiedzi pozostają eksperymentalne, niezatwierdzone do działań na maszynie.</p>
''' + alert + '<div class="table-scroll"><table><thead><tr><th>Model</th><th>Stan eksperymentu</th><th>Kroki uczenia</th><th>Etap</th><th>Dane</th></tr></thead><tbody>' + "".join(statuses) + '''</tbody></table></div>
<div class="legend"><span class="first">Slayer 1.05</span><span class="second">Slayer 1.06</span></div><div class="panels">''' + "".join(panels) + '</div>' + "".join(answers) + '''
<p class="fine">Kropki oznaczają pomiary, a łączące je odcinki nie są dodatkowymi obserwacjami. Brak wartości nie jest zerem. Zbiór ma odrębne pytania, lecz współdzielone fakty i dokumenty; nie dowodzi generalizacji na inne maszyny. Oryginalne logi i odpowiedzi są zapisane lokalnie.</p>
</main></body></html>'''


def write_dashboard(root, output, refresh=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = render_dashboard(root, refresh)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    try:
        for attempt in range(8):
            try:
                os.replace(temporary, output)
                break
            except PermissionError as error:
                # Windows readers/antivirus may briefly deny replacement of an open HTML snapshot.
                # Preserve the previous complete snapshot; permanent failures still propagate.
                if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 7:
                    raise
                time.sleep(min(0.05 * (attempt + 1), 0.5))
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports/machine_training"))
    parser.add_argument("--output", type=Path, default=Path("reports/machine_training/dashboard.html"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    write_dashboard(args.root, args.output, args.interval if args.watch else None)
    print(f"Dashboard ready: {args.output.resolve()}", flush=True)
    while args.watch:
        time.sleep(args.interval)
        write_dashboard(args.root, args.output, args.interval)


if __name__ == "__main__":
    main()
