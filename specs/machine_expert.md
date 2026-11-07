# SLAYER Machine Expert — offline documentation experiment

## Scope and decision

The original baseline experiment searches selected LinuxCNC 2.9.4 G-code sections. It does not diagnose a real machine, execute G-code, provide a repair procedure, interpret images, or claim coverage of a complete CNC manual. A separate real Graphiti/Kuzu comparison is available; see [Graphiti experiment](graphiti_experiment.md). Those retrieval experiments used no further SFT. The subsequent controlled dual-release training and opt-in unverified generation are described separately in [grounded SFT](grounded_sft.md).

- Source language: original English. Questions and reference answers: Polish, hand-authored for this project.
- Retrieval: SQLite FTS5/BM25 plus an explicit Polish-English domain glossary and exact code filtering. No embeddings, translation API, model, GPU or network is needed for search.
- G-code questions prioritize section identities, preserving G90 versus G90.1 and exact decimal codes. Unknown code/alarm queries return no evidence rather than a numeric near-match. Named Haas/FANUC/Siemens/Heidenhain queries are outside scope.
- These rules do not establish a universal abstention/safety classifier. An unrelated natural-language question can still match shared vocabulary.
- Search returns original source excerpts and commit-pinned links with section and source line numbers, not fabricated PDF pages.
- Model outputs are experimental `generated_unverified` completions, never validated manufacturer instructions. Source association is not proof that the generated answer is entailed by the source.

## Source and reuse conditions

Pinned upstream: [LinuxCNC v2.9.4](https://github.com/LinuxCNC/linuxcnc/commit/b8ccac8f835c0610233fdf799255c63f2b9738ac).

Original files are retained under `data/machine_expert/original/`:

- [G-code documentation](https://github.com/LinuxCNC/linuxcnc/blob/b8ccac8f835c0610233fdf799255c63f2b9738ac/docs/src/gcode/g-code.adoc)
- [Attribution and permission](https://github.com/LinuxCNC/linuxcnc/blob/b8ccac8f835c0610233fdf799255c63f2b9738ac/docs/src/common/overleaf.adoc)
- [Legal section including GNU FDL 1.1](https://github.com/LinuxCNC/linuxcnc/blob/b8ccac8f835c0610233fdf799255c63f2b9738ac/docs/src/common/gpld-copyright.adoc)

LinuxCNC documentation grants GNU FDL 1.1-or-later permission without invariant sections or front/back cover texts. Original attribution and license are retained verbatim; do not strip them when distributing the corpus. The derived corpus is a modified compilation of selected sections, not a new manufacturer manual. Original content remains English, unchanged within each extracted line span. The manifest records SHA256, URLs, revision, acquisition time and extraction method. Illustrations and include expansion are not supported. Review applicable license obligations before wider redistribution; public accessibility alone is not a reuse license for other manuals such as Haas.

## Run from project root (PowerShell)

One-time network acquisition and local index:

```powershell
.venv\Scripts\python.exe scripts\fetch_machine_docs.py
.venv\Scripts\python.exe scripts\machine_expert.py index
```

All following commands operate offline:

```powershell
.venv\Scripts\python.exe scripts\machine_expert.py search --question "W jakich jednostkach długości działa G21?"
.venv\Scripts\python.exe scripts\machine_expert.py search --question "Maszyna zgłasza alarm E-247. Co oznacza?"
.venv\Scripts\python.exe scripts\machine_expert.py benchmark
```

Default artifacts:

- `data/machine_expert/chunks.jsonl`: nine complete selected sections.
- `data/machine_expert/manifest.json`: upstream source provenance and checksums.
- `data/machine_expert/questions.jsonl`: 30 hand-authored Polish questions (24 supported, six out of scope).
- `data/machine_expert/index.sqlite`: rebuildable local FTS index.
- `reports/machine_expert_retrieval.json`: section hit rates, unknown-query abstentions and every query result.

This is a narrow development set, not an independent industrial benchmark. Hit@1 and Hit@3 mean that an expected section was found, not that an answer is correct. Six successful unknown-query rejections would not prove general safe refusal.

## Grounding experiment

```powershell
.venv\Scripts\python.exe scripts\evaluate_machine_expert.py --model-dir models\Slayer1B-1.05 --output reports\machine_expert_1.05_smoke.jsonl --limit 4 --max-new-tokens 32
.venv\Scripts\python.exe scripts\evaluate_machine_expert.py --model-dir models\Slayer1B-1.06 --output reports\machine_expert_1.06_smoke.jsonl --limit 4 --max-new-tokens 32
```

Each question is exercised in two modes:

- `oracle`: reference evidence IDs from the benchmark, bypassing retrieval. This measures reading/generation with ideal evidence.
- `retrieved`: top result from offline search, with no access to expected evidence IDs for selection.

Unknown oracle questions have no reference evidence by construction; their refusal is an application guard, not proof of model abstention. No-evidence retrieved cases also bypass generation. Raw generation is retained with references and source metadata for manual assessment; the evaluator does not present token overlap as factual accuracy.

The input budget defaults to 1024 tokens with 48 output tokens. All 24 supported reference prompts fit this budget (largest measured: 825 tokens with the 1.05 tokenizer). Complete evidence and warnings are never silently trimmed: over-budget examples are recorded as `context_budget_exceeded`. To evaluate all 30 questions, remove `--limit 4`; this takes substantially longer on CPU. Set `--max-input-tokens` only within the model's context capacity. No training adapter or experimental INT8 path is used.

## Observed results

- Retrieval: expected section at rank 1 for 24/24 supported questions; no hits for 6/6 unsupported questions. This development set was used to debug code matching, so these are not held-out industrial accuracy figures.
- Real BF16 CPU smoke: first four questions, two evidence modes, both Slayer 1.05 and 1.06, 32 new tokens per generation. There are 12 generated records and four no-evidence application-guard records.
- Both releases incorrectly call G53 modal even when the complete source explicitly says it is not modal. Supplying the correct source alone does not ensure a trustworthy answer.
- 1.06 correctly distinguishes G20/inches from G21/millimeters, but continues invented dialogue. 1.05 does not give that distinction in this run.
- Both copy part of the English G90 explanation; the 32-token cap prevents a complete G91 comparison. Do not count this as an unrestricted-length quality benchmark.
- All 10 regression tests passed, including code identity, punctuation after unknown alarms, wrong-controller scope and Windows SQLite handle cleanup.

Raw evidence: `reports/machine_expert_1.05_smoke.jsonl`, `reports/machine_expert_1.06_smoke.jsonl`, `reports/machine_expert_retrieval.json`. Manual interpretation: `reports/machine_expert_review.json`. Context refusal smoke: `reports/machine_expert_context_guard.jsonl`.

Decision from the original grounding smoke: cited excerpts remain the useful interface, while Slayer-generated explanations are unverified. Graphiti has now been added as an independent retrieval and relationship experiment, not a fix for the observed G53 contradiction. Its installation, curated-versus-automatic ingestion distinction, measured comparison and backend limitations are documented in [Graphiti experiment](graphiti_experiment.md).

## Decision routing and evidence gate

`scripts/machine_router.py` is the routed entry point. The original retrieval and generation evaluators remain independent baseline experiments, not aliases for the new policy.

- `lexical`: direct questions about a documented code or one source topic. Runs SQLite search, with explicit topic/code subqueries where required.
- `graph`: relationship questions. Runs actual Graphiti search, retaining the first three distinct semantic source sections, plus up to two native topology hops from identified entities.
- `hybrid`: multi-section comparisons. Runs both backends and retains each backend's results and errors.
- `refuse`: unknown codes, different named controllers, machine-specific data outside this corpus, explicit bypass/actuation requests, or insufficiently specified questions. Ambiguous questions receive `clarification_required`; no retrieval backend is called.

Routing uses documentation code families and a small explicit domain vocabulary. It does not load reference answers or routing-case expected IDs during a normal question. It is not a general-purpose safety classifier.

The evidence gate checks the corpus SHA256 and revision, exact retrieved source content and citation metadata, required-section coverage, and graph quote/source provenance. Backend failure stays visible as `backend_incomplete`; it is not hidden by silently changing routes. Missing sections produce `partial_evidence`, and inconsistent provenance produces `source_integrity_failed`.

**`target_coverage_complete` means only that all explicitly identified source topics were found.** It does not prove semantic entailment, complete warning coverage, applicability to a physical machine, or the truth of a generated answer. The gate retains `generation_allowed: false` and approves source excerpts only. An explicit `ask --model-dir ...` may now add an independently labelled `experimental_answer`; it does not override this lack of validated technical approval. The router itself does not repair the previously observed G53 contradiction; new training outcomes are measured separately.

Run from the project root:

```powershell
.venv\Scripts\python.exe scripts\machine_router.py ask --question "Czy G53 jest modalny?"
.venv-graphiti\Scripts\python.exe scripts\machine_router.py ask --question "Z jakimi kodami G53 ma wspólne ograniczenia dotyczące kompensacji promienia frezu?"
.venv-graphiti\Scripts\python.exe scripts\machine_router.py ask --question "Porównaj G53 i G54."
.venv\Scripts\python.exe scripts\machine_router.py ask --question "Co oznacza E-247?" --policy laya
```

The last command is refused before initializing the classifier, even when `--policy laya` was requested. Graph routes need the already installed optional Graphiti environment and curated database described in [Graphiti experiment](graphiti_experiment.md).

### Actual local classifier comparison

The optional `scripts/routing_classifier.py` uses the installed `laya==0.3.4` Python API and cached `convaiinnovations/laya-multilingual`, snapshot `052592a15d198d9ad47da779604259b10b47b7aa`. It performs real CPU model inference, not a second heuristic advertised as a model. The `routing` optional dependencies in `pyproject.toml` record the tested versions; no installation or model download is needed on this workstation.

Only the multilingual cached snapshot is loaded, offline. Input exceeding the model budget is rejected instead of silently truncated. Scores are recorded but are **not calibrated for CNC routing**. Hard scope guards cannot be overridden by a model proposal. No confidence threshold has been validated.

```powershell
.venv-graphiti\Scripts\python.exe scripts\machine_router.py ask --question "Jaką jednostkę długości wybiera G21?" --policy laya --output reports\machine_expert_router_classifier_lookup.json
.venv-graphiti\Scripts\python.exe scripts\machine_router.py benchmark --classifier laya --execute
```

The second command compares both selectors and executes the **default rule-selected** routes against real backends. `--policy laya --execute` instead executes classifier-selected routes after hard scope guards. Classification errors are recorded, not disguised as successful model decisions.

`data/machine_expert/router_cases.jsonl` contains 24 hand-authored development cases, six per route. The rules and vocabulary were developed against this set; results are **not held-out industrial accuracy**. Final observed results:

| Selector | Correct route |
|---|---:|
| Explicit rules, default | 24/24 |
| Local Laya classifier, raw | 7/24 |
| Laya proposal with hard scope guards | 12/24 |

Laya selected lexical search for all relationship and hybrid examples. It is therefore available for experiments, **not enabled by default**. Median selector-only time was 0.462 ms for rules versus 435.208 ms for the warmed classifier; classifier loading separately took 43.514 s. These are single-workstation development measurements, excluding retrieval from selector timing.

All 18 source-bearing rule-routed cases retrieved their expected sections and passed the provenance/topic-coverage gate. Five cases were refused and one requested clarification, with no backend executed. This establishes neither answer correctness nor general refusal reliability.

Artifacts: `reports/machine_expert_router.json` stores every decision, actual classifier scores, backend execution, source excerpts, graph facts, missing sections, errors and timing. Separate actual CLI smokes are recorded in `machine_expert_router_lookup.json`, `machine_expert_router_refusal.json` and `machine_expert_router_classifier_lookup.json` under `reports/`.

Verification: 26/26 tests passed in `.venv-graphiti`, including decimal-code boundaries, time-versus-length units, guard precedence, partial evidence, altered corpus/citations, unsupported graph quotes and backend failures. Real CLI runs exercised direct lookup in the base environment, guarded refusal, all graph/hybrid development cases and local classifier inference.
