# Graphiti comparison — SLAYER Machine Expert

## What is real, and what is curated

This experiment uses the actual `graphiti-core==0.31.2` package, its Kuzu graph driver, Graphiti entity/episode/edge persistence, and `Graphiti.search()` hybrid reciprocal-rank fusion. It is not a SQLite table renamed as a graph and does not replace Graphiti with a custom graph library.

The full nine-section graph is **human-curated structured input**: 42 explicit relations over 56 entities. Each relation has an exact original source quote, source section ID, pinned URL and extraction label. `scripts/curate_machine_graph.py` checks that each evidence quote occurs in the retained source; this is an integrity check, not an automatic semantic proof. No benchmark reference answers are used to build graph facts. Original English episodes and GNU FDL source/attribution files remain under `data/machine_expert/`.

Automatic extraction is a **separate experiment**, using real local Qwen generation through Graphiti `add_episode()` on the short G20/G21 section. It does not silently fall back to curated relations when generation fails. Its database, report and raw model-call log are separate. Qwen is an extraction dependency, not Slayer and not evidence that Slayer became more capable.

## Local prerequisites and dependency isolation

Installed in `.venv-graphiti`, without changing the inference `.venv`:

- Graphiti 0.31.2 and Kuzu 0.11.3.
- PyTorch 2.14.0 and Transformers 4.57.6, pinned to avoid loading incompatible inherited audio/vision native libraries with a newer ML stack.
- Cached `intfloat/multilingual-e5-small`: actual 384-dimensional normalized CPU embeddings. Both queries and facts use the E5 symmetric `query:` prefix convention.
- Cached `Qwen/Qwen2.5-1.5B-Instruct`: BF16 CPU, lazy-loaded only for automatic extraction. Structured outputs are parsed and validated against Graphiti's requested Pydantic schema; no invented JSON repairs. Generation is bounded to 384 output tokens, 4096 input tokens and 20 model calls per experiment.
- Default search is Graphiti RRF; the supplied cosine reranker is local and real but is not used by RRF. There is no cloud reranker/API.

To construct a fresh optional environment, use the existing base Python and install the optional dependency versions from `pyproject.toml` (or explicitly):

```powershell
uv venv --system-site-packages --python .venv\Scripts\python.exe .venv-graphiti
uv pip install --python .venv-graphiti\Scripts\python.exe "graphiti-core[kuzu]==0.31.2" "torch==2.14.0" "transformers==4.57.6"
```

Do not recreate the environment already installed here. Model snapshots must already be cached: runtime sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and disables Graphiti telemetry. The initial package installation and one-time Kuzu FTS extension installation use the network; normal model loading, search, comparison and traversal use local files only.

**Backend limitation:** Kuzu is deprecated by Graphiti and its upstream is no longer maintained. It avoids a Docker/server prerequisite for this local experiment; choose a maintained Graphiti backend for production. See [Graphiti requirements](https://github.com/getzep/graphiti#installation).

## Rebuild structured corpus and graph

Run from the project root:

```powershell
.venv\Scripts\python.exe scripts\curate_machine_graph.py
.venv-graphiti\Scripts\python.exe scripts\graphiti_machine.py ingest --install-fts
```

The graph already exists at `data/machine_expert/graphiti_curated.kuzu`. Ingestion refuses to overwrite an existing graph; use `--db` pointing to a new path for a new experiment. There is no silent delete/reset command.

The graph stores statements about the **LinuxCNC 2.9.4 documentation revision**, not the current state of a physical machine. The release timestamp is the reference time. This experiment does not establish temporal change-history performance.

Kuzu integration details: structured import calls Graphiti's `add_nodes_and_edges_bulk`; FTS indexes are created through Kuzu's real Graphiti maintenance operations after population. In 0.31.2 the driver's top-level index method is a no-op. `with_database(group)` explicitly binds the driver for the `add_episode` path, avoiding an unset `_database` in the stock Kuzu constructor. Windows connections are explicitly closed after each CLI operation.

## Search and graph traversal

```powershell
.venv-graphiti\Scripts\python.exe scripts\graphiti_machine.py search --question "Czy G53 jest poleceniem modalnym?"
.venv-graphiti\Scripts\python.exe scripts\graphiti_machine.py search --question "Co oznacza alarm E-247?" --guarded
.venv-graphiti\Scripts\python.exe scripts\graphiti_machine.py neighbors --entity G53 --hops 2 --output reports\machine_expert_graphiti_neighbors.json
```

Raw Graphiti search returns nearest facts; proximity is not proof that a question is answerable. `--guarded` separately rejects explicit unknown codes and named out-of-scope controllers. It does not calibrate general semantic abstention. These policy checks are not Graphiti's learned ability.

Two-hop traversal executes graph queries against the persisted Graphiti topology. From G53 it reaches the shared `cutter compensation` concept and the coordinate-system selection commands G54–G59.3 in a different source section. The result carries each relation's source URL and exact quote. A shared constraint is not proof of a causal machine fault.

For automatic selection between direct lookup, Graphiti relationship traversal, hybrid search and refusal, use the [decision router](machine_expert.md#decision-routing-and-evidence-gate). It retains source provenance and reports incomplete coverage separately; it does not enable Slayer-generated technical advice.

## Reproduce comparison

```powershell
.venv-graphiti\Scripts\python.exe scripts\graphiti_machine.py compare --relationships data\machine_expert\relationship_questions.jsonl
```

Both retrievers receive the identical 30-question development set plus four relationship diagnostics. Baseline gets its existing domain glossary/exact-code handling. Graphiti raw gets the original question, English curated fact embeddings and its native hybrid search. Graphiti guarded adds explicit code/controller scope checks. Results are collapsed to distinct source sections so top-three comparisons are at the same granularity; complete evidence coverage is also reported for multi-section questions.

Default artifacts:

- `reports/machine_expert_graphiti_ingest.json`: package/model provenance, graph counts, hashes and import duration.
- `reports/machine_expert_graphiti_compare.json`: every question, section ranks, native facts, abstentions and timing.
- `reports/machine_expert_graphiti_neighbors.json`: actual two-hop traversal.
- `reports/machine_expert_graphiti_auto.json`: separate automatic extraction outcome.
- `reports/linuxcnc_2_9_4_auto_llm.jsonl`: actual local model responses and schema-validation outcomes when model calls occur.

The benchmark is a small hand-authored development set, mostly code-identity queries. Curated relation design and baseline vocabulary both incorporate domain knowledge. Do not present the scores as independent generalization, manufacturer approval or an industrial safety certificate.

## Automatic extraction probe

```powershell
.venv-graphiti\Scripts\python.exe scripts\graphiti_machine.py extract --section g20-g21 --db data\machine_expert\graphiti_auto_probe.kuzu
```

Use a new database path for each probe; the command does not overwrite earlier evidence. Failures produce a report and nonzero exit status, never fabricated nodes or a hidden manual fallback. Automatic probe output is unverified, not included in the curated graph benchmark. Read the report's model-call count to distinguish infrastructure failure before generation from a model/schema failure.

All outputs remain informational. No machine connection, G-code execution or generated repair recommendation is implemented.

## Measured curated-graph comparison

Original 30-question development set:

| Measure | Baseline SQLite | Raw Graphiti | Graphiti with scope guard |
|---|---:|---:|---:|
| Expected section at rank 1 (24 supported) | 24 | 22 | 22 |
| Expected section in top 3 (24 supported) | 24 | 23 | 23 |
| No hits for unknown questions (6) | 6 | 0 | 4 |

Four relationship diagnostics: Graphiti chose an expected section first for 4/4, versus baseline 3/4. Both included all required source sections within the top three for 3/4; both missed complete evidence on the uncoded distance-mode versus arc-offset question. There is no demonstrated improvement in complete multi-source coverage on this small set.

Across all 34 queries, observed median retrieval time was 2.06 ms for the baseline and 72.62 ms for Graphiti, excluding model/database initialization. Curated import persisted 42 facts and took 7.182 seconds excluding interpreter startup. These are local observations, not a hardware-independent benchmark.

Actual G53 two-hop traversal returned four direct relations plus nine relations sharing the cutter-compensation concept with the G54–G59.3 section. This provides explicit, cited relationship navigation without relying on Slayer to invent the connection. It does not establish causality or a repair procedure.

Decision from this run: retain the precise lexical/code baseline for direct lookup and use Graphiti as an additional relationship exploration path, not as a replacement or an automatic abstention mechanism. Unknown-query raw nearest-neighbor results must not be presented as documented answers.

## Observed automatic extraction outcome

The separate real `add_episode()` probe on the eight-line G20/G21 section failed after two local Qwen calls (608.853 seconds excluding process startup):

1. Entity extraction: schema-valid response naming G20 and G21, 1223 input tokens / 72 output tokens.
2. Edge extraction: 2150 input tokens / 384 output tokens. The model copied JSON Schema definitions rather than returning extracted relationships; generation hit the explicit output cap and left an incomplete fenced object. Parsing failed and the CLI exited nonzero.

No automatic relations were claimed or substituted into the curated graph. The output cap is a confounder, so this single bounded probe is not proof that automatic extraction can never work. It does show that this local model/prompt/configuration is not a reliable extractor as configured. Raw output and failure details are retained in `reports/linuxcnc_2_9_4_auto_llm.jsonl` and `reports/machine_expert_graphiti_auto.json`.

The first infrastructure attempt hit Kuzu's unset `_database` before any LLM call; the public `with_database(group)` binding fixed that prerequisite. The reported model outcome above is from the later attempt that genuinely ran inference, not the infrastructure failure.

Verification of the initial Graphiti experiment: isolated Graphiti environment passed 12/12 unit tests; the original inference environment passed 10 tests and skipped the two optional Graphiti tests. Real smoke runs covered graph population, all 34 retrieval queries, cited two-hop traversal, the guarded unknown-alarm CLI and automatic extraction failure reporting. Numerical review split by original versus relationship suites: `reports/machine_expert_graphiti_review.json`. The later router verification is documented separately in the linked decision-router section.
