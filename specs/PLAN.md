# Slayer1B inference

1. ✅ **Research** — Confirmed Slayer1B-1.05 and 1.06 architecture, weights, tokenizer, and loaders.
2. ✅ **Runtime** — Implemented a PyTorch runtime with low-memory loading and an architecture-preserving KV cache.
3. ✅ **Benchmark** — Cached decoding matches reference logits for both releases; CPU generation and memory measurements completed.
4. ✅ **Memory path** — BF16/FP16 loading is implemented; INT8/INT4 is intentionally left out until a quality baseline is defined.

Current direction: SLAYER Machine Expert, offline source retrieval plus a separately measured grounded SFT experiment for Slayer1B-1.05 and 1.06. Model output is explicitly opt-in and unverified.
9. ✅ **Evaluation** — Fixed 32-prompt Polish continuation suite and 2048-token external Polish proxy perplexity measurement completed for both releases.
10. **Next** — Use `scripts/machine_router.py` for routed evidence lookup; `--model-dir` explicitly adds an unverified Slayer completion. Rules remain the default selector. Validate routing and the new grounded SFT on independently authored, broader documentation before making general quality claims.

11. ✅ **SFT data** — 48 Polish instruction/response records for training and 12 held-out records for validation; JSONL schema validated with the Slayer tokenizer.
12. ✅ **SFT full** — 48-step LoRA run completed in two resumable 24-step phases; output: `adapters/slayer1b-1.05-polish-sft-full.pt`.
13. ✅ **SFT validation** — Final adapter loaded successfully; 12/12 held-out greedy outputs matched the base model, with 3 dash-only outputs and 8 marker-loop outputs unchanged.
14. ✅ **Training measurements** — SFT now writes append-only JSONL events for run start, every step, epoch, validation, and run end, including loss, learning rate, gradient norm, elapsed time, and RSS memory.
15. ✅ **Inference speed experiment** — Added `--max-context-tokens` and CPU dynamic INT8 benchmarking. BF16 remains preferred: 8.86 tok/s and ~2.18 GB RSS on the short prompt versus INT8 5.55 tok/s and ~6.53 GB RSS; a 301-token prompt dropped from 29.23 s to 6.20 s prefill when capped at 64 tokens.

16. **Machine Expert** — Completed a pinned LinuxCNC 2.9.4 excerpt corpus, local SQLite FTS5 search, 30-question Polish retrieval benchmark, and oracle/retrieved grounding smoke on both releases. Search found 24/24 supported sections and rejected 6/6 out-of-scope queries on the development set; both models contradicted the supplied G53 modality evidence. Commands, source/license attribution and limitations: `specs/machine_expert.md`.

17. **Graphiti experiment** — Installed real Graphiti 0.31.2 with local Kuzu and E5 embeddings in `.venv-graphiti`. Persisted 56 entities / 42 source-backed curated relations. Compared identical 30 questions plus four relationship diagnostics: original-set top-3 section hits were 24/24 baseline versus 23/24 Graphiti; native Graphiti did not abstain on six unknown queries. Two-hop traversal linked G53 and G54–G59.3 through a shared constraint. Separate automatic Qwen extraction probe and setup/backend caveats: `specs/graphiti_experiment.md`.

18. **Decision routing** — Added lexical/Graphiti/hybrid/refusal selection, independent provenance and source-topic coverage checks, actual optional local Laya classification and 24 development cases. Rules selected 24/24 expected routes; raw Laya 7/24, or 12/24 with hard guards. All 18 source-bearing rule-routed cases found their expected sections; five refused and one requested clarification. Real CLI scenarios and 26/26 tests passed. Reports and commands: `specs/machine_expert.md#decision-routing-and-evidence-gate`.

19. **Grounded SFT for both releases — completed local CPU experiment** — Added the shared source-conditioned prompt, strict complete-example/EOS supervision, adapter base/precision compatibility checks, a question-disjoint 24/9/9 dataset, before/after evaluation, sequential dual-release controller and local HTML/SVG charts. User selected local CPU; both releases completed three epochs and 72 LoRA steps, without cloud resources. Epoch 3 was selected for each by validation loss. Protocol, measured results, artifacts and commands: `specs/grounded_sft.md`.

   The first run finished on 2026-11-07 after about 3 h 29 min. Validation/test loss fell, but full-answer test match remained 0/9 for both; correct citation ID sets rose to 6/9 for 1.05 and 4/9 for 1.06 without ensuring correct content. Actual answers remain incomplete or contradictory. Both selected adapters were loaded through the real source-gated router; 1.06 still contradicted the supplied G53 modality evidence. Neither becomes the default or approved machine advice. All 43 regression tests passed, including nonzero-LoRA cached/full-reference parity on tiny reference models. Actual completed charts and raw-answer expansion were exercised in a browser; an unknown alarm refused generation before touching nonexistent model paths. Both baselines and trained evaluations used the same mixed CPU precision. Full logs and the frozen dataset hashes were checked.
