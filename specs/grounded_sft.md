# Grounded SFT: Slayer1B 1.05 and 1.06

## What this experiment measures

This is local **LoRA supervised fine-tuning**, not pretraining a billion-parameter model from scratch. Both original releases remain unchanged on disk. Separate adapters teach short Polish answers conditioned on explicit LinuxCNC evidence and a source identifier.

The operational source-only router remains the default. Supplying `machine_router.py ask --model-dir ...` explicitly enables an **experimental, unverified** Slayer completion after the source gate. A retrieved citation is not proof of a correct answer or an approved repair procedure.

The user chose **local CPU execution**, without renting GPU resources. Releases run sequentially to limit memory use. Run configuration and all measurements are stored locally rather than uploaded to an external tracking service.

## Environment prerequisites

The existing local environment is already prepared; do not reinstall it during training. For a new checkout, create the base `.venv` as described in the README and install the base dependencies before running model or training commands:

```powershell
uv pip install --python .venv\Scripts\python.exe -r pyproject.toml --extra training
```

This package-installation step requires network access and Python 3.11 for the pinned training group. NumPy is an explicit dependency because the releases' reference Python module imports it. The measured workstation uses Python 3.11.9, PyTorch 2.14.0, NumPy 2.3.5, tokenizers 0.23.3 and safetensors 0.8.0; the `training` extra pins those four core libraries. Transitive dependencies are not fully locked, so this is not a bit-for-bit environment reproduction guarantee. A dry-run of the unpinned base group selected newer Torch/NumPy; do not substitute that installation for the controlled training group.

The original `config.json`, `tokenizer.json`, `model.safetensors` and `train_gpt_ref_r23.py` must already be present in each release folder under `models/`. The repository does not ship the model weights or claim that training works in an otherwise empty standard-library-only environment. Later evaluation/training reads the local assets; no paid API or cloud endpoint is used.


## Dataset and split limitations

```powershell
.venv\Scripts\python.exe scripts\prepare_machine_sft.py
```

Data already prepared on this workstation need not be regenerated during a run. Files are under `data/machine_expert/sft/`:

| Split | Questions | Purpose |
|---|---:|---|
| Train | 24 | Supervised parameter updates |
| Validation | 9 | Before/after-epoch measurements and checkpoint selection |
| Test | 9 | Fixed baseline and final measurement, not checkpoint selection |

Training questions and reference statements originate in the existing explicit documentation benchmark. Held-out paraphrases and the additional reference phrasing were authored by the AI assistant, with exact source spans checked by the builder; there has been no independent human review in this workflow. The builder does not call a model to label data at runtime. This is not a human-certified benchmark. Questions are checked for overlap between splits. The initial ambiguous `generated_labels` flag was clarified to `label_generation_at_runtime` plus explicit authorship metadata; no sample bytes or training settings were changed.

**The split is question-disjoint, not document- or fact-disjoint.** All splits use the same nine LinuxCNC sections and share many facts and canonical answer phrases. Results measure this narrow experiment, not generalization to unknown machines, manuals or industrial faults. The public test set is not a blind evaluation.

Each training example contains complete selected source sections, the question, and a Polish reference answer with `[source-id]`. No source or response is silently truncated. Both real release tokenizers were checked: the largest complete example including EOS uses 868 tokens; the longest reference uses 40 tokens. The actual run uses a 1024-token sequence budget and a 64-token generation budget.

The manifest records source revision, corpus SHA256 and hashes of every generated data file. Dataset content is immutable for a run; the controller verifies these hashes before execution. Derived source-containing data retain the source/attribution context described in [the documentation experiment](machine_expert.md#source-and-reuse-conditions).

## Shared prompt, response masking and EOS

`grounded_slayer.build_instruction()` constructs the same source-conditioned instruction for dataset generation, evaluation and opt-in runtime use. `sft_slayer.format_sft_prompt()` supplies the common `Użytkownik: ... / Asystent:` wrapper.

Only response tokens and EOS contribute to supervised loss. Tokenizer offsets identify the response boundary, including tokens that span a leading space. EOS is token 0, confirmed as `<|endoftext|>` for both releases. Empty targets and over-budget complete examples are rejected. This replaces the earlier generic trainer's silent slicing behavior.

Validation response loss is the mean of per-example response losses, not perplexity of the entire source passage. Loss includes answer formatting, source identifiers and EOS; it must not be interpreted as factual accuracy.

## CPU precision and adapter compatibility

The experiment uses:

- BF16 hidden states and most frozen backbone weights;
- frozen FP32 output projection;
- frozen FP32 linear projections in the final two blocks, converting their output back to the surrounding hidden-state dtype;
- FP32 LoRA parameters only, rank 4, alpha 8, targeting `qkv`, `proj` and `down` in the final two blocks.

The FP32 frozen projections avoid the very slow native BF16 backward path observed on this CPU. They do **not** make base weights trainable. A real short forward/backward probe completed with this policy; tests verify that an optimizer update changes LoRA while leaving the base weights frozen. The new numerical policy can differ slightly from the old all-BF16 path.

**Both new baselines and their trained evaluations use the same precision policy.** Do not directly attribute differences from older continuation/grounding experiments to training: those used different prompts and numerical configurations.

New checkpoints store the base `config.json` SHA256, precision policy, LoRA structure, optimizer state and global step. New grounded inference rejects a different release, an unidentified legacy adapter or an incompatible precision policy before loading model weights. Resume also checks base identity and precision. Old generic adapters without the fingerprint are not silently reused for this new run.

## Execute the controlled run

First completed experiment command:

```powershell
.venv\Scripts\python.exe scripts\run_machine_training.py --epochs 3 --max-new-tokens 64
```

Do not start a second copy while the existing one is running. The controller refuses to overwrite a prior run or adapters. For a separate new experiment, use new directories for **both** outputs:

```powershell
.venv\Scripts\python.exe scripts\run_machine_training.py --epochs 3 --max-new-tokens 64 --output-root reports\machine_training_run02 --adapters-root adapters\machine_expert_run02
```

The fixed initial hyperparameters are learning rate 0.001, four CPU threads and seed 20261106, incremented per epoch for data order. Three epochs over the training split produce 72 optimizer steps per release. The controller reuses the existing trainer and its resumable optimizer checkpoints for successive epochs.

For each release the controller performs:

1. Baseline validation and test generation, with response-loss measurement.
2. One epoch of LoRA updates, with every step flushed to JSONL.
3. Validation generation and response-loss measurement after that epoch; repeat for the requested epochs.
4. Select the trained checkpoint with lowest validation response loss, without using final test results for selection.
5. Evaluate that selected adapter on the fixed test questions and save its before/after summary.

Even if all trained checkpoints are worse than the base model, the report still shows the trained result. There is no hidden switch back to the baseline and no promise that training improves quality.

The controller exits on errors and records `experiment_failed` where possible. `run_end` from an individual trainer invocation is not a completed multi-epoch experiment. A hard process termination may prevent a final event; the viewer reflects recorded events, not an OS process-health probe.

## What the automatic answer metrics do and do not establish

`evaluate_machine_sft.py` receives complete **oracle-selected** source sections for each case. This isolates reading/generation from retrieval ranking. The reference answer is used for scoring and supervised loss, **never included in the generation prompt**. This is not an end-to-end Graphiti retrieval accuracy score; real router integration is exercised separately.

- **Strict full-answer reference match:** case-insensitive full-answer equality after whitespace normalization. It can reject semantically correct paraphrases and does not replace factual review. Appending an unsupported claim fails this metric even if the expected sentence is present.
- **Citation identifier match:** the set of generated bracketed identifiers equals the required source IDs, with no invented extra IDs. A false statement can have a correct source identifier; this is explicitly separate from answer correctness.
- **Response loss:** teacher-forced loss on the canonical response and EOS.
- **Generation time/tokens:** actual observed local inference measurements, not a controlled cross-hardware throughput benchmark.

Inspect the saved full answers and original sources before making claims about truthfulness or safety. No external LLM judge or keyword-overlap score is presented as verified factual correctness.

## Live plots and saved artifacts

The live viewer on this workstation is served at:

**http://127.0.0.1:8876/dashboard.html**

To start it manually when it is not already running, use separate terminals:

```powershell
.venv\Scripts\python.exe -u scripts\plot_machine_training.py --watch
```

```powershell
.venv\Scripts\python.exe -u -m http.server 8876 --bind 127.0.0.1 --directory reports\machine_training
```

The viewer uses only local HTML/SVG and standard-library Python. There is no CDN, tracking SDK, external API or JavaScript chart dependency. Watch mode refreshes every five seconds. After the first run completed, its watcher was stopped and a final static snapshot regenerated; the local HTTP server remains available. On Windows, bounded retries handle temporary file-sharing conflicts when atomically replacing the HTML snapshot; permanent errors are not hidden.

Plots separately show training loss, validation loss, validation reference match, test reference match and test citation IDs. Only actual measurement points are plotted. Missing data remain missing, not zero. Validation can select checkpoints; the test curve contains only baseline and final observations, not invented intermediate scores. The viewer also exposes raw before/after test answers.

The HTML snapshot can also be regenerated once, without a live server:

```powershell
.venv\Scripts\python.exe scripts\plot_machine_training.py
```

Artifacts under `reports/machine_training/`:

- `run.json`: requested settings and dataset manifest.
- `1.05/metrics.jsonl`, `1.06/metrics.jsonl`: append-only step, phase and evaluation events.
- Per-release `baseline-valid.jsonl`, `baseline-test.jsonl`, `epoch-001-valid.jsonl` and subsequent epoch files, `final-test.jsonl`: actual completions, references, sources and diagnostics.
- Corresponding `.summary.json` files and per-release `summary.json`: observed aggregates and selected adapter.
- Root `summary.json`: created after both requested release experiments finish.
- `dashboard.html`: self-contained chart snapshot.

Adapters are saved under `adapters/machine_expert/1.05/` and `1.06/`, separately for each epoch. Base model files remain unchanged. The original `reports/`, adapters, environments and downloaded weights remain excluded from Git. A portable snapshot of this completed run is explicitly published under [`results/grounded-sft-2026-11-07/`](../results/grounded-sft-2026-11-07/README.md): actual metrics and completions, with only project-absolute path strings normalized to project-relative paths. Its publication manifest records the transformation and hashes. No weights or adapters are included.

## Use a trained adapter through the real router

Use the selected adapter path recorded in the release's `summary.json`. For example, the last epoch checkpoint for release 1.06 can be exercised as follows after it exists:

```powershell
.venv\Scripts\python.exe scripts\machine_router.py ask --question "Czy G53 jest modalny?" --model-dir models\Slayer1B-1.06 --adapter adapters\machine_expert\1.06\epoch-003.pt --max-new-tokens 64 --output reports\machine_training\router_1.06.json
```

The same command without `--adapter` uses the corresponding base release with the controlled precision policy. Change both release paths together for 1.05; cross-release adapters are rejected.

The new `experimental_answer` result is always `generated_unverified`. The independent evidence gate's `generation_allowed: false` remains a statement that the default/source-only policy has not approved a validated technical answer; explicitly requested research output does not turn that into a safety certification.

Unknown alarms, other controllers, incomplete/provenance-failed evidence and backend failures do not load or run Slayer, even when a model and adapter were requested. Complete input exceeding the generation budget raises an explicit error rather than trimming evidence or warnings. Do not use generated text to operate or repair equipment.

## Verification scope

The complete local regression suite passed 43/43 tests. Cache parity tests instantiate tiny models from the releases' real reference GPT Python module; they do not load billion-parameter weights. The additional case uses nonzero LoRA and the mixed CPU precision policy and checks both prefill and subsequent cached tokens against full reference forward passes, including selected token identities. Source-code availability, not full weight availability, controls those tests' skip condition.

Other regressions cover strict response/EOS packing, incompatible adapters, frozen base versus trainable LoRA, unknown-scope refusal, evidence integrity, metric separation and incomplete-versus-corrupt JSONL records. Actual full-weight training and answer measurements are separate from unit-test success; their observed first-run results follow.


## Observed first run: 2026-11-07

The real dual-release CPU experiment ran from 2026-11-06 22:03:48 UTC to 2026-11-07 01:32:30 UTC, about 3 h 29 min. Both releases completed all 72 consecutive optimizer steps, three epochs, baseline validation/test, validation after each epoch and final test. All six dataset file hashes still matched the recorded manifest. Epoch 3 was selected independently for each release by lowest validation response loss, not by final test results.

| Measurement | 1.05 before → after | 1.06 before → after |
|---|---:|---:|
| Validation response loss | 2.6304 → 1.3881 | 2.5196 → 1.3602 |
| Test response loss | 2.3452 → 1.1985 | 2.2388 → 1.1376 |
| Strict full-answer test match | 0/9 → 0/9 | 0/9 → 0/9 |
| Required citation ID set on test | 0/9 → 6/9 | 0/9 → 4/9 |

These numbers **do not demonstrate a reliable technical-answer improvement**. The final outputs include the bare citation `[g4]` without the requested condition, missing unit/system definitions, and incorrect negations for G53 and G61.1. Release 1.06's G61 answer contradicts itself. Exact match can reject valid shorter answers, but inspecting the raw responses also confirms substantive failures; the issue is not merely strict wording.

Real `machine_router.py ask` runs loaded both selected adapters against retrieved G53 evidence. Release 1.05 returned `Nie. [g53]` to “Czy G53 jest modalny?”, while 1.06 returned the contradictory `Nie. G53 jest modalny. [g53]`. Both were correctly labelled `generated_unverified`, with the original source excerpt retained and `generation_allowed: false`. Reports: `reports/machine_training/router_1.05.json` and `router_1.06.json`.

The combined Graphiti → source gate → trained 1.05 route was also exercised with “Porównaj G53 i G54.” in `.venv-graphiti`, using `--max-input-tokens 1536`. Both lexical and Graphiti backends ran without backend errors, the required sections were covered, and 1,229 complete input tokens reached the model. It returned only `G53`, not a useful comparison; the output remains visible as `generated_unverified` in `reports/machine_training/router_graphiti_1.05.json`. This proves runtime integration, not answer quality. A real cross-release CLI probe rejected the 1.05 checkpoint with the 1.06 base before full weight loading, with `ValueError: adapter belongs to a different or unidentified base model`.

Selected checkpoints are `adapters/machine_expert/1.05/epoch-003.pt` and `adapters/machine_expert/1.06/epoch-003.pt`. Full aggregates are in `reports/machine_training/summary.json`; raw baseline/final responses and every update remain in the per-release JSONL files. The actual final dashboard was opened in a real browser: both runs showed 72/72 and completion, all five plots rendered without data errors, and the expandable before/after table exposed all nine responses. The inspection tab was closed; the local report server was left running.

Do not interpret shorter generation times as a throughput gain: trained completions often stop with very little content. Do not promote either adapter to approved machine advice. A further quality experiment needs broader independently reviewed examples and source-entailment evaluation; a falling canonical-answer loss or increased citation-ID rate is insufficient.

