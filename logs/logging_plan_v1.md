# dissertation logging upgrade plan (v1)

## purpose

define all changes required so each training run automatically produces:

- raw training outputs (checkpoints, trainer state, adapters)
- compact analysis-ready outputs (json/csv summaries)
- reproducibility/provenance metadata
- stability diagnostics
- comparison-ready rows for lora vs qlora analysis

goal: no manual metric extraction from `trainer_state.json` for dissertation reporting.

---

## target outcome (what a finished run should produce)

each run should write into a single run root (existing `training.output_dir`) with two subtrees:

- `raw/` (heavy artifacts, HF trainer outputs)
- `reports/` (lightweight dissertation-ready artifacts)

example:

```text
outputs/<run_id>/
  raw/
    checkpoints/...
    final_adapter/...
    trainer_state.json (or copied from best/final checkpoint)
    baseline_eval_metrics.json
    training_summary.json
    cuda_memory_*.json
    run.log
  reports/
    run_manifest.json
    resolved_config.yaml
    environment.json
    dataset_summary.json
    budget_summary.json
    metrics_history.jsonl
    eval_summary.csv
    final_metrics.json
    memory_summary.json
    timing_summary.json
    stability_summary.json
    stability_events.jsonl
    comparison_row.json
```

note:
- current HF output files can remain, but post-processing should copy/normalize them into `reports/`.
- if you prefer not to change HF output paths, `raw/` can initially be logical rather than physical (documented in `run_manifest.json`).

---

## implementation split (where changes should live)

### 1) `src/train_lora.py` (minimal orchestration changes)

put only runtime-only measurements here (or wire callbacks into it):

- initialize run-level artifact directories (`raw/`, `reports/`)
- save `resolved_config.yaml` (after any launcher overrides)
- save start/end status (`completed`, `oom`, `failed`)
- register callbacks for:
  - step timing collection
  - stability/anomaly event capture
  - structured metric history emission
- phase-correct VRAM resets/snapshots:
  - baseline eval
  - training
  - periodic eval
  - final
- write a compact `run_manifest.json` shell for provenance + paths
- always emit artifacts on failure paths (especially OOM)

keep `train_lora.py` as the coordinator; move logic to helpers.

### 2) new `src/*.py` modules (main implementation surface)

recommended new files:

- `src/artifacts.py`
  - standardized json/jsonl/csv write helpers
  - schema version tagging (`logging_schema_version`)
  - path helpers for `raw/` and `reports/`

- `src/callbacks_metrics.py`
  - step-time tracking (mean, std, p50, p95 per eval interval)
  - metrics history logger (append structured rows to jsonl)

- `src/callbacks_stability.py`
  - non-finite loss detection
  - non-finite grad norm detection
  - loss spike detection (configurable rule)
  - divergence/early termination event capture
  - event log + counters

- `src/provenance.py`
  - git commit hash + dirty flag
  - package versions (`torch`, `transformers`, `peft`, `bitsandbytes`, `datasets`)
  - hardware snapshot (gpu name, total vram)
  - runtime metadata (hostname optional, timestamps, command, pid)

- `src/postprocess_run.py`
  - parse raw outputs after training
  - derive summary artifacts (`final_metrics`, `memory_summary`, `timing_summary`, `comparison_row`)
  - export `eval_summary.csv`
  - validate required fields exist

- `src/compare_runs.py` (or `scripts/compare_runs.py`)
  - aggregate `comparison_row.json` across runs
  - emit comparison csv/json (+ optional markdown table)

### 3) bash launcher scripts (`scripts/run_*.sh`)

bash should orchestrate only:

- generate unique `RUN_ID`
- resolve output dir for run
- persist launcher env + command line
- tee stdout/stderr to `raw/run.log`
- invoke training
- invoke postprocess (even if training fails, where possible)

avoid parsing trainer json in bash.

### 4) configs (`configs/*.yaml`)

add explicit metadata fields for analysis and reproducibility:

- `experiment.variant`: `lora` / `qlora`
- `experiment.protocol_version`: e.g. `protocol_v1`
- `experiment.logging_schema_version`: e.g. `logging_v1`
- `experiment.notes` (optional)
- `training.eval_steps`
- `training.save_steps`
- stability thresholds (optional, if not hard-coded in protocol)

these are not strictly required for training, but they make downstream analysis cleaner.

---

## required artifact schema (minimum fields)

## 1. `reports/run_manifest.json`

single source of truth for the run.

required fields:

- `run_id`
- `experiment_name`
- `variant` (`lora`/`qlora`)
- `status` (`completed`/`oom`/`failed`)
- `start_time_utc`
- `end_time_utc`
- `duration_s`
- `paths` (raw/report/config/checkpoints/final_adapter)
- `logging_schema_version`
- `protocol_version`

## 2. `reports/resolved_config.yaml`

exact config used after launcher overrides (including `output_dir`).

## 3. `reports/environment.json`

required fields:

- `git_commit`
- `git_dirty`
- `python_version`
- `torch_version`
- `transformers_version`
- `peft_version`
- `bitsandbytes_version` (if installed)
- `datasets_version`
- `cuda_available`
- `cuda_device_name`
- `cuda_total_memory_bytes`
- `torch_cuda_version` (if available)
- `command`
- `cwd`

## 4. `reports/dataset_summary.json`

required fields:

- `data_type`
- `corpus_dir`
- `split_json`
- `max_seq_length`
- `train_docs_total`
- `val_docs_total`
- `val_docs_layer_a`
- `val_docs_layer_b`
- `train_chunks_total`
- `val_chunks_total`
- `val_chunks_layer_a`
- `val_chunks_layer_b`
- `tokens_by_split` (if available from split manifest)
- `split_seed` (if available)
- `stratification` (if available)

note:
- some fields can be read from `data/splits/intrinsic_splits.json`; if absent, set `null` and record source limitations.

## 5. `reports/budget_summary.json`

required fields (for objective 2 fairness):

- `max_steps_planned`
- `max_steps_completed`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `effective_batch_size_sequences`
- `max_seq_length`
- `tokens_per_optimizer_step`
- `planned_tokens_processed`
- `realized_tokens_processed`
- `eval_steps`
- `save_steps`
- `seed`
- `budget_match_key` (hash/string summarizing fixed-budget settings)

## 6. `reports/metrics_history.jsonl`

append one normalized record per log event.

required fields:

- `step`
- `epoch`
- `event_type` (`train_log`, `eval`, `baseline_eval`, `system`)
- `metrics` (object)
- `timestamp_utc`

goal:
- stop relying on nested/heterogeneous HF `log_history` structure for analysis.

## 7. `reports/eval_summary.csv`

one row per evaluation point (`baseline`, each scheduled eval, final if distinct).

columns:

- `run_id`
- `step`
- `phase` (`baseline`, `eval`)
- `eval_loss`
- `eval_perplexity`
- `eval_layer_a_loss`
- `eval_layer_a_perplexity`
- `eval_layer_b_loss`
- `eval_layer_b_perplexity`
- `train_step_time_mean_s_window`
- `train_step_time_p50_s_window`
- `train_step_time_p95_s_window`
- `train_step_time_std_s_window`
- `train_peak_vram_allocated_gb_window` (if tracked by window)
- `eval_peak_vram_allocated_gb`

## 8. `reports/final_metrics.json`

dissertation-facing final intrinsic metrics.

required fields:

- `baseline`:
  - `perplexity_all`
  - `perplexity_layer_a`
  - `perplexity_layer_b`
- `best_eval`:
  - `step`
  - `perplexity_all`
  - `perplexity_layer_a`
  - `perplexity_layer_b`
- `final_eval` (if same as best, still include)
- `deltas_from_baseline`:
  - `delta_ppl_all`
  - `delta_ppl_layer_a`
  - `delta_ppl_layer_b`
- `selection_metric` (`eval_perplexity`)

## 9. `reports/memory_summary.json`

phase-separated memory reporting (do not overload `eval_peak_*` semantics).

required fields:

- `baseline_eval_peak_vram_allocated_gb`
- `baseline_eval_peak_vram_reserved_gb`
- `train_peak_vram_allocated_gb`
- `train_peak_vram_reserved_gb`
- `max_eval_peak_vram_allocated_gb`
- `max_eval_peak_vram_reserved_gb`
- `run_peak_vram_allocated_gb` (optional if cumulative tracked)
- `run_peak_vram_reserved_gb`
- `memory_measurement_notes`

## 10. `reports/timing_summary.json`

required fields:

- `train_runtime_s`
- `train_steps_per_second`
- `optimizer_step_time_mean_s`
- `optimizer_step_time_p50_s`
- `optimizer_step_time_p95_s`
- `optimizer_step_time_std_s`
- `num_step_time_samples`
- `timing_window_definition`

## 11. `reports/stability_summary.json`

required fields (mapped to dissertation indicators):

- `num_loss_spike_events`
- `num_divergence_events`
- `num_nan_loss_events`
- `num_inf_loss_events`
- `num_nonfinite_grad_norm_events`
- `num_oom_events`
- `optimizer_reset_count`
- `terminated_early`
- `termination_reason`
- `last_completed_step`
- `stability_rule_config`

## 12. `reports/stability_events.jsonl`

one row per event:

- `step`
- `event_type`
- `severity`
- `value`
- `threshold`
- `message`
- `timestamp_utc`

## 13. `reports/comparison_row.json`

single-row normalized record for cross-run aggregation.

required fields:

- `run_id`
- `variant`
- `base_model`
- `use_4bit`
- `gradient_checkpointing`
- `seed`
- `max_steps`
- `effective_batch_size_sequences`
- `max_seq_length`
- `baseline_ppl_all`
- `baseline_ppl_layer_a`
- `baseline_ppl_layer_b`
- `best_ppl_all`
- `best_ppl_layer_a`
- `best_ppl_layer_b`
- `delta_ppl_all`
- `delta_ppl_layer_a`
- `delta_ppl_layer_b`
- `train_peak_vram_allocated_gb`
- `train_peak_vram_reserved_gb`
- `optimizer_step_time_mean_s`
- `optimizer_step_time_p95_s`
- `num_loss_spike_events`
- `num_nonfinite_events`
- `status`

---

## logging semantics to fix (important)

## 1. peak memory semantics

current issue:
- peak CUDA stats are reset before training, then later logged during eval.
- reported `eval_peak_vram_*` may represent peak since reset (train + eval), not eval-only.

required change:

- define and log explicitly:
  - `baseline_eval_peak_*`
  - `train_peak_*`
  - `eval_peak_*` (eval-only, reset before each eval if desired)
  - `run_peak_*` (optional cumulative)

## 2. step time semantics

current issue:
- `eval_mean_train_step_time_s` is useful but ambiguous.

required change:

- rename (or duplicate) in reports as:
  - `train_step_time_mean_s_window`
- add:
  - `p50`, `p95`, `std`, `num_samples`
- define unit explicitly:
  - optimizer step wall-clock time (post gradient accumulation)

## 3. stability semantics

current issue:
- loss + grad_norm exist in trainer logs, but anomalies are not first-class outputs.

required change:

- emit explicit events/counters and threshold definitions.

---

## what should be runtime instrumentation vs post-processing

## runtime only (must be captured live)

- step-time sample collection
- CUDA peak resets by phase
- anomaly detection (non-finite loss/grad norms, OOM)
- failure status + termination reason
- environment snapshot at run start

## post-process (derive after training)

- `eval_summary.csv`
- `final_metrics.json`
- `budget_summary.json` (partly derived)
- `memory_summary.json` (from raw snapshots + eval logs)
- `timing_summary.json` (from callback summary + training summary)
- `comparison_row.json`
- multi-run comparison tables

---

## bash launcher changes (minimal but important)

required changes to `scripts/run_*.sh`:

- set a deterministic/unique `RUN_ID` (timestamp + variant + seed)
- write launcher metadata:
  - selected config path
  - temp config path
  - command line
  - env/cache paths
- capture stdout/stderr to file:
  - `raw/run.log`
- call post-process script after train exits:
  - success path
  - failure path (best effort)
- avoid manual file deletion in launcher

optional:
- `--keep-raw-checkpoints` / `--prune-raw` flag handling later

---

## raw artifact retention policy (to stop manual deletions)

recommended policy:

- always keep:
  - `reports/*`
  - `final_adapter/*`
  - `training_summary.json`
  - `baseline_eval_metrics.json`
  - `run.log`
- keep checkpoints only as configured for model selection:
  - best + final (already close to current behavior)
- do not manually delete files during/after run; use a scripted prune command if needed

future optional script:

- `scripts/prune_run_outputs.py --run <run_id> --keep reports final_adapter best final`

---

## analysis automation (multi-run)

add a comparison aggregator so dissertation tables/plots read from `reports/` only.

inputs:

- multiple `reports/comparison_row.json`

outputs:

- `outputs/reports_index/all_runs_comparison.csv`
- `outputs/reports_index/all_runs_comparison.json`
- optional `outputs/reports_index/all_runs_comparison.md`

minimum grouping keys:

- `variant`
- `base_model`
- `seed`
- `r`
- `alpha`
- `use_4bit`
- `gradient_checkpointing`

---

## phased implementation checklist

## phase 1 (high value, low risk) - do first

- [ ] create `reports/` subtree and `run_manifest.json`
- [ ] save `resolved_config.yaml`
- [ ] save `environment.json`
- [ ] emit `metrics_history.jsonl`
- [ ] add `postprocess_run.py`
- [ ] emit `eval_summary.csv`
- [ ] emit `final_metrics.json`
- [ ] emit `comparison_row.json`
- [ ] capture `run.log` from launcher

result:
- no more manual metric extraction for perplexity comparisons.

## phase 2 (measurement quality upgrades)

- [ ] phase-separated VRAM peak tracking (`baseline/train/eval`)
- [ ] richer step-time stats (`mean/p50/p95/std`)
- [ ] `memory_summary.json`
- [ ] `timing_summary.json`

result:
- strong evidence for objective 2 (memory + efficiency).

## phase 3 (stability instrumentation)

- [ ] explicit stability callback
- [ ] `stability_events.jsonl`
- [ ] `stability_summary.json`
- [ ] document spike/divergence thresholds in protocol

result:
- dissertation-ready stability reporting, not just anecdotal logs.

## phase 4 (cross-run automation)

- [ ] `compare_runs.py`
- [ ] aggregated csv/json/md reports
- [ ] optional seed-aggregate summaries (mean/std) if multi-seed later

result:
- one-command generation of comparison tables for writing.

---

## acceptance criteria (ready for real dissertation runs)

the pipeline is "ready" when a single run can be launched and, without manual parsing:

- produce baseline + stratified held-out perplexity summaries
- produce train/eval memory + step-time summaries with unambiguous semantics
- produce stability summaries and event logs
- capture full provenance (config/environment/git)
- produce a single `comparison_row.json`
- be aggregated with other runs into a comparison csv/json

if any one of these still requires opening `trainer_state.json` manually, logging is not yet complete.

---

## notes specific to your protocol

- calibration configs (`3B`, temporary GC toggles, etc.) are valid for plumbing/stability testing and should not be treated as the final comparison design.
- final dissertation comparison should keep configs identical across LoRA/QLoRA except quantisation (`use_4bit`) unless feasibility constraints force a documented protocol change.
- if feasibility forces changes (e.g., batch size), log them explicitly in `budget_summary.json` and `comparison_row.json`.

