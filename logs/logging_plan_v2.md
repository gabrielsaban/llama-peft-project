# dissertation logging plan (v2)

## purpose

document the logging decisions actually implemented for queue-ready runs, with explicit tradeoffs and remaining gaps.

goals:

- no manual metric extraction from `trainer_state.json`
- baseline + train runs produce structured artifacts automatically
- comparison-ready outputs for LoRA vs QLoRA
- explicit metric semantics to avoid over-claiming

---

## decided output model

### run output layout

- keep Hugging Face trainer outputs in `training.output_dir` (no risky relocation).
- write normalized analysis artifacts to `training.output_dir/reports/`.
- keep launcher stdout/stderr in `training.output_dir/raw/run.log` when launcher supports it.

justification:
- minimizes pipeline break risk while still giving clean dissertation artifacts.

---

## decided implementation split

### `src/train_lora.py` (orchestrator)

- owns run orchestration, baseline/training flow, and artifact emission timing.
- now includes:
  - baseline-only mode (`--baseline-only`)
  - baseline-before-adapter semantics
  - run manifest lifecycle (`running` -> `completed` / `oom` / `failed`)

### helper modules in `src/`

- `src/artifacts.py`: json/yaml/jsonl/csv writers
- `src/provenance.py`: environment/provenance snapshot
- `src/callbacks_metrics.py`: step-time stats + train peak vram sampling
- `src/callbacks_stability.py`: objective stability events/counters
- `src/compare_runs.py`: cross-run aggregation

### script policy

- prefer `python3 -m src.compare_runs` as canonical aggregator entrypoint.
- keep script wrappers only for convenience/compatibility.

justification:
- separates concerns without introducing a high-risk full postprocess split yet.

---

## decided schema status

### implemented

- `reports/run_manifest.json`
- `reports/resolved_config.yaml`
- `reports/environment.json`
- `reports/dataset_summary.json`
- `reports/budget_summary.json` (includes `budget_match_key`)
- `reports/metrics_history.jsonl`
- `reports/eval_summary.csv`
- `reports/final_metrics.json`
- `reports/memory_summary.json`
- `reports/timing_summary.json`
- `reports/stability_summary.json`
- `reports/stability_events.jsonl`
- `reports/comparison_row.json`

### notable metadata decisions

- manifest includes:
  - `variant`
  - `protocol_version`
  - `logging_schema_version`
  - `duration_s`

justification:
- ensures run provenance and comparability are explicit in artifacts, not inferred.

---

## decided metric semantics

### perplexity

- baseline perplexity is measured on the unadapted base model.
- train run still records baseline first, then applies LoRA for training.
- overall + layer-stratified perplexity retained.

### timing

- store summary statistics only:
  - mean, p50, p95, std, count
- no raw per-step timing logs.

justification:
- captures useful efficiency signal with low storage and low overhead.

### memory

- explicit semantics adopted:
  - `train_peak_vram_*`: sampled during training steps
  - strict eval-only peaks via CUDA peak reset immediately before each eval pass
  - `*_peak_vram_*_since_last_reset` retained as backward-compatible aliases

justification:
- low-risk implementation with explicit semantics and backward compatibility.

---

## decided stability scope

### included objective events

- `nan_loss`
- `inf_loss`
- `nonfinite_grad_norm`
- `oom`
- `early_termination` (via manifest/status path)

### explicitly de-scoped for now

- heuristic `loss_spike`
- abstract `divergence` counters

justification:
- objective events are reproducible and defensible; spike/divergence definitions are protocol-sensitive.

---

## decided failure handling

- OOM path: writes `oom` status + stability update + OOM artifacts.
- non-OOM exceptions: minimal top-level wrapper writes `failed` manifest + early termination summary, then re-raises.

justification:
- preserves failure visibility without swallowing errors.

---

## aggregation decisions

- aggregate from `reports/comparison_row.json` files.
- outputs:
  - `all_runs_comparison.csv`
  - `all_runs_comparison.json`
  - `all_runs_summary.json`

justification:
- direct path from run artifacts to dissertation tables.

---

## implications

- current pipeline is queue-usable and comparison-ready for:
  - perplexity
  - step-time summaries
  - objective stability events
  - memory with strict eval-only peak isolation + backward-compatible aliases
- postprocess remains integrated in `train_lora.py`, so architecture is improved but not fully decoupled.

---

## next plans (short)

1. make active launcher(s) repo-output-only (no `/tmp` output override).
2. keep strict eval-only VRAM fields stable in downstream analysis scripts (no schema drift).
3. keep stability scope objective unless protocol explicitly defines spike/divergence thresholds.
4. run one baseline-only smoke run and one short train smoke run to validate final artifact semantics before queueing full experiments.
