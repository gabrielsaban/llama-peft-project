# Quantisation Effects in PEFT Domain Adaptation

This repository contains the implementation and reproducibility artefacts for a
final-year dissertation comparing LoRA and QLoRA for domain-adaptive continued
pretraining on UK employment-law text.

The final experiment set is `protocol_v3`: a matched intrinsic comparison using
`meta-llama/Meta-Llama-3-8B`, a fixed UK employment-law corpus, a fixed
document-level validation split, and seed-repeated LoRA/QLoRA configurations.

## What Is In This Repository

- `src/`: training, evaluation, reporting, provenance, and aggregation code.
- `preprocess/`: corpus extraction, cleaning, tokenisation, and split utilities.
- `configs/`: frozen experiment configurations. The active final configs are
  the `*protocol_v3*` files at the top level.
- `data/domain_corpus/`: the fixed cleaned corpus and corpus manifest.
- `data/splits/intrinsic_splits.json`: the fixed document-level split used by
  every reportable run.
- `scripts/l40/`: local L40 launch wrappers for final protocol runs.
- `jobs/protocol_v3.sbatch`: Slurm wrapper for the final protocol on the L40
  cluster environment.
- `outputs/experiments/protocol_v3/`: completed per-run artefacts, including
  resolved configs, environment snapshots, metrics, memory summaries, timing
  summaries, and final adapters.
- `outputs/reports_index/protocol_v3/`: aggregate CSV/JSON summaries generated
  from the per-run artefacts.
- `logs/protocol_v3.md`: frozen protocol definition and rationale.

The dissertation source/PDF and local analysis-render folders are intentionally
not part of the reproducibility path. The experiment can be inspected from the
tracked code, configs, data, logs, and output artefacts listed above.

## Quick Inspection

Use the wrapper as the main entry point:

```bash
scripts/reproduce_protocol_v3.sh inspect
```

This checks the expected final configs, frozen corpus files, split file,
completed run artefacts, and aggregate summaries. It also prints the condition
summary if `outputs/reports_index/protocol_v3/condition_summary.csv` is present.

To regenerate aggregate CSV/JSON summaries from the saved per-run reports:

```bash
scripts/reproduce_protocol_v3.sh aggregate
```

This runs:

```bash
python3 -m src.compare_runs --root outputs --out-dir outputs/reports_index/protocol_v3
```

## Environment

Two dependency files are provided:

```bash
conda env create -f environment.yaml
conda activate llama-peft-project
```

or, for a pip-style setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The final training runs were executed on L40-class GPU hardware with Hugging
Face access to `meta-llama/Meta-Llama-3-8B`. Authentication is expected to come
from `HF_TOKEN` or an existing Hugging Face login; tokens are not stored in this
repository.

## Final Protocol

The final reportable matrix contains twelve training runs:

- Phase 1: `LoRA r16` versus `QLoRA r16`, seeds `42`, `43`, and `44`.
- Phase 2: QLoRA rank sweep at `r32` and `r64`, seeds `42`, `43`, and `44`.

The frozen recipe is documented in `logs/protocol_v3.md`. The corresponding
YAML configs are:

```text
configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml
configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed43.yaml
configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed44.yaml
configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed42.yaml
configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed43.yaml
configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed44.yaml
configs/llama3_8b_qlora_l40_protocol_v3_phase2_r32_seed42.yaml
configs/llama3_8b_qlora_l40_protocol_v3_phase2_r32_seed43.yaml
configs/llama3_8b_qlora_l40_protocol_v3_phase2_r32_seed44.yaml
configs/llama3_8b_qlora_l40_protocol_v3_phase2_r64_seed42.yaml
configs/llama3_8b_qlora_l40_protocol_v3_phase2_r64_seed43.yaml
configs/llama3_8b_qlora_l40_protocol_v3_phase2_r64_seed44.yaml
```

## Re-running Experiments

Run one config locally on suitable GPU hardware:

```bash
scripts/reproduce_protocol_v3.sh single configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml
```

Run the final phase groups locally:

```bash
scripts/reproduce_protocol_v3.sh phase1
scripts/reproduce_protocol_v3.sh phase2
```

Submit through Slurm:

```bash
scripts/reproduce_protocol_v3.sh slurm phase1
scripts/reproduce_protocol_v3.sh slurm phase2
```

The local wrappers write run logs to each run's `raw/run.log`, structured
reports to each run's `reports/` directory, and aggregate summaries to
`outputs/reports_index/protocol_v3/`.

## Data Reproducibility

The final training/evaluation path uses the already cleaned corpus in
`data/domain_corpus/corpus_final`, the manifest at
`data/domain_corpus/corpus_manifest.json`, and the fixed split at
`data/splits/intrinsic_splits.json`.

Corpus construction scripts are retained in `preprocess/` for auditability, but
the final comparison should be reproduced from the frozen corpus and split, not
by re-scraping public sources. This avoids accidental source drift and preserves
the matched-data design of the study.

## Result Traceability

For each completed run, inspect:

- `reports/resolved_config.yaml`: effective configuration used by the run.
- `reports/environment.json`: runtime environment snapshot.
- `reports/dataset_summary.json`: dataset and split quantities.
- `reports/eval_summary.csv`: checkpoint-level evaluation metrics.
- `reports/memory_summary.json`: memory measurements.
- `reports/timing_summary.json`: runtime measurements.
- `reports/stability_summary.json`: numerical stability events.
- `reports/comparison_row.json`: row consumed by the aggregate comparison.

The aggregate files in `outputs/reports_index/protocol_v3/` are derived from
those per-run reports. They are the compact inspection point for the final
condition-level comparison.
