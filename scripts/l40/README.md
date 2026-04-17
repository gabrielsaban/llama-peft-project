# L40 Protocol Launchers

The active frozen campaign is `logs/protocol_v3.md`.
The `protocol_v2` launcher is retained only for provenance reruns.

## Single config runner

Run full baseline+train flow (default):

```bash
scripts/l40/run_l40_config.sh configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml
```

Run baseline-only (no training):

```bash
scripts/l40/run_l40_config.sh configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed42.yaml --baseline-only
```

Skip aggregation:

```bash
scripts/l40/run_l40_config.sh <config.yaml> --no-aggregate
```

## Protocol matrix runner

Core comparison:

```bash
scripts/l40/run_protocol_v3.sh phase1
```

Rank sweep:

```bash
scripts/l40/run_protocol_v3.sh phase2
```

Baselines only:

```bash
scripts/l40/run_protocol_v3.sh baseline
```

Everything in order:

```bash
scripts/l40/run_protocol_v3.sh all
```

## Notes

- `phase1` runs the 6 core training jobs in `protocol_v3`:
  `LoRA r16` seeds `42/43/44` and `QLoRA r16` seeds `42/43/44`.
- `phase2` runs the 6 QLoRA rank-sweep jobs:
  `r32` and `r64`, each with seeds `42/43/44`.
- `all` starts with standalone baseline-only runs for the two seed-42 phase-1 configs, then runs phase 1 and phase 2.
  Use it only if you explicitly want those extra baseline-only artifacts in addition to the baseline-first evaluation already done inside every training run.

## Output behavior

- All run outputs are repo-local under each config `training.output_dir`.
- Structured artifacts are emitted under `training.output_dir/reports/`.
- Launcher logs are appended to `training.output_dir/raw/run.log`.
- Aggregation writes to `outputs/reports_index/`.
