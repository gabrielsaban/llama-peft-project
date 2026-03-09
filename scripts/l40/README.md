# L40 Protocol v2 Launchers

These are the active launch scripts for L40 runs under `logs/protocol_v2.md`.

## Single config runner

Run full baseline+train flow (default):

```bash
scripts/l40/run_l40_config.sh configs/llama3_8b_lora_l40_protocol_v2_phase1_r16_seed42.yaml
```

Run baseline-only (no training):

```bash
scripts/l40/run_l40_config.sh configs/llama3_8b_qlora_l40_protocol_v2_phase1_r16_seed42.yaml --baseline-only
```

Skip aggregation:

```bash
scripts/l40/run_l40_config.sh <config.yaml> --no-aggregate
```

## Protocol matrix runner

Core comparison:

```bash
scripts/l40/run_protocol_v2.sh phase1
```

Rank sweep:

```bash
scripts/l40/run_protocol_v2.sh phase2
```

Baselines only:

```bash
scripts/l40/run_protocol_v2.sh baseline
```

Everything in order:

```bash
scripts/l40/run_protocol_v2.sh all
```

## Output behavior

- All run outputs are repo-local under each config `training.output_dir`.
- Structured artifacts are emitted under `training.output_dir/reports/`.
- Launcher logs are appended to `training.output_dir/raw/run.log`.
- Aggregation writes to `outputs/reports_index/`.
