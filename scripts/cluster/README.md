# Cluster Helpers

## bootstrap environment

```bash
scripts/cluster/bootstrap_env.sh
```

Optional:

```bash
scripts/cluster/bootstrap_env.sh --with-requirements
```

## preflight checks

```bash
scripts/cluster/preflight.sh configs/llama3_8b_lora_l40_protocol_v2_smoke_seed42.yaml
```

## run jobs

Single smoke config:

```bash
scripts/cluster/run_job.sh --config configs/llama3_8b_lora_l40_protocol_v2_smoke_seed42.yaml
```

Baseline-only:

```bash
scripts/cluster/run_job.sh --config configs/llama3_8b_lora_l40_protocol_v2_smoke_seed42.yaml --baseline-only
```

Protocol matrix:

```bash
scripts/cluster/run_job.sh --matrix phase1
```
