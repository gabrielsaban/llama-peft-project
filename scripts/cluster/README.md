# Cluster Helpers

These helpers are intended for the frozen `protocol_v3` campaign by default.
Use `--protocol-version v2` only when you explicitly want an older provenance rerun.

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
scripts/cluster/preflight.sh configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml
```

## run jobs

Single protocol-v3 config:

```bash
scripts/cluster/run_job.sh --config configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml
```

Baseline-only:

```bash
scripts/cluster/run_job.sh --config configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml --baseline-only
```

Protocol-v3 matrix:

```bash
scripts/cluster/run_job.sh --matrix phase1
```

QLoRA rank sweep:

```bash
scripts/cluster/run_job.sh --matrix phase2
```

Older protocol-v2 matrix, if needed:

```bash
scripts/cluster/run_job.sh --matrix phase1 --protocol-version v2
```
