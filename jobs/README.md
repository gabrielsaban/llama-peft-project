# Slurm Jobs

`jobs/protocol_v3.sbatch` is the active submission wrapper for the frozen `protocol_v3` campaign.

Common submissions:

```bash
sbatch jobs/protocol_v3.sbatch
sbatch jobs/protocol_v3.sbatch phase1
sbatch jobs/protocol_v3.sbatch phase2
sbatch jobs/protocol_v3.sbatch --config configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml
sbatch jobs/protocol_v3.sbatch --config configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed42.yaml --baseline-only
```

What each mode does:

- `phase1`: runs the 6 core training configs from `logs/protocol_v3.md`
- `phase2`: runs the 6 QLoRA rank-sweep training configs
- `baseline`: runs standalone baseline-only eval for the two seed-42 phase-1 configs
- `all`: runs `baseline`, then `phase1`, then `phase2`

Notes:

- The wrapper defaults to `phase1`.
- It forwards extra flags to `scripts/cluster/run_job.sh`.
- `PROJECT_ROOT` defaults to `${SCRATCH}/repos/llama-peft-project` when `SCRATCH` is set, otherwise `${HOME}/repos/llama-peft-project`.
- `CONDA_ENV_NAME` can override the conda env name if needed.
- Hugging Face auth should come from `HF_TOKEN` or an existing cached login; the job script does not embed tokens.

The older `smoke_*.sbatch` and `phase1_*.sbatch` files are retained as earlier protocol-v2 examples and should not be used for the frozen v3 campaign.
