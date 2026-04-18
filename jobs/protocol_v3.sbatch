#!/bin/bash
#SBATCH --job-name=protocol_v3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=protocol_v3-%j.out
#SBATCH --error=protocol_v3-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
usage:
  sbatch jobs/protocol_v3.sbatch
  sbatch jobs/protocol_v3.sbatch phase1
  sbatch jobs/protocol_v3.sbatch phase2
  sbatch jobs/protocol_v3.sbatch baseline
  sbatch jobs/protocol_v3.sbatch all
  sbatch jobs/protocol_v3.sbatch --config configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml

notes:
  - default is: --matrix phase1
  - any extra flags are forwarded to scripts/cluster/run_job.sh
  - examples of forwarded flags: --skip-preflight, --baseline-only, --no-aggregate
EOF
}

if [[ $# -ge 1 ]] && [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

if type module >/dev/null 2>&1; then
  module load "${MODULE_NAME:-miniforge/24.7.1}"
fi

PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH:-${HOME}}/repos/llama-peft-project}"
cd "${PROJECT_ROOT}"

SLURM_LOG_DIR="${PROJECT_ROOT}/outputs/slurm_logs"
mkdir -p "${SLURM_LOG_DIR}"
exec >>"${SLURM_LOG_DIR}/protocol_v3-${SLURM_JOB_ID:-manual}.out" \
     2>>"${SLURM_LOG_DIR}/protocol_v3-${SLURM_JOB_ID:-manual}.err"

if [[ $# -eq 0 ]]; then
  set -- --matrix phase1
elif [[ "$1" == "baseline" || "$1" == "phase1" || "$1" == "phase2" || "$1" == "all" ]]; then
  mode="$1"
  shift
  set -- --matrix "${mode}" "$@"
fi

echo "[info] start_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[info] project_root: ${PROJECT_ROOT}"
echo "[info] hostname: $(hostname)"
echo "[info] slurm_job_id: ${SLURM_JOB_ID:-unset}"
echo "[info] args: $*"

scripts/cluster/run_job.sh --protocol-version v3 "$@"
