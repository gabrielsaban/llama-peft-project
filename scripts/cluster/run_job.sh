#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
usage:
  scripts/cluster/run_job.sh --config <config.yaml> [--baseline-only] [--no-aggregate] [--skip-preflight]
  scripts/cluster/run_job.sh --matrix <baseline|phase1|phase2|all> [--protocol-version <v2|v3>] [--skip-preflight]

options:
  --env-name NAME             conda env name (default: llama-peft or $CONDA_ENV_NAME)
  --protocol-version <v2|v3>  matrix launcher/config set to use (default: v3)
EOF
}

ENV_NAME="${CONDA_ENV_NAME:-llama-peft}"
PROTOCOL_VERSION="v3"
MODE=""
CONFIG=""
MATRIX_MODE=""
BASELINE_ONLY=0
NO_AGGREGATE=0
SKIP_PREFLIGHT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || { echo "[fatal] --env-name requires a value" >&2; exit 1; }
      ENV_NAME="$2"
      shift
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "[fatal] --config requires a value" >&2; exit 1; }
      MODE="config"
      CONFIG="$2"
      shift
      ;;
    --matrix)
      [[ $# -ge 2 ]] || { echo "[fatal] --matrix requires a value" >&2; exit 1; }
      MODE="matrix"
      MATRIX_MODE="$2"
      shift
      ;;
    --protocol-version)
      [[ $# -ge 2 ]] || { echo "[fatal] --protocol-version requires a value" >&2; exit 1; }
      PROTOCOL_VERSION="$2"
      shift
      ;;
    --baseline-only)
      BASELINE_ONLY=1
      ;;
    --no-aggregate)
      NO_AGGREGATE=1
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[fatal] unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if [[ -z "${MODE}" ]]; then
  echo "[fatal] choose one mode: --config or --matrix" >&2
  usage >&2
  exit 1
fi

case "${PROTOCOL_VERSION}" in
  v2|v3)
    ;;
  *)
    echo "[fatal] invalid --protocol-version value: ${PROTOCOL_VERSION}" >&2
    usage >&2
    exit 1
    ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "[fatal] conda not found in PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if [[ -n "${HF_TOKEN:-}" ]] && command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli login --token "${HF_TOKEN}" >/dev/null 2>&1 || true
fi

# Keep HF caches out of tracked repo paths while leaving experiment outputs in-repo.
# Default to per-Slurm-job cache isolation so concurrent jobs do not contend on shared
# Hugging Face lock files during tokenizer/model startup.
JOB_CACHE_NAMESPACE="${SLURM_JOB_ID:-shared}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/llama-peft/hf/${JOB_CACHE_NAMESPACE}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_XET_CACHE}"

echo "[info] hf_home: ${HF_HOME}"
echo "[info] hf_hub_cache: ${HUGGINGFACE_HUB_CACHE}"

PREFLIGHT="${ROOT_DIR}/scripts/cluster/preflight.sh"
RUN_SINGLE="${ROOT_DIR}/scripts/l40/run_l40_config.sh"
RUN_MATRIX="${ROOT_DIR}/scripts/l40/run_protocol_${PROTOCOL_VERSION}.sh"

if [[ ! -x "${RUN_MATRIX}" ]]; then
  echo "[fatal] matrix runner missing or not executable: ${RUN_MATRIX}" >&2
  exit 1
fi

if [[ "${MODE}" == "config" ]]; then
  [[ -n "${CONFIG}" ]] || { echo "[fatal] --config value missing" >&2; exit 1; }
  [[ -f "${CONFIG}" ]] || { echo "[fatal] config not found: ${CONFIG}" >&2; exit 1; }

  if [[ "${SKIP_PREFLIGHT}" -eq 0 ]]; then
    "${PREFLIGHT}" "${CONFIG}"
  fi

  CMD=("${RUN_SINGLE}" "${CONFIG}")
  if [[ "${BASELINE_ONLY}" -eq 1 ]]; then
    CMD+=(--baseline-only)
  fi
  if [[ "${NO_AGGREGATE}" -eq 1 ]]; then
    CMD+=(--no-aggregate)
  fi

  echo -n "[info] command:"
  printf " %q" "${CMD[@]}"
  echo
  "${CMD[@]}"
  exit 0
fi

case "${MATRIX_MODE}" in
  baseline|phase1|phase2|all)
    ;;
  *)
    echo "[fatal] invalid --matrix mode: ${MATRIX_MODE}" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ "${SKIP_PREFLIGHT}" -eq 0 ]]; then
  CHECK_CONFIGS=()
  if [[ "${PROTOCOL_VERSION}" == "v2" ]]; then
    case "${MATRIX_MODE}" in
      baseline|phase1)
        CHECK_CONFIGS+=(
          "configs/llama3_8b_lora_l40_protocol_v2_phase1_r16_seed42.yaml"
          "configs/llama3_8b_qlora_l40_protocol_v2_phase1_r16_seed42.yaml"
        )
        ;;
      phase2)
        CHECK_CONFIGS+=(
          "configs/llama3_8b_qlora_l40_protocol_v2_phase2_r32_seed42.yaml"
          "configs/llama3_8b_qlora_l40_protocol_v2_phase2_r64_seed42.yaml"
        )
        ;;
      all)
        CHECK_CONFIGS+=(
          "configs/llama3_8b_lora_l40_protocol_v2_phase1_r16_seed42.yaml"
          "configs/llama3_8b_qlora_l40_protocol_v2_phase1_r16_seed42.yaml"
          "configs/llama3_8b_qlora_l40_protocol_v2_phase2_r32_seed42.yaml"
          "configs/llama3_8b_qlora_l40_protocol_v2_phase2_r64_seed42.yaml"
        )
        ;;
    esac
  else
    case "${MATRIX_MODE}" in
      baseline|phase1)
        CHECK_CONFIGS+=(
          "configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml"
          "configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed42.yaml"
        )
        ;;
      phase2)
        CHECK_CONFIGS+=(
          "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r32_seed42.yaml"
          "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r64_seed42.yaml"
        )
        ;;
      all)
        CHECK_CONFIGS+=(
          "configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml"
          "configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed42.yaml"
          "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r32_seed42.yaml"
          "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r64_seed42.yaml"
        )
        ;;
    esac
  fi

  for cfg in "${CHECK_CONFIGS[@]}"; do
    "${PREFLIGHT}" "${cfg}"
  done
fi

CMD=("${RUN_MATRIX}" "${MATRIX_MODE}")
echo -n "[info] command:"
printf " %q" "${CMD[@]}"
echo
echo "[info] protocol_version: ${PROTOCOL_VERSION}"
"${CMD[@]}"
