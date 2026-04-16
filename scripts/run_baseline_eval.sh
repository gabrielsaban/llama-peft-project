#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUNNER="${ROOT_DIR}/scripts/l40/run_l40_config.sh"
DEFAULT_CONFIGS=(
  "configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed42.yaml"
)

usage() {
  cat <<'EOF'
usage: scripts/run_baseline_eval.sh [config.yaml ...] [--no-aggregate]

defaults:
  if no config is supplied, runs baseline-only for both protocol-v3 phase1 seed42 configs.

options:
  --no-aggregate  skip final src.compare_runs aggregation
EOF
}

if [[ $# -ge 1 ]] && [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -x "${RUNNER}" ]]; then
  echo "[fatal] runner missing or not executable: ${RUNNER}" >&2
  exit 1
fi

DO_AGGREGATE=1
CONFIGS=()
for arg in "$@"; do
  case "${arg}" in
    --no-aggregate)
      DO_AGGREGATE=0
      ;;
    *)
      CONFIGS+=("${arg}")
      ;;
  esac
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "[fatal] config not found: ${config}" >&2
    exit 1
  fi
done

echo "[info] running baseline-only perplexity eval"
echo "[info] cwd: ${ROOT_DIR}"
echo "[info] configs (${#CONFIGS[@]}):"
for config in "${CONFIGS[@]}"; do
  echo "[info]  - ${config}"
done

for config in "${CONFIGS[@]}"; do
  echo "[info] baseline-only -> ${config}"
  "${RUNNER}" "${config}" --baseline-only --no-aggregate
done

if [[ "${DO_AGGREGATE}" -eq 1 ]]; then
  if ! python3 -m src.compare_runs --root outputs --out-dir outputs/reports_index; then
    echo "[warn] compare_runs aggregation failed"
  fi
fi
