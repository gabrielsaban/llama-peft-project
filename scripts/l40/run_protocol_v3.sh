#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUNNER="${ROOT_DIR}/scripts/l40/run_l40_config.sh"

usage() {
  cat <<'EOF'
usage: scripts/l40/run_protocol_v3.sh [baseline|phase1|phase2|all]

modes:
  baseline  run baseline-only evaluations for LoRA and QLoRA phase-1 seed42 configs
  phase1    run core comparison training runs across seeds (LoRA r16, QLoRA r16)
  phase2    run QLoRA rank sweep training runs across seeds (r32, r64)
  all       run baseline, phase1, then phase2 in sequence
EOF
}

if [[ ! -x "${RUNNER}" ]]; then
  echo "[fatal] runner missing or not executable: ${RUNNER}" >&2
  exit 1
fi

MODE="${1:-phase1}"
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 1
fi

BASELINE_CONFIGS=(
  "configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed42.yaml"
)

PHASE1_CONFIGS=(
  "configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml"
  "configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed43.yaml"
  "configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed44.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed42.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed43.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed44.yaml"
)

PHASE2_CONFIGS=(
  "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r32_seed42.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r32_seed43.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r32_seed44.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r64_seed42.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r64_seed43.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r64_seed44.yaml"
)

run_baseline_group() {
  for config in "${BASELINE_CONFIGS[@]}"; do
    echo "[info] baseline-only -> ${config}"
    "${RUNNER}" "${config}" --baseline-only
  done
}

run_train_group() {
  local phase="$1"
  shift
  for config in "$@"; do
    echo "[info] ${phase} -> ${config}"
    "${RUNNER}" "${config}"
  done
}

case "${MODE}" in
  baseline)
    run_baseline_group
    ;;
  phase1)
    run_train_group "phase1" "${PHASE1_CONFIGS[@]}"
    ;;
  phase2)
    run_train_group "phase2" "${PHASE2_CONFIGS[@]}"
    ;;
  all)
    run_baseline_group
    run_train_group "phase1" "${PHASE1_CONFIGS[@]}"
    run_train_group "phase2" "${PHASE2_CONFIGS[@]}"
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "[fatal] unknown mode: ${MODE}" >&2
    usage >&2
    exit 1
    ;;
esac
