#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
usage: scripts/reproduce_protocol_v3.sh [inspect|aggregate|single|phase1|phase2|all|slurm] [...]

modes:
  inspect                 check final protocol files and print aggregate summary
  aggregate               rebuild outputs/reports_index from saved per-run reports
  single <config> [args]  run one config through scripts/l40/run_l40_config.sh
  phase1                  run the six LoRA r16 vs QLoRA r16 final configs locally
  phase2                  run the six QLoRA r32/r64 rank-sweep configs locally
  all                     run phase1 then phase2 locally
  slurm [args]            submit jobs/protocol_v3.sbatch with forwarded args

The default mode is inspect. Training modes require suitable GPU/HPC resources.
EOF
}

PROTOCOL_CONFIGS=(
  "configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed42.yaml"
  "configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed43.yaml"
  "configs/llama3_8b_lora_l40_protocol_v3_phase1_r16_seed44.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed42.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed43.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase1_r16_seed44.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r32_seed42.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r32_seed43.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r32_seed44.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r64_seed42.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r64_seed43.yaml"
  "configs/llama3_8b_qlora_l40_protocol_v3_phase2_r64_seed44.yaml"
)

REQUIRED_FILES=(
  "logs/protocol_v3.md"
  "data/domain_corpus/corpus_manifest.json"
  "data/splits/intrinsic_splits.json"
  "src/train_lora.py"
  "src/compare_runs.py"
  "scripts/l40/run_l40_config.sh"
  "scripts/l40/run_protocol_v3.sh"
  "jobs/protocol_v3.sbatch"
)

inspect() {
  local missing=0

  echo "[inspect] repository root: ${ROOT_DIR}"
  echo "[inspect] checking required files"
  for path in "${REQUIRED_FILES[@]}" "${PROTOCOL_CONFIGS[@]}"; do
    if [[ -e "${path}" ]]; then
      echo "  ok      ${path}"
    else
      echo "  missing ${path}"
      missing=1
    fi
  done

  echo
  echo "[inspect] final protocol configs: ${#PROTOCOL_CONFIGS[@]}"

  local report_count=0
  if [[ -d outputs/experiments/protocol_v3 ]]; then
    report_count="$(
      find outputs/experiments/protocol_v3 -mindepth 2 -maxdepth 2 -type d -name reports | wc -l
    )"
  fi
  echo "[inspect] completed run report dirs: ${report_count}"

  local comparison_rows=0
  if [[ -d outputs/experiments/protocol_v3 ]]; then
    comparison_rows="$(
      find outputs/experiments/protocol_v3 -path '*/reports/comparison_row.json' -type f | wc -l
    )"
  fi
  echo "[inspect] comparison rows: ${comparison_rows}"

  local summary="outputs/reports_index/protocol_v3/condition_summary.csv"
  if [[ -f "${summary}" ]]; then
    echo
    echo "[inspect] condition summary (${summary})"
    python3 - "${summary}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields = [
        "condition",
        "n_runs",
        "best_ppl_all_mean",
        "final_ppl_all_mean",
        "train_peak_vram_allocated_gb_mean",
        "step_time_mean_s_mean",
    ]
    rows = list(reader)

print("\t".join(fields))
for row in rows:
    condition = f"{row.get('variant', '')} r{row.get('rank', '')}"
    values = {
        "condition": condition,
        "n_runs": row.get("n_runs", ""),
        "best_ppl_all_mean": row.get("best_ppl_all_mean", ""),
        "final_ppl_all_mean": row.get("final_ppl_all_mean", ""),
        "train_peak_vram_allocated_gb_mean": row.get("train_peak_vram_allocated_gb_mean", ""),
        "step_time_mean_s_mean": row.get("step_time_mean_s_mean", ""),
    }
    print("\t".join(values.get(field, "") for field in fields))
PY
  else
    echo "[inspect] aggregate condition summary not found: ${summary}"
    echo "[inspect] run 'scripts/reproduce_protocol_v3.sh aggregate' after reports exist"
  fi

  return "${missing}"
}

aggregate() {
  python3 -m src.compare_runs --root outputs --out-dir outputs/reports_index/protocol_v3
}

MODE="${1:-inspect}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${MODE}" in
  inspect)
    inspect
    ;;
  aggregate)
    aggregate
    ;;
  single)
    if [[ $# -lt 1 ]]; then
      echo "[fatal] single mode requires a config path" >&2
      usage >&2
      exit 1
    fi
    scripts/l40/run_l40_config.sh "$@"
    ;;
  phase1|phase2)
    if [[ $# -gt 0 ]]; then
      echo "[fatal] mode '${MODE}' does not accept extra arguments" >&2
      usage >&2
      exit 1
    fi
    scripts/l40/run_protocol_v3.sh "${MODE}"
    ;;
  all)
    if [[ $# -gt 0 ]]; then
      echo "[fatal] mode '${MODE}' does not accept extra arguments" >&2
      usage >&2
      exit 1
    fi
    scripts/l40/run_protocol_v3.sh phase1
    scripts/l40/run_protocol_v3.sh phase2
    ;;
  slurm)
    if ! command -v sbatch >/dev/null 2>&1; then
      echo "[fatal] sbatch not found on PATH; use this mode on the Slurm cluster" >&2
      exit 1
    fi
    sbatch jobs/protocol_v3.sbatch "$@"
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
