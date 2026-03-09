#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
usage: scripts/l40/run_l40_config.sh <config.yaml> [--baseline-only] [--no-aggregate]

options:
  --baseline-only   run held-out baseline evaluation only (no training)
  --no-aggregate    skip post-run src.compare_runs aggregation
EOF
}

if [[ $# -ge 1 ]] && [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

CONFIG="$1"
shift

RUN_MODE="train"
DO_AGGREGATE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-only)
      RUN_MODE="baseline_only"
      ;;
    --no-aggregate)
      DO_AGGREGATE=0
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

if [[ ! -f "${CONFIG}" ]]; then
  echo "[fatal] config not found: ${CONFIG}" >&2
  exit 1
fi

OUTPUT_DIR="$(
  python3 - "${CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["training"]["output_dir"])
PY
)"

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "[fatal] failed to resolve training.output_dir from ${CONFIG}" >&2
  exit 1
fi

RAW_DIR="${OUTPUT_DIR}/raw"
mkdir -p "${RAW_DIR}"
LOG_FILE="${RAW_DIR}/run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[info] start_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[info] run_mode: ${RUN_MODE}"
echo "[info] config: ${CONFIG}"
echo "[info] output_dir: ${OUTPUT_DIR}"
echo "[info] run_log: ${LOG_FILE}"
echo "[info] cwd: ${ROOT_DIR}"
echo "[info] python: $(command -v python3)"
echo "[info] python_version: $(python3 --version 2>&1)"

CMD=(python3 -m src.train_lora --config "${CONFIG}")
if [[ "${RUN_MODE}" == "baseline_only" ]]; then
  CMD+=(--baseline-only)
fi

echo -n "[info] command:"
printf " %q" "${CMD[@]}"
echo
"${CMD[@]}"

if [[ "${DO_AGGREGATE}" -eq 1 ]]; then
  echo "[info] aggregating comparison rows"
  python3 -m src.compare_runs --root outputs --out-dir outputs/reports_index
fi
