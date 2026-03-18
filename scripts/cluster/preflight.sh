#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
usage: scripts/cluster/preflight.sh <config.yaml> [--allow-missing-hf-auth]

checks:
  - config readability
  - cuda visibility
  - output directory write access
  - hf auth presence (HF_TOKEN env or cached token file)
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

CONFIG="$1"
shift
ALLOW_MISSING_HF_AUTH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-missing-hf-auth)
      ALLOW_MISSING_HF_AUTH=1
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

if ! command -v python3 >/dev/null 2>&1; then
  echo "[fatal] python3 not found" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[fatal] nvidia-smi not found" >&2
  exit 1
fi

mapfile -t CFG_INFO < <(
  python3 - "${CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg["training"]["output_dir"])
print(cfg["model"]["base_model"])
print("1" if bool(cfg.get("hardware", {}).get("use_4bit", False)) else "0")
PY
)

OUTPUT_DIR="${CFG_INFO[0]}"
BASE_MODEL="${CFG_INFO[1]}"
USE_4BIT="${CFG_INFO[2]}"

mkdir -p "${OUTPUT_DIR}/raw" "${OUTPUT_DIR}/reports"
WRITE_PROBE="${OUTPUT_DIR}/raw/.write_probe"
touch "${WRITE_PROBE}"
rm -f "${WRITE_PROBE}"

echo "[info] config: ${CONFIG}"
echo "[info] base_model: ${BASE_MODEL}"
echo "[info] use_4bit: ${USE_4BIT}"
echo "[info] output_dir: ${OUTPUT_DIR}"
echo "[info] gpu summary:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python3 - "${USE_4BIT}" <<'PY'
import importlib
import sys
import torch

use_4bit = sys.argv[1] == "1"
if not torch.cuda.is_available():
    raise SystemExit("[fatal] torch reports cuda unavailable")

for mod in ("transformers", "datasets", "peft", "yaml"):
    importlib.import_module(mod)

if use_4bit:
    importlib.import_module("bitsandbytes")

print(f"[info] torch_cuda_available={torch.cuda.is_available()}")
print(f"[info] torch_cuda_device_count={torch.cuda.device_count()}")
print(f"[info] torch_cuda_device_name={torch.cuda.get_device_name(0)}")
PY

HF_AUTH_OK=0
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_AUTH_OK=1
else
  for token_path in \
    "${HF_TOKEN_PATH:-}" \
    "${HOME}/.cache/huggingface/token" \
    "${HOME}/.huggingface/token"
  do
    if [[ -n "${token_path}" && -s "${token_path}" ]]; then
      HF_AUTH_OK=1
      break
    fi
  done
fi

if [[ "${HF_AUTH_OK}" -eq 0 && "${ALLOW_MISSING_HF_AUTH}" -eq 0 ]]; then
  echo "[fatal] no HF auth found (set HF_TOKEN or cached token file)" >&2
  exit 1
fi

if [[ "${HF_AUTH_OK}" -eq 1 ]]; then
  echo "[info] hf_auth: present"
else
  echo "[warn] hf_auth: missing (allowed by --allow-missing-hf-auth)"
fi

echo "[info] preflight checks passed"
