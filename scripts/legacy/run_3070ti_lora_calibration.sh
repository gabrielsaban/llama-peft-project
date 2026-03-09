#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/llama3_3b_lora_domain_calibration.yaml"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[fatal] config not found: ${CONFIG}" >&2
  exit 1
fi

echo "[info] starting LoRA calibration on 3070 Ti"
echo "[info] config: ${CONFIG}"
echo "[info] cwd: ${ROOT_DIR}"

python3 -m src.train_lora --config "${CONFIG}"
