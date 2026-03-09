#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/llama3_3b_lora_domain_calibration.yaml"
TMP_BASE="/tmp/${USER}/llama-peft"
TMP_CONFIG="${TMP_BASE}/configs/llama3_3b_lora_domain_calibration.4070.tmp.yaml"
TMP_OUTPUT_DIR="${TMP_BASE}/outputs/llama3_3b_lora_domain_calibration"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[fatal] config not found: ${CONFIG}" >&2
  exit 1
fi

# Keep all download/cache/temp activity off home quota.
mkdir -p \
  "${TMP_BASE}/hf-cache/hub" \
  "${TMP_BASE}/hf-cache/transformers" \
  "${TMP_BASE}/hf-cache/xet" \
  "${TMP_BASE}/xdg-cache" \
  "${TMP_BASE}/tmp" \
  "${TMP_BASE}/configs" \
  "${TMP_OUTPUT_DIR}"

export HF_HOME="${TMP_BASE}/hf-cache"
export HUGGINGFACE_HUB_CACHE="${TMP_BASE}/hf-cache/hub"
export TRANSFORMERS_CACHE="${TMP_BASE}/hf-cache/transformers"
export HF_XET_CACHE="${TMP_BASE}/hf-cache/xet"
export XDG_CACHE_HOME="${TMP_BASE}/xdg-cache"
export TMPDIR="${TMP_BASE}/tmp"
export PYTHONNOUSERSITE=1

cp "${CONFIG}" "${TMP_CONFIG}"
sed -i "s#^  output_dir: .*#  output_dir: \"${TMP_OUTPUT_DIR}\"#" "${TMP_CONFIG}"

echo "[info] starting LoRA calibration on 4070"
echo "[info] base config: ${CONFIG}"
echo "[info] temp config: ${TMP_CONFIG}"
echo "[info] output_dir override: ${TMP_OUTPUT_DIR}"
echo "[info] cwd: ${ROOT_DIR}"
echo "[info] HF_HOME: ${HF_HOME}"
echo "[info] TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE}"
echo "[info] HF_XET_CACHE: ${HF_XET_CACHE}"
echo "[info] TMPDIR: ${TMPDIR}"
echo "[info] PYTHONNOUSERSITE: ${PYTHONNOUSERSITE}"
echo "[info] python: $(command -v python)"
echo "[info] python version: $(python --version 2>&1)"
echo "[info] pip: $(python -m pip --version 2>&1)"

python -m src.train_lora --config "${TMP_CONFIG}"
