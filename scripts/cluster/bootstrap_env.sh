#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
usage: scripts/cluster/bootstrap_env.sh [--env-name NAME] [--env-file FILE] [--with-requirements]

options:
  --env-name NAME         conda env name (default: llama-peft or $CONDA_ENV_NAME)
  --env-file FILE         environment yaml (default: environment.yaml)
  --with-requirements     also run pip install -r requirements.txt after conda update/create
EOF
}

ENV_NAME="${CONDA_ENV_NAME:-llama-peft}"
ENV_FILE="environment.yaml"
WITH_REQUIREMENTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || { echo "[fatal] --env-name requires a value" >&2; exit 1; }
      ENV_NAME="$2"
      shift
      ;;
    --env-file)
      [[ $# -ge 2 ]] || { echo "[fatal] --env-file requires a value" >&2; exit 1; }
      ENV_FILE="$2"
      shift
      ;;
    --with-requirements)
      WITH_REQUIREMENTS=1
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

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[fatal] env file not found: ${ENV_FILE}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[fatal] conda not found in PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_EXISTS="$(
  conda env list --json | python3 - "${ENV_NAME}" <<'PY'
import json
import os
import sys

target = sys.argv[1]
data = json.load(sys.stdin)
suffix = os.sep + target
exists = any(p.endswith(suffix) for p in data.get("envs", []))
print("1" if exists else "0")
PY
)"

if [[ "${ENV_EXISTS}" == "1" ]]; then
  echo "[info] updating conda env: ${ENV_NAME}"
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "[info] creating conda env: ${ENV_NAME}"
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"

if [[ "${WITH_REQUIREMENTS}" -eq 1 ]]; then
  if [[ ! -f "requirements.txt" ]]; then
    echo "[fatal] requirements.txt not found but --with-requirements set" >&2
    exit 1
  fi
  echo "[info] installing pip requirements"
  python -m pip install -r requirements.txt
fi

echo "[info] running dependency import checks"
python - <<'PY'
import importlib

mods = ["torch", "transformers", "datasets", "peft", "bitsandbytes", "yaml"]
for mod in mods:
    importlib.import_module(mod)

import torch
print("[info] import_check=ok")
print(f"[info] torch_version={torch.__version__}")
print(f"[info] cuda_available={torch.cuda.is_available()}")
PY
