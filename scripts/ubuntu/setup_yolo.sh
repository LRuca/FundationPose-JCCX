#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${JXCX_YOLO_ENV:-jxcx-yolo}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or Mambaforge first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.9
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install \
  torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "$ROOT/requirements-yolo.txt"

python - <<'PY'
import torch
import ultralytics
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"ultralytics={ultralytics.__version__}")
PY

