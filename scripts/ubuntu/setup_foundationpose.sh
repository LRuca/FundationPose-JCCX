#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FP_DIR="${FOUNDATIONPOSE_DIR:-$ROOT/third_party/FoundationPose}"
FP_COMMIT="e3d597b8c6b851d053094ebd6fa240191c5238f8"

if [[ ! -d "$FP_DIR/.git" ]]; then
  mkdir -p "$(dirname "$FP_DIR")"
  git clone https://github.com/NVlabs/FoundationPose.git "$FP_DIR"
fi

git -C "$FP_DIR" fetch --all --tags
git -C "$FP_DIR" checkout "$FP_COMMIT"
cp -a "$ROOT/foundationpose_overlay/." "$FP_DIR/"

echo "FoundationPose source prepared at: $FP_DIR"
echo "Next choose one environment path:"
echo "  1. Official Docker workflow in $FP_DIR/readme.md (recommended)."
echo "  2. Conda workflow in $FP_DIR/readme.md using Python 3.9 and CUDA 11.8."
echo "After installing dependencies, run: bash $FP_DIR/build_all_conda.sh"
echo "Download the official scorer/refiner weights using docs/HANDOFF_UBUNTU.md."
