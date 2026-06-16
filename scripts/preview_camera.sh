#!/usr/bin/env bash
# 实时相机预览：后台采集 + 预览窗口。按 q 关窗即停止采集。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="$ROOT/third_party/orbbec_sdk_v1/OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_x64_release/OrbbecSDK_v1.10.27"
LIVE_DIR="$ROOT/third_party/FoundationPose/live_orbbec"
PUB="$ROOT/build/tools/orbbec_live_publisher"
YOLO_PY="$HOME/miniconda3/envs/jxcx-yolo/bin/python"
ROI="${ROI:-230,90,450,320}"

mkdir -p "$LIVE_DIR" "$ROOT/logs"

cleanup() { [[ -n "${CAP_PID:-}" ]] && kill "$CAP_PID" 2>/dev/null; }
trap cleanup EXIT

# 采集器不存在则编译
if [[ ! -x "$PUB" ]]; then
  echo "[preview] building publisher..."
  mkdir -p "$ROOT/build/tools"
  g++ -std=c++11 "$ROOT/tools/orbbec_live_publisher.cpp" \
    -I"$SDK/SDK/include" -L"$SDK/SDK/lib" -lOrbbecSDK \
    -Wl,-rpath,"$SDK/SDK/lib" -o "$PUB" || { echo "build failed"; exit 1; }
fi

echo "[preview] starting camera publisher..."
"$PUB" --out_dir "$LIVE_DIR" --max_frames 0 --warmup 5 \
  > "$ROOT/logs/orbbec_live_publisher.log" 2>&1 &
CAP_PID=$!

# 等首帧
for _ in $(seq 1 100); do
  [[ -s "$LIVE_DIR/color.ppm" ]] && break
  sleep 0.1
done
if [[ ! -s "$LIVE_DIR/color.ppm" ]]; then
  echo "[preview] no frame produced; see logs/orbbec_live_publisher.log"; tail -n 20 "$ROOT/logs/orbbec_live_publisher.log"; exit 1
fi

echo "[preview] opening window (press q to quit)..."
ROI="$ROI" "$YOLO_PY" "$ROOT/tools/live_preview.py" --live_dir "$LIVE_DIR" --roi "$ROI"
