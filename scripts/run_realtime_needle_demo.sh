#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="$ROOT/third_party/orbbec_sdk_v1/OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_x64_release/OrbbecSDK_v1.10.27"
LIVE_DIR="$ROOT/third_party/FoundationPose/live_orbbec"
DEBUG_DIR="$ROOT/third_party/FoundationPose/debug_needle_live"
YOLO_PY="$HOME/miniconda3/envs/jxcx-yolo/bin/python"
FP_PY="$HOME/miniconda3/envs/foundationpose/bin/python"

MAX_FRAMES="${MAX_FRAMES:-180}"
CONF="${CONF:-0.05}"
IMG_SIZE="${IMG_SIZE:-960}"
EST_REFINE_ITER="${EST_REFINE_ITER:-3}"
TRACK_REFINE_ITER="${TRACK_REFINE_ITER:-1}"
REPAIR_RADIUS="${REPAIR_RADIUS:-60}"
ROI="${ROI:-230,90,450,320}"
MIN_BBOX_HEIGHT="${MIN_BBOX_HEIGHT:-40}"

cleanup() {
  set +e
  if [[ -n "${YOLO_PID:-}" ]]; then kill "$YOLO_PID" 2>/dev/null; fi
  if [[ -n "${CAP_PID:-}" ]]; then kill "$CAP_PID" 2>/dev/null; fi
}
trap cleanup EXIT

mkdir -p "$ROOT/build/tools" "$LIVE_DIR" "$ROOT/report" "$ROOT/logs"

g++ -std=c++11 \
  "$ROOT/tools/orbbec_live_publisher.cpp" \
  -I"$SDK/SDK/include" \
  -L"$SDK/SDK/lib" \
  -lOrbbecSDK \
  -Wl,-rpath,"$SDK/SDK/lib" \
  -o "$ROOT/build/tools/orbbec_live_publisher"

rm -rf "$DEBUG_DIR"
rm -f "$LIVE_DIR"/mask_yolo.png "$LIVE_DIR"/mask_yolo.json

"$ROOT/build/tools/orbbec_live_publisher" \
  --out_dir "$LIVE_DIR" \
  --max_frames 0 \
  --warmup 5 \
  > "$ROOT/logs/orbbec_live_publisher.log" 2>&1 &
CAP_PID=$!

for _ in $(seq 1 80); do
  if [[ -s "$LIVE_DIR/color.ppm" && -s "$LIVE_DIR/depth.pgm" && -s "$LIVE_DIR/cam_K.txt" ]]; then
    break
  fi
  sleep 0.1
done

"$YOLO_PY" "$ROOT/scripts/yolo_mug_mask_bridge.py" \
  --image "$LIVE_DIR/color.ppm" \
  --mask "$LIVE_DIR/mask_yolo.png" \
  --meta "$LIVE_DIR/mask_yolo.json" \
  --model "$ROOT/runs/needle_inbox_seg/yolov8n_seg_combined/weights/best.pt" \
  --class_name needle_inbox \
  --conf "$CONF" \
  --imgsz "$IMG_SIZE" \
  --min_area 50 \
  --roi "$ROI" \
  --min_bbox_height "$MIN_BBOX_HEIGHT" \
  --device 0 \
  --clear_on_fail \
  --loop \
  --poll_interval 0.05 \
  --log "$ROOT/logs/yolo_mug_mask_bridge.log" &
YOLO_PID=$!

for _ in $(seq 1 160); do
  if [[ -s "$LIVE_DIR/mask_yolo.png" ]]; then
    break
  fi
  sleep 0.1
done

cd "$ROOT/third_party/FoundationPose"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$FP_PY" run_orbbec_mug_live.py \
  --live_dir live_orbbec \
  --mesh_file ../../model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl \
  --mask_file live_orbbec/mask_yolo.png \
  --debug_dir debug_needle_live \
  --max_frames "$MAX_FRAMES" \
  --min_mask_area 50 \
  --min_valid_depth_points 5 \
  --repair_mask_depth \
  --repair_radius "$REPAIR_RADIUS" \
  --est_refine_iter "$EST_REFINE_ITER" \
  --track_refine_iter "$TRACK_REFINE_ITER" \
  --debug 1

cd "$ROOT"
"$FP_PY" tools/make_realtime_demo_video.py \
  --frames "$DEBUG_DIR/track_vis" \
  --out "$ROOT/report/needle_realtime_demo.mp4" \
  --fps 15

"$FP_PY" tools/evaluate_pose_tracking.py \
  --pose-dir "$DEBUG_DIR/ob_in_cam" \
  --fps 15 \
  --out-dir "$ROOT/report/realtime_eval"

echo "demo video: $ROOT/report/needle_realtime_demo.mp4"
