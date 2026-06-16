#!/usr/bin/env bash
# 录制一条 demo 镜次（多镜次互不覆盖）。
# 用法: [MAX_FRAMES=180] [ROI=...] [REPAIR_RADIUS=60] [EST_REFINE_ITER=5] [TRACK_REFINE_ITER=2] bash scripts/record_take.sh [镜次名]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="$ROOT/third_party/orbbec_sdk_v1/OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_x64_release/OrbbecSDK_v1.10.27"
LIVE_DIR="$ROOT/third_party/FoundationPose/live_orbbec"
PUB="$ROOT/build/tools/orbbec_live_publisher"
YOLO_PY="$HOME/miniconda3/envs/jxcx-yolo/bin/python"
FP_PY="$HOME/miniconda3/envs/foundationpose/bin/python"

MAX_FRAMES="${MAX_FRAMES:-180}"
CONF="${CONF:-0.05}"
IMG_SIZE="${IMG_SIZE:-960}"
EST_REFINE_ITER="${EST_REFINE_ITER:-5}"
TRACK_REFINE_ITER="${TRACK_REFINE_ITER:-2}"
REPAIR_RADIUS="${REPAIR_RADIUS:-60}"
ROI="${ROI:-230,90,450,320}"
MIN_BBOX_HEIGHT="${MIN_BBOX_HEIGHT:-40}"
FPS="${FPS:-15}"
PREVIEW="${PREVIEW:-1}"

NAME="${1:-take_$(date +%Y%m%d_%H%M%S)}"
TAKE_DIR="$ROOT/report/takes/$NAME"
DEBUG_DIR="$TAKE_DIR/fp_debug"
mkdir -p "$TAKE_DIR" "$ROOT/logs"

echo "================ 录制镜次: $NAME ================"
echo "输出目录: $TAKE_DIR  | 帧数: $MAX_FRAMES | ROI: $ROI | est/track iter: $EST_REFINE_ITER/$TRACK_REFINE_ITER"

cleanup() {
  set +e
  [[ -n "${PREVIEW_PID:-}" ]] && kill "$PREVIEW_PID" 2>/dev/null
  [[ -n "${YOLO_PID:-}" ]] && kill "$YOLO_PID" 2>/dev/null
  [[ -n "${CAP_PID:-}" ]] && kill "$CAP_PID" 2>/dev/null
}
trap cleanup EXIT

# 采集器
[[ -x "$PUB" ]] || { echo "publisher 未编译，先跑一次 scripts/run_realtime_needle_demo.sh 或 preview_camera.sh"; exit 1; }
rm -f "$LIVE_DIR"/mask_yolo.png "$LIVE_DIR"/mask_yolo.json
"$PUB" --out_dir "$LIVE_DIR" --max_frames 0 --warmup 5 \
  > "$ROOT/logs/orbbec_live_publisher.log" 2>&1 &
CAP_PID=$!
for _ in $(seq 1 100); do
  [[ -s "$LIVE_DIR/color.ppm" && -s "$LIVE_DIR/depth.pgm" && -s "$LIVE_DIR/cam_K.txt" ]] && break
  sleep 0.1
done

# YOLO mask 桥接（只为首帧注册提供 mask）
"$YOLO_PY" "$ROOT/scripts/yolo_mug_mask_bridge.py" \
  --image "$LIVE_DIR/color.ppm" --mask "$LIVE_DIR/mask_yolo.png" --meta "$LIVE_DIR/mask_yolo.json" \
  --model "$ROOT/runs/needle_inbox_seg/yolov8n_seg_combined/weights/best.pt" \
  --class_name needle_inbox --conf "$CONF" --imgsz "$IMG_SIZE" --min_area 50 \
  --roi "$ROI" --min_bbox_height "$MIN_BBOX_HEIGHT" --device 0 \
  --clear_on_fail --loop --poll_interval 0.05 --log "$ROOT/logs/yolo_mug_mask_bridge.log" &
YOLO_PID=$!
for _ in $(seq 1 160); do [[ -s "$LIVE_DIR/mask_yolo.png" ]] && break; sleep 0.1; done

# 实时相机预览窗（显示 ROI + 注册状态；不抢占相机，只读发布文件）
if [[ "$PREVIEW" == "1" ]]; then
  ROI="$ROI" "$YOLO_PY" "$ROOT/tools/live_preview.py" \
    --live_dir "$LIVE_DIR" --roi "$ROI" --track_dir "$DEBUG_DIR/track_vis" \
    > "$ROOT/logs/live_preview.log" 2>&1 &
  PREVIEW_PID=$!
fi

# FoundationPose 注册 + 跟踪
cd "$ROOT/third_party/FoundationPose"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$FP_PY" run_orbbec_mug_live.py \
  --live_dir live_orbbec \
  --mesh_file ../../model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl \
  --mask_file live_orbbec/mask_yolo.png \
  --debug_dir "$DEBUG_DIR" \
  --max_frames "$MAX_FRAMES" --min_mask_area 50 --min_valid_depth_points 5 \
  --repair_mask_depth --repair_radius "$REPAIR_RADIUS" \
  --est_refine_iter "$EST_REFINE_ITER" --track_refine_iter "$TRACK_REFINE_ITER" --debug 1
RC=$?
cd "$ROOT"
[[ $RC -ne 0 ]] && { echo "跟踪进程异常退出 rc=$RC，看 logs/"; exit $RC; }

# 视频 + 评估
"$FP_PY" tools/make_realtime_demo_video.py --frames "$DEBUG_DIR/track_vis" --out "$TAKE_DIR/demo.mp4" --fps "$FPS"
"$FP_PY" tools/evaluate_pose_tracking.py --pose-dir "$DEBUG_DIR/ob_in_cam" --fps "$FPS" --out-dir "$TAKE_DIR/eval"

NFR=$(ls "$DEBUG_DIR/track_vis" 2>/dev/null | wc -l)
echo "================ 镜次完成: $NAME ================"
echo "  跟踪帧数: $NFR"
echo "  视频:     $TAKE_DIR/demo.mp4"
echo "  位姿序列: $DEBUG_DIR/ob_in_cam"
echo "  评估:     $TAKE_DIR/eval"
