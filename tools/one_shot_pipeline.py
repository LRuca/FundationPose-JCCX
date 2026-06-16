#!/usr/bin/env python
"""One-shot: grab a live frame, run YOLO mask + FoundationPose, combine results."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Add FoundationPose to path
ROOT = Path(r"C:\Users\lenovo\Desktop\JXCX")
sys.path.insert(0, str(ROOT / "FoundationPose"))

from estimater import *
from ultralytics import YOLO

LIVE_DIR = ROOT / "FoundationPose" / "live_orbbec"
COLOR_FILE = LIVE_DIR / "color.png"
DEPTH_FILE = LIVE_DIR / "depth.png"
K_FILE = LIVE_DIR / "cam_K.txt"
META_FILE = LIVE_DIR / "frame.json"
MASK_FILE = LIVE_DIR / "mask_yolo.png"
MESH_FILE = ROOT / "model" / "fixed_unnamed_object_3" / "needle_structured_tail_reconstruction_v3.stl"
MODEL_PT = ROOT / "runs" / "needle_inbox_seg" / "yolov8n_seg_combined" / "weights" / "best.pt"
OUTPUT = ROOT / "test_combined_result.png"


def wait_for_stable_file(path, checks=5, interval=0.1):
    path = Path(path)
    last = None
    for _ in range(checks):
        stat = path.stat()
        cur = (stat.st_size, stat.st_mtime_ns)
        if last and cur == last:
            return
        last = cur
        time.sleep(interval)


def read_frame():
    wait_for_stable_file(COLOR_FILE, checks=5, interval=0.1)
    color = cv2.imread(str(COLOR_FILE), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(DEPTH_FILE), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("Depth image is None")
    depth = depth.astype(np.float32) / 1000.0
    K = np.loadtxt(str(K_FILE)).reshape(3, 3).astype(np.float32)
    if META_FILE.exists():
        with open(META_FILE, "r", encoding="utf-8-sig") as f:
            frame_idx = json.load(f).get("frame_index", "?")
    else:
        frame_idx = "?"
    return frame_idx, color, depth, K


def run_yolo(color):
    print("[YOLO] running...")
    model = YOLO(str(MODEL_PT))
    cv2.imwrite(str(COLOR_FILE), color)  # ensure latest
    results = model.predict(str(COLOR_FILE), imgsz=960, conf=0.25, device="0", verbose=False)
    r = results[0]
    if r.masks is None or len(r.masks) == 0:
        raise RuntimeError("No masks detected")
    # pick best
    confs = r.boxes.conf.detach().cpu().numpy()
    masks = r.masks.data.detach().cpu().numpy()
    best_i = int(np.argmax(confs))
    mask = masks[best_i]
    h, w = color.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    cv2.imwrite(str(MASK_FILE), mask_u8)
    print(f"[YOLO] mask saved, conf={float(confs[best_i]):.3f}, area={int(np.count_nonzero(mask_u8))}")
    return mask_u8, float(confs[best_i])


def run_foundationpose(color, depth, K, mask_u8):
    print("[FP] initializing...")
    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(str(MESH_FILE))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    mesh_f32 = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=vertex_normals, process=False)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_f32)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=vertices,
        model_normals=vertex_normals,
        mesh=mesh_f32,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(ROOT / "FoundationPose" / "debug_orbbec_mug"),
        debug=0,
        glctx=glctx,
    )
    mask_bool = mask_u8 > 0
    print("[FP] registering...")
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask_bool, iteration=5)
    print(f"[FP] pose:\n{pose.reshape(4,4)}")

    # draw pose overlay
    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=color.copy(), ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.05, K=K, thickness=2, transparency=0, is_input_rgb=True)
    return vis


def combine_display(original, yolo_mask, pose_vis, conf):
    """Side-by-side: [original + YOLO overlay]  [pose visualization]"""
    h, w = original.shape[:2]
    # resize all to same height
    target_h = 480
    scale = target_h / h
    target_w = int(w * scale)

    def resize(im):
        return cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # panel 1: original + mask overlay
    mask_color = np.zeros_like(original)
    mask_color[:, :, 1] = yolo_mask  # green mask
    overlay = cv2.addWeighted(original, 0.7, mask_color, 0.3, 0)
    # draw contour
    contours, _ = cv2.findContours(yolo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    cv2.putText(overlay, f"needle_inbox conf={conf:.3f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    panel1 = resize(overlay)

    # panel 2: pose visualization
    cv2.putText(pose_vis, "FoundationPose", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    panel2 = resize(pose_vis)

    # combine
    gap = np.zeros((target_h, 4, 3), dtype=np.uint8)
    combined = np.hstack([panel1, gap, panel2])
    cv2.putText(combined, "YOLO Mask", (10, target_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(combined, "Pose Track", (target_w + 14, target_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return combined


def main():
    print(f"=== needle_inbox one-shot pipeline ===")

    # 1. Read frame
    frame_idx, color, depth, K = read_frame()
    print(f"[FRAME] idx={frame_idx}, color={color.shape}, depth={depth.shape}")

    # 2. YOLO mask
    mask_u8, conf = run_yolo(color)

    # 3. FoundationPose
    pose_vis = run_foundationpose(color, depth, K, mask_u8)

    # 4. Combine & save
    combined = combine_display(color, mask_u8, pose_vis, conf)
    cv2.imwrite(str(OUTPUT), combined)
    print(f"[DONE] saved to {OUTPUT}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Frame: {frame_idx}")
    print(f"YOLO: needle_inbox conf={conf:.3f}")
    print(f"Depth: valid={int((depth>0.001).sum())}, max={depth.max():.4f}m")
    print(f"Image: {OUTPUT}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
