#!/usr/bin/env python
"""Continuous tracking + YOLO mask overlay → save frames for video."""
import argparse
from pathlib import Path
import sys, os, cv2, numpy as np, trimesh, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimater import *

parser = argparse.ArgumentParser()
parser.add_argument("--live_dir", default="live_orbbec")
parser.add_argument("--mesh_file", required=True)
parser.add_argument("--output_dir", default="video_frames")
parser.add_argument("--frames", type=int, default=150)
args = parser.parse_args()

N_FRAMES = args.frames
LIVE = str(Path(args.live_dir).resolve())
MESH = str(Path(args.mesh_file).resolve())
OUT_DIR = str(Path(args.output_dir).resolve())

os.makedirs(OUT_DIR, exist_ok=True)
for f in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, f))

def read_live():
    color = cv2.imread(f"{LIVE}/color.png")
    depth_raw = cv2.imread(f"{LIVE}/depth.png", -1)
    if depth_raw is None:
        return None, None, None, None, None
    depth = depth_raw.astype(np.float32) / 1000.0
    K = np.loadtxt(f"{LIVE}/cam_K.txt").reshape(3, 3).astype(np.float32)
    mask_im = cv2.imread(f"{LIVE}/mask_yolo.png", -1)
    mask = (mask_im > 0) if mask_im is not None else None
    if META_FILE := f"{LIVE}/frame.json":
        try:
            with open(META_FILE) as f:
                idx = json.load(f).get("frame_index", "?")
        except:
            idx = "?"
    else:
        idx = "?"
    return color, depth, K, mask, str(idx)

def overlay_mask(color, mask):
    ov = color.copy()
    if mask is not None and mask.sum() > 0:
        ov[:, :, 1] = np.clip(ov[:, :, 1].astype(np.int16) + (mask.astype(np.uint8) * 128).astype(np.int16), 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, contours, -1, (0, 255, 0), 2)
    return ov

def wait_for_new_frame(last_mtime):
    deadline = time.time() + 30
    cf = f"{LIVE}/color.png"
    while time.time() < deadline:
        try:
            mt = os.path.getmtime(cf)
            if mt != last_mtime:
                return mt
        except:
            pass
        time.sleep(0.05)
    raise RuntimeError("timed out waiting for new frame")

set_logging_format()
set_seed(0)

# Init FoundationPose
print("[INIT] loading mesh & models...")
mesh = trimesh.load(MESH)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
v = np.asarray(mesh.vertices, np.float32)
n = np.asarray(mesh.vertex_normals, np.float32)
m = trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.faces), vertex_normals=n, process=False)

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=v, model_normals=n, mesh=m,
    scorer=scorer, refiner=refiner,
    debug_dir=f"{LIVE}/../debug_orbbec_mug", debug=0, glctx=glctx)

# Wait for first frame + mask
print("[WAIT] waiting for frame + mask...")
for _ in range(300):
    color, depth, K, mask, idx = read_live()
    if color is not None and depth is not None and mask is not None and mask.sum() > 50:
        break
    time.sleep(0.1)
else:
    raise RuntimeError("no valid frame/mask")

print(f"[INIT] frame={idx} depth_valid={(depth>0.001).sum()} mask_area={mask.sum()}")

# Register initial pose
pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
print("[INIT] registration done, starting tracking...")

last_mtime = os.path.getmtime(f"{LIVE}/color.png")

for i in range(N_FRAMES):
    t0 = time.time()
    try:
        new_mt = wait_for_new_frame(last_mtime)
        last_mtime = new_mt
    except:
        print(f"[{i}] no new frame, reusing")
    color, depth, K, mask, idx = read_live()
    if color is None:
        continue

    pose = est.track_one(rgb=color, depth=depth, K=K, iteration=2)

    cpose = pose @ np.linalg.inv(to_origin)
    vis = overlay_mask(color.copy(), mask)
    vis = draw_posed_3d_box(K, img=vis, ob_in_cam=cpose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=cpose, scale=0.05, K=K, thickness=2,
                         transparency=0, is_input_rgb=True)
    cv2.putText(vis, f"Frame {i+1}/{N_FRAMES}  YOLO + FoundationPose", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    out = f"{OUT_DIR}/{i:05d}.png"
    cv2.imwrite(out, vis)

    elapsed = time.time() - t0
    fps = 1.0 / elapsed if elapsed > 0 else 0
    if i % 10 == 0:
        print(f"[{i+1}/{N_FRAMES}] {fps:.1f} fps  saved {out}")

print(f"\n[DONE] {N_FRAMES} frames saved to {OUT_DIR}/")
