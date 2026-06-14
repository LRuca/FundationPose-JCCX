#!/usr/bin/env python
"""Run FoundationPose registration N times, save the best result."""
import argparse
from pathlib import Path
import sys, os, cv2, numpy as np, trimesh, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimater import *

parser = argparse.ArgumentParser()
parser.add_argument("--live_dir", default="live_orbbec")
parser.add_argument("--mesh_file", required=True)
parser.add_argument("--runs", type=int, default=5)
args = parser.parse_args()

N = args.runs
LIVE = str(Path(args.live_dir).resolve())
MESH = str(Path(args.mesh_file).resolve())

color = cv2.imread(f"{LIVE}/color.png")
depth = cv2.imread(f"{LIVE}/depth.png", -1).astype(np.float32) / 1000.0
with open(f"{LIVE}/cam_K.txt") as f:
    K = np.loadtxt(f).reshape(3, 3).astype(np.float32)
mask = cv2.imread(f"{LIVE}/mask_yolo.png", -1) > 0

mesh = trimesh.load(MESH)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
v = np.asarray(mesh.vertices, np.float32)
n = np.asarray(mesh.vertex_normals, np.float32)
m = trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.faces), vertex_normals=n, process=False)

set_logging_format()

best_score = -1e9
best_vis = None
best_pose = None

for i in range(N):
    set_seed(i * 137 + 42)  # different seed each time
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=v, model_normals=n, mesh=m,
        scorer=scorer, refiner=refiner,
        debug_dir=f"{LIVE}/../debug_orbbec_mug", debug=0, glctx=glctx,
    )
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)

    # Read add_errs if available
    err_file = f"{LIVE}/../debug_orbbec_mug/add_errs.txt"
    if os.path.exists(err_file):
        scores = np.loadtxt(err_file)
        score = float(np.max(scores))
    else:
        score = -1.0

    cpose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=color.copy(), ob_in_cam=cpose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=cpose, scale=0.05, K=K, thickness=2,
                         transparency=0, is_input_rgb=True)

    print(f"[{i+1}/{N}] seed={i*137+42} score={score:.4f}")

    if score > best_score:
        best_score = score
        best_vis = vis
        best_pose = pose

cv2.imwrite(f"{LIVE}/track_vis_single.png", best_vis)
np.savetxt(f"{LIVE}/pose_single.txt", best_pose.reshape(4, 4))
print(f"\nBest: score={best_score:.4f}")
print("DONE")
