#!/usr/bin/env python
"""FoundationPose 实时效率基准（纯计算测量，不需要相机）。

测量内容：
- 首帧注册耗时（est_refine_iter 扫描）
- 后续帧跟踪耗时与可达 FPS（track_refine_iter 扫描）
- 每种配置的峰值显存

输出：JSON + CSV，供 tools/plot_efficiency_charts.py 绘制中文图表。

需在 foundationpose 环境、third_party/FoundationPose 目录可导入 estimater 的前提下运行。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 允许从 third_party/FoundationPose 导入 estimater
FP_DIR = Path(__file__).resolve().parents[1] / "third_party" / "FoundationPose"
sys.path.insert(0, str(FP_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, set_seed  # noqa: E402


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


def read_frame(live_dir):
    color_file = first_existing(f"{live_dir}/color.ppm", f"{live_dir}/color.png")
    depth_file = first_existing(f"{live_dir}/depth.pgm", f"{live_dir}/depth.png")
    color = imageio.imread(color_file)[..., :3]
    depth = cv2.imread(depth_file, -1).astype(np.float32) / 1000.0
    K = np.loadtxt(f"{live_dir}/cam_K.txt").reshape(3, 3).astype(np.float32)
    return color, depth, K


def read_mask(mask_file, target_shape):
    mask = cv2.imread(mask_file, -1)
    if mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def repair_depth_under_mask(depth, mask, radius, min_valid_points=5):
    mask_bool = mask.astype(bool)
    current_valid = (depth >= 0.001) & mask_bool
    if int(current_valid.sum()) >= min_valid_points:
        return depth
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    search = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
    source = search & (depth >= 0.001)
    if int(source.sum()) < min_valid_points:
        return depth
    fill_value = float(np.median(depth[source]))
    repaired = depth.copy()
    repaired[mask_bool & (repaired < 0.001)] = fill_value
    return repaired


def force_mesh_float32(mesh):
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces),
        vertex_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
        process=False,
    )


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_vram_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live_dir", default=str(FP_DIR / "live_orbbec"))
    ap.add_argument("--mesh_file", default=str(Path(__file__).resolve().parents[1] /
                    "model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl"))
    ap.add_argument("--mask_file", default=None, help="默认用 live_dir/mask_yolo.png")
    ap.add_argument("--repair_radius", type=int, default=60)
    ap.add_argument("--est_iters", default="1,3,5,10")
    ap.add_argument("--track_iters", default="1,2,5")
    ap.add_argument("--track_reps", type=int, default=40, help="每个 track 配置重复帧数（含预热）")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] /
                    "report/realtime_eval/efficiency/pose_efficiency.json"))
    args = ap.parse_args()

    mask_file = args.mask_file or f"{args.live_dir}/mask_yolo.png"
    est_iters = [int(x) for x in args.est_iters.split(",") if x.strip()]
    track_iters = [int(x) for x in args.track_iters.split(",") if x.strip()]

    set_seed(0)
    color, depth, K = read_frame(args.live_dir)
    mask = read_mask(mask_file, depth.shape[:2])
    depth = repair_depth_under_mask(depth, mask, radius=args.repair_radius)
    valid = int(((depth >= 0.001) & mask).sum())
    print(f"frame: color={color.shape} depth={depth.shape} mask_px={int(mask.sum())} mask_depth_valid={valid}",
          flush=True)
    if valid < 5:
        raise SystemExit(f"mask/depth 重合点太少({valid})，无法注册；换一帧或调大 repair_radius。")

    mesh = force_mesh_float32(trimesh.load(args.mesh_file))
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices.astype(np.float32),
        model_normals=mesh.vertex_normals.astype(np.float32),
        mesh=mesh, scorer=scorer, refiner=refiner,
        debug_dir="/tmp/fp_bench_debug", debug=0, glctx=glctx,
    )
    print("estimator ready", flush=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    # ---- 注册耗时扫描（首帧 6D 初始化）----
    register_rows = []
    # 预热一次（编译/分配显存）
    est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=max(est_iters))
    for it in est_iters:
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        cuda_sync(); t0 = time.perf_counter()
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=it)
        cuda_sync(); dt = time.perf_counter() - t0
        register_rows.append({"est_refine_iter": it, "register_ms": dt * 1000.0,
                              "peak_vram_mb": peak_vram_mb()})
        print(f"[register] iter={it:2d}  {dt*1000:8.1f} ms  vram={register_rows[-1]['peak_vram_mb']:.0f}MB",
              flush=True)

    # ---- 跟踪耗时扫描（后续帧 track_one）----
    track_rows = []
    for it in track_iters:
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        # 先用一次 register 复位 pose，保证跟踪输入一致
        est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
        times = []
        for r in range(args.track_reps):
            cuda_sync(); t0 = time.perf_counter()
            est.track_one(rgb=color, depth=depth, K=K, iteration=it)
            cuda_sync(); dt = time.perf_counter() - t0
            if r >= args.warmup:
                times.append(dt)
        times = np.array(times)
        row = {
            "track_refine_iter": it,
            "track_ms_mean": float(times.mean() * 1000),
            "track_ms_std": float(times.std() * 1000),
            "track_ms_p50": float(np.percentile(times, 50) * 1000),
            "track_ms_p95": float(np.percentile(times, 95) * 1000),
            "fps_mean": float(1.0 / times.mean()),
            "peak_vram_mb": peak_vram_mb(),
            "reps": int(len(times)),
        }
        track_rows.append(row)
        print(f"[track]    iter={it:2d}  {row['track_ms_mean']:8.1f} ms  "
              f"FPS={row['fps_mean']:5.1f}  vram={row['peak_vram_mb']:.0f}MB", flush=True)

    result = {
        "gpu": gpu_name,
        "mesh_file": os.path.basename(args.mesh_file),
        "frame_shape": list(color.shape),
        "mask_depth_valid": valid,
        "register": register_rows,
        "track": track_rows,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 同时写 CSV 方便查看
    import csv
    with open(out.with_name("pose_register.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(register_rows[0].keys())); w.writeheader(); w.writerows(register_rows)
    with open(out.with_name("pose_track.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(track_rows[0].keys())); w.writeheader(); w.writerows(track_rows)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
