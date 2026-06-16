#!/usr/bin/env python
"""实时位姿精度/稳定性评估（无绝对6D真值时的替代指标）。

基于稳定化后的位姿序列，评估：
- 平移轨迹（物体原点在相机系 x/y/z）
- 针朝向相对首帧的倾斜角（可观测的真实旋转）
- 帧间抖动：平移步长(mm)、可观测旋转步长(度)
产出中文多panel图 + JSON。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import cv2
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
    if Path(p).exists():
        F = FontProperties(fname=p); break
plt.rcParams["axes.unicode_minus"] = False


def stabilize_R(R_list, sym):
    out, prev_u = [], None
    for R in R_list:
        a = R[:, sym] / (np.linalg.norm(R[:, sym]) + 1e-9)
        u = R[:, (sym + 1) % 3] if prev_u is None else prev_u
        u = u - (u @ a) * a
        u = u / (np.linalg.norm(u) + 1e-9)
        w = np.cross(a, u)
        cols = [None, None, None]; cols[sym] = a; cols[(sym + 1) % 3] = u; cols[(sym + 2) % 3] = w
        out.append(np.stack(cols, 1)); prev_u = u
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", default="report/takes/take_20260615_231544")
    ap.add_argument("--mesh", default="model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=250)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out", default="report/assets/realtime/精度_实时跟踪稳定性.png")
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    sym = int(np.argmax(extents)); inv = np.linalg.inv(to_origin)
    oc = Path(args.take) / "fp_debug" / "ob_in_cam"
    names = sorted(p.stem for p in oc.glob("*.txt"))[args.start:args.end]
    poses = [np.loadtxt(oc / f"{n}.txt").reshape(4, 4) for n in names]
    t_cm = np.array([p[:3, 3] for p in poses]) * 100.0  # cm
    center_R = [(p @ inv)[:3, :3] for p in poses]
    Rs = stabilize_R(center_R, sym)

    axis_dir = np.array([R[:, sym] for R in Rs])
    a0 = axis_dir[0] / np.linalg.norm(axis_dir[0])
    tilt = np.degrees(np.arccos(np.clip([np.dot(a / np.linalg.norm(a), a0) for a in axis_dir], -1, 1)))

    trans_step = np.linalg.norm(np.diff(t_cm, axis=0), axis=1) * 10.0  # mm
    rot_step = []
    for i in range(len(Rs) - 1):
        rv, _ = cv2.Rodrigues(Rs[i].T @ Rs[i + 1])
        rv = rv.ravel(); rv[sym] = 0.0  # 只看可观测(垂直)旋转
        rot_step.append(np.degrees(np.linalg.norm(rv)))
    rot_step = np.array(rot_step)
    tsec = np.arange(len(names)) / args.fps

    summary = {
        "take": args.take, "frames": len(names), "duration_s": float(len(names) / args.fps),
        "trans_range_cm": {"x": float(t_cm[:, 0].ptp()), "y": float(t_cm[:, 1].ptp()), "z": float(t_cm[:, 2].ptp())},
        "max_tilt_deg": float(tilt.max()),
        "trans_jitter_mm": {"mean": float(trans_step.mean()), "p95": float(np.percentile(trans_step, 95)), "max": float(trans_step.max())},
        "obs_rot_jitter_deg": {"mean": float(rot_step.mean()), "p95": float(np.percentile(rot_step, 95)), "max": float(rot_step.max())},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(2, 2, figsize=(13, 8), dpi=160)
    ax[0, 0].plot(tsec, t_cm[:, 0], label="X", color="#dc2626")
    ax[0, 0].plot(tsec, t_cm[:, 1], label="Y", color="#16a34a")
    ax[0, 0].plot(tsec, t_cm[:, 2], label="Z(深度)", color="#2563eb")
    ax[0, 0].set_title("针在相机坐标系的位置轨迹", fontproperties=F, fontsize=13)
    ax[0, 0].set_xlabel("时间 (秒)", fontproperties=F); ax[0, 0].set_ylabel("位置 (cm)", fontproperties=F)
    ax[0, 0].legend(prop=F)

    ax[0, 1].plot(tsec, tilt, color="#7c3aed")
    ax[0, 1].set_title("针朝向相对首帧的倾斜角（可观测旋转）", fontproperties=F, fontsize=13)
    ax[0, 1].set_xlabel("时间 (秒)", fontproperties=F); ax[0, 1].set_ylabel("倾斜角 (度)", fontproperties=F)

    ax[1, 0].plot(tsec[1:], trans_step, color="#0f766e")
    ax[1, 0].axhline(trans_step.mean(), color="#dc2626", ls="--", lw=1)
    ax[1, 0].set_title(f"帧间平移抖动 (均值 {trans_step.mean():.2f} mm)", fontproperties=F, fontsize=13)
    ax[1, 0].set_xlabel("时间 (秒)", fontproperties=F); ax[1, 0].set_ylabel("每帧平移 (mm)", fontproperties=F)

    ax[1, 1].plot(tsec[1:], rot_step, color="#f59e0b")
    ax[1, 1].axhline(rot_step.mean(), color="#dc2626", ls="--", lw=1)
    ax[1, 1].set_title(f"帧间可观测旋转抖动 (均值 {rot_step.mean():.2f}°)", fontproperties=F, fontsize=13)
    ax[1, 1].set_xlabel("时间 (秒)", fontproperties=F); ax[1, 1].set_ylabel("每帧旋转 (度)", fontproperties=F)

    for a in ax.ravel():
        a.grid(alpha=0.25, ls="--")
        for t in a.get_xticklabels() + a.get_yticklabels():
            t.set_fontproperties(F)
    fig.suptitle("实时针位姿跟踪稳定性评估（稳定化后，take_231544 前250帧）", fontproperties=F, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(args.out); plt.close(fig)
    print("saved", args.out); print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
