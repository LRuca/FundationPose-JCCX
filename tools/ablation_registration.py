#!/usr/bin/env python
"""注册阶段消融（在固定测试帧上，foundationpose 环境）：
A. 注册精化迭代次数 -> 位姿收敛性(相对iter=10的偏差) + 耗时
B. mesh 版本(轴对称/v2/v3) -> 注册耗时 + 几何复杂度
C. 细针深度补全半径 -> mask 内有效深度点数(模拟细针深度缺失) + 是否满足注册阈值
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FP = ROOT / "third_party" / "FoundationPose"
sys.path.insert(0, str(FP))
import numpy as np, cv2, trimesh, torch
import imageio.v2 as imageio
import nvdiffrast.torch as dr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, set_seed

A = ROOT / "report" / "assets" / "realtime"
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
    if Path(p).exists():
        F = FontProperties(fname=p); break
plt.rcParams["axes.unicode_minus"] = False
FR = ROOT / "report" / "ablation_frame"
MESHES = {
    "轴对称": ROOT / "model/fixed_unnamed_object_3/needle_axisymmetric_reconstruction.stl",
    "v2": ROOT / "model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v2.stl",
    "v3(采用)": ROOT / "model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl",
}


def f32(mesh):
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, np.float32), faces=np.asarray(mesh.faces),
                           vertex_normals=np.asarray(mesh.vertex_normals, np.float32), process=False)


def build(mesh):
    return FoundationPose(model_pts=mesh.vertices.astype(np.float32),
                          model_normals=mesh.vertex_normals.astype(np.float32), mesh=mesh,
                          scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                          debug_dir="/tmp/fp_abl", debug=0, glctx=dr.RasterizeCudaContext())


def repair(depth, mask, radius, minv=5):
    mb = mask.astype(bool)
    if int(((depth >= 0.001) & mb).sum()) >= minv:
        return depth, int(((depth >= 0.001) & mb).sum())
    k = np.ones((radius * 2 + 1,) * 2, np.uint8)
    src = cv2.dilate(mb.astype(np.uint8), k).astype(bool) & (depth >= 0.001)
    if int(src.sum()) < minv:
        return depth, int(((depth >= 0.001) & mb).sum())
    out = depth.copy(); out[mb & (out < 0.001)] = float(np.median(depth[src]))
    return out, int(((out >= 0.001) & mb).sum())


def rot_deg(R1, R2):
    c = (np.trace(R1.T @ R2) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    set_seed(0)
    color = imageio.imread(FR / "color.ppm")[..., :3]
    depth = cv2.imread(str(FR / "depth.pgm"), -1).astype(np.float32) / 1000.0
    K = np.loadtxt(FR / "cam_K.txt").reshape(3, 3).astype(np.float32)
    mask = cv2.imread(str(FR / "mask.png"), -1) > 0
    depth_v3, _ = repair(depth, mask, 60)

    mesh_v3 = f32(trimesh.load(str(MESHES["v3(采用)"])))
    est = build(mesh_v3)
    print("estimator(v3) ready", flush=True)

    # A. 注册迭代收敛
    iters = [1, 2, 3, 5, 7, 10]
    poses, times = {}, {}
    for it in iters:
        t0 = time.perf_counter()
        p = est.register(K=K, rgb=color, depth=depth_v3, ob_mask=mask, iteration=it)
        torch.cuda.synchronize()
        times[it] = (time.perf_counter() - t0) * 1000
        poses[it] = p.copy()
    ref = poses[10]
    conv = [{"iter": it, "dev_trans_mm": float(np.linalg.norm(poses[it][:3, 3] - ref[:3, 3]) * 1000),
             "dev_rot_deg": rot_deg(poses[it][:3, :3], ref[:3, :3]), "time_ms": times[it]} for it in iters]

    # C. 深度补全半径（模拟细针深度缺失：mask 内深度置零）
    depth_stressed = depth.copy(); depth_stressed[mask] = 0.0
    radii = [0, 15, 30, 45, 60]
    rep = []
    for r in radii:
        _, valid = repair(depth_stressed, mask, r) if r > 0 else (depth_stressed, int(((depth_stressed >= 0.001) & mask).sum()))
        rep.append({"radius": r, "valid_in_mask": valid, "enough": int(valid >= 5)})

    # B. mesh 版本
    meshrows = []
    for name, path in MESHES.items():
        mesh = f32(trimesh.load(str(path)))
        ext = trimesh.bounds.oriented_bounds(mesh)[1]
        e = est if name == "v3(采用)" else build(mesh)
        t0 = time.perf_counter()
        e.register(K=K, rgb=color, depth=depth_v3, ob_mask=mask, iteration=5)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        meshrows.append({"mesh": name, "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
                         "length_mm": float(max(ext) * 1000), "diameter_mm": float(np.linalg.norm(ext) * 1000),
                         "register_ms": dt})
        print(f"mesh {name}: V={len(mesh.vertices)} 长={max(ext)*1000:.1f}mm 注册={dt:.0f}ms", flush=True)

    out = {"convergence": conv, "repair_radius": rep, "mesh": meshrows}
    (A.parent / "realtime_eval" / "efficiency" / "registration_ablation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))

    # 图 A
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=170)
    xi = np.arange(len(iters))
    ax.plot(xi, [c["dev_trans_mm"] for c in conv], "-o", color="#0f766e", label="平移偏差(相对iter10) mm")
    ax.plot(xi, [c["dev_rot_deg"] for c in conv], "-s", color="#dc2626", label="旋转偏差(相对iter10) °")
    ax.set_xticks(xi); ax.set_xticklabels([str(i) for i in iters])
    ax.set_xlabel("注册精化迭代次数 est_refine_iter", fontproperties=F); ax.set_ylabel("相对收敛位姿的偏差", fontproperties=F)
    ax.set_title("注册迭代收敛性消融（偏差越小越收敛）", fontproperties=F, fontsize=14)
    ax2 = ax.twinx(); ax2.plot(xi, [c["time_ms"] for c in conv], "--^", color="#2563eb", label="注册耗时 ms")
    ax2.set_ylabel("注册耗时 (ms)", fontproperties=F)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, prop=F, loc="upper right")
    ax.grid(alpha=0.25, ls="--")
    for t in ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels(): t.set_fontproperties(F)
    fig.tight_layout(); fig.savefig(A / "消融_注册迭代收敛.png"); plt.close(fig)

    # 图 C
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=170)
    xr = np.arange(len(radii)); vals = [r["valid_in_mask"] for r in rep]
    bars = ax.bar(xr, vals, color=["#dc2626" if r["enough"] == 0 else "#16a34a" for r in rep], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02 + 1, str(v), ha="center", fontproperties=F)
    ax.axhline(5, color="#334155", ls="--", lw=1.5)
    ax.text(len(radii) - 1, 7, "注册阈值=5", fontproperties=F, fontsize=10, ha="right", color="#334155")
    ax.set_xticks(xr); ax.set_xticklabels([f"r={r}" for r in radii])
    ax.set_xlabel("深度补全半径（r=0 即不补全）", fontproperties=F); ax.set_ylabel("mask 内有效深度点数", fontproperties=F)
    ax.set_title("细针深度补全消融（模拟针身深度缺失）\n红=不足以注册, 绿=可注册", fontproperties=F, fontsize=13)
    ax.grid(alpha=0.25, ls="--", axis="y")
    for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_fontproperties(F)
    fig.tight_layout(); fig.savefig(A / "消融_深度补全半径.png"); plt.close(fig)

    # 图 B
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=170)
    names = [m["mesh"] for m in meshrows]; xb = np.arange(len(names))
    ax.bar(xb - 0.2, [m["register_ms"] for m in meshrows], 0.4, color="#f59e0b", label="注册耗时 ms")
    ax2 = ax.twinx()
    ax2.bar(xb + 0.2, [m["vertices"] for m in meshrows], 0.4, color="#2563eb", label="网格顶点数")
    ax.set_xticks(xb); ax.set_xticklabels(names)
    ax.set_ylabel("注册耗时 (ms)", fontproperties=F); ax2.set_ylabel("网格顶点数", fontproperties=F)
    ax.set_title("mesh 版本消融：几何复杂度 vs 注册耗时", fontproperties=F, fontsize=14)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, prop=F, loc="upper left")
    ax.grid(alpha=0.25, ls="--", axis="y")
    for t in ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels(): t.set_fontproperties(F)
    fig.tight_layout(); fig.savefig(A / "消融_mesh版本.png"); plt.close(fig)

    print("saved 3 charts"); print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
