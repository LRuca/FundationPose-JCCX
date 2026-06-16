#!/usr/bin/env python
"""时序平滑(EMA)对位姿抖动的改善 vs 引入的延迟 —— 精度/实时性权衡。
在已录位姿序列上做平滑强度扫描，量化抖动下降与延迟上升。"""
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
        u = u - (u @ a) * a; u = u / (np.linalg.norm(u) + 1e-9)
        w = np.cross(a, u)
        cols = [None, None, None]; cols[sym] = a; cols[(sym + 1) % 3] = u; cols[(sym + 2) % 3] = w
        out.append(np.stack(cols, 1)); prev_u = u
    return out


def orthonormal(M):
    x = M[:, 0] / (np.linalg.norm(M[:, 0]) + 1e-9)
    y = M[:, 1] - (M[:, 1] @ x) * x; y = y / (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y)
    return np.stack([x, y, z], 1)


def ema(t, Rs, alpha):
    """alpha=新观测权重(1=不平滑)。返回平滑后的 t, R。"""
    ts, Rss = [t[0].copy()], [Rs[0].copy()]
    for i in range(1, len(t)):
        ts.append(alpha * t[i] + (1 - alpha) * ts[-1])
        Rss.append(orthonormal(alpha * Rs[i] + (1 - alpha) * Rss[-1]))
    return np.array(ts), Rss


def jitter(t_cm, Rs, sym):
    ts = np.linalg.norm(np.diff(t_cm, axis=0), axis=1) * 10.0
    rs = []
    for i in range(len(Rs) - 1):
        rv, _ = cv2.Rodrigues(Rs[i].T @ Rs[i + 1]); rv = rv.ravel(); rv[sym] = 0
        rs.append(np.degrees(np.linalg.norm(rv)))
    return ts.mean(), np.mean(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", default="report/takes/take_20260615_231544")
    ap.add_argument("--mesh", default="model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=250)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out", default="report/assets/realtime/权衡_时序平滑.png")
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    sym = int(np.argmax(extents)); inv = np.linalg.inv(to_origin)
    oc = Path(args.take) / "fp_debug" / "ob_in_cam"
    names = sorted(p.stem for p in oc.glob("*.txt"))[args.start:args.end]
    poses = [np.loadtxt(oc / f"{n}.txt").reshape(4, 4) for n in names]
    t_cm = np.array([p[:3, 3] for p in poses]) * 100.0
    Rs0 = stabilize_R([(p @ inv)[:3, :3] for p in poses], sym)

    alphas = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
    rows = []
    for a in alphas:
        ts, Rss = ema(t_cm, Rs0, a)
        tj, rj = jitter(ts, Rss, sym)
        lat_frames = (1 - a) / a
        rows.append({"alpha": a, "trans_jitter_mm": tj, "rot_jitter_deg": rj,
                     "latency_frames": lat_frames, "latency_ms": lat_frames / args.fps * 1000})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    al = [r["alpha"] for r in rows]
    tj = [r["trans_jitter_mm"] for r in rows]
    rj = [r["rot_jitter_deg"] for r in rows]
    lat = [r["latency_ms"] for r in rows]
    x = np.arange(len(al)); labels = [f"α={a}\n{'不平滑' if a==1 else ''}" for a in al]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=170)
    ax[0].plot(x, tj, "-o", color="#0f766e", label="平移抖动 (mm)")
    ax[0].plot(x, rj, "-s", color="#f59e0b", label="旋转抖动 (度)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_title("时序平滑强度 → 抖动下降", fontproperties=F, fontsize=13)
    ax[0].set_xlabel("EMA 系数 α（越小越平滑）", fontproperties=F); ax[0].set_ylabel("平均每帧抖动", fontproperties=F)
    ax[0].legend(prop=F); ax[0].grid(alpha=0.25, ls="--")

    ax[1].plot(lat, tj, "-o", color="#2563eb")
    for r in rows:
        ax[1].annotate(f"α={r['alpha']}", (r["latency_ms"], r["trans_jitter_mm"]),
                       fontproperties=F, fontsize=9, xytext=(4, 4), textcoords="offset points")
    ax[1].set_title("精度↔实时性权衡：平移抖动 vs 引入延迟", fontproperties=F, fontsize=13)
    ax[1].set_xlabel("EMA 引入的延迟 (ms @15fps)", fontproperties=F); ax[1].set_ylabel("平移抖动 (mm)", fontproperties=F)
    ax[1].grid(alpha=0.25, ls="--")
    for a in ax:
        for t in a.get_xticklabels() + a.get_yticklabels():
            t.set_fontproperties(F)
    fig.tight_layout(); fig.savefig(args.out); plt.close(fig)
    print("saved", args.out)
    for r in rows:
        print(f"  α={r['alpha']:<4} 平移抖动={r['trans_jitter_mm']:.2f}mm 旋转抖动={r['rot_jitter_deg']:.2f}° 延迟={r['latency_ms']:.0f}ms")


if __name__ == "__main__":
    main()
