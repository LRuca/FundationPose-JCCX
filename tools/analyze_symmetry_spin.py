#!/usr/bin/env python
"""分解针位姿的帧间旋转：绕对称轴的自转(不可观测/无意义) vs 垂直方向真实倾斜；
并对比稳定化前后的旋转抖动。产出中文图表 + JSON。"""
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


def steps(Rs, sym):
    tot, spin = [], []
    for i in range(len(Rs) - 1):
        Rrel = Rs[i].T @ Rs[i + 1]
        rv, _ = cv2.Rodrigues(Rrel)
        rv = rv.ravel()
        tot.append(np.degrees(np.linalg.norm(rv)))
        spin.append(np.degrees(abs(rv[sym])))
    return np.array(tot), np.array(spin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", default="report/takes/take_20260615_231544")
    ap.add_argument("--mesh", default="model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=250)
    ap.add_argument("--out", default="report/assets/realtime/精度_对称轴自转分解.png")
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    sym = int(np.argmax(extents)); inv = np.linalg.inv(to_origin)
    oc = Path(args.take) / "fp_debug" / "ob_in_cam"
    names = sorted(p.stem for p in oc.glob("*.txt"))[args.start:args.end]
    center = [(np.loadtxt(oc / f"{n}.txt").reshape(4, 4) @ inv)[:3, :3] for n in names]
    Rs_stab = stabilize_R(center, sym)

    tot0, spin0 = steps(center, sym)
    tot1, spin1 = steps(Rs_stab, sym)

    summary = {
        "frames": len(names), "sym_axis_col": sym,
        "orig_rot_step_mean_deg": float(tot0.mean()), "orig_rot_step_p95_deg": float(np.percentile(tot0, 95)),
        "orig_spin_step_mean_deg": float(spin0.mean()),
        "spin_share_pct": float(100 * spin0.sum() / (tot0.sum() + 1e-9)),
        "stab_rot_step_mean_deg": float(tot1.mean()), "stab_rot_step_p95_deg": float(np.percentile(tot1, 95)),
        "rot_jitter_reduction_pct": float(100 * (1 - tot1.mean() / (tot0.mean() + 1e-9))),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=170)
    x = np.arange(len(tot0))
    ax[0].plot(x, tot0, color="#dc2626", lw=1.3, label="原始：总旋转步长")
    ax[0].plot(x, spin0, color="#f59e0b", lw=1.1, label="其中：绕对称轴自转")
    ax[0].plot(x, tot1, color="#16a34a", lw=1.6, label="稳定化后：总旋转步长")
    ax[0].set_title("帧间旋转步长：稳定化前后对比", fontproperties=F, fontsize=14)
    ax[0].set_xlabel("帧序号", fontproperties=F); ax[0].set_ylabel("每帧旋转角 (度)", fontproperties=F)
    ax[0].legend(prop=F); ax[0].grid(alpha=0.25, ls="--")
    for t in ax[0].get_xticklabels() + ax[0].get_yticklabels(): t.set_fontproperties(F)

    bars = ["原始\n总旋转", "其中\n自转分量", "稳定化后\n总旋转"]
    vals = [tot0.mean(), spin0.mean(), tot1.mean()]
    cols = ["#dc2626", "#f59e0b", "#16a34a"]
    b = ax[1].bar(bars, vals, color=cols, width=0.6)
    for bi, v in zip(b, vals):
        ax[1].text(bi.get_x() + bi.get_width() / 2, v + 0.03, f"{v:.2f}°", ha="center", fontproperties=F)
    ax[1].set_title(f"平均每帧旋转抖动\n自转占原始旋转 {summary['spin_share_pct']:.0f}%，稳定化降抖 {summary['rot_jitter_reduction_pct']:.0f}%",
                    fontproperties=F, fontsize=13)
    ax[1].set_ylabel("平均每帧旋转角 (度)", fontproperties=F); ax[1].grid(alpha=0.25, ls="--", axis="y")
    for t in ax[1].get_xticklabels() + ax[1].get_yticklabels(): t.set_fontproperties(F)
    fig.tight_layout(); fig.savefig(args.out); plt.close(fig)
    print("saved", args.out); print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
