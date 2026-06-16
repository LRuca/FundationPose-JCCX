#!/usr/bin/env python
"""精度提升方法对比：One-Euro 自适应滤波 vs 固定 EMA。
在抖动(平滑度) ↔ 保真度(对原始信号RMSE，延迟代理)的权衡上比较两种滤波。
One-Euro 在快动时少滞后、慢动时强平滑，期望 Pareto 更优。基于已录平移序列。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "report" / "assets" / "realtime"
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
    if Path(p).exists():
        F = FontProperties(fname=p); break
plt.rcParams["axes.unicode_minus"] = False


def ema(x, alpha):
    y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def one_euro(x, fps, mincutoff, beta=0.7, dcutoff=1.0):
    def a(cut):
        tau = 1.0 / (2 * np.pi * cut)
        te = 1.0 / fps
        return 1.0 / (1.0 + tau / te)
    y = np.empty_like(x); y[0] = x[0]
    dx_prev = 0.0
    for i in range(1, len(x)):
        dx = (x[i] - y[i - 1]) * fps
        ad = a(dcutoff)
        dx_hat = ad * dx + (1 - ad) * dx_prev
        cut = mincutoff + beta * abs(dx_hat)
        af = a(cut)
        y[i] = af * x[i] + (1 - af) * y[i - 1]
        dx_prev = dx_hat
    return y


def metrics(sig, raw):
    jit = np.linalg.norm(np.diff(sig, axis=0), axis=1).mean() * 10  # mm
    rmse = np.sqrt((np.linalg.norm(sig - raw, axis=1) ** 2).mean()) * 10  # mm
    return jit, rmse


def main():
    take = ROOT / "report/takes/take_20260615_231544/fp_debug/ob_in_cam"
    names = sorted(p.stem for p in take.glob("*.txt"))[:250]
    raw = np.array([np.loadtxt(take / f"{n}.txt").reshape(4, 4)[:3, 3] for n in names]) * 100  # cm
    fps = 15.0

    ema_rows = [{"param": al, **dict(zip(("jitter", "rmse"), metrics(np.stack([ema(raw[:, k], al) for k in range(3)], 1), raw)))}
                for al in [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1]]
    oe_rows = [{"param": mc, **dict(zip(("jitter", "rmse"), metrics(np.stack([one_euro(raw[:, k], fps, mc) for k in range(3)], 1), raw)))}
               for mc in [8, 4, 2, 1.2, 0.8, 0.5, 0.3]]
    (A.parent / "realtime_eval" / "efficiency").mkdir(parents=True, exist_ok=True)
    (ROOT / "report/realtime_eval/oneeuro_vs_ema.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "report/realtime_eval/oneeuro_vs_ema.json").write_text(
        json.dumps({"ema": ema_rows, "one_euro": oe_rows}, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=170)
    ax.plot([r["jitter"] for r in ema_rows], [r["rmse"] for r in ema_rows], "-o", color="#f59e0b", lw=2, label="固定 EMA")
    ax.plot([r["jitter"] for r in oe_rows], [r["rmse"] for r in oe_rows], "-s", color="#16a34a", lw=2, label="One-Euro 自适应")
    ax.set_xlabel("残余抖动 (mm/帧，越小越平滑)", fontproperties=F)
    ax.set_ylabel("对原始轨迹 RMSE (mm，越小越跟手/低延迟)", fontproperties=F)
    ax.set_title("位姿滤波 Pareto 对比：One-Euro vs 固定 EMA\n(左下角更优：同等平滑度下 One-Euro 更跟手)", fontproperties=F, fontsize=13)
    ax.legend(prop=F); ax.grid(alpha=0.25, ls="--")
    ax.annotate("不平滑", (ema_rows[0]["jitter"], ema_rows[0]["rmse"]), textcoords="offset points",
                xytext=(6, 6), fontproperties=F, fontsize=9)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(F)
    fig.tight_layout(); out = A / "提升_OneEuro对比EMA.png"; fig.savefig(out); plt.close(fig)
    print("saved", out)
    print("EMA   :", [(round(r["jitter"], 2), round(r["rmse"], 2)) for r in ema_rows])
    print("OneEuro:", [(round(r["jitter"], 2), round(r["rmse"], 2)) for r in oe_rows])


if __name__ == "__main__":
    main()
