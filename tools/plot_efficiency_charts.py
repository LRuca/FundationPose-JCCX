#!/usr/bin/env python
"""读取效率基准 JSON，绘制统一中文风格的效率图表。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
EFF_DIR = ROOT / "report" / "realtime_eval" / "efficiency"
OUT_DIR = ROOT / "report" / "assets" / "realtime"


def font():
    for p in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]:
        if Path(p).exists():
            return FontProperties(fname=p)
    raise FileNotFoundError("缺少中文字体")


F = font()
plt.rcParams["axes.unicode_minus"] = False
C = {"teal": "#0f766e", "blue": "#2563eb", "orange": "#f97316",
     "red": "#dc2626", "green": "#16a34a", "purple": "#7c3aed", "slate": "#64748b"}


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontproperties=F, fontsize=14, pad=12)
    ax.set_xlabel(xlabel, fontproperties=F, fontsize=11)
    ax.set_ylabel(ylabel, fontproperties=F, fontsize=11)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(F)
    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_fontproperties(F)
    ax.grid(alpha=0.25, linestyle="--")


def chart_register(pose):
    rows = pose["register"]
    its = [r["est_refine_iter"] for r in rows]
    ms = [r["register_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
    bars = ax.bar([str(i) for i in its], ms, color=C["orange"], width=0.6, edgecolor="#9a3412")
    for b, v in zip(bars, ms):
        ax.text(b.get_x() + b.get_width() / 2, v + max(ms) * 0.02, f"{v:.0f}",
                ha="center", fontproperties=F, fontsize=11)
    style(ax, f"首帧注册耗时随精化迭代次数变化\n(GPU: {pose['gpu']})", "注册精化迭代次数 est_refine_iter", "首帧注册耗时 (ms)")
    fig.tight_layout(); out = OUT_DIR / "效率_首帧注册耗时.png"; fig.savefig(out); plt.close(fig); return out


def chart_track(pose):
    rows = pose["track"]
    its = [r["track_refine_iter"] for r in rows]
    ms = [r["track_ms_mean"] for r in rows]
    fps = [r["fps_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
    x = np.arange(len(its))
    bars = ax.bar(x, ms, color=C["teal"], width=0.55, edgecolor="#134e4a", label="逐帧跟踪耗时")
    for b, v in zip(bars, ms):
        ax.text(b.get_x() + b.get_width() / 2, v + max(ms) * 0.02, f"{v:.1f}ms",
                ha="center", fontproperties=F, fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in its])
    style(ax, "逐帧跟踪耗时与可达帧率", "跟踪精化迭代次数 track_refine_iter", "单帧跟踪耗时 (ms)")
    ax2 = ax.twinx()
    ax2.plot(x, fps, color=C["red"], marker="o", lw=2, label="可达 FPS")
    for xi, v in zip(x, fps):
        ax2.text(xi, v + max(fps) * 0.03, f"{v:.0f}", ha="center", color=C["red"], fontproperties=F, fontsize=10)
    ax2.set_ylabel("纯计算可达 FPS", fontproperties=F, fontsize=11)
    for t in ax2.get_yticklabels():
        t.set_fontproperties(F)
    ax2.axhline(30, color=C["slate"], ls=":", lw=1.5)
    ax2.text(len(its) - 1, 33, "相机 30 FPS 上限", color=C["slate"], fontproperties=F, fontsize=9, ha="right")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ["逐帧跟踪耗时", "可达 FPS"]
    ax.legend(lines, labels, loc="upper left", prop=F)
    fig.tight_layout(); out = OUT_DIR / "效率_逐帧跟踪耗时与帧率.png"; fig.savefig(out); plt.close(fig); return out


def chart_yolo(y):
    rows = y["rows"]
    sz = [r["imgsz"] for r in rows]
    pre = [r["preprocess_ms"] for r in rows]
    inf = [r["inference_ms"] for r in rows]
    post = [r["postprocess_ms"] for r in rows]
    fps = [r["fps_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
    x = np.arange(len(sz))
    ax.bar(x, pre, width=0.55, color=C["blue"], label="预处理")
    ax.bar(x, inf, width=0.55, bottom=pre, color=C["teal"], label="推理")
    ax.bar(x, post, width=0.55, bottom=np.array(pre) + np.array(inf), color=C["orange"], label="后处理")
    tot = np.array(pre) + np.array(inf) + np.array(post)
    for xi, v in zip(x, tot):
        ax.text(xi, v + max(tot) * 0.02, f"{v:.1f}ms", ha="center", fontproperties=F, fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in sz])
    style(ax, "YOLO-seg 各阶段耗时随输入分辨率变化", "输入分辨率 imgsz (像素)", "单帧耗时 (ms)")
    ax.legend(prop=F, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(x, fps, color=C["red"], marker="s", lw=2)
    for xi, v in zip(x, fps):
        ax2.text(xi, v, f"{v:.0f} FPS", ha="center", va="bottom", color=C["red"], fontproperties=F, fontsize=9)
    ax2.set_ylabel("端到端 FPS", fontproperties=F, fontsize=11)
    for t in ax2.get_yticklabels():
        t.set_fontproperties(F)
    fig.tight_layout(); out = OUT_DIR / "效率_YOLO各阶段耗时.png"; fig.savefig(out); plt.close(fig); return out


def chart_budget(pose, y):
    """端到端单帧时间预算：YOLO + 跟踪 vs 相机 33ms 预算。"""
    yolo_ms = next(r for r in y["rows"] if r["imgsz"] == 960)["total_ms_mean"]
    track1 = next(r for r in pose["track"] if r["track_refine_iter"] == 1)["track_ms_mean"]
    track2 = next(r for r in pose["track"] if r["track_refine_iter"] == 2)["track_ms_mean"]
    track5 = next(r for r in pose["track"] if r["track_refine_iter"] == 5)["track_ms_mean"]
    configs = [
        ("仅跟踪 iter1", [("跟踪", track1, C["teal"])]),
        ("仅跟踪 iter2", [("跟踪", track2, C["teal"])]),
        ("仅跟踪 iter5", [("跟踪", track5, C["teal"])]),
        ("YOLO@960+跟踪iter2\n(每帧都跑YOLO)", [("YOLO", yolo_ms, C["blue"]), ("跟踪", track2, C["teal"])]),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=170)
    labels = [c[0] for c in configs]
    yv = np.arange(len(configs))[::-1]
    for yi, (_, segs) in zip(yv, configs):
        left = 0
        for name, val, col in segs:
            ax.barh(yi, val, left=left, color=col, edgecolor="white", height=0.5,
                    label=name if yi == yv[-1] or name not in ax.get_legend_handles_labels()[1] else None)
            left += val
        ax.text(left + 0.5, yi, f"{left:.1f}ms", va="center", fontproperties=F, fontsize=10)
    ax.axvline(33.3, color=C["red"], ls="--", lw=2)
    ax.text(33.3, len(configs) - 0.4, "相机 30FPS 帧预算 33.3ms", color=C["red"], fontproperties=F, fontsize=10, ha="left")
    ax.set_yticks(yv); ax.set_yticklabels(labels)
    for t in ax.get_yticklabels():
        t.set_fontproperties(F)
    style(ax, "端到端单帧时间预算（GPU计算 vs 相机帧预算）", "单帧耗时 (ms)", "")
    # 去重图例
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        if ll and ll not in seen:
            seen[ll] = hh
    ax.legend(seen.values(), seen.keys(), prop=F, loc="lower right")
    fig.tight_layout(); out = OUT_DIR / "效率_端到端时间预算.png"; fig.savefig(out); plt.close(fig); return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pose = json.loads((EFF_DIR / "pose_efficiency.json").read_text(encoding="utf-8"))
    y = json.loads((EFF_DIR / "yolo_efficiency.json").read_text(encoding="utf-8"))
    outs = [chart_register(pose), chart_track(pose), chart_yolo(y), chart_budget(pose, y)]
    for o in outs:
        print("已生成图表:", o)


if __name__ == "__main__":
    main()
