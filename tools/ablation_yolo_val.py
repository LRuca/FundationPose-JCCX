#!/usr/bin/env python
"""YOLO 输入分辨率精度消融：在 combined 验证集上测不同 imgsz 的分割 mAP / 精确率 / 召回率。
与效率图(imgsz->速度)配成完整的精度-效率权衡。需 jxcx-yolo 环境。"""
from __future__ import annotations
import os, json
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "report" / "assets" / "realtime"
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
    if Path(p).exists():
        F = FontProperties(fname=p); break
plt.rcParams["axes.unicode_minus"] = False

# 生成本地 yaml（原 yaml 是 Windows 路径）
DS = ROOT / "datasets" / "needle_inbox_combined"
yaml_local = ROOT / "report" / "ablation_frame" / "needle_combined_local.yaml"
yaml_local.parent.mkdir(parents=True, exist_ok=True)
yaml_local.write_text(f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n  0: needle_inbox\n")

MODEL = ROOT / "runs/needle_inbox_seg/yolov8n_seg_combined/weights/best.pt"


def main():
    rows = []
    for imgsz in [640, 960, 1280]:
        m = YOLO(str(MODEL))
        r = m.val(data=str(yaml_local), imgsz=imgsz, conf=0.001, iou=0.6,
                  split="val", verbose=False, plots=False, save_json=False,
                  project=str(ROOT / "runs/smoke"), name=f"val_imgsz{imgsz}", exist_ok=True)
        rows.append({
            "imgsz": imgsz,
            "seg_map50": float(r.seg.map50), "seg_map": float(r.seg.map),
            "box_map50": float(r.box.map50), "precision": float(r.box.mp), "recall": float(r.box.mr),
        })
        print(f"imgsz={imgsz}: seg mAP50={r.seg.map50:.3f} mAP50-95={r.seg.map:.3f} "
              f"P={r.box.mp:.3f} R={r.box.mr:.3f}", flush=True)
    (A.parent / "realtime_eval" / "efficiency").mkdir(parents=True, exist_ok=True)
    (A.parent / "realtime_eval" / "efficiency" / "yolo_val_imgsz.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))

    sz = [r["imgsz"] for r in rows]
    x = np.arange(len(sz))
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=170)
    ax.plot(x, [r["seg_map50"] for r in rows], "-o", color="#2563eb", lw=2, label="分割 mAP50")
    ax.plot(x, [r["seg_map"] for r in rows], "-s", color="#16a34a", lw=2, label="分割 mAP50-95")
    ax.plot(x, [r["precision"] for r in rows], "-^", color="#f59e0b", lw=2, label="精确率 P")
    ax.plot(x, [r["recall"] for r in rows], "-v", color="#dc2626", lw=2, label="召回率 R")
    for r, xi in zip(rows, x):
        ax.annotate(f"{r['seg_map50']:.2f}", (xi, r["seg_map50"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontproperties=F, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in sz])
    ax.set_title("YOLO 输入分辨率精度消融（combined 验证集，24 张）", fontproperties=F, fontsize=14)
    ax.set_xlabel("输入分辨率 imgsz (像素)", fontproperties=F); ax.set_ylabel("指标值", fontproperties=F)
    ax.legend(prop=F, loc="best"); ax.grid(alpha=0.25, ls="--"); ax.set_ylim(0, 1.02)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(F)
    fig.tight_layout(); out = A / "消融_YOLO分辨率精度.png"; fig.savefig(out); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
