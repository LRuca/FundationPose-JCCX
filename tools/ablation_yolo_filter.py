#!/usr/bin/env python
"""YOLO 候选过滤消融：无过滤 / ROI / ROI+高度 三档下的候选数与假阳性数。
直接针对历史失败模式（误检显示器文字、线缆等横向/工作区外目标）。"""
from __future__ import annotations
import os, argparse, json
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="report/_inspect/filter_test.ppm")
    ap.add_argument("--model", default="runs/needle_inbox_seg/yolov8n_seg_combined/weights/best.pt")
    ap.add_argument("--roi", default="230,90,450,320")
    ap.add_argument("--min_h", type=int, default=40)
    ap.add_argument("--true_idx", type=int, default=-1, help="真针候选下标(按conf排序后)，-1=最高conf")
    ap.add_argument("--out", default="report/assets/realtime/消融_YOLO候选过滤.png")
    args = ap.parse_args()

    img = cv2.imread(args.image); H, W = img.shape[:2]
    x1r, y1r, x2r, y2r = [int(v) for v in args.roi.split(",")]
    m = YOLO(args.model)
    cid = [k for k, v in m.names.items() if str(v).lower() == "needle_inbox"][0]
    r = m.predict(source=img, conf=0.05, imgsz=960, device=0, verbose=False)[0]
    cls = r.boxes.cls.cpu().numpy().astype(int); conf = r.boxes.conf.cpu().numpy(); xy = r.boxes.xyxy.cpu().numpy()
    cands = []
    for i in range(len(cls)):
        if cls[i] != cid:
            continue
        b = [int(v) for v in xy[i]]
        cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        cands.append({"conf": float(conf[i]), "bbox": b, "cx": cx, "cy": cy, "h": b[3] - b[1]})
    cands.sort(key=lambda c: c["conf"], reverse=True)
    true_i = (len(cands) - 1 if args.true_idx == -1 and False else 0)  # 最高conf=真针

    def passes(c, use_roi, use_h):
        if use_roi and not (x1r <= c["cx"] <= x2r and y1r <= c["cy"] <= y2r):
            return False
        if use_h and c["h"] < args.min_h:
            return False
        return True

    stages = [("无过滤", False, False), ("仅ROI", True, False), ("ROI+高度", True, True)]
    res = []
    for name, ur, uh in stages:
        kept = [i for i, c in enumerate(cands) if passes(c, ur, uh)]
        fp = sum(1 for i in kept if i != true_i)
        res.append({"stage": name, "kept": len(kept), "false_pos": fp, "true_kept": int(true_i in kept)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"image": args.image, "n_candidates": len(cands), "candidates": cands, "stages": res}, ensure_ascii=False, indent=2))

    names = [s["stage"] for s in res]
    kept = [s["kept"] for s in res]
    fp = [s["false_pos"] for s in res]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=170)
    ax.bar(x - 0.2, kept, 0.4, color="#2563eb", label="保留候选总数")
    ax.bar(x + 0.2, fp, 0.4, color="#dc2626", label="其中假阳性")
    for xi, (k, f) in enumerate(zip(kept, fp)):
        ax.text(xi - 0.2, k + 0.03, str(k), ha="center", fontproperties=F)
        ax.text(xi + 0.2, f + 0.03, str(f), ha="center", fontproperties=F)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title(f"YOLO 候选过滤消融（共检出 {len(cands)} 个候选，真针1个）\nROI={args.roi}, 最小高度={args.min_h}px",
                 fontproperties=F, fontsize=13)
    ax.set_ylabel("数量", fontproperties=F); ax.legend(prop=F); ax.grid(alpha=0.25, ls="--", axis="y")
    ax.set_ylim(0, max(kept) + 1)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(F)
    fig.tight_layout(); fig.savefig(args.out); plt.close(fig)
    print("saved", args.out)
    for s in res:
        print(f"  {s['stage']}: 保留={s['kept']} 假阳性={s['false_pos']} 真针保留={'是' if s['true_kept'] else '否'}")


if __name__ == "__main__":
    main()
