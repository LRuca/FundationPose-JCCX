#!/usr/bin/env python
"""YOLO-seg 实时效率基准（不需要相机，用已有图像）。

测量：不同输入分辨率(imgsz)下的预处理/推理/后处理耗时、可达 FPS、峰值显存。
这同时就是"输入分辨率"效率消融。

需在 jxcx-yolo 环境运行。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "runs/needle_inbox_seg/yolov8n_seg_combined/weights/best.pt"))
    ap.add_argument("--image", default=str(ROOT / "third_party/FoundationPose/live_orbbec/color.ppm"))
    ap.add_argument("--imgsz", default="640,960,1280")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--device", default="0")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "report/realtime_eval/efficiency/yolo_efficiency.json"))
    args = ap.parse_args()

    imgsz_list = [int(x) for x in args.imgsz.split(",") if x.strip()]
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"无法读取图像: {args.image}")

    model = YOLO(args.model)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    rows = []
    for imgsz in imgsz_list:
        # 预热
        for _ in range(args.warmup):
            model.predict(source=image, conf=args.conf, imgsz=imgsz, device=args.device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pre, inf, post, wall = [], [], [], []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            r = model.predict(source=image, conf=args.conf, imgsz=imgsz, device=args.device, verbose=False)[0]
            wall.append(time.perf_counter() - t0)
            pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
        pre, inf, post, wall = map(np.array, (pre, inf, post, wall))
        vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        row = {
            "imgsz": imgsz,
            "preprocess_ms": float(pre.mean()),
            "inference_ms": float(inf.mean()),
            "postprocess_ms": float(post.mean()),
            "total_ms_mean": float(wall.mean() * 1000),
            "total_ms_p95": float(np.percentile(wall, 95) * 1000),
            "fps_mean": float(1.0 / wall.mean()),
            "peak_vram_mb": float(vram),
        }
        rows.append(row)
        print(f"[yolo] imgsz={imgsz:5d}  推理={row['inference_ms']:6.1f}ms  端到端={row['total_ms_mean']:6.1f}ms  "
              f"FPS={row['fps_mean']:5.1f}  vram={row['peak_vram_mb']:.0f}MB", flush=True)

    result = {
        "gpu": gpu_name,
        "model": os.path.relpath(args.model, ROOT),
        "image_shape": list(image.shape),
        "rows": rows,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out.with_name("yolo_efficiency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
