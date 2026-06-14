#!/usr/bin/env python
"""Smoke test both needle_inbox models on the live camera frame."""
from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO

root = Path(r"C:\Users\lenovo\Desktop\JXCX")
img = root / "FoundationPose" / "live_orbbec" / "color.png"

models = {
    "inbox_pure": root / "runs" / "needle_inbox_seg" / "yolov8n_seg_inbox" / "weights" / "best.pt",
    "combined": root / "runs" / "needle_inbox_seg" / "yolov8n_seg_combined" / "weights" / "best.pt",
}

for name, pt in models.items():
    print(f"\n=== {name} ===")
    model = YOLO(str(pt))
    results = model.predict(
        str(img),
        imgsz=960,
        conf=0.25,
        save=True,
        project=str(root / "runs" / "needle_inbox_seg"),
        name=f"predict_{name}",
        exist_ok=True,
    )
    r = results[0]
    if r.masks is not None and len(r.masks) > 0:
        print(f"Detected {len(r.masks)} instance(s)")
        for i, (box, mask) in enumerate(zip(r.boxes, r.masks)):
            cls_name = r.names[int(box.cls)]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"  [{i}] class={cls_name}, conf={conf:.3f}, box=[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}]")
    else:
        print("No mask detected")
    print(f"Saved to: {r.save_dir}")
