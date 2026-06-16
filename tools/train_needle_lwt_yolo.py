#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation for the needle_lwt dataset.")
    parser.add_argument("--model", default=str(root / "yolov8n-seg.pt"))
    parser.add_argument(
        "--data",
        default=str(root / "data" / "needle_lwt" / "needle_lwt_yolo_split" / "needle_lwt_train.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(root / "runs" / "needle_lwt_seg"))
    parser.add_argument("--name", default="yolov8n_seg_lwt")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    try:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            workers=args.workers,
            amp=args.amp,
            val=args.val,
            task="segment",
            exist_ok=True,
        )
    except NotImplementedError:
        print("[WARNING] Final validation skipped (NMS CUDA compatibility). Weights already saved.")


if __name__ == "__main__":
    main()
