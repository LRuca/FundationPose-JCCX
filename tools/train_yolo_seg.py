#!/usr/bin/env python
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


DATASETS = {
    "lwt": ("data/needle_lwt/needle_lwt_yolo_split", "needle"),
    "inbox": ("datasets/needle_inbox", "needle_inbox"),
    "combined": ("datasets/needle_inbox_combined", "needle_inbox"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable YOLO segmentation trainer.")
    parser.add_argument("--dataset", choices=DATASETS, default="combined")
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/ablation")
    parser.add_argument("--name", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset_rel, class_name = DATASETS[args.dataset]
    dataset_root = (root / dataset_rel).resolve()
    for split in ("images/train", "images/val", "labels/train", "labels/val"):
        if not (dataset_root / split).is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {dataset_root / split}")

    config = {
        "path": dataset_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    run_name = args.name or f"{args.dataset}_{Path(args.model).stem}_img{args.imgsz}_seed{args.seed}"
    project = (root / args.project).resolve()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8", delete=False) as f:
        yaml.safe_dump(config, f, sort_keys=False)
        data_yaml = f.name

    model = YOLO(args.model)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        amp=args.amp,
        val=args.val,
        task="segment",
        project=str(project),
        name=run_name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
