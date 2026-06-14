#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible YOLO ablation grid.")
    parser.add_argument("--dataset", choices=("inbox", "lwt", "combined"), default="combined")
    parser.add_argument("--models", nargs="+", default=["yolov8n-seg.pt"])
    parser.add_argument("--imgsz", nargs="+", type=int, default=[960])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    trainer = Path(__file__).with_name("train_yolo_seg.py")
    for model, size, seed in product(args.models, args.imgsz, args.seeds):
        name = f"{args.dataset}_{Path(model).stem}_img{size}_seed{seed}"
        cmd = [
            sys.executable, str(trainer), "--dataset", args.dataset,
            "--model", model, "--imgsz", str(size), "--seed", str(seed),
            "--epochs", str(args.epochs), "--batch", str(args.batch),
            "--device", args.device, "--name", name,
            "--amp" if args.amp else "--no-amp",
        ]
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
