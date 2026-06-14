#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "lwt train images": (ROOT / "data/needle_lwt/needle_lwt_yolo_split/images/train", 95),
    "lwt val images": (ROOT / "data/needle_lwt/needle_lwt_yolo_split/images/val", 24),
    "inbox train images": (ROOT / "datasets/needle_inbox/images/train", 45),
    "inbox val images": (ROOT / "datasets/needle_inbox/images/val", 5),
    "combined train images": (ROOT / "datasets/needle_inbox_combined/images/train", 145),
    "combined val images": (ROOT / "datasets/needle_inbox_combined/images/val", 24),
}


def is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[:80].startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-foundationpose", action="store_true")
    args = parser.parse_args()
    errors = []

    for label, (directory, expected_count) in EXPECTED.items():
        files = [p for p in directory.glob("*") if p.is_file()]
        print(f"{label}: {len(files)} files")
        if len(files) != expected_count:
            errors.append(f"{label}: expected {expected_count}, found {len(files)}")
        pointers = [p for p in files[:5] if is_lfs_pointer(p)]
        if pointers:
            errors.append(f"{label}: Git LFS objects not pulled")

    required = [ROOT / "model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl"]
    for path in required:
        print(f"asset: {path.relative_to(ROOT)}")
        if not path.is_file():
            errors.append(f"missing asset: {path.relative_to(ROOT)}")
        elif is_lfs_pointer(path):
            errors.append(f"Git LFS object not pulled: {path.relative_to(ROOT)}")

    weights = [
        ROOT / "yolov8n-seg.pt",
        ROOT / "runs/needle_lwt_seg/yolov8n_seg_lwt/weights/best.pt",
        ROOT / "runs/needle_inbox_seg/yolov8n_seg_inbox/weights/best.pt",
        ROOT / "runs/needle_inbox_seg/yolov8n_seg_combined/weights/best.pt",
    ]
    missing_weights = [path.relative_to(ROOT) for path in weights if not path.is_file()]
    if missing_weights:
        print("optional weight package is not extracted; missing:")
        for path in missing_weights:
            print(f"- {path}")

    if args.check_foundationpose:
        fp = ROOT / "third_party/FoundationPose"
        for relative in ("run_orbbec_mug_live.py", "run_orbbec_live_folder.py", "Utils.py"):
            if not (fp / relative).is_file():
                errors.append(f"FoundationPose setup incomplete: {fp / relative}")

    if errors:
        print("\nFAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nHandoff validation passed.")


if __name__ == "__main__":
    main()
