#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取彩色图: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"无法读取深度图: {path}")
    if depth.ndim != 2:
        raise RuntimeError(f"深度图不是单通道: {path}, shape={depth.shape}")
    return depth


def depth_vis(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo = float(np.percentile(depth[valid], 2))
    hi = float(np.percentile(depth[valid], 98))
    scaled = np.clip(depth.astype(np.float32), lo, hi)
    scaled = ((scaled - lo) / max(hi - lo, 1.0) * 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_bbox(image: np.ndarray, box, color, thickness=2):
    out = image.copy()
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="诊断实时 color/depth/mask 是否对齐。")
    parser.add_argument("--live-dir", default="third_party/FoundationPose/live_orbbec")
    parser.add_argument("--out", default="report/realtime_eval/live_alignment_diagnosis/实时对齐诊断图.png")
    args = parser.parse_args()

    live = Path(args.live_dir)
    color_path = live / "color.ppm"
    depth_path = live / "depth.pgm"
    if not color_path.exists():
        color_path = live / "color.png"
    if not depth_path.exists():
        depth_path = live / "depth.png"

    color = read_color(color_path)
    depth = read_depth(depth_path)
    mask = cv2.imread(str(live / "mask_yolo.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("无法读取 mask_yolo.png")
    if mask.shape != depth.shape:
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

    valid = depth > 0
    mask_bool = mask > 0
    overlap = valid & mask_bool
    mask_box = bbox_from_mask(mask_bool.astype(np.uint8))
    valid_box = bbox_from_mask(valid.astype(np.uint8))
    overlap_box = bbox_from_mask(overlap.astype(np.uint8))

    overlay = color.copy()
    overlay[valid] = (0.65 * overlay[valid] + 0.35 * np.array([40, 180, 255])).astype(np.uint8)
    overlay[mask_bool] = (0.50 * overlay[mask_bool] + 0.50 * np.array([255, 70, 40])).astype(np.uint8)
    overlay[overlap] = (0.35 * overlay[overlap] + 0.65 * np.array([30, 220, 80])).astype(np.uint8)
    overlay = draw_bbox(overlay, mask_box, (255, 255, 255), 2)
    overlay = draw_bbox(overlay, valid_box, (0, 255, 255), 2)
    overlay = draw_bbox(overlay, overlap_box, (0, 255, 0), 3)

    stats = {
        "color": str(color_path),
        "depth": str(depth_path),
        "mask_area": int(mask_bool.sum()),
        "depth_valid": int(valid.sum()),
        "overlap": int(overlap.sum()),
        "mask_box": mask_box,
        "valid_depth_box": valid_box,
        "overlap_box": overlap_box,
        "depth_max_mm": int(depth.max()) if depth.size else 0,
        "depth_median_mm": float(np.median(depth[valid])) if valid.any() else 0.0,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (out_path.with_suffix(".json")).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    panels = [color, depth_vis(depth), overlay]
    labels = ["彩色图", "有效深度图", "对齐诊断"]
    h, w = color.shape[:2]
    canvas = Image.new("RGB", (w * 3, h + 92), (248, 248, 248))
    for i, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (i * w, 58))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 24)
    small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 17)
    for i, label in enumerate(labels):
        draw.text((i * w + 16, 14), label, fill=(30, 30, 30), font=font)
    msg = f"mask {stats['mask_area']} | 深度有效 {stats['depth_valid']} | 重合 {stats['overlap']}"
    draw.text((2 * w + 150, 20), msg, fill=(70, 70, 70), font=small)
    draw.text((2 * w + 16, h + 64), "白框=YOLO，黄框=有效深度，绿框=两者重合", fill=(50, 50, 50), font=small)
    canvas.save(out_path)
    print(out_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
