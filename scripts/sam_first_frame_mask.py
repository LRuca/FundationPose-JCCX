import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


def atomic_write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp.png")
    if not cv2.imwrite(str(tmp_path), image):
        raise RuntimeError(f"Failed to write temporary image: {tmp_path}")
    os.replace(tmp_path, path)


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def wait_for_stable_file(path, checks=3, interval=0.05):
    path = Path(path)
    last = None
    for _ in range(checks):
        stat = path.stat()
        current = (stat.st_size, stat.st_mtime_ns)
        if last is not None and current != last:
            last = current
            time.sleep(interval)
            continue
        last = current
        time.sleep(interval)
    return path


def parse_bbox(text):
    parts = [int(round(float(x.strip()))) for x in text.replace(";", ",").split(",") if x.strip()]
    if len(parts) != 4:
        raise ValueError("--bbox must be four numbers: x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def clamp_bbox(bbox, width, height, min_size=2):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        raise ValueError(f"Bounding box is too small after clamping: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def load_sam(model_type, checkpoint, device):
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "segment-anything is not installed. Install it in the runtime env with: "
            "python -m pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc

    if not Path(checkpoint).exists():
        raise RuntimeError(f"SAM checkpoint not found: {checkpoint}")
    if model_type not in sam_model_registry:
        available = ", ".join(sorted(sam_model_registry.keys()))
        raise RuntimeError(f"Unknown SAM model type '{model_type}'. Available: {available}")

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def select_mask(masks, scores, min_area):
    candidates = []
    for idx, mask in enumerate(masks):
        mask_u8 = mask.astype(np.uint8) * 255
        area = int(np.count_nonzero(mask_u8))
        if area < min_area:
            continue
        candidates.append((float(scores[idx]), area, mask_u8, idx))
    if not candidates:
        raise RuntimeError(f"SAM returned no mask with area >= {min_area}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0]


def overlay_mask(image_rgb, mask_u8, bbox):
    overlay = image_rgb.copy()
    color = np.zeros_like(overlay)
    color[:, :, 1] = 255
    alpha = (mask_u8 > 0).astype(np.float32)[:, :, None] * 0.45
    overlay = (overlay * (1 - alpha) + color * alpha).astype(np.uint8)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 128, 0), 2)
    return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Generate a first-frame object mask from a user bbox using SAM.")
    parser.add_argument("--image", default="FoundationPose/live_orbbec/color.png")
    parser.add_argument("--bbox", required=True, help="x1,y1,x2,y2 in image pixel coordinates")
    parser.add_argument("--mask", default="FoundationPose/live_orbbec/mask_yolo.png")
    parser.add_argument("--meta", default="FoundationPose/live_orbbec/mask_yolo.json")
    parser.add_argument("--preview", default="FoundationPose/live_orbbec/mask_sam_preview.png")
    parser.add_argument("--checkpoint", default="downloads/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_area", type=int, default=80)
    args = parser.parse_args()

    image_path = wait_for_stable_file(args.image)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    height, width = image_bgr.shape[:2]
    bbox = clamp_bbox(parse_bbox(args.bbox), width=width, height=height)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    predictor = load_sam(args.model_type, args.checkpoint, args.device)
    predictor.set_image(image_rgb)
    masks, scores, logits = predictor.predict(
        box=np.array(bbox, dtype=np.float32),
        multimask_output=True,
    )
    score, area, mask_u8, mask_index = select_mask(masks, scores, args.min_area)

    atomic_write_image(args.mask, mask_u8)
    if args.preview:
        atomic_write_image(args.preview, overlay_mask(image_rgb, mask_u8, bbox))
    atomic_write_json(
        args.meta,
        {
            "image": str(image_path),
            "mask": str(Path(args.mask)),
            "preview": str(Path(args.preview)) if args.preview else None,
            "source": "sam_bbox",
            "bbox_xyxy": list(map(int, bbox)),
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "device": args.device,
            "score": score,
            "area": area,
            "mask_index": int(mask_index),
            "height": height,
            "width": width,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(f"wrote {args.mask} bbox={bbox} score={score:.4f} area={area}", flush=True)


if __name__ == "__main__":
    main()
