import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import cv2
import numpy as np
from ultralytics import YOLO


def append_log(log_file, message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def atomic_write_mask(mask_file, mask):
    mask_path = Path(mask_file)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = mask_path.with_suffix(mask_path.suffix + ".tmp.png")
    ok = cv2.imwrite(str(tmp_path), mask)
    if not ok:
        raise RuntimeError(f"Failed to write temporary mask: {tmp_path}")
    os.replace(tmp_path, mask_path)


def atomic_write_json(json_file, payload):
    json_path = Path(json_file)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, json_path)


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
    return path.stat().st_mtime


def read_image_retry(image_file, attempts=8, interval=0.08):
    last_err = None
    for _ in range(attempts):
        try:
            wait_for_stable_file(image_file)
            image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            if image is not None:
                return image
            last_err = RuntimeError("cv2.imread returned None")
        except Exception as exc:
            last_err = exc
        time.sleep(interval)
    raise RuntimeError(f"Failed to read stable image: {image_file}; last error: {last_err}")


def resolve_class_id(model, class_name):
    names = model.names
    for class_id, name in names.items():
        if str(name).lower() == class_name.lower():
            return int(class_id)
    available = ", ".join(str(v) for v in names.values())
    raise RuntimeError(f"Class '{class_name}' not found in model names: {available}")


def parse_roi(value):
    if not value:
        return None
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--roi must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_center_in_roi(box, roi):
    if roi is None:
        return True
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def predict_mask(model, image_file, class_id, conf, imgsz, min_area, device, roi, min_bbox_height):
    image = read_image_retry(image_file)
    height, width = image.shape[:2]

    results = model.predict(
        source=image,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO returned no results")

    result = results[0]
    if result.masks is None or result.boxes is None:
        raise RuntimeError("YOLO found no segmentation masks")

    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy()
    masks = result.masks.data.detach().cpu().numpy()

    candidates = []
    for i, det_class in enumerate(classes):
        if det_class != class_id:
            continue
        mask = masks[i]
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        area = int(np.count_nonzero(mask_u8))
        if area < min_area:
            continue
        box = bbox_from_mask(mask_u8)
        if box is None or not bbox_center_in_roi(box, roi):
            continue
        bbox_height = box[3] - box[1] + 1
        if bbox_height < min_bbox_height:
            continue
        candidates.append(
            {
                "mask": mask_u8,
                "conf": float(confs[i]),
                "area": area,
                "bbox": box,
                "class_id": int(det_class),
            }
        )

    if not candidates:
        raise RuntimeError(
            f"No target masks passed filters class_id={class_id}, conf>={conf}, min_area={min_area}"
        )

    candidates.sort(key=lambda item: (item["conf"], item["area"]), reverse=True)
    return candidates[0], len(candidates), (height, width)


def run_once(args, model, class_id, image_mtime=None):
    image_path = Path(args.image)
    if not image_path.exists():
        raise RuntimeError(f"Image does not exist: {image_path}")
    mtime = wait_for_stable_file(image_path)
    if image_mtime is not None and mtime <= image_mtime:
        return image_mtime, False

    selected, num_candidates, shape = predict_mask(
        model=model,
        image_file=image_path,
        class_id=class_id,
        conf=args.conf,
        imgsz=args.imgsz,
        min_area=args.min_area,
        device=args.device,
        roi=args.roi,
        min_bbox_height=args.min_bbox_height,
    )
    atomic_write_mask(args.mask, selected["mask"])

    meta_file = args.meta or str(Path(args.mask).with_suffix(".json"))
    atomic_write_json(
        meta_file,
        {
            "image": str(image_path),
            "mask": str(Path(args.mask)),
            "class_name": args.class_name,
            "class_id": selected["class_id"],
            "confidence": selected["conf"],
            "area": selected["area"],
            "bbox": selected["bbox"],
            "roi": args.roi,
            "num_candidates": num_candidates,
            "height": shape[0],
            "width": shape[1],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    append_log(
        args.log,
        f"wrote {args.mask} class={args.class_name} conf={selected['conf']:.3f} "
        f"area={selected['area']} bbox={selected['bbox']}",
    )
    return mtime, True


def clear_mask(args, reason):
    image = read_image_retry(args.image, attempts=2, interval=0.02)
    empty = np.zeros(image.shape[:2], dtype=np.uint8)
    atomic_write_mask(args.mask, empty)
    meta_file = args.meta or str(Path(args.mask).with_suffix(".json"))
    atomic_write_json(
        meta_file,
        {
            "image": str(args.image),
            "mask": str(args.mask),
            "class_name": args.class_name,
            "class_id": None,
            "confidence": 0.0,
            "area": 0,
            "num_candidates": 0,
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "cleared": True,
            "reason": str(reason),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="YOLO-seg file bridge for mug/cup masks.")
    parser.add_argument("--image", default="FoundationPose/live_orbbec/color.png")
    parser.add_argument("--mask", default="FoundationPose/live_orbbec/mask_yolo.png")
    parser.add_argument("--meta", default=None)
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--class_name", default="cup")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min_area", type=int, default=300)
    parser.add_argument("--roi", default=None, help="Keep masks whose bbox center is inside x1,y1,x2,y2.")
    parser.add_argument("--min_bbox_height", type=int, default=0)
    parser.add_argument("--device", default=None, help="Ultralytics device, for example cpu or 0.")
    parser.add_argument("--clear_on_fail", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll_interval", type=float, default=0.2)
    parser.add_argument("--log", default="FoundationPose/live_orbbec/yolo_mug_mask_bridge.log")
    args = parser.parse_args()
    args.roi = parse_roi(args.roi)

    model = YOLO(args.model)
    class_id = resolve_class_id(model, args.class_name)
    append_log(args.log, f"loaded model={args.model} class={args.class_name} class_id={class_id}")

    last_mtime = None
    while True:
        try:
            last_mtime, _ = run_once(args, model, class_id, image_mtime=last_mtime)
        except Exception as exc:
            if args.clear_on_fail:
                try:
                    clear_mask(args, exc)
                except Exception as clear_exc:
                    append_log(args.log, f"failed to clear mask: {type(clear_exc).__name__}: {clear_exc}")
            append_log(args.log, f"no mask update: {type(exc).__name__}: {exc}")
        if not args.loop:
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
