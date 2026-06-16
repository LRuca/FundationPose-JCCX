#!/usr/bin/env python
"""实时相机预览：读取采集器发布的最新帧，画出 ROI 工作区 + 中心十字 + 中心深度(米)。

用于摆针/调角度。按 q 退出。不直接打开相机（由采集器独占），只读它写出的文件。
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def parse_roi(s):
    if not s:
        return None
    x1, y1, x2, y2 = [int(v) for v in s.split(",")]
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live_dir", default="third_party/FoundationPose/live_orbbec")
    ap.add_argument("--roi", default="230,90,450,320")
    ap.add_argument("--track_dir", default=None,
                    help="若给定，检测到该目录出现跟踪帧即提示『已注册，可以移动针』")
    args = ap.parse_args()
    live = Path(args.live_dir)
    roi = parse_roi(args.roi)
    track_dir = Path(args.track_dir) if args.track_dir else None

    win = "needle camera preview (q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    last_t = time.time()
    fps = 0.0
    print("preview started; press q in the window to quit", flush=True)
    while True:
        color = cv2.imread(str(live / "color.ppm"), cv2.IMREAD_COLOR)
        if color is None:
            color = cv2.imread(str(live / "color.png"), cv2.IMREAD_COLOR)
        if color is None:
            time.sleep(0.05)
            continue
        depth = cv2.imread(str(live / "depth.pgm"), cv2.IMREAD_UNCHANGED)
        h, w = color.shape[:2]
        vis = color.copy()

        # ROI 工作区
        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(vis, "ROI (put needle here)", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        else:
            cx, cy = w // 2, h // 2
        # 中心十字
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)

        # 中心深度
        dtxt = "depth@center: N/A"
        if depth is not None:
            patch = depth[max(0, cy - 4):cy + 5, max(0, cx - 4):cx + 5].astype(np.float32)
            valid = patch[patch > 0]
            if valid.size:
                dtxt = f"depth@center: {np.median(valid)/1000.0:.2f} m"
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - last_t))
        last_t = now
        bar = f"{w}x{h}  view_fps:{fps:4.1f}  {dtxt}  | keep needle 0.4-1.5m, avoid screens/reflection"
        cv2.rectangle(vis, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(vis, bar, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 注册状态横幅（配合录制）
        if track_dir is not None:
            tracked = track_dir.exists() and any(track_dir.glob("*.png"))
            if tracked:
                msg, col = "REGISTERED - move the needle now (tilt / rotate)", (0, 200, 0)
            else:
                msg, col = "REGISTERING... hold the needle STILL", (0, 180, 255)
            cv2.rectangle(vis, (0, h - 30), (w, h), col, -1)
            cv2.putText(vis, msg, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, vis)
        if (cv2.waitKey(20) & 0xFF) in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
