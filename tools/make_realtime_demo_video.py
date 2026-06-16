#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def collect_frames(frame_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    frames = [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    frames.sort(key=lambda p: (len(p.stem), p.stem))
    if not frames:
        raise FileNotFoundError(f"没有找到可视化帧: {frame_dir}")
    return frames


def write_with_ffmpeg(frames: list[Path], out_file: Path, fps: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, frame in enumerate(frames):
            (tmp_dir / f"{i:06d}{frame.suffix.lower()}").symlink_to(frame.resolve())
        pattern = str(tmp_dir / f"%06d{frames[0].suffix.lower()}")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_file),
        ]
        subprocess.run(cmd, check=True)
    return True


def write_with_cv2(frames: list[Path], out_file: Path, fps: float) -> None:
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"无法读取视频帧: {frames[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {out_file}")
    try:
        writer.write(first)
        for frame in frames[1:]:
            image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"无法读取视频帧: {frame}")
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="把 FoundationPose track_vis 图片序列合成为 demo 视频。")
    parser.add_argument("--frames", default="third_party/FoundationPose/debug_needle_live/track_vis")
    parser.add_argument("--out", default="report/needle_realtime_demo.mp4")
    parser.add_argument("--fps", type=float, default=30)
    args = parser.parse_args()

    frame_dir = (ROOT / args.frames).resolve() if not Path(args.frames).is_absolute() else Path(args.frames)
    out_file = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    frames = collect_frames(frame_dir)
    try:
        used_ffmpeg = write_with_ffmpeg(frames, out_file, args.fps)
        if not used_ffmpeg:
            write_with_cv2(frames, out_file, args.fps)
    except Exception:
        if out_file.exists():
            out_file.unlink()
        raise

    print(f"已生成视频: {out_file}")
    print(f"帧数: {len(frames)}")
    print(f"帧率: {args.fps}")


if __name__ == "__main__":
    main()
