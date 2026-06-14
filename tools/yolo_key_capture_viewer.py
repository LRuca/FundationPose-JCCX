#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit("Missing dependency: Pillow. Install with: python -m pip install pillow") from exc


def is_complete_png(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 16:
        return False
    try:
        with path.open("rb") as f:
            f.seek(-12, 2)
            return f.read(12) == b"\x00\x00\x00\x00IEND\xaeB`\x82"
    except OSError:
        return False


def read_frame_index(frame_json: Path) -> str:
    if not frame_json.exists():
        return ""
    try:
        import json

        with frame_json.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        value = data.get("frame_index")
        return f"{int(value):06d}" if value is not None else ""
    except Exception:
        return ""


class KeyCaptureViewer:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.live_dir = Path(args.live_dir)
        self.out_dir = Path(args.out_dir)
        self.color_path = self.live_dir / "color.png"
        self.frame_json = self.live_dir / "frame.json"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.capture_proc: subprocess.Popen | None = None
        self.last_mtime_ns: int | None = None
        self.current_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.saved_count = 0
        self.last_saved_file: Path | None = None

        self.root.title("YOLO Source Key Capture")
        self.root.geometry("1040x720")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status_var = tk.StringVar(value="Waiting for live frame...")
        self.count_var = tk.StringVar(value="Saved: 0")

        self._build_ui()
        self._bind_keys()

        if args.start_camera:
            self.start_camera()
        self.refresh()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(top, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT)
        ttk.Button(top, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(top, text="Save Frame", command=self.save_frame).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(top, textvariable=self.count_var).pack(side=tk.LEFT, padx=(12, 0))

        # output directory chooser
        out_frame = ttk.Frame(self.root)
        out_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(out_frame, text="Output:").pack(side=tk.LEFT)
        self.out_dir_var = tk.StringVar(value=str(self.out_dir))
        ttk.Entry(out_frame, textvariable=self.out_dir_var, width=60).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(out_frame, text="Browse...", command=self.choose_output_dir).pack(side=tk.LEFT, padx=(4, 0))

        self.canvas = tk.Canvas(self.root, bg="#202020", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        self.canvas.bind("<Configure>", lambda _e: self.draw())

        bottom = ttk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(bottom, text="Space/S: save   C: start camera   X: stop camera   Q/Esc: quit").pack(side=tk.LEFT)
        ttk.Label(bottom, textvariable=self.status_var, anchor=tk.E).pack(side=tk.RIGHT)

    def _bind_keys(self) -> None:
        for key in ("<space>", "s", "S"):
            self.root.bind(key, lambda _e: self.save_frame())
        self.root.bind("c", lambda _e: self.start_camera())
        self.root.bind("C", lambda _e: self.start_camera())
        self.root.bind("x", lambda _e: self.stop_camera())
        self.root.bind("X", lambda _e: self.stop_camera())
        self.root.bind("q", lambda _e: self.close())
        self.root.bind("Q", lambda _e: self.close())
        self.root.bind("<Escape>", lambda _e: self.close())

    def choose_output_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.out_dir, mustexist=False)
        if chosen:
            self.out_dir = Path(chosen)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.out_dir_var.set(str(self.out_dir))
            self.status_var.set(f"Output dir: {self.out_dir}")

    def start_camera(self) -> None:
        if self.capture_proc and self.capture_proc.poll() is None:
            self.status_var.set("Camera capture is already running.")
            return

        script = Path(self.args.project_root) / "scripts" / "orbbec_capture_loop.ps1"
        if not script.exists():
            self.status_var.set(f"Missing camera script: {script}")
            return

        log_dir = Path(self.args.project_root) / "logs" / "key_capture"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = (log_dir / "orbbec_capture.out.log").open("w", encoding="utf-8")
        stderr = (log_dir / "orbbec_capture.err.log").open("w", encoding="utf-8")
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-OutDir",
            str(self.live_dir),
        ]
        self.capture_proc = subprocess.Popen(
            cmd,
            cwd=self.args.project_root,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.status_var.set("Camera capture started.")

    def stop_camera(self) -> None:
        if self.capture_proc and self.capture_proc.poll() is None:
            self.capture_proc.terminate()
            try:
                self.capture_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.capture_proc.kill()
        self.capture_proc = None
        self.status_var.set("Camera capture stopped.")

    def refresh(self) -> None:
        try:
            if is_complete_png(self.color_path):
                mtime_ns = self.color_path.stat().st_mtime_ns
                if mtime_ns != self.last_mtime_ns:
                    with Image.open(self.color_path) as img:
                        self.current_image = img.convert("RGB")
                    self.last_mtime_ns = mtime_ns
                    frame_id = read_frame_index(self.frame_json)
                    suffix = f" frame {frame_id}" if frame_id else ""
                    self.status_var.set(f"Live{suffix}: {self.current_image.width}x{self.current_image.height}")
                    self.draw()
            elif self.capture_proc and self.capture_proc.poll() is not None:
                self.status_var.set("Camera process exited. Check logs\\key_capture.")
        except Exception as exc:
            self.status_var.set(f"Preview error: {type(exc).__name__}: {exc}")
        self.root.after(self.args.poll_ms, self.refresh)

    def draw(self) -> None:
        self.canvas.delete("all")
        if self.current_image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                self.canvas.winfo_height() / 2,
                text="No live frame yet",
                fill="#dddddd",
                font=("Segoe UI", 18),
            )
            return

        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        img_w, img_h = self.current_image.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        draw_w = max(int(img_w * scale), 1)
        draw_h = max(int(img_h * scale), 1)
        resized = self.current_image.resize((draw_w, draw_h), Image.Resampling.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(resized)
        x = (canvas_w - draw_w) // 2
        y = (canvas_h - draw_h) // 2
        self.canvas.create_image(x, y, image=self.tk_image, anchor=tk.NW)

    def save_frame(self) -> None:
        if self.current_image is None or not is_complete_png(self.color_path):
            self.status_var.set("No complete frame to save yet.")
            return

        frame_id = read_frame_index(self.frame_json) or f"{self.saved_count + 1:06d}"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1) * 1000)
        out_file = self.out_dir / f"{self.args.prefix}_{stamp}_{millis:03d}_{frame_id}.png"
        tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
        shutil.copy2(self.color_path, tmp_file)
        tmp_file.replace(out_file)
        self.saved_count += 1
        self.last_saved_file = out_file
        self.count_var.set(f"Saved: {self.saved_count}")
        self.status_var.set(f"Saved: {out_file.name}")

    def close(self) -> None:
        if self.args.stop_camera_on_exit:
            self.stop_camera()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live preview and key-triggered YOLO source capture.")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--live-dir", default=str(project_root / "FoundationPose" / "live_orbbec"))
    parser.add_argument("--out-dir", default=str(project_root / "data" / "needle_raw"))
    parser.add_argument("--prefix", default="needle")
    parser.add_argument("--poll-ms", type=int, default=80)
    parser.add_argument("--start-camera", action="store_true")
    parser.add_argument("--stop-camera-on-exit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    KeyCaptureViewer(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
