#!/usr/bin/env python
"""
Portable YOLO segmentation annotator for the puncture-needle pipeline.

Dependencies:
  pip install pillow

Optional packaging:
  pyinstaller --onefile --windowed --name NeedleAnnotator tools/needle_yolo_annotator.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: Pillow. Install with: python -m pip install pillow"
    ) from exc

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASS_NAME = "needle"
DEFAULT_CLASS_NAMES = [DEFAULT_CLASS_NAME]
CLASS_COLORS = [
    ("#00ff66", "#194d2e"),
    ("#00ccff", "#173a4d"),
    ("#ff66cc", "#4d173d"),
    ("#ff9933", "#4d3217"),
    ("#ccff33", "#3f4d17"),
    ("#aa88ff", "#2f2750"),
]


@dataclass
class Polygon:
    points: List[Tuple[float, float]]
    class_id: int = 0


@dataclass
class ImageRecord:
    source: str
    export_name: str
    split: str = "train"
    polygons: List[Polygon] = field(default_factory=list)


def stable_export_name(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}_{digest}{path.suffix.lower()}"


def norm_point(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    return (
        min(max(x / max(width, 1), 0.0), 1.0),
        min(max(y / max(height, 1), 0.0), 1.0),
    )


def parse_yolo_seg(label_file: Path, width: int, height: int) -> List[Polygon]:
    polygons: List[Polygon] = []
    if not label_file.exists():
        return polygons
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        pts = []
        for i in range(0, len(coords) - 1, 2):
            pts.append((coords[i] * width, coords[i + 1] * height))
        if len(pts) >= 3:
            polygons.append(Polygon(points=pts, class_id=class_id))
    return polygons


class NeedleAnnotator:
    def __init__(self, root: tk.Tk, image_dir: Optional[Path], dataset_dir: Optional[Path], class_name: str = DEFAULT_CLASS_NAME):
        self.root = root
        self.root.title("Needle YOLO Segmentation Annotator")
        self.root.geometry("1280x820")

        self.image_dir = image_dir
        self.dataset_dir = dataset_dir
        self.records: List[ImageRecord] = []
        self.index = 0
        self.current_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.current_points: List[Tuple[float, float]] = []
        self.selected_polygon: Optional[int] = None

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.fit_mode = True
        self.dragging = False
        self.pan_moved = False
        self.last_drag = (0, 0)

        self.class_names: List[str] = [class_name]
        self.current_class = tk.StringVar(value=class_name)
        self.new_class_name = tk.StringVar(value="")
        self.split_var = tk.StringVar(value="train")
        self.status_var = tk.StringVar(value="Open an image folder to begin.")

        self._build_ui()
        self._bind_shortcuts()

        if self.image_dir and self.dataset_dir:
            self.load_project(self.image_dir, self.dataset_dir)
        else:
            self.choose_folders()

    @property
    def record(self) -> Optional[ImageRecord]:
        if not self.records:
            return None
        return self.records[self.index]

    @property
    def project_file(self) -> Path:
        assert self.dataset_dir is not None
        return self.dataset_dir / "needle_annotator_project.json"

    def _build_ui(self) -> None:
        root = ttk.Frame(self.root)
        root.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        ttk.Button(toolbar, text="Open", command=self.choose_folders).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Save", command=self.save_current).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(toolbar, text="Export YAML", command=self.write_dataset_yaml).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(toolbar, text="Label").pack(side=tk.LEFT)
        self.class_box = ttk.Combobox(toolbar, textvariable=self.current_class, width=16, state="readonly")
        self.class_box.pack(side=tk.LEFT, padx=(4, 4))
        self.class_box.bind("<<ComboboxSelected>>", lambda _e: self.on_class_selected())
        ttk.Entry(toolbar, textvariable=self.new_class_name, width=12).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Button(toolbar, text="Add Label", command=self.add_class).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(toolbar, text="Split").pack(side=tk.LEFT)
        split_box = ttk.Combobox(toolbar, textvariable=self.split_var, width=8, values=["train", "val"], state="readonly")
        split_box.pack(side=tk.LEFT, padx=(4, 8))
        split_box.bind("<<ComboboxSelected>>", lambda _e: self.set_split())
        ttk.Button(toolbar, text="Prev", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(toolbar, text="Undo Point", command=self.undo_point).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(toolbar, text="Close Poly", command=self.close_polygon).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(toolbar, text="Delete Poly", command=self.delete_selected_polygon).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(toolbar, text="Fit", command=self.fit_image).pack(side=tk.LEFT, padx=(12, 0))
        self.sync_class_combo()

        body = ttk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 4), pady=4)
        left.pack_propagate(False)

        self.file_list = tk.Listbox(left, exportselection=False)
        self.file_list.pack(fill=tk.BOTH, expand=True)
        self.file_list.bind("<<ListboxSelect>>", self.on_file_select)

        help_text = (
            "Left click: add point\n"
            "Enter/right click: close polygon\n"
            "Mouse wheel: zoom\n"
            "Middle/right drag: pan\n"
            "Z: undo point\n"
            "Del: delete selected polygon\n"
            "N/P: next/previous\n"
            "Ctrl+S: save"
        )
        ttk.Label(left, text=help_text, justify=tk.LEFT).pack(fill=tk.X, pady=(8, 0))

        canvas_frame = ttk.Frame(body)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6), pady=4)

        self.canvas = tk.Canvas(canvas_frame, bg="#202020", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.redraw())
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        self.canvas.bind("<B2-Motion>", self.pan)
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<ButtonRelease-3>", self.end_right_pan)
        self.canvas.bind("<B3-Motion>", self.pan)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, 1.1))
        self.canvas.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 1 / 1.1))

        status = ttk.Label(root, textvariable=self.status_var, anchor=tk.W)
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(0, 4))

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-s>", lambda _e: self.save_current())
        self.root.bind("<Return>", lambda _e: self.close_polygon())
        self.root.bind("z", lambda _e: self.undo_point())
        self.root.bind("Z", lambda _e: self.undo_point())
        self.root.bind("<Delete>", lambda _e: self.delete_selected_polygon())
        self.root.bind("n", lambda _e: self.next_image())
        self.root.bind("N", lambda _e: self.next_image())
        self.root.bind("p", lambda _e: self.prev_image())
        self.root.bind("P", lambda _e: self.prev_image())
        self.root.bind("f", lambda _e: self.fit_image())
        self.root.bind("F", lambda _e: self.fit_image())

    def choose_folders(self) -> None:
        image_dir = filedialog.askdirectory(title="Choose source image folder")
        if not image_dir:
            return
        dataset_dir = filedialog.askdirectory(title="Choose YOLO dataset output folder")
        if not dataset_dir:
            return
        self.load_project(Path(image_dir), Path(dataset_dir))

    def load_project(self, image_dir: Path, dataset_dir: Path) -> None:
        self.image_dir = image_dir.resolve()
        self.dataset_dir = dataset_dir.resolve()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in self.image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not image_paths:
            messagebox.showerror("No images", f"No supported images found in:\n{self.image_dir}")
            return

        old_records: Dict[str, ImageRecord] = {}
        if self.project_file.exists():
            try:
                raw = json.loads(self.project_file.read_text(encoding="utf-8"))
                class_names = raw.get("class_names")
                if isinstance(class_names, list) and class_names:
                    self.class_names = [str(name).strip() or f"class_{i}" for i, name in enumerate(class_names)]
                else:
                    self.class_names = [str(raw.get("class_name", DEFAULT_CLASS_NAME)).strip() or DEFAULT_CLASS_NAME]
                self.current_class.set(self.class_names[0])
                self.sync_class_combo()
                for item in raw.get("records", []):
                    polygons = [
                        Polygon(points=[tuple(pt) for pt in poly.get("points", [])], class_id=int(poly.get("class_id", 0)))
                        for poly in item.get("polygons", [])
                    ]
                    old_records[item["source"]] = ImageRecord(
                        source=item["source"],
                        export_name=item.get("export_name", stable_export_name(Path(item["source"]))),
                        split=item.get("split", "train"),
                        polygons=polygons,
                    )
            except Exception as exc:
                messagebox.showwarning("Project read failed", str(exc))

        self.records = []
        for path in image_paths:
            key = str(path.resolve())
            rec = old_records.get(key)
            if rec is None:
                rec = ImageRecord(source=key, export_name=stable_export_name(path))
                label_file = self.label_path_for(rec)
                try:
                    with Image.open(path) as img:
                        rec.polygons = parse_yolo_seg(label_file, img.width, img.height)
                except Exception:
                    pass
            self.records.append(rec)

        self.index = min(self.index, len(self.records) - 1)
        self.refresh_file_list()
        self.load_image(self.index)
        self.sync_class_combo()

    def refresh_file_list(self) -> None:
        self.file_list.delete(0, tk.END)
        for i, rec in enumerate(self.records):
            mark = "*" if rec.polygons else " "
            name = Path(rec.source).name
            self.file_list.insert(tk.END, f"{mark} [{rec.split}] {i + 1:04d} {name}")
        self.file_list.selection_clear(0, tk.END)
        if self.records:
            self.file_list.selection_set(self.index)
            self.file_list.see(self.index)

    def load_image(self, index: int) -> None:
        if not self.records:
            return
        self.save_current(silent=True)
        self.index = max(0, min(index, len(self.records) - 1))
        rec = self.record
        assert rec is not None
        self.current_image = Image.open(rec.source).convert("RGB")
        self.current_points = []
        self.selected_polygon = None
        self.split_var.set(rec.split)
        self.fit_image(save=False)
        self.refresh_file_list()
        self.redraw()
        self.update_status()

    def image_path_for(self, rec: ImageRecord) -> Path:
        assert self.dataset_dir is not None
        return self.dataset_dir / "images" / rec.split / rec.export_name

    def label_path_for(self, rec: ImageRecord) -> Path:
        assert self.dataset_dir is not None
        return self.dataset_dir / "labels" / rec.split / f"{Path(rec.export_name).stem}.txt"

    def save_current(self, silent: bool = False) -> None:
        rec = self.record
        if rec is None or self.current_image is None or self.dataset_dir is None:
            return

        if len(self.current_points) >= 3:
            rec.polygons.append(Polygon(points=list(self.current_points), class_id=self.current_class_id()))
            self.current_points = []
            self.selected_polygon = len(rec.polygons) - 1

        for split in ("train", "val"):
            old_label = self.dataset_dir / "labels" / split / f"{Path(rec.export_name).stem}.txt"
            if old_label != self.label_path_for(rec) and old_label.exists():
                old_label.unlink()
            old_image = self.dataset_dir / "images" / split / rec.export_name
            if old_image != self.image_path_for(rec) and old_image.exists():
                old_image.unlink()

        out_image = self.image_path_for(rec)
        out_label = self.label_path_for(rec)
        out_image.parent.mkdir(parents=True, exist_ok=True)
        out_label.parent.mkdir(parents=True, exist_ok=True)

        if not out_image.exists():
            shutil.copy2(rec.source, out_image)

        width, height = self.current_image.size
        lines = []
        for poly in rec.polygons:
            if len(poly.points) < 3:
                continue
            vals = [str(poly.class_id)]
            for x, y in poly.points:
                nx, ny = norm_point(x, y, width, height)
                vals.extend([f"{nx:.6f}", f"{ny:.6f}"])
            lines.append(" ".join(vals))
        out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.write_project()
        self.write_dataset_yaml()
        self.refresh_file_list()
        self.redraw()
        if not silent:
            if lines:
                self.update_status(f"Saved {len(lines)} polygon(s): {out_label}")
            else:
                self.update_status(f"Saved empty label: no closed polygon with at least 3 points.")

    def write_project(self) -> None:
        assert self.dataset_dir is not None
        payload = {
            "class_name": self.class_names[0] if self.class_names else DEFAULT_CLASS_NAME,
            "class_names": self.class_names,
            "records": [
                {
                    "source": rec.source,
                    "export_name": rec.export_name,
                    "split": rec.split,
                    "polygons": [
                        {"class_id": poly.class_id, "points": [[x, y] for x, y in poly.points]}
                        for poly in rec.polygons
                    ],
                }
                for rec in self.records
            ],
        }
        self.project_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_dataset_yaml(self) -> None:
        if self.dataset_dir is None:
            return
        names = self.class_names or list(DEFAULT_CLASS_NAMES)
        yaml_text = (
            f"path: {self.dataset_dir.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n\n"
            "names:\n"
            + "".join(f"  {i}: {name}\n" for i, name in enumerate(names))
        )
        out = self.dataset_dir / "needle.yaml"
        out.write_text(yaml_text, encoding="utf-8")
        self.update_status(f"Wrote dataset YAML: {out}")

    def sync_class_combo(self) -> None:
        if not hasattr(self, "class_box"):
            return
        self.class_box["values"] = self.class_names
        if not self.current_class.get() and self.class_names:
            self.current_class.set(self.class_names[0])

    def current_class_id(self) -> int:
        name = self.current_class.get().strip()
        if name in self.class_names:
            return self.class_names.index(name)
        if not name:
            name = DEFAULT_CLASS_NAME
        self.class_names.append(name)
        self.sync_class_combo()
        self.current_class.set(name)
        return len(self.class_names) - 1

    def class_name_for(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def add_class(self) -> None:
        name = self.new_class_name.get().strip()
        if not name:
            self.update_status("Type a label name before Add Label.")
            return
        if name not in self.class_names:
            self.class_names.append(name)
        self.current_class.set(name)
        self.new_class_name.set("")
        self.sync_class_combo()
        self.write_project()
        self.write_dataset_yaml()
        self.update_status(f"Current label: {name}")

    def on_class_selected(self) -> None:
        rec = self.record
        if rec is not None and self.selected_polygon is not None:
            if 0 <= self.selected_polygon < len(rec.polygons):
                rec.polygons[self.selected_polygon].class_id = self.current_class_id()
                self.save_current(silent=True)
                self.redraw()
                self.update_status("Updated selected polygon label.")
                return
        self.update_status(f"Current label: {self.current_class.get()}")

    def set_split(self) -> None:
        rec = self.record
        if rec is None:
            return
        rec.split = self.split_var.get()
        self.save_current(silent=True)
        self.refresh_file_list()
        self.update_status()

    def fit_image(self, save: bool = True) -> None:
        if self.current_image is None:
            return
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        iw, ih = self.current_image.size
        self.scale = min(cw / iw, ch / ih) * 0.96
        self.offset_x = (cw - iw * self.scale) / 2
        self.offset_y = (ch - ih * self.scale) / 2
        self.fit_mode = True
        if save:
            self.redraw()

    def image_to_canvas(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        return pt[0] * self.scale + self.offset_x, pt[1] * self.scale + self.offset_y

    def canvas_to_image(self, x: float, y: float) -> Tuple[float, float]:
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale

    def redraw(self) -> None:
        self.canvas.delete("all")
        if self.current_image is None:
            return
        iw, ih = self.current_image.size
        disp_size = (max(1, int(iw * self.scale)), max(1, int(ih * self.scale)))
        display = self.current_image.resize(disp_size, Image.Resampling.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(display)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)

        rec = self.record
        if rec:
            for idx, poly in enumerate(rec.polygons):
                self.draw_polygon(poly.points, complete=True, selected=(idx == self.selected_polygon), class_id=poly.class_id)
        self.draw_polygon(self.current_points, complete=False, selected=False, class_id=self.current_class_id())

    def draw_polygon(self, points: List[Tuple[float, float]], complete: bool, selected: bool, class_id: int) -> None:
        if not points:
            return
        coords = []
        for pt in points:
            cx, cy = self.image_to_canvas(pt)
            coords.extend([cx, cy])

        color, fill = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        if selected:
            color, fill = "#ffcc00", "#4d3f19"
        if len(points) >= 2:
            if complete:
                self.canvas.create_polygon(*coords, outline=color, fill=fill, width=2)
            else:
                self.canvas.create_line(*coords, fill="#00ccff", width=2)
        if complete:
            cx, cy = self.image_to_canvas(points[0])
            self.canvas.create_text(
                cx + 6,
                cy - 12,
                anchor=tk.W,
                fill=color,
                text=self.class_name_for(class_id),
                font=("TkDefaultFont", 10, "bold"),
            )
        for i, pt in enumerate(points):
            cx, cy = self.image_to_canvas(pt)
            r = 4 if i else 5
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, fill=color)

    def on_left_click(self, event: tk.Event) -> None:
        if self.current_image is None:
            return
        x, y = self.canvas_to_image(event.x, event.y)
        iw, ih = self.current_image.size
        if x < 0 or y < 0 or x >= iw or y >= ih:
            return

        hit = self.find_polygon_at(x, y)
        if hit is not None and not self.current_points:
            self.selected_polygon = hit
            rec = self.record
            if rec is not None:
                self.current_class.set(self.class_name_for(rec.polygons[hit].class_id))
        else:
            self.selected_polygon = None
            self.current_points.append((x, y))
        self.redraw()
        self.update_status()

    def start_pan(self, event: tk.Event) -> None:
        self.dragging = True
        self.pan_moved = False
        self.last_drag = (event.x, event.y)

    def pan(self, event: tk.Event) -> None:
        dx = event.x - self.last_drag[0]
        dy = event.y - self.last_drag[1]
        if abs(dx) + abs(dy) > 2:
            self.pan_moved = True
        self.offset_x += dx
        self.offset_y += dy
        self.last_drag = (event.x, event.y)
        self.fit_mode = False
        self.redraw()

    def end_pan(self, _event: tk.Event) -> None:
        self.dragging = False

    def end_right_pan(self, _event: tk.Event) -> None:
        should_close = not self.pan_moved
        self.dragging = False
        self.pan_moved = False
        if should_close:
            self.close_polygon()

    def on_mouse_wheel(self, event: tk.Event) -> None:
        factor = 1.1 if event.delta > 0 else 1 / 1.1
        self.zoom_at(event.x, event.y, factor)

    def zoom_at(self, cx: float, cy: float, factor: float) -> None:
        if self.current_image is None:
            return
        ix, iy = self.canvas_to_image(cx, cy)
        self.scale = min(max(self.scale * factor, 0.05), 20.0)
        self.offset_x = cx - ix * self.scale
        self.offset_y = cy - iy * self.scale
        self.fit_mode = False
        self.redraw()

    def close_polygon(self) -> None:
        rec = self.record
        if rec is None:
            return
        if len(self.current_points) < 3:
            self.update_status("Need at least 3 points to close a polygon.")
            return
        rec.polygons.append(Polygon(points=list(self.current_points), class_id=self.current_class_id()))
        self.current_points = []
        self.selected_polygon = len(rec.polygons) - 1
        self.save_current(silent=True)
        self.redraw()
        self.update_status("Polygon saved.")

    def undo_point(self) -> None:
        if self.current_points:
            self.current_points.pop()
            self.redraw()
            self.update_status()

    def delete_selected_polygon(self) -> None:
        rec = self.record
        if rec is None or self.selected_polygon is None:
            return
        if 0 <= self.selected_polygon < len(rec.polygons):
            del rec.polygons[self.selected_polygon]
            self.selected_polygon = None
            self.save_current(silent=True)
            self.redraw()
            self.update_status("Deleted selected polygon.")

    def find_polygon_at(self, x: float, y: float) -> Optional[int]:
        rec = self.record
        if rec is None:
            return None
        for idx in range(len(rec.polygons) - 1, -1, -1):
            if point_in_poly(x, y, rec.polygons[idx].points):
                return idx
        return None

    def on_file_select(self, _event: tk.Event) -> None:
        selection = self.file_list.curselection()
        if selection:
            self.load_image(selection[0])

    def next_image(self) -> None:
        if self.records:
            self.load_image(min(self.index + 1, len(self.records) - 1))

    def prev_image(self) -> None:
        if self.records:
            self.load_image(max(self.index - 1, 0))

    def update_status(self, extra: Optional[str] = None) -> None:
        rec = self.record
        if rec is None:
            return
        text = (
            f"{self.index + 1}/{len(self.records)} | {Path(rec.source).name} | "
            f"split={rec.split} | polygons={len(rec.polygons)} | current_points={len(self.current_points)}"
        )
        if extra:
            text += f" | {extra}"
        self.status_var.set(text)


def point_in_poly(x: float, y: float, points: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(points)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Needle YOLO segmentation annotation UI.")
    parser.add_argument("--images", type=Path, default=None, help="Source image folder.")
    parser.add_argument("--dataset", type=Path, default=None, help="YOLO dataset output folder.")
    parser.add_argument("--class", dest="class_name", type=str, default=DEFAULT_CLASS_NAME, help="Default class name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    try:
        NeedleAnnotator(root, args.images, args.dataset, class_name=args.class_name)
        root.mainloop()
    except Exception as exc:
        messagebox.showerror("Annotator error", str(exc))
        raise


if __name__ == "__main__":
    main()
