# Needle YOLO Segmentation Annotator

This is a portable local annotation UI for training a YOLO segmentation model for the puncture needle.

## Run

From `C:\Users\lenovo\Desktop\JXCX`:

```powershell
python -m pip install pillow
python tools\needle_yolo_annotator.py --images data\needle_raw --dataset datasets\needle_seg
```

Or double-click:

```text
start_needle_annotator.cmd
```

The default launcher opens `FoundationPose\live_orbbec` as the image source and writes YOLO data to `datasets\needle_seg`.

## Controls

- Left click: add polygon point or select an existing polygon
- Enter / right click: close the current polygon
- Label dropdown: choose the class for new polygons
- Add Label: add a new segmentation class
- Select an existing polygon, then change the Label dropdown to relabel it
- Mouse wheel: zoom
- Middle/right drag: pan
- `Z`: undo current point
- `Delete`: delete selected polygon
- `N` / `P`: next / previous image
- `Ctrl+S`: save

## Output

The tool writes a YOLO segmentation dataset:

```text
datasets\needle_seg\
  images\train
  images\val
  labels\train
  labels\val
  needle.yaml
  needle_annotator_project.json
```

Train with:

```bash
cd /mnt/c/Users/lenovo/Desktop/JXCX
/opt/conda/envs/yolo_mugseg/bin/yolo segment train \
  model=yolov8n-seg.pt \
  data=datasets/needle_seg/needle.yaml \
  imgsz=960 \
  epochs=100 \
  batch=4 \
  device=0 \
  project=runs/needle_seg \
  name=yolov8n_needle
```

After training, use the best checkpoint for automatic masks:

```bash
/opt/conda/envs/yolo_mugseg/bin/yolo segment predict \
  model=runs/needle_seg/yolov8n_needle/weights/best.pt \
  source=FoundationPose/live_orbbec/color.png \
  imgsz=960 \
  conf=0.25
```

## Build EXE

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build_needle_annotator.ps1
```

The portable executable is written to:

```text
dist\NeedleAnnotator.exe
```
