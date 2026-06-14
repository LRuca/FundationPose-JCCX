# YOLO 权重包

`JXCX_YOLO_WEIGHTS_FINAL.zip` 不上传 GitHub，由项目交接人单独发送。

在 Ubuntu 上将它解压到仓库根目录：

```bash
cd JXCX
unzip /path/to/JXCX_YOLO_WEIGHTS_FINAL.zip
python tools/validate_handoff.py
```

压缩包包含：

| 文件 | 用途 |
|---|---|
| `yolov8n-seg.pt` | Ultralytics YOLOv8n-seg 预训练初始化权重 |
| `runs/needle_lwt_seg/yolov8n_seg_lwt/weights/best.pt` | LWT 数据集训练结果 |
| `runs/needle_inbox_seg/yolov8n_seg_inbox/weights/best.pt` | Inbox 数据集训练结果 |
| `runs/needle_inbox_seg/yolov8n_seg_combined/weights/best.pt` | Combined 数据集训练结果，当前推荐基线 |

训练新模型时，即使没有 `yolov8n-seg.pt`，Ultralytics 也可联网自动下载官方初始化权重。已有实验权重必须从本压缩包获取。
