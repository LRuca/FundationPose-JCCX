# 消融实验建议

## YOLO 分割

先固定随机种子和 train/val split，每次只改一个变量。建议优先级：

| 组别 | 变量 | 建议取值 |
|---|---|---|
| 输入分辨率 | `imgsz` | 640, 960, 1280 |
| 模型规模 | `model` | yolov8n-seg, yolov8s-seg, yolov8m-seg |
| 数据组成 | `dataset` | inbox, lwt, combined |
| 精度模式 | `amp` | false, true |
| 增强 | mosaic / scale / flip | 默认、关闭 mosaic、减小 scale |
| 训练长度 | `epochs` | 50, 100, 200 |

统一记录 box/mask mAP50、mAP50-95、precision、recall、训练时间、峰值显存，并至少运行 3 个随机种子。基础命令：

```bash
python tools/run_yolo_ablation.py \
  --dataset combined \
  --models yolov8n-seg.pt yolov8s-seg.pt \
  --imgsz 640 960 \
  --seeds 0 1 2 \
  --epochs 100 --device 0
```

## FoundationPose

建议在固定 RGB-D 帧、固定 mask 和固定 mesh 上比较：

| 组别 | 变量 | 建议取值 |
|---|---|---|
| 注册迭代 | `est_refine_iter` | 1, 3, 5, 10 |
| 跟踪迭代 | `track_refine_iter` | 1, 2, 5 |
| mask 来源 | 输入 | GT/人工、YOLO、SAM |
| mesh 版本 | 网格 | axisymmetric, v2, v3 |
| 输入质量 | 深度处理 | 原始、孔洞过滤、边缘腐蚀 |

若没有真实 6D 标注，至少报告重复注册的位姿方差、连续帧平移/旋转抖动、失败率、单帧耗时和显存。不要仅凭可视化挑选结果。
