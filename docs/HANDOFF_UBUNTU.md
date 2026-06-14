# Ubuntu 交接说明

## 1. 已验证基线

原机器在 WSL2 Ubuntu 中实际使用如下组合：

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
- NVIDIA driver: 591.74
- Python: 3.9
- PyTorch: 2.0.0+cu118
- torchvision: 0.15.1+cu118
- torchaudio: 2.0.1+cu118
- Ultralytics: 8.0.120
- NumPy: 1.26.4
- OpenCV: 4.9.0.80
- trimesh: 4.2.2
- FoundationPose upstream commit: `e3d597b8c6b851d053094ebd6fa240191c5238f8`

WSL 下没有另一份项目文件。历史运行直接访问 Windows 目录 `/mnt/c/Users/lenovo/Desktop/JXCX`，因此在纯 Ubuntu 上只需克隆本仓库并重建环境。

## 2. 克隆和数据

本仓库使用 Git LFS 保存 `.pt` 权重和数据集图片。克隆前安装 Git LFS：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs
git lfs install
git clone <repo-url> JXCX
cd JXCX
git lfs pull
unzip /path/to/JXCX_YOLO_WEIGHTS_FINAL.zip
python tools/validate_handoff.py
```

`.pt` 文件不上传 GitHub，由项目交接人单独发送 `JXCX_YOLO_WEIGHTS_FINAL.zip`。解压后保持原目录结构，详见 `docs/WEIGHTS_PACKAGE.md`。

若 `validate_handoff.py` 报文件内容是 LFS pointer，说明尚未执行 `git lfs pull`。

## 3. YOLO 训练

```bash
bash scripts/ubuntu/setup_yolo.sh
conda activate jxcx-yolo

# 先做 1 epoch 冒烟测试
python tools/train_yolo_seg.py --dataset combined --epochs 1 --batch 2 --device 0 --name smoke

# 复现 combined 基线
python tools/train_yolo_seg.py --dataset combined --epochs 100 --imgsz 960 --batch 4 \
  --device 0 --no-amp --no-val --name yolov8n_seg_combined_repro

# 复现 needle_lwt 基线
python tools/train_yolo_seg.py --dataset lwt --epochs 100 --imgsz 960 --batch 4 \
  --device 0 --no-amp --no-val --name yolov8n_seg_lwt_repro

# 仅使用 inbox 小数据集
python tools/train_yolo_seg.py --dataset inbox --epochs 100 --imgsz 960 --batch 4 \
  --device 0 --no-amp --name yolov8n_seg_inbox_repro
```

脚本运行时生成绝对路径 YAML，因此不依赖原 Windows 路径。输出默认写入 `runs/ablation/`。

## 4. FoundationPose

推荐优先使用官方 Docker；若需要复现原 Conda 组合，执行：

```bash
bash scripts/ubuntu/setup_foundationpose.sh
conda activate foundationpose
```

脚本会克隆固定上游提交到 `third_party/FoundationPose`，再覆盖 `foundationpose_overlay/` 中的本地修复和 Orbbec/文件夹入口。

FoundationPose scorer/refiner 是 NVlabs 官方权重，不在本仓库重复托管。请按照上游 README 的 **Data prepare** 段落，从官方链接下载全部网络权重：

- 上游说明：<https://github.com/NVlabs/FoundationPose#data-prepare>
- README 中给出的官方权重目录：<https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i>

本项目使用的两个文件应放到：

```text
third_party/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth
third_party/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth
```

其中 `2023-10-28-18-33-37` 是 refiner，`2024-01-11-20-02-45` 是 scorer。

项目主网格为：

```text
model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl
```

离线帧输入至少包含 `color.png`、16-bit 毫米单位 `depth.png`、`cam_K.txt` 和二值 `mask_yolo.png`。运行示例：

```bash
cd third_party/FoundationPose
python run_orbbec_mug_live.py \
  --live_dir ../../sample_orbbec \
  --mesh_file ../../model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl \
  --mask_file ../../sample_orbbec/mask_yolo.png \
  --max_frames 1
```

## 5. 已知注意事项

- 原实验将 `amp=false`、`val=false`，原因是当时的 CUDA/NMS 兼容问题；消融时应单独测试 AMP 和验证流程。
- FoundationPose 对 mesh、K、pose 的 dtype 敏感。本仓库 overlay 强制关键张量为 `float32`，不要删除此修复。
- 深度必须为毫米单位的 16-bit PNG，脚本读取后除以 1000 转米。
- 初始 mask 必须与有效深度有重合；入口会检查有效点数量。
- Orbbec 实时采集属于 Windows 历史链路，Ubuntu 同学做离线消融不需要复制 WSL、PowerShell、Windows SDK 或 USB 转发配置。

## 6. Git 约定

- 不提交 `runs/` 下新的批次图、预测图、日志、缓存和 `last.pt`。
- 需要共享的新最佳权重可用 `git lfs track` 后提交。
- 每组实验应保存命令、随机种子、数据版本、`args.yaml` 和最终指标 CSV。
- 不直接修改 `third_party/FoundationPose` 后丢失改动；需要保留的修改同步到 `foundationpose_overlay/`。
