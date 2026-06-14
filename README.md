# JXCX Needle Pose Project

本仓库用于针状目标的 YOLO 实例分割、三维模型处理，以及 YOLO mask 到 FoundationPose 6D 位姿估计的实验。

## Ubuntu 快速开始

```bash
git lfs install
git clone <repo-url> JXCX
cd JXCX
git lfs pull

# 将单独收到的 JXCX_YOLO_WEIGHTS_FINAL.zip 解压到仓库根目录
# unzip JXCX_YOLO_WEIGHTS_FINAL.zip

bash scripts/ubuntu/setup_yolo.sh
conda activate jxcx-yolo
python tools/validate_handoff.py
python tools/train_yolo_seg.py --dataset combined --epochs 1 --device 0
```

`.pt` 权重不上传 GitHub。权重包 `JXCX_YOLO_WEIGHTS_FINAL.zip` 的内容和解压路径见 [docs/WEIGHTS_PACKAGE.md](docs/WEIGHTS_PACKAGE.md)。

FoundationPose 环境和本项目对上游的修改由下面的脚本重建：

```bash
bash scripts/ubuntu/setup_foundationpose.sh
conda activate foundationpose
python tools/validate_handoff.py --check-foundationpose
```

完整交接说明见 [docs/HANDOFF_UBUNTU.md](docs/HANDOFF_UBUNTU.md)，消融实验建议见 [docs/ABLATION_EXPERIMENTS.md](docs/ABLATION_EXPERIMENTS.md)。

## 关键资产

- `data/needle_lwt/needle_lwt_yolo_split`: 95 张训练图、24 张验证图，类别 `needle`。
- `datasets/needle_inbox`: 45 张训练图、3 张验证图，类别 `needle_inbox`。
- `datasets/needle_inbox_combined`: 145 张训练图、24 张验证图，类别 `needle_inbox`。
- `runs/**/weights/best.pt`: 已训练 YOLO 分割权重。
- `model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl`: 当前推荐 FoundationPose 网格。
- `foundationpose_overlay`: 基于 NVlabs/FoundationPose `e3d597b` 的项目修改。官方 FoundationPose 权重按交接文档下载，不在本仓库重复托管。

Windows 下 Orbbec 采集 GUI 和 SDK 文件保留在原工作目录中，但不进入 Git；Ubuntu 消融实验不依赖 WSL 或 Windows SDK。
