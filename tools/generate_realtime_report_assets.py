#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "report" / "assets" / "realtime"
REPORT_PATH = ROOT / "report" / "realtime_pose_work_report.md"


def font_path() -> Path:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("缺少中文字体，无法生成中文图表。")


FONT_PATH = font_path()
MPL_FONT = FontProperties(fname=str(FONT_PATH))


def set_labels(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontproperties=MPL_FONT, fontsize=13)
    ax.set_xlabel(xlabel, fontproperties=MPL_FONT)
    ax.set_ylabel(ylabel, fontproperties=MPL_FONT)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(MPL_FONT)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontproperties(MPL_FONT)
    ax.grid(alpha=0.22)


def read_results() -> dict[str, pd.DataFrame]:
    result = {}
    for path in sorted((ROOT / "runs").glob("*/*/results.csv")):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        result[path.parent.name] = df
    return result


def make_pipeline_diagram() -> Path:
    out = ASSET_DIR / "实时位姿解算流程图.png"
    w, h = 2200, 820
    im = Image.new("RGB", (w, h), "#f8fafc")
    draw = ImageDraw.Draw(im)
    title_font = ImageFont.truetype(str(FONT_PATH), 54)
    box_font = ImageFont.truetype(str(FONT_PATH), 32)
    small_font = ImageFont.truetype(str(FONT_PATH), 23)
    draw.text((70, 48), "针状目标实时 6D 位姿解算流程", fill="#0f172a", font=title_font)

    boxes = [
        ("RGB-D 相机", "color.png / depth.png / K", "#dbeafe"),
        ("YOLO 分割", "首帧或低频更新 mask", "#dcfce7"),
        ("FPose 注册", "第一帧得到初始 6D 位姿", "#ffedd5"),
        ("FPose 跟踪", "后续帧 track_one 更新", "#ede9fe"),
        ("输出结果", "位姿矩阵 / 可视化 / demo 视频", "#fce7f3"),
    ]
    x0, y0, bw, bh, gap = 70, 220, 335, 220, 70
    for i, (title, sub, color) in enumerate(boxes):
        x = x0 + i * (bw + gap)
        draw.rounded_rectangle([x, y0, x + bw, y0 + bh], radius=18, fill=color, outline="#334155", width=3)
        draw.text((x + 28, y0 + 48), title, fill="#0f172a", font=box_font)
        draw.text((x + 28, y0 + 116), sub, fill="#475569", font=small_font)
        if i < len(boxes) - 1:
            ax = x + bw + 8
            ay = y0 + bh // 2
            draw.line([ax, ay, ax + gap - 18, ay], fill="#334155", width=5)
            draw.polygon([(ax + gap - 18, ay - 14), (ax + gap + 5, ay), (ax + gap - 18, ay + 14)], fill="#334155")

    notes = [
        "原则 1：第一帧注册，后续帧跟踪，不要每帧重新初始化。",
        "原则 2：mask 与有效深度必须重合，错误要暴露，不要用假数据糊过去。",
        "原则 3：精度和速度一起记录，单看视频不算实验。",
    ]
    for i, text in enumerate(notes):
        draw.text((90, 560 + i * 52), text, fill="#1e293b", font=small_font)
    im.save(out)
    return out


def make_metric_diagram() -> Path:
    out = ASSET_DIR / "精度与效率指标总览.png"
    w, h = 1600, 860
    im = Image.new("RGB", (w, h), "#ffffff")
    draw = ImageDraw.Draw(im)
    title_font = ImageFont.truetype(str(FONT_PATH), 50)
    head_font = ImageFont.truetype(str(FONT_PATH), 36)
    text_font = ImageFont.truetype(str(FONT_PATH), 27)
    draw.text((70, 45), "实时位姿解算评估指标", fill="#111827", font=title_font)
    columns = [
        ("精度", ["YOLO mask mAP", "重投影误差", "平移误差 / 旋转误差", "连续帧位姿抖动", "跟踪失败率"], "#e0f2fe"),
        ("效率", ["YOLO 单帧耗时", "首帧注册耗时", "后续帧跟踪耗时", "端到端 FPS", "峰值显存"], "#fef3c7"),
        ("稳定性", ["mask/depth 有效点数", "丢帧次数", "位姿突跳次数", "长序列漂移", "重复实验方差"], "#dcfce7"),
    ]
    for i, (head, items, color) in enumerate(columns):
        x = 70 + i * 505
        draw.rounded_rectangle([x, 160, x + 430, 745], radius=20, fill=color, outline="#334155", width=3)
        draw.text((x + 35, 205), head, fill="#0f172a", font=head_font)
        for j, item in enumerate(items):
            y = 300 + j * 76
            draw.ellipse([x + 38, y + 8, x + 58, y + 28], fill="#0f766e")
            draw.text((x + 78, y), item, fill="#1f2937", font=text_font)
    im.save(out)
    return out


def make_yolo_comparison(results: dict[str, pd.DataFrame]) -> Path:
    out = ASSET_DIR / "YOLO历史结果对比.png"
    rows = []
    for name, df in results.items():
        last = df.iloc[-1]
        rows.append(
            {
                "实验": name.replace("yolov8n_seg_", ""),
                "分割损失": float(last["train/seg_loss"]),
                "mask mAP50": float(last["metrics/mAP50(M)"]),
                "mask mAP50-95": float(last["metrics/mAP50-95(M)"]),
                "训练耗时": float(last["time"]),
            }
        )
    data = pd.DataFrame(rows).sort_values("实验")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    x = np.arange(len(data))
    axes[0].bar(x - 0.18, data["分割损失"], width=0.36, color="#0f766e", label="训练分割损失")
    axes[0].bar(x + 0.18, data["mask mAP50-95"], width=0.36, color="#f97316", label="mask mAP50-95")
    axes[0].set_xticks(x, data["实验"])
    axes[0].legend()
    set_labels(axes[0], "已有 YOLO 实验结果", "实验组", "数值")

    axes[1].bar(data["实验"], data["训练耗时"] / 60.0, color="#1d4ed8")
    set_labels(axes[1], "训练耗时对比", "实验组", "分钟")

    fig.suptitle("历史训练结果：combined 组才有可用验证指标", fontproperties=MPL_FONT, fontsize=16)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_tradeoff_chart() -> Path:
    out = ASSET_DIR / "精度效率权衡示意图.png"
    labels = ["640/iter1", "640/iter2", "960/iter2", "960/iter5", "1280/iter5"]
    fps = np.array([30, 24, 16, 10, 6])
    score = np.array([0.64, 0.69, 0.75, 0.80, 0.82])
    size = np.array([120, 150, 220, 290, 360])

    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.scatter(fps, score, s=size, c=["#16a34a", "#0f766e", "#2563eb", "#f97316", "#dc2626"], alpha=0.86)
    for x, y, label in zip(fps, score, labels):
        ax.text(x + 0.4, y, label, fontproperties=MPL_FONT, fontsize=9)
    set_labels(ax, "精度与效率权衡示意", "实时速度 FPS", "综合精度分数")
    ax.text(
        6,
        0.61,
        "说明：这是实验设计示意图，不是假装已有结果。\n真实数值需要跑完消融后替换。",
        fontproperties=MPL_FONT,
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_ablation_matrix() -> Path:
    out = ASSET_DIR / "消融实验矩阵.png"
    rows = ["输入分辨率", "模型规模", "数据来源", "mask 后处理", "注册迭代", "跟踪迭代", "mesh 版本", "深度过滤"]
    cols = ["精度", "效率", "稳定性", "工作量展示"]
    values = np.array(
        [
            [3, 5, 3, 4],
            [4, 4, 3, 4],
            [5, 2, 4, 5],
            [3, 2, 5, 4],
            [4, 3, 4, 5],
            [3, 5, 4, 5],
            [5, 1, 4, 5],
            [3, 2, 5, 4],
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    im = ax.imshow(values, cmap="YlGnBu", vmin=1, vmax=5)
    ax.set_xticks(np.arange(len(cols)), cols, fontproperties=MPL_FONT)
    ax.set_yticks(np.arange(len(rows)), rows, fontproperties=MPL_FONT)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, str(values[i, j]), ha="center", va="center", color="#0f172a", fontsize=11)
    ax.set_title("建议优先做的消融实验矩阵", fontproperties=MPL_FONT, fontsize=15)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("影响程度", fontproperties=MPL_FONT)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def collect_status(results: dict[str, pd.DataFrame]) -> dict:
    weights = [
        ROOT / "yolov8n-seg.pt",
        ROOT / "runs/needle_lwt_seg/yolov8n_seg_lwt/weights/best.pt",
        ROOT / "runs/needle_inbox_seg/yolov8n_seg_inbox/weights/best.pt",
        ROOT / "runs/needle_inbox_seg/yolov8n_seg_combined/weights/best.pt",
    ]
    fp_weights = [
        ROOT / "third_party/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth",
        ROOT / "third_party/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth",
    ]
    return {
        "datasets": {
            "lwt_train": len(list((ROOT / "data/needle_lwt/needle_lwt_yolo_split/images/train").glob("*"))),
            "lwt_val": len(list((ROOT / "data/needle_lwt/needle_lwt_yolo_split/images/val").glob("*"))),
            "inbox_train": len(list((ROOT / "datasets/needle_inbox/images/train").glob("*"))),
            "inbox_val": len(list((ROOT / "datasets/needle_inbox/images/val").glob("*"))),
            "combined_train": len(list((ROOT / "datasets/needle_inbox_combined/images/train").glob("*"))),
            "combined_val": len(list((ROOT / "datasets/needle_inbox_combined/images/val").glob("*"))),
        },
        "weights_missing": [str(p.relative_to(ROOT)) for p in weights if not p.exists()],
        "foundationpose_weights_missing": [str(p.relative_to(ROOT)) for p in fp_weights if not p.exists()],
        "foundationpose_ready": (ROOT / "third_party/FoundationPose").exists(),
        "conda_env": {
            "yolo_name": "jxcx-yolo",
            "yolo_path": "/home/ly/miniconda3/envs/jxcx-yolo",
            "yolo_created": Path("/home/ly/miniconda3/envs/jxcx-yolo/bin/python").exists(),
            "foundationpose_name": "foundationpose",
            "foundationpose_path": "/home/ly/miniconda3/envs/foundationpose",
            "foundationpose_created": Path("/home/ly/miniconda3/envs/foundationpose/bin/python").exists(),
            "python": "3.9.25",
            "dependency_status": "环境已创建；torch/ultralytics 等依赖因 Python 包源下载速度约 15 KB/s 暂未装完。",
        },
        "history_runs": list(results.keys()),
    }


def write_report(charts: list[Path], status: dict) -> None:
    missing = "\n".join(f"- `{item}`" for item in status["weights_missing"]) or "- 无"
    fp_missing = "\n".join(f"- `{item}`" for item in status["foundationpose_weights_missing"]) or "- 无"
    weight_note = (
        "YOLO 和 FoundationPose 权重已经补齐。真实 demo 仍需要可用的 Python 推理依赖和实时 RGB-D 输入。"
        if not status["weights_missing"] and not status["foundationpose_weights_missing"]
        else "权重仍未完全补齐；缺失项会阻止真实 demo 推理。"
    )
    charts_md = "\n".join(f"- `{path.relative_to(ROOT)}`" for path in charts)
    text = f"""# 实时针状目标位姿解算阶段报告

生成日期：2026-06-15

## 1. 当前目标

把已有的离线 FoundationPose 复现扩展成实时 RGB-D 位姿解算链路：YOLO 提供目标 mask，FoundationPose 首帧注册，后续帧跟踪输出 6D 位姿，并生成可视化 demo 视频。

## 2. 当前仓库状态

- LWT 数据集：{status['datasets']['lwt_train']} 张训练图，{status['datasets']['lwt_val']} 张验证图。
- Inbox 数据集：{status['datasets']['inbox_train']} 张训练图，{status['datasets']['inbox_val']} 张验证图。
- Combined 数据集：{status['datasets']['combined_train']} 张训练图，{status['datasets']['combined_val']} 张验证图。
- 推荐 mesh：`model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl`。
- FoundationPose 目录是否存在：`{status['foundationpose_ready']}`。
- Conda 环境：`{status['conda_env']['yolo_name']}`，已创建：`{status['conda_env']['yolo_created']}`，Python：`{status['conda_env']['python']}`。
- Conda 环境：`{status['conda_env']['foundationpose_name']}`，已创建：`{status['conda_env']['foundationpose_created']}`，Python：`{status['conda_env']['python']}`。
- 依赖状态：{status['conda_env']['dependency_status']}

缺失权重：

{missing}

缺失 FoundationPose 权重：

{fp_missing}

{weight_note}

## 3. 已补充的工作

- 新增 `tools/evaluate_pose_tracking.py`：读取 FoundationPose 输出的 `ob_in_cam/*.txt`，计算平移抖动、旋转抖动、稳定性曲线。
- 新增 `tools/make_realtime_demo_video.py`：读取 `track_vis/*.png`，合成 `report/needle_realtime_demo.mp4`。
- 新增 `tools/generate_realtime_report_assets.py`：生成中文图表和本报告。

## 4. 已生成图表

{charts_md}

## 5. 实时 demo 执行命令

补齐权重后先校验：

```bash
unzip /path/to/JXCX_YOLO_WEIGHTS_FINAL.zip
python tools/validate_handoff.py
```

准备 FoundationPose：

```bash
bash scripts/ubuntu/setup_foundationpose.sh
```

运行 YOLO mask 桥接：

```bash
python scripts/yolo_mug_mask_bridge.py \\
  --image third_party/FoundationPose/live_orbbec/color.png \\
  --mask third_party/FoundationPose/live_orbbec/mask_yolo.png \\
  --model runs/needle_inbox_seg/yolov8n_seg_combined/weights/best.pt \\
  --class_name needle_inbox \\
  --device 0 \\
  --loop
```

运行 FoundationPose 跟踪：

```bash
cd third_party/FoundationPose
python run_orbbec_mug_live.py \\
  --live_dir live_orbbec \\
  --mesh_file ../../model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl \\
  --mask_file live_orbbec/mask_yolo.png \\
  --debug_dir debug_needle_live \\
  --max_frames 300
```

合成 demo 视频：

```bash
python tools/make_realtime_demo_video.py \\
  --frames third_party/FoundationPose/debug_needle_live/track_vis \\
  --out report/needle_realtime_demo.mp4 \\
  --fps 30
```

评估位姿稳定性：

```bash
python tools/evaluate_pose_tracking.py \\
  --pose-dir third_party/FoundationPose/debug_needle_live/ob_in_cam \\
  --fps 30 \\
  --out-dir report/realtime_eval
```

## 6. 精度与效率评估设计

精度不要只看视频。至少记录 YOLO mask mAP、重投影误差、连续帧平移抖动、连续帧旋转抖动、跟踪失败率。

效率至少记录 YOLO 单帧耗时、首帧注册耗时、后续帧跟踪耗时、端到端 FPS、峰值显存。

没有真实 6D 标注时，先用稳定性指标和重投影误差。等有标定工装或人工关键帧标注后，再报告绝对平移/旋转误差。

## 7. 消融实验安排

优先跑这些变量，每次只改一个：

| 实验 | 取值 | 主要观察 |
|---|---|---|
| 输入分辨率 | 640 / 960 / 1280 | mask 精度与 FPS |
| 模型规模 | yolov8n / yolov8s / yolov8m | 精度与显存 |
| 数据来源 | lwt / inbox / combined | 泛化能力 |
| mask 后处理 | 无 / 开闭运算 / 边缘腐蚀 | 跟踪稳定性 |
| 注册迭代 | 1 / 3 / 5 / 10 | 首帧精度与耗时 |
| 跟踪迭代 | 1 / 2 / 5 | 实时 FPS 与抖动 |
| mesh 版本 | axisymmetric / v2 / v3 | 细长针几何约束 |
| 深度过滤 | 原始 / 去噪 / 有效点阈值 | 失败率 |

## 8. 风险

最大风险不是算法理论，是工程链路：权重缺失、深度图无效、mask 与 depth 不重合、相机写文件竞争。代码要直接暴露这些错误，不要写一堆 fallback 假装成功。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    results = read_results()
    charts = [
        make_pipeline_diagram(),
        make_metric_diagram(),
        make_yolo_comparison(results),
        make_tradeoff_chart(),
        make_ablation_matrix(),
    ]
    status = collect_status(results)
    (ASSET_DIR / "当前状态.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(charts, status)
    print(f"已生成报告: {REPORT_PATH}")
    for chart in charts:
        print(f"已生成图表: {chart}")


if __name__ == "__main__":
    main()
