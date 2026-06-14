from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report"
ASSET_DIR = OUT_DIR / "assets"
PDF_PATH = OUT_DIR / "needle_pose_experiment_report.pdf"
MD_PATH = OUT_DIR / "needle_pose_experiment_report.md"


def find_font() -> Path:
    candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No CJK font found in C:\\Windows\\Fonts")


FONT_PATH = find_font()
MPL_CJK_FONT = FontProperties(fname=str(FONT_PATH))


def rel(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def count_files(path: Path, pattern: str = "*") -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in exts)


def read_csv_tail(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[-1]


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def make_yolo_curves() -> dict:
    csv_path = ROOT / "runs/needle_lwt_seg/yolov8n_seg_lwt/results.csv"
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=180)
    x = df["epoch"]
    plots = [
        (axes[0, 0], ["train/box_loss", "train/seg_loss"], "定位与分割损失"),
        (axes[0, 1], ["train/cls_loss", "train/dfl_loss"], "分类与 DFL 损失"),
        (axes[1, 0], ["lr/pg0", "lr/pg1", "lr/pg2"], "学习率"),
        (axes[1, 1], ["metrics/mAP50(M)", "metrics/mAP50-95(M)"], "验证指标记录"),
    ]
    for ax, cols, title in plots:
        for col in cols:
            if col in df:
                ax.plot(x, df[col], label=col.replace("train/", "").replace("metrics/", ""))
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    for ax in axes.reshape(-1):
        ax.title.set_fontproperties(MPL_CJK_FONT)
        ax.xaxis.label.set_fontproperties(MPL_CJK_FONT)
        ax.yaxis.label.set_fontproperties(MPL_CJK_FONT)
    fig.suptitle("YOLOv8n-seg needle_lwt 训练曲线", fontproperties=MPL_CJK_FONT)
    fig.tight_layout()
    out = ASSET_DIR / "yolo_training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    first = df.iloc[0]
    last = df.iloc[-1]
    best_seg_loss_idx = int(df["train/seg_loss"].idxmin())
    return {
        "path": out,
        "epochs": int(df["epoch"].max()),
        "first_box": float(first["train/box_loss"]),
        "last_box": float(last["train/box_loss"]),
        "first_seg": float(first["train/seg_loss"]),
        "last_seg": float(last["train/seg_loss"]),
        "first_cls": float(first["train/cls_loss"]),
        "last_cls": float(last["train/cls_loss"]),
        "best_seg_epoch": int(df.loc[best_seg_loss_idx, "epoch"]),
        "best_seg_loss": float(df.loc[best_seg_loss_idx, "train/seg_loss"]),
        "final_lr": float(last["lr/pg0"]),
    }


def make_evidence_contact_sheet() -> Path:
    items = [
        ("扫描清理后", ROOT / "model/fixed_unnamed_object_3/object_without_flat_sheet_main_preview.png"),
        ("轴对称重建", ROOT / "model/fixed_unnamed_object_3/needle_axisymmetric_reconstruction_preview.png"),
        ("结构尾部 v1", ROOT / "model/fixed_unnamed_object_3/needle_structured_tail_material_render.png"),
        ("结构尾部 v2", ROOT / "model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v2_material_render.png"),
        ("最终 v3", ROOT / "model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3_material_render_fresh.png"),
        ("尾部近景", ROOT / "model/fixed_unnamed_object_3/needle_structured_tail_v3_tail_closeup_rerender.png"),
    ]
    thumb_w, thumb_h = 480, 270
    pad = 28
    title_h = 48
    cols = 2
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * pad, rows * (thumb_h + title_h) + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONT_PATH), 28)
    small = ImageFont.truetype(str(FONT_PATH), 22)
    for idx, (label, path) in enumerate(items):
        row, col = divmod(idx, cols)
        x = pad + col * (thumb_w + pad)
        y = pad + row * (thumb_h + title_h + pad)
        draw.text((x, y), label, fill=(35, 35, 35), font=font)
        if path.exists():
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                px = x + (thumb_w - im.width) // 2
                py = y + title_h + (thumb_h - im.height) // 2
                sheet.paste(im, (px, py))
        else:
            draw.rectangle([x, y + title_h, x + thumb_w, y + title_h + thumb_h], outline=(180, 180, 180))
            draw.text((x + 20, y + title_h + 100), "missing", fill=(120, 120, 120), font=small)
    out = ASSET_DIR / "model_iteration_contact_sheet.png"
    sheet.save(out, quality=92)
    return out


def extract_log_evidence() -> dict:
    log_dir = ROOT / "logs/mug_pipeline"
    texts = []
    for path in log_dir.glob("*.err"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        texts.append((path, text))

    patterns = {
        "double_dtype": r"expected scalar type Float but found Double",
        "frame_timeout": r"Timed out waiting for new frame",
        "zero_depth": r"Depth image has no valid nonzero pixels",
        "mask_depth_overlap": r"Mask/depth overlap has too few valid depth pixels",
        "file_race": r"MoveFileInfoItemIOError|RemoveItemUnauthorizedAccessError|PathNotFound|Invalid argument",
    }
    counts = {k: 0 for k in patterns}
    examples: dict[str, str] = {}
    for path, text in texts:
        for key, pattern in patterns.items():
            if re.search(pattern, text, flags=re.I):
                counts[key] += 1
                examples.setdefault(key, f"{path.name}")
    return {"counts": counts, "examples": examples}


def collect_summary(yolo: dict, contact_sheet: Path) -> dict:
    frame = read_json(ROOT / "FoundationPose/live_orbbec/frame.json")
    mask = read_json(ROOT / "FoundationPose/live_orbbec/mask_yolo.json")
    header, tail = read_csv_tail(ROOT / "runs/needle_lwt_seg/yolov8n_seg_lwt/results.csv")
    mesh_dir = ROOT / "model/fixed_unnamed_object_3"
    pose_count = count_files(ROOT / "FoundationPose/debug_orbbec_mug/ob_in_cam", "*.txt")
    vis_count = count_files(ROOT / "FoundationPose/debug_orbbec_mug/track_vis", "*.png")

    return {
        "generated_at": "2026-05-04",
        "font": str(FONT_PATH),
        "dataset": {
            "train_images": count_images(ROOT / "data/needle_lwt/needle_lwt_yolo_split/images/train"),
            "val_images": count_images(ROOT / "data/needle_lwt/needle_lwt_yolo_split/images/val"),
            "train_labels": count_files(ROOT / "data/needle_lwt/needle_lwt_yolo_split/labels/train", "*.txt"),
            "val_labels": count_files(ROOT / "data/needle_lwt/needle_lwt_yolo_split/labels/val", "*.txt"),
        },
        "yolo": {**yolo, "path": rel(yolo["path"]), "results_tail": dict(zip(header, tail))},
        "live_frame": frame,
        "mask": mask,
        "pose_outputs": {"pose_txt": pose_count, "track_vis_png": vis_count},
        "model_assets": {
            "final_stl": rel(mesh_dir / "needle_structured_tail_reconstruction_v3.stl"),
            "final_glb": rel(mesh_dir / "needle_structured_tail_reconstruction_v3.glb"),
            "final_obj": rel(mesh_dir / "needle_structured_tail_reconstruction_v3.obj"),
            "contact_sheet": rel(contact_sheet),
        },
        "log_evidence": extract_log_evidence(),
    }


def para(text: str, style: ParagraphStyle):
    return Paragraph(text.replace("\n", "<br/>"), style)


def add_image(story, path: Path, width_mm: float, caption: str, styles):
    if not path.exists():
        story.append(para(f"缺失图片：{rel(path)}", styles["Body"]))
        return
    w, h = image_size(path)
    width = width_mm * mm
    height = width * h / w
    story.append(RLImage(str(path), width=width, height=height))
    story.append(para(caption, styles["Caption"]))
    story.append(Spacer(1, 5 * mm))


def make_styles() -> dict:
    pdfmetrics.registerFont(TTFont("CJK", str(FONT_PATH)))
    pdfmetrics.registerFont(TTFont("CJKBold", str(FONT_PATH)))
    styles = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontName="CJKBold",
            fontSize=22,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=16,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontName="CJKBold",
            fontSize=15,
            leading=21,
            textColor=colors.HexColor("#263238"),
            spaceBefore=10,
            spaceAfter=7,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontName="CJKBold",
            fontSize=12.5,
            leading=18,
            textColor=colors.HexColor("#37474F"),
            spaceBefore=7,
            spaceAfter=5,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName="CJK",
            fontSize=10.5,
            leading=17,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=styles["BodyText"],
            fontName="CJK",
            fontSize=8.6,
            leading=12,
            textColor=colors.HexColor("#546E7A"),
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "Small": ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontName="CJK",
            fontSize=8.7,
            leading=12,
            textColor=colors.HexColor("#455A64"),
        ),
    }


def table(data, col_widths):
    cell_style = ParagraphStyle(
        "TableCell",
        fontName="CJK",
        fontSize=8.4,
        leading=10.8,
        wordWrap="CJK",
        splitLongWords=True,
    )
    header_style = ParagraphStyle(
        "TableHeader",
        parent=cell_style,
        fontName="CJKBold",
        textColor=colors.HexColor("#263238"),
    )

    wrapped = []
    for r, row in enumerate(data):
        wrapped_row = []
        for cell in row:
            if isinstance(cell, str):
                style = header_style if r == 0 else cell_style
                wrapped_row.append(Paragraph(cell, style))
            else:
                wrapped_row.append(cell)
        wrapped.append(wrapped_row)

    t = Table(wrapped, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "CJK"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4),
                ("LEADING", (0, 0), (-1, -1), 10.8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ECEFF1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#263238")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B0BEC5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def write_markdown(summary: dict) -> None:
    y = summary["yolo"]
    d = summary["dataset"]
    log = summary["log_evidence"]["counts"]
    text = f"""# 细长针目标位姿与模型修复实验报告

生成日期：2026-05-04

## 关键结论

- FoundationPose 复现最终可运行，成功输出 {summary['pose_outputs']['pose_txt']} 个位姿矩阵和 {summary['pose_outputs']['track_vis_png']} 张跟踪可视化图。
- 原始扫描模型经过扫描清理、去平面背景、轴对称重建、结构化尾部 v1/v2/v3 三轮修正，最终模型为 `{summary['model_assets']['final_stl']}`。
- YOLOv8n-seg 使用 {d['train_images']} 张训练图和 {d['val_images']} 张验证图训练 100 epoch；训练中关闭验证，因此 metrics/mAP 曲线为 0，主要依据训练损失下降判断收敛。
- YOLO 分割损失从 {y['first_seg']:.4f} 降至 {y['last_seg']:.4f}，最佳训练分割损失 {y['best_seg_loss']:.4f} 出现在 epoch {y['best_seg_epoch']}。
- 主要困难集中在 WSL/Conda/CUDA 依赖、本地 Orbbec 帧文件竞争、深度图全零、mask 与有效深度不重合、FoundationPose mesh dtype 不一致。

## 证据文件

- YOLO 曲线：`{y['path']}`
- 模型迭代图：`{summary['model_assets']['contact_sheet']}`
- 最终 PDF：`{rel(PDF_PATH)}`

## 错误计数摘要

- Float/Double 类型问题：{log['double_dtype']}
- 等待新帧超时：{log['frame_timeout']}
- 深度图全零：{log['zero_depth']}
- mask/depth 有效点不足：{log['mask_depth_overlap']}
- 文件复制/替换竞争：{log['file_race']}
"""
    MD_PATH.write_text(text, encoding="utf-8")


def build_pdf(summary: dict) -> None:
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="细长针目标位姿与模型修复实验报告",
        author="Codex",
    )
    story = []
    y = summary["yolo"]
    d = summary["dataset"]
    frame = summary["live_frame"]
    mask = summary["mask"]
    logs = summary["log_evidence"]["counts"]

    story.append(para("细长针目标位姿与模型修复实验报告", styles["Title"]))
    story.append(
        para(
            "基于本工作空间内的脚本、日志、训练结果、模型迭代产物和可视化图片整理。报告覆盖纯算法复现、"
            "从原始扫描模型到最终修正模型的优化路径，以及 YOLO 训练曲线和实验证据。",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        table(
            [
                ["项目", "结果"],
                ["最终模型", "needle_structured_tail_reconstruction_v3.stl / glb / obj"],
                ["实时跟踪输出", f"{summary['pose_outputs']['pose_txt']} 个 pose txt，{summary['pose_outputs']['track_vis_png']} 张 track_vis"],
                ["YOLO 数据集", f"train {d['train_images']} 图/{d['train_labels']} 标注；val {d['val_images']} 图/{d['val_labels']} 标注"],
                ["最后一帧深度", f"valid_pixels={frame['depth_valid_pixels']}，max={frame['depth_max_mm']} mm"],
                ["最后一次 YOLO mask", f"class={mask['class_name']}，conf={mask['confidence']:.3f}，area={mask['area']}"],
            ],
            [40 * mm, 128 * mm],
        )
    )

    story.append(para("一、纯算法复现过程", styles["H1"]))
    story.append(
        para(
            "复现链路采用 Windows 负责 Orbbec 采集和 GUI，WSL/Ubuntu 负责 FoundationPose 的 CUDA 推理。核心路径是："
            "Windows 端 Orbbec SDK 的 SaveToDisk 周期性导出 color/depth，采集脚本把完整 PNG 原子发布到 "
            "FoundationPose/live_orbbec；YOLO 或 SAM 写出 mask_yolo.png；WSL 中 FoundationPose 读取 color、depth、K、mask，"
            "完成首帧 register 和后续 track_one。",
            styles["Body"],
        )
    )
    story.append(
        para(
            "环境复现的主要困难不是算法本身，而是依赖 ABI 与跨系统 I/O。工作区保留了 torch/torchvision/torchaudio cu118 "
            "Linux wheel 到 wheelhouse，并通过 WSL 路径 /mnt/c/Users/lenovo/Desktop/JXCX 复用 Windows 工作区。"
            "FoundationPose 依赖 mycpp、nvdiffrast、pytorch3d、CUDA rasterizer，实际做法是固定 Python 3.9 + torch 2.0.0/cu118，"
            "编译 mycpp.cpython-39-x86_64-linux-gnu.so，并用 check_foundationpose_env.py 检查 torch、open3d、trimesh、pyrender、mycpp 等模块。",
            styles["Body"],
        )
    )
    story.append(
        table(
            [
                ["困难", "日志/现象", "解决"],
                ["CUDA/Conda 依赖体积大且下载不稳定", "wheelhouse/download.log 记录 2.11GB torch cu118 wheel 下载，曾出现 SIGHUP", "改用国内镜像、断点续传、把 wheel 落盘到 wheelhouse 后在 WSL 环境安装"],
                ["FoundationPose mesh dtype 不一致", "RuntimeError: expected scalar type Float but found Double", "在 run_orbbec_mug_live.py 中加入 force_mesh_float32，将 vertices/normals/K 显式转 float32"],
                ["采集文件被并发读写", "MoveFileInfoItemIOError、RemoveItemUnauthorizedAccessError、Invalid argument", "采集脚本增加 PNG 完整性检测、tmp 文件、Copy-Atomic/Write-AtomicText 和稳定文件等待"],
                ["深度图全零或 mask 与深度不重合", "Depth image has no valid nonzero pixels；Mask/depth overlap has too few valid depth pixels", "增加 depth_valid 诊断，GUI 中先验证 depth.png，再要求重画 bbox/mask 或调整目标距离"],
                ["实时帧等待超时", "Timed out waiting for new frame", "发布 frame.json 并用 frame_index 判断新帧，Orbbec 循环失败时重启 SaveToDisk"],
            ],
            [34 * mm, 58 * mm, 76 * mm],
        )
    )

    story.append(para("二、从原始扫描模型到最终修正模型", styles["H1"]))
    story.append(
        para(
            "原始模型来自 model/未命名对象 3_export 和 raw ObjectMaskOn USDZ。扫描结果包含纹理、点云/网格、深度帧和大量背景平面；"
            "直接用于 FoundationPose 会带来尺度、外形和背景几何干扰。修复目标不是简单平滑，而是得到一个稳定、轻量、与实物外观一致的针形 CAD-like 网格。",
            styles["Body"],
        )
    )
    story.append(
        table(
            [
                ["阶段", "输入/输出", "处理逻辑", "困难与解决"],
                ["扫描清理", "raw USDZ -> textured_cleaned / cleaned_smoothed", "用 USD API 读取三角面、UV 和纹理；trimesh 连通域过滤小碎片；拉普拉斯式平滑", "扫描有噪声、碎片和贴图；保留大连通域，导出 OBJ/PLY/STL 兼容后续工具"],
                ["去背景平面", "cleaned_smoothed -> object_without_flat_sheet_main", "根据 face normal 过滤接近平面的背景，拆分组件并保留主组件", "扫描里背景板占比高；用法向阈值和最大组件过滤，得到细长核心"],
                ["轴对称重建", "object_without_flat_sheet_main -> needle_axisymmetric", "PCA/协方差估计主轴，按径向百分位剔除离群点，生成细长旋转体", "仅能表达针杆，尾部结构丢失；作为尺度和主轴估计基线"],
                ["结构尾部 v1", "参考 RGB + 主轴 -> structured_tail", "按轴向 profile 生成长针杆、套筒、环和后端帽，添加顶点颜色", "尾端只是圆柱/环组合，结构不像参考图"],
                ["结构尾部 v2", "v1 + 新参考 -> structured_tail_v2", "加入六边形套筒、球形过渡、后端盘状帽和沟槽", "尾端过大且更像圆形按钮，继续根据参考裁剪尺寸"],
                ["最终 v3", "v2 -> structured_tail_v3", "压缩尾部长度，使用六面主块、短暗腰、窄环、圆角矩形端部块和凹槽", "在视觉相似度与 mesh 稳定性之间折中，不做布尔切割，保证可导出 STL/GLB/OBJ"],
            ],
            [25 * mm, 37 * mm, 52 * mm, 54 * mm],
        )
    )
    add_image(story, Path(summary["model_assets"]["contact_sheet"]), 168, "图 1：模型从扫描清理、轴对称基线到结构化尾部 v3 的迭代证据。", styles)
    add_image(story, ROOT / "model/fixed_unnamed_object_3/reference_image_implement_sheet.png", 168, "图 2：用于比对尾部结构的 RGB 参考图集合。", styles)
    story.append(PageBreak())

    story.append(para("三、实时识别与位姿结果", styles["H1"]))
    story.append(
        para(
            "最终链路使用训练后的 YOLO needle_lwt checkpoint 作为 mask 入口。最后一次记录中，YOLO 在 640x360 图像上识别到 1 个 needle 候选，"
            f"置信度 {mask['confidence']:.3f}，mask 面积 {mask['area']} 像素；对应深度帧有 {frame['depth_valid_pixels']} 个有效深度像素，"
            f"最大深度 {frame['depth_max_mm']} mm。FoundationPose debug_orbbec_mug 中已落盘 pose 矩阵和 track_vis 可视化，说明 register/track 流程已经跑通。",
            styles["Body"],
        )
    )
    add_image(story, ROOT / "FoundationPose/live_orbbec/color.png", 126, "图 3：Orbbec 实时 color.png。", styles)
    add_image(story, ROOT / "FoundationPose/live_orbbec/mask_sam_preview.png", 126, "图 4：首帧 SAM bbox mask 预览，用于人工初始化/纠偏。", styles)
    add_image(story, ROOT / "FoundationPose/debug_orbbec_mug/track_vis/000057.png", 126, "图 5：FoundationPose 最终跟踪可视化样例。", styles)

    story.append(para("四、YOLO 训练曲线", styles["H1"]))
    story.append(
        para(
            f"训练配置来自 runs/needle_lwt_seg/yolov8n_seg_lwt/args.yaml：YOLOv8n-seg，imgsz=960，batch=4，epochs=100，device=0，amp=false，val=false。"
            "由于训练时关闭 val，results.csv 中 precision/recall/mAP 和 val loss 均为 0；因此曲线重点展示 train/box_loss、train/seg_loss、"
            "train/cls_loss、train/dfl_loss 与学习率。",
            styles["Body"],
        )
    )
    story.append(
        table(
            [
                ["指标", "起始", "最终", "变化"],
                ["box_loss", f"{y['first_box']:.4f}", f"{y['last_box']:.4f}", f"{(y['last_box'] - y['first_box']):.4f}"],
                ["seg_loss", f"{y['first_seg']:.4f}", f"{y['last_seg']:.4f}", f"{(y['last_seg'] - y['first_seg']):.4f}"],
                ["cls_loss", f"{y['first_cls']:.4f}", f"{y['last_cls']:.4f}", f"{(y['last_cls'] - y['first_cls']):.4f}"],
                ["best seg_loss", "-", f"{y['best_seg_loss']:.4f}", f"epoch {y['best_seg_epoch']}"],
                ["final lr", "-", f"{y['final_lr']:.6f}", "线性衰减至低学习率"],
            ],
            [42 * mm, 38 * mm, 38 * mm, 50 * mm],
        )
    )
    add_image(story, Path(y["path"]), 168, "图 6：YOLO 训练曲线。验证曲线为 0 是因为本轮训练配置 val=false，不代表验证性能为 0。", styles)
    add_image(story, ROOT / "runs/needle_lwt_seg/yolov8n_seg_lwt/train_batch2160.jpg", 126, "图 7：训练后期 batch 可视化样例。", styles)
    add_image(story, ROOT / "runs/needle_lwt_seg/predict_smoke_cpu/needle_20260428_231351_152_000016_0e974ebb.jpg", 126, "图 8：训练权重 CPU smoke predict 输出样例。", styles)

    story.append(para("五、结论与后续建议", styles["H1"]))
    story.append(
        para(
            "当前工作区已经形成完整闭环：模型侧有最终 v3 CAD-like 网格，视觉侧有针类 YOLO 分割模型，实时侧有 Orbbec -> mask -> FoundationPose 的脚本化链路。"
            "最关键的工程修正是把跨系统实时数据发布做成原子文件协议，并在 FoundationPose 入口强制检查 depth/mask overlap，避免错误在 CUDA 推理阶段才爆出。",
            styles["Body"],
        )
    )
    story.append(
        para(
            "后续若继续提高质量，优先补一次带验证的 YOLO 训练并输出真实 mAP；其次采集更多角度下的深度有效样本，专门验证细针远端在深度图中的缺失率；"
            "最后可将最终 v3 模型与参考图进行尺度标定，避免纯视觉调参造成尾部比例偏差。",
            styles["Body"],
        )
    )
    story.append(
        table(
            [
                ["错误类别", "出现次数"],
                ["Float/Double dtype", str(logs["double_dtype"])],
                ["等待新帧超时", str(logs["frame_timeout"])],
                ["深度全零", str(logs["zero_depth"])],
                ["mask/depth overlap 不足", str(logs["mask_depth_overlap"])],
                ["文件竞争/路径异常", str(logs["file_race"])],
            ],
            [70 * mm, 35 * mm],
        )
    )

    def page(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("CJK", 8)
        canvas.setFillColor(colors.HexColor("#78909C"))
        canvas.drawRightString(195 * mm, 9 * mm, f"第 {doc_obj.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=page, onLaterPages=page)


def render_pdf_pages() -> list[Path]:
    doc = fitz.open(PDF_PATH)
    out_paths = []
    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        out = OUT_DIR / f"render_page_{idx:02d}.png"
        pix.save(out)
        out_paths.append(out)
    return out_paths


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    ASSET_DIR.mkdir(exist_ok=True)
    yolo = make_yolo_curves()
    contact_sheet = make_evidence_contact_sheet()
    summary = collect_summary(yolo, contact_sheet)
    (OUT_DIR / "evidence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary)
    build_pdf(summary)
    rendered = render_pdf_pages()
    print(f"PDF: {PDF_PATH.resolve()}")
    print(f"Markdown: {MD_PATH.resolve()}")
    print(f"Summary: {(OUT_DIR / 'evidence_summary.json').resolve()}")
    print(f"Rendered pages: {len(rendered)}")
    for path in rendered:
        print(path.resolve())


if __name__ == "__main__":
    main()
