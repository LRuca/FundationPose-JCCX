#!/usr/bin/env python
"""贡献汇总图（各模块→实测效果，真实数字）+ YOLO 数据来源对比图。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "report" / "assets" / "realtime"
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
    if Path(p).exists():
        FONT = p; break
MPL = FontProperties(fname=FONT)
plt.rcParams["axes.unicode_minus"] = False


def contribution_table():
    rows = [
        ("精度", "对称轴去自转", "保持长轴方向, 垂直轴随上帧最小旋转", "自转占帧间旋转54%; 旋转抖动 2.93°→2.17°/帧 (↓26%)"),
        ("精度", "时序平滑 EMA(α=0.5)", "位姿低通滤波", "平移抖动 3.23→2.50mm (↓23%); 仅 +67ms 延迟"),
        ("鲁棒", "细针深度补全", "mask 邻域有效深度中位数填充", "mask 内有效深度 0→数百像素; 注册得以成功"),
        ("鲁棒", "YOLO候选过滤 ROI+高度", "限定工作区 + 剔除横向小目标", "假阳性 1→0; 专治误检显示器文字/线缆"),
        ("鲁棒", "combined 数据训练", "LWT + Inbox 合并训练", "mask mAP50=0.72, mAP50-95=0.39 (唯一可用)"),
        ("效率", "跟踪迭代 track_iter=1~2", "后续帧精化次数", "5.5~9.8 ms/帧 → 102~183 FPS (纯计算)"),
        ("效率", "注册迭代 est_iter=3~5", "首帧精化次数", "首帧注册 0.65~0.96 s"),
        ("效率", "YOLO 分辨率 imgsz=640~960", "分割输入尺寸", "1.7~2.1 ms/帧, 显存 37~100MB"),
        ("效率", "端到端", "YOLO + 跟踪", "≈8 ms/帧 ≪ 相机 33ms预算 → 瓶颈在相机非算法"),
    ]
    cat_color = {"精度": "#dbeafe", "效率": "#fef3c7", "鲁棒": "#dcfce7"}
    cat_dot = {"精度": "#2563eb", "效率": "#d97706", "鲁棒": "#16a34a"}
    W = 2000
    head_h, row_h, top = 150, 96, 200
    H = top + row_h * len(rows) + 50
    im = Image.new("RGB", (W, H), "#ffffff"); d = ImageDraw.Draw(im)
    f_title = ImageFont.truetype(FONT, 56); f_head = ImageFont.truetype(FONT, 30)
    f_cell = ImageFont.truetype(FONT, 28); f_tech = ImageFont.truetype(FONT, 30)
    d.text((50, 45), "各模块/技术的作用与实测贡献", fill="#0f172a", font=f_title)
    cols_x = [50, 230, 700, 1180]
    headers = ["类别", "技术 / 参数", "做法", "实测效果（真实数据）"]
    d.rectangle([40, top - 50, W - 40, top - 6], fill="#0f172a")
    for x, h in zip(cols_x, headers):
        d.text((x + 12, top - 46), h, fill="#ffffff", font=f_head)
    for i, (cat, tech, how, eff) in enumerate(rows):
        y = top + i * row_h
        d.rectangle([40, y, W - 40, y + row_h - 8], fill=cat_color[cat], outline="#cbd5e1", width=2)
        d.ellipse([cols_x[0] + 14, y + row_h // 2 - 18, cols_x[0] + 46, y + row_h // 2 + 14], fill=cat_dot[cat])
        d.text((cols_x[0] + 56, y + row_h // 2 - 18), cat, fill="#0f172a", font=f_cell)
        d.text((cols_x[1] + 12, y + 14), tech, fill="#0f172a", font=f_tech)
        d.text((cols_x[2] + 12, y + 20), how, fill="#334155", font=f_cell)
        d.text((cols_x[3] + 12, y + 20), eff, fill="#0f172a", font=f_cell)
    out = A / "贡献汇总表.png"; im.save(out); return out


def datasource_chart():
    data = {"lwt": 0.0, "inbox": 0.0, "combined": 0.723}
    data95 = {"lwt": 0.0, "inbox": 0.0, "combined": 0.387}
    import pandas as pd
    for name, path in [("lwt", "runs/needle_lwt_seg/yolov8n_seg_lwt/results.csv"),
                       ("inbox", "runs/needle_inbox_seg/yolov8n_seg_inbox/results.csv"),
                       ("combined", "runs/needle_inbox_seg/yolov8n_seg_combined/results.csv")]:
        fp = ROOT / path
        if fp.exists():
            df = pd.read_csv(fp); df.columns = [c.strip() for c in df.columns]
            if "metrics/mAP50(M)" in df.columns:
                data[name] = float(df.iloc[-1]["metrics/mAP50(M)"])
                data95[name] = float(df.iloc[-1]["metrics/mAP50-95(M)"])
    names = ["lwt", "inbox", "combined"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
    ax.bar(x - 0.2, [data[n] for n in names], 0.4, color="#2563eb", label="mask mAP50")
    ax.bar(x + 0.2, [data95[n] for n in names], 0.4, color="#f59e0b", label="mask mAP50-95")
    for xi, n in enumerate(names):
        ax.text(xi - 0.2, data[n] + 0.01, f"{data[n]:.2f}", ha="center", fontproperties=MPL)
        ax.text(xi + 0.2, data95[n] + 0.01, f"{data95[n]:.2f}", ha="center", fontproperties=MPL)
    ax.set_xticks(x); ax.set_xticklabels(["LWT", "Inbox", "Combined"])
    ax.set_title("YOLO 训练数据来源消融（针分割验证 mAP）", fontproperties=MPL, fontsize=14)
    ax.set_ylabel("mask mAP", fontproperties=MPL); ax.legend(prop=MPL); ax.grid(alpha=0.25, ls="--", axis="y")
    ax.set_ylim(0, 0.85)
    ax.text(0.5, 0.45, "LWT / Inbox 单独训练验证 mAP≈0\n仅 Combined 合并训练得到可用模型",
            fontproperties=MPL, fontsize=10, color="#475569", ha="center")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(MPL)
    fig.tight_layout(); out = A / "消融_YOLO数据来源.png"; fig.savefig(out); plt.close(fig); return out


if __name__ == "__main__":
    print("saved", contribution_table())
    print("saved", datasource_chart())
