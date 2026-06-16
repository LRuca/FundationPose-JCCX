#!/usr/bin/env python
"""汇报用全景图：系统架构、中期规划对照、工程量统计、实时性vs论文、穿刺针挑战对策。"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "report" / "assets" / "realtime"
A.mkdir(parents=True, exist_ok=True)
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
    if Path(p).exists():
        FONT = p; break
MPL = FontProperties(fname=FONT)
plt.rcParams["axes.unicode_minus"] = False


def fnt(sz):
    return ImageFont.truetype(FONT, sz)


def rrect(d, box, r, fill, outline="#334155", w=3):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=w)


def arrow(d, x1, y1, x2, y2, color="#334155", w=5):
    d.line([x1, y1, x2, y2], fill=color, width=w)
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    L = 16
    for s in (0.5, -0.5):
        d.line([x2, y2, x2 - L * math.cos(ang - s), y2 - L * math.sin(ang - s)], fill=color, width=w)


# ---------- 1. 系统架构图 ----------
def architecture():
    W, H = 2200, 1180
    im = Image.new("RGB", (W, H), "#f8fafc"); d = ImageDraw.Draw(im)
    d.text((60, 40), "实时穿刺针 6D 位姿解算 —— 系统架构与多进程数据流", fill="#0f172a", font=fnt(52))
    d.text((60, 110), "三进程 + 文件级 IPC（原子写入避免读写竞争），GPU 计算远快于相机帧率", fill="#475569", font=fnt(28))

    # 进程框
    cam = (80, 220, 560, 470)
    rrect(d, cam, 18, "#dbeafe"); d.text((cam[0]+24, cam[1]+24), "① 采集进程 (C++)", fill="#0f172a", font=fnt(34))
    for k, t in enumerate(["Orbbec DaBai DCW RGB-D", "warmup + 对齐 + 发布", "原子写 color/depth/K/json"]):
        d.text((cam[0]+28, cam[1]+80+k*46), "• "+t, fill="#1e293b", font=fnt(26))

    yolo = (80, 560, 560, 810)
    rrect(d, yolo, 18, "#dcfce7"); d.text((yolo[0]+24, yolo[1]+24), "② YOLO 分割进程", fill="#0f172a", font=fnt(34))
    for k, t in enumerate(["YOLOv8n-seg (combined)", "ROI+高度过滤 抗误检", "输出 mask_yolo.png"]):
        d.text((yolo[0]+28, yolo[1]+80+k*46), "• "+t, fill="#1e293b", font=fnt(26))

    fp = (820, 360, 1380, 720)
    rrect(d, fp, 18, "#ffedd5", outline="#9a3412", w=4)
    d.text((fp[0]+24, fp[1]+24), "③ FoundationPose 跟踪进程", fill="#0f172a", font=fnt(34))
    for k, t in enumerate(["首帧: register 注册初始位姿", "后续帧: track_one 跟踪", "细针深度补全 (邻域中位数)", "对称轴去自转 + 时序平滑", "保存逐帧 RGB-D / 位姿"]):
        d.text((fp[0]+28, fp[1]+82+k*52), "• "+t, fill="#1e293b", font=fnt(26))

    out = (1640, 360, 2120, 720)
    rrect(d, out, 18, "#fce7f3"); d.text((out[0]+24, out[1]+24), "④ 输出 / 评估", fill="#0f172a", font=fnt(34))
    for k, t in enumerate(["6D 位姿序列 ob_in_cam", "可视化 / demo 视频", "精度·效率·消融评估", "中文图表自动生成"]):
        d.text((out[0]+28, out[1]+82+k*52), "• "+t, fill="#1e293b", font=fnt(26))

    # 共享文件 IPC
    ipc = (640, 560, 740, 810)
    rrect(d, ipc, 14, "#e2e8f0"); d.text((ipc[0]+14, ipc[1]+90), "共享\n文件\nIPC", fill="#0f172a", font=fnt(26))

    arrow(d, cam[2], 345, fp[0], 430)             # cam -> fp
    arrow(d, 310, cam[3], 310, yolo[1])           # cam -> yolo
    arrow(d, yolo[2], 685, ipc[0], 685)           # yolo -> ipc
    arrow(d, ipc[2], 660, fp[0], 600)             # ipc -> fp
    arrow(d, fp[2], 540, out[0], 540)             # fp -> out

    # 底部关键指标条
    d.text((80, 900), "关键性能（实测 @RTX 5070 Ti）", fill="#0f172a", font=fnt(34))
    stats = [("YOLO 单帧", "1.7~2.1 ms"), ("首帧注册", "0.35~1.7 s"), ("逐帧跟踪", "5.5~22 ms"),
             ("纯计算帧率", "45~183 FPS"), ("峰值显存", "≈3.4 GB"), ("相机帧预算", "33 ms")]
    for k, (a, b) in enumerate(stats):
        x = 80 + k * 350
        rrect(d, (x, 960, x+320, 1090), 16, "#ffffff")
        d.text((x+24, 980), a, fill="#475569", font=fnt(28))
        d.text((x+24, 1028), b, fill="#0f766e", font=fnt(40))
    im.save(A / "汇报_系统架构图.png"); return A / "汇报_系统架构图.png"


# ---------- 2. 中期规划 -> 本阶段交付 对照 ----------
def roadmap():
    W, H = 2100, 1080
    im = Image.new("RGB", (W, H), "#ffffff"); d = ImageDraw.Draw(im)
    d.text((60, 40), "中期规划『后续四步』 → 本阶段交付对照", fill="#0f172a", font=fnt(52))
    d.text((60, 112), "中期答辩(2026.5.7)提出的下一阶段计划，本阶段逐项落地", fill="#475569", font=fnt(28))
    rows = [
        ("第1步 提升识别精度", "扩展数据标注 + 训练 YOLO 分割", "Combined 合并训练，mask mAP50=0.72 / mAP50-95=0.39", "已完成"),
        ("第2步 动态优化", "实现动态跟踪与识别；蒸馏提效", "实时链路打通；跟踪达 183 FPS（论文 32Hz）。蒸馏接口预留", "已完成*"),
        ("第3步 鲁棒性提升", "深度滤波 + mask时序补偿 + 滤波/物理先验", "细针深度补全 + 时序平滑(EMA) + 对称轴去自转", "已完成"),
        ("第4步 实物实验", "相机标定、真值对齐、记录动态轨迹与分析", "真实穿刺针动态序列已录(1500帧)并量化分析；绝对真值待标定工装", "部分完成"),
    ]
    colors = {"已完成": "#16a34a", "已完成*": "#0d9488", "部分完成": "#d97706"}
    y = 220
    for step, plan, deliver, status in rows:
        rrect(d, (50, y, W-50, y+185), 18, "#f1f5f9")
        d.text((80, y+22), step, fill="#0f172a", font=fnt(36))
        d.text((80, y+82), "规划：" + plan, fill="#475569", font=fnt(27))
        d.text((80, y+130), "交付：" + deliver, fill="#0f172a", font=fnt(27))
        bx = (W-300, y+40, W-90, y+120)
        rrect(d, bx, 14, colors[status], outline=colors[status])
        d.text((bx[0]+28, bx[1]+18), status, fill="#ffffff", font=fnt(34))
        y += 205
    d.text((60, y+6), "* 蒸馏按分工由团队另行完成；本阶段已验证实时性与提效空间。", fill="#64748b", font=fnt(24))
    im.save(A / "汇报_中期规划对照.png"); return A / "汇报_中期规划对照.png"


# ---------- 3. 工程量统计 ----------
def workload():
    W, H = 2000, 760
    im = Image.new("RGB", (W, H), "#0f172a"); d = ImageDraw.Draw(im)
    d.text((60, 50), "本阶段工程量统计", fill="#ffffff", font=fnt(56))
    d.text((60, 130), "从“离线估计”扩展到“实时动态解算”的完整工程实现与实验", fill="#94a3b8", font=fnt(28))
    cards = [
        ("17", "新增工具/脚本", "#38bdf8"), ("~2100", "行新增代码", "#34d399"),
        ("24", "张中文图表", "#fbbf24"), ("46", "组实验配置", "#f472b6"),
        ("5", "条真实录制镜次", "#a78bfa"), ("1500", "帧跟踪位姿", "#fb7185"),
        ("10", "项专题实验", "#22d3ee"), ("183", "FPS 实时峰值", "#4ade80"),
    ]
    for k, (num, label, col) in enumerate(cards):
        r, c = divmod(k, 4)
        x = 60 + c * 470; y = 230 + r * 240
        rrect(d, (x, y, x+430, y+200), 20, "#1e293b", outline="#334155", w=2)
        d.text((x+30, y+30), num, fill=col, font=fnt(76))
        d.text((x+30, y+135), label, fill="#cbd5e1", font=fnt(30))
    im.save(A / "汇报_工程量统计.png"); return A / "汇报_工程量统计.png"


# ---------- 4. 实时性 vs 论文 ----------
def realtime_vs_paper():
    labels = ["论文 FoundationPose\n(报告值)", "本方法\ntrack_iter=5", "本方法\ntrack_iter=2", "本方法\ntrack_iter=1"]
    fps = [32, 45, 102, 183]
    cols = ["#94a3b8", "#0f766e", "#2563eb", "#16a34a"]
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=170)
    b = ax.bar(np.arange(len(labels)), fps, color=cols, width=0.6)
    for bi, v in zip(b, fps):
        ax.text(bi.get_x()+bi.get_width()/2, v+3, f"{v} FPS", ha="center", fontproperties=MPL, fontsize=12)
    ax.axhline(32, color="#dc2626", ls="--", lw=1.5)
    ax.axhline(30, color="#0ea5e9", ls=":", lw=1.5)
    ax.text(3.4, 33.5, "论文 32Hz", color="#dc2626", fontproperties=MPL, fontsize=10, ha="right")
    ax.text(3.4, 24, "相机 30FPS", color="#0ea5e9", fontproperties=MPL, fontsize=10, ha="right")
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels)
    ax.set_title("实时跟踪帧率：本方法纯计算 vs 论文基线（均超实时）", fontproperties=MPL, fontsize=14)
    ax.set_ylabel("跟踪帧率 FPS", fontproperties=MPL); ax.grid(alpha=0.25, ls="--", axis="y")
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontproperties(MPL)
    fig.tight_layout(); out = A / "汇报_实时性对比.png"; fig.savefig(out); plt.close(fig); return out


# ---------- 5. 穿刺针挑战 -> 对策 ----------
def challenges():
    W, H = 2050, 760
    im = Image.new("RGB", (W, H), "#ffffff"); d = ImageDraw.Draw(im)
    d.text((60, 45), "穿刺针固有挑战 → 本阶段对策 → 实测效果", fill="#0f172a", font=fnt(50))
    items = [
        ("细长·深度缺失", "深度相机在针像素常给0", "mask 邻域深度补全", "有效深度 0→数百，注册成功"),
        ("轴对称·位姿翻转", "绕长轴旋转不可观测→自转", "对称轴去自转稳定化", "自转占旋转54%，抖动↓26%"),
        ("低纹理·反光", "mask 边界断裂/缺帧→位姿抖动", "时序平滑(EMA) + combined训练", "抖动↓23%，仅+67ms延迟"),
        ("背景干扰·误检", "屏幕文字/线缆被误判为针", "ROI工作区+最小高度过滤", "假阳性 1→0"),
    ]
    cw = 480
    for k, (ch, why, sol, eff) in enumerate(items):
        x = 50 + k * 495
        rrect(d, (x, 150, x+cw, 290), 16, "#fee2e2", outline="#dc2626", w=2)
        d.text((x+24, 168), "挑战", fill="#dc2626", font=fnt(26)); d.text((x+24, 206), ch, fill="#0f172a", font=fnt(30))
        d.text((x+24, 250), why, fill="#7f1d1d", font=fnt(21))
        arrow(d, x+cw//2, 295, x+cw//2, 335, "#334155", 4)
        rrect(d, (x, 340, x+cw, 470), 16, "#dbeafe", outline="#2563eb", w=2)
        d.text((x+24, 358), "对策", fill="#2563eb", font=fnt(26)); d.text((x+24, 396), sol, fill="#0f172a", font=fnt(28))
        arrow(d, x+cw//2, 475, x+cw//2, 515, "#334155", 4)
        rrect(d, (x, 520, x+cw, 660), 16, "#dcfce7", outline="#16a34a", w=2)
        d.text((x+24, 538), "效果", fill="#16a34a", font=fnt(26)); d.text((x+24, 580), eff, fill="#0f172a", font=fnt(26))
    im.save(A / "汇报_挑战与对策.png"); return A / "汇报_挑战与对策.png"


if __name__ == "__main__":
    for f in [architecture, roadmap, workload, realtime_vs_paper, challenges]:
        print("saved", f())
