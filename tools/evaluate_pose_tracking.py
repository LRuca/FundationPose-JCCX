#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]


def find_cjk_font() -> FontProperties | None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for item in candidates:
        path = Path(item)
        if path.exists():
            return FontProperties(fname=str(path))
    return None


FONT = find_cjk_font()


def apply_chinese_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    if FONT:
        ax.set_title(title, fontproperties=FONT)
        ax.set_xlabel(xlabel, fontproperties=FONT)
        ax.set_ylabel(ylabel, fontproperties=FONT)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontproperties(FONT)
    else:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.24)


def load_poses(pose_dir: Path) -> list[tuple[str, np.ndarray]]:
    items: list[tuple[str, np.ndarray]] = []
    for path in sorted(pose_dir.glob("*.txt")):
        mat = np.loadtxt(path, dtype=np.float64)
        if mat.size != 16:
            raise ValueError(f"位姿矩阵不是 4x4: {path}")
        items.append((path.stem, mat.reshape(4, 4)))
    if len(items) < 2:
        raise ValueError(f"至少需要 2 个位姿文件，当前目录不足: {pose_dir}")
    return items


def rotation_angle_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    rel = r2 @ r1.T
    cos_theta = (np.trace(rel) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def evaluate(poses: list[tuple[str, np.ndarray]], fps: float | None) -> tuple[pd.DataFrame, dict]:
    rows = []
    first_name, first_pose = poses[0]
    first_t = first_pose[:3, 3]
    for i, (name, pose) in enumerate(poses):
        t = pose[:3, 3]
        row = {
            "frame": name,
            "index": i,
            "tx_m": float(t[0]),
            "ty_m": float(t[1]),
            "tz_m": float(t[2]),
            "distance_from_first_mm": float(np.linalg.norm(t - first_t) * 1000.0),
        }
        if i == 0:
            row.update(
                {
                    "step_translation_mm": 0.0,
                    "step_rotation_deg": 0.0,
                    "translation_speed_mm_s": 0.0,
                    "rotation_speed_deg_s": 0.0,
                }
            )
        else:
            prev_pose = poses[i - 1][1]
            step_t = float(np.linalg.norm(t - prev_pose[:3, 3]) * 1000.0)
            step_r = rotation_angle_deg(prev_pose[:3, :3], pose[:3, :3])
            row.update(
                {
                    "step_translation_mm": step_t,
                    "step_rotation_deg": step_r,
                    "translation_speed_mm_s": step_t * fps if fps else np.nan,
                    "rotation_speed_deg_s": step_r * fps if fps else np.nan,
                }
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    valid = df.iloc[1:]
    summary = {
        "pose_dir": "",
        "num_frames": int(len(df)),
        "fps_assumption": fps,
        "mean_step_translation_mm": float(valid["step_translation_mm"].mean()),
        "p95_step_translation_mm": float(valid["step_translation_mm"].quantile(0.95)),
        "max_step_translation_mm": float(valid["step_translation_mm"].max()),
        "mean_step_rotation_deg": float(valid["step_rotation_deg"].mean()),
        "p95_step_rotation_deg": float(valid["step_rotation_deg"].quantile(0.95)),
        "max_step_rotation_deg": float(valid["step_rotation_deg"].max()),
    }
    if fps:
        summary["estimated_duration_s"] = float((len(df) - 1) / fps)
    return df, summary


def make_chart(df: pd.DataFrame, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=180)

    axes[0, 0].plot(df["index"], df["step_translation_mm"], color="#0f766e", linewidth=2)
    apply_chinese_style(axes[0, 0], "相邻帧平移变化", "帧序号", "毫米")

    axes[0, 1].plot(df["index"], df["step_rotation_deg"], color="#b45309", linewidth=2)
    apply_chinese_style(axes[0, 1], "相邻帧旋转变化", "帧序号", "角度")

    axes[1, 0].plot(df["index"], df["distance_from_first_mm"], color="#1d4ed8", linewidth=2)
    apply_chinese_style(axes[1, 0], "相对首帧位移", "帧序号", "毫米")

    axes[1, 1].hist(df["step_translation_mm"].iloc[1:], bins=24, color="#64748b", edgecolor="white")
    apply_chinese_style(axes[1, 1], "平移抖动分布", "毫米", "帧数")

    if FONT:
        fig.suptitle("实时位姿序列稳定性评估", fontproperties=FONT, fontsize=18)
    else:
        fig.suptitle("实时位姿序列稳定性评估", fontsize=18)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 FoundationPose 输出的 ob_in_cam 位姿序列稳定性。")
    parser.add_argument("--pose-dir", default="third_party/FoundationPose/debug_needle_live/ob_in_cam")
    parser.add_argument("--fps", type=float, default=None, help="录制或处理帧率；不知道就不填。")
    parser.add_argument("--out-dir", default="report/realtime_eval")
    args = parser.parse_args()

    pose_dir = (ROOT / args.pose_dir).resolve() if not Path(args.pose_dir).is_absolute() else Path(args.pose_dir)
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    poses = load_poses(pose_dir)
    df, summary = evaluate(poses, args.fps)
    summary["pose_dir"] = str(pose_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "pose_sequence_metrics.csv", index=False)
    (out_dir / "pose_sequence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_chart(df, out_dir / "pose_sequence_stability.png")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"已输出: {out_dir}")


if __name__ == "__main__":
    main()
