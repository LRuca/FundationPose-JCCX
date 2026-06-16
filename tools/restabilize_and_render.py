#!/usr/bin/env python
"""从已录制的 ob_in_cam 位姿序列重渲染 demo，消除针绕自身对称轴(长轴)的自转。

做三件事：
1) 时序稳定化：保持对称轴方向不变，让垂直于对称轴的两根坐标轴随上一帧最小旋转 -> 去掉自转。
2) 色键 + inpaint：把旧 track_vis 上烧进去的纯色坐标轴/包围盒抠掉，恢复干净背景。
3) 用稳定后的位姿重绘坐标轴(+可选包围盒)，合成视频。

仅依赖已保存数据：track_vis(做背景) + ob_in_cam(位姿)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def load_mesh_frame(mesh_file):
    mesh = trimesh.load(mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    sym = int(np.argmax(extents))  # 最长 extent = 针长轴 = 对称轴
    return np.asarray(to_origin, float), np.asarray(extents, float), sym


def stabilize(R_list, sym):
    """去除绕 sym 轴的滚转：保持 sym 列方向，垂直方向跟随上一帧。"""
    out = []
    prev_u = None
    for R in R_list:
        a = R[:, sym] / (np.linalg.norm(R[:, sym]) + 1e-9)
        if prev_u is None:
            u = R[:, (sym + 1) % 3]
        else:
            u = prev_u
        u = u - (u @ a) * a
        if np.linalg.norm(u) < 1e-6:
            u = R[:, (sym + 1) % 3] - (R[:, (sym + 1) % 3] @ a) * a
        u = u / (np.linalg.norm(u) + 1e-9)
        w = np.cross(a, u)
        cols = [None, None, None]
        cols[sym] = a
        cols[(sym + 1) % 3] = u
        cols[(sym + 2) % 3] = w
        Rs = np.stack(cols, axis=1)
        out.append(Rs)
        prev_u = u
    return out


def clean_background(bgr):
    """色键抠掉纯红/绿/蓝叠加线，inpaint 补背景。"""
    b, g, r = bgr[..., 0].astype(int), bgr[..., 1].astype(int), bgr[..., 2].astype(int)
    red = (r > 140) & (g < 90) & (b < 90)
    green = (g > 140) & (r < 110) & (b < 110)
    blue = (b > 140) & (r < 110) & (g < 110)
    mask = (red | green | blue).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)


def project(K, P):
    p = (K @ P.T).T
    return p[:, :2] / p[:, 2:3]


def draw_axes(img, K, center_pose, scale, sym, hide_sym, thickness=3):
    R = center_pose[:3, :3]
    t = center_pose[:3, 3]
    origin = t.reshape(1, 3)
    ends = np.stack([t + R[:, i] * scale for i in range(3)], axis=0)
    pts = project(K, np.concatenate([origin, ends], axis=0)).astype(int)
    o = tuple(pts[0])
    # X=红(BGR 0,0,255) Y=绿 Z=蓝
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i in range(3):
        if hide_sym and i == sym:
            continue
        cv2.arrowedLine(img, o, tuple(pts[1 + i]), colors[i], thickness, cv2.LINE_AA, tipLength=0.18)
    return img


def draw_box(img, K, center_pose, extents, color=(0, 200, 255), lw=2):
    c = extents / 2.0
    corners = np.array([[sx * c[0], sy * c[1], sz * c[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    cam = (center_pose[:3, :3] @ corners.T).T + center_pose[:3, 3]
    uv = project(K, cam).astype(int)
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for a, b in edges:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, lw, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True, help="report/takes/<name> 目录")
    ap.add_argument("--mesh", default=str(ROOT / "model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl"))
    ap.add_argument("--K", default=str(ROOT / "third_party/FoundationPose/live_orbbec/cam_K.txt"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--scale", type=float, default=0.06)
    ap.add_argument("--hide_sym", action="store_true", help="不画对称轴(长轴)那根箭头")
    ap.add_argument("--no_box", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "report/needle_realtime_demo.mp4"))
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--dump_frames", default=None)
    args = ap.parse_args()

    take = Path(args.take)
    tv = take / "fp_debug" / "track_vis"
    oc = take / "fp_debug" / "ob_in_cam"
    K = np.loadtxt(args.K).reshape(3, 3)
    to_origin, extents, sym = load_mesh_frame(args.mesh)
    inv_to_origin = np.linalg.inv(to_origin)
    print(f"对称轴(长轴)= 第 {sym} 列 (0=X红,1=Y绿,2=Z蓝); extents={np.round(extents,4).tolist()}")

    names = sorted(p.stem for p in oc.glob("*.txt"))
    names = names[args.start:args.end]
    poses = [np.loadtxt(oc / f"{n}.txt").reshape(4, 4) for n in names]
    center = [p @ inv_to_origin for p in poses]
    Rs = stabilize([c[:3, :3] for c in center], sym)

    writer = None
    dump = Path(args.dump_frames) if args.dump_frames else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        bg = cv2.imread(str(tv / f"{n}.png"))
        if bg is None:
            continue
        bg = clean_background(bg)
        cp = center[i].copy()
        cp[:3, :3] = Rs[i]
        if not args.no_box:
            bg = draw_box(bg, K, cp, extents)
        bg = draw_axes(bg, K, cp, args.scale, sym, args.hide_sym)
        if writer is None:
            h, w = bg.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(bg)
        if dump:
            cv2.imwrite(str(dump / f"{i:06d}.png"), bg)
    if writer:
        writer.release()
    print(f"saved {args.out}  ({len(names)} frames)")


if __name__ == "__main__":
    main()
