from __future__ import annotations

import numpy as np


class _PointCloud:
  def __init__(self):
    self.points = np.empty((0, 3), dtype=np.float64)
    self.colors = np.empty((0, 3), dtype=np.float64)
    self.normals = np.empty((0, 3), dtype=np.float64)

  def voxel_down_sample(self, voxel_size):
    pts = np.asarray(self.points, dtype=np.float64)
    if len(pts) == 0 or voxel_size <= 0:
      return self
    keys = np.floor(pts / float(voxel_size)).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(keep)
    out = _PointCloud()
    out.points = pts[keep]
    colors = np.asarray(self.colors)
    normals = np.asarray(self.normals)
    if len(colors) == len(pts):
      out.colors = colors[keep]
    if len(normals) == len(pts):
      out.normals = normals[keep]
    return out

  def select_by_index(self, indices):
    idx = np.asarray(indices, dtype=np.int64)
    out = _PointCloud()
    out.points = np.asarray(self.points)[idx]
    colors = np.asarray(self.colors)
    normals = np.asarray(self.normals)
    if len(colors) == len(self.points):
      out.colors = colors[idx]
    if len(normals) == len(self.points):
      out.normals = normals[idx]
    return out

  def __iadd__(self, other):
    self.points = np.concatenate([np.asarray(self.points), np.asarray(other.points)], axis=0)
    if len(np.asarray(self.colors)) or len(np.asarray(other.colors)):
      self.colors = np.concatenate([np.asarray(self.colors), np.asarray(other.colors)], axis=0)
    if len(np.asarray(self.normals)) or len(np.asarray(other.normals)):
      self.normals = np.concatenate([np.asarray(self.normals), np.asarray(other.normals)], axis=0)
    return self


class _Geometry:
  PointCloud = _PointCloud


class _Utility:
  @staticmethod
  def Vector3dVector(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr.reshape((-1, 3))


class _IO:
  @staticmethod
  def write_point_cloud(path, pcd):
    pts = np.asarray(pcd.points, dtype=np.float64).reshape((-1, 3))
    colors = np.asarray(getattr(pcd, "colors", []), dtype=np.float64)
    has_colors = len(colors) == len(pts)
    with open(path, "w", encoding="utf-8") as f:
      f.write("ply\nformat ascii 1.0\n")
      f.write(f"element vertex {len(pts)}\n")
      f.write("property float x\nproperty float y\nproperty float z\n")
      if has_colors:
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
      f.write("end_header\n")
      for i, p in enumerate(pts):
        if has_colors:
          c = np.clip(colors[i] * (255 if colors[i].max() <= 1.0 else 1), 0, 255).astype(np.uint8)
          f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
          f.write(f"{p[0]} {p[1]} {p[2]}\n")
    return True

  @staticmethod
  def read_point_cloud(path):
    pcd = _PointCloud()
    pts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
      in_body = False
      for line in f:
        if not in_body:
          if line.strip() == "end_header":
            in_body = True
          continue
        parts = line.split()
        if len(parts) >= 3:
          pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    pcd.points = np.asarray(pts, dtype=np.float64).reshape((-1, 3))
    return pcd


geometry = _Geometry()
utility = _Utility()
io = _IO()
