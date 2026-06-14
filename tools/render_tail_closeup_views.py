import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps_model_fix"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


MESH = Path("model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.ply")
OUT = Path("model/fixed_unnamed_object_3/needle_structured_tail_v3_tail_closeup_rerender.png")


def colors(mesh, light):
    vc = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
    fc = vc[mesh.faces].mean(axis=1)
    n = np.asarray(mesh.face_normals, dtype=np.float32)
    light = light / np.linalg.norm(light)
    shade = 0.36 + 0.64 * np.clip(n @ light, 0, 1)[:, None]
    return np.clip(fc * shade, 0, 1)


def add_mesh(ax, mesh, view, title):
    tri = mesh.vertices[mesh.faces]
    coll = Poly3DCollection(
        tri,
        facecolors=np.c_[colors(mesh, np.array([0.2, -0.45, 0.87])), np.ones(len(tri))],
        edgecolors=(0, 0, 0, 0.12),
        linewidths=0.08,
        antialiased=True,
    )
    ax.add_collection3d(coll)
    tail = mesh.vertices[mesh.vertices[:, 0] > np.percentile(mesh.vertices[:, 0], 70)]
    mn, mx = tail.min(axis=0), tail.max(axis=0)
    center = (mn + mx) / 2
    radius = (mx - mn).max() * 0.58
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(*view)
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    ax.set_facecolor((0.96, 0.96, 0.94))


def main():
    mesh = trimesh.load_mesh(MESH, process=False)
    fig = plt.figure(figsize=(13, 5), dpi=220)
    fig.patch.set_facecolor((0.96, 0.96, 0.94))
    views = [
        ((0, -90), "side: thin stacked tail"),
        ((90, -90), "top: round groove between blocks"),
        ((5, -8), "end: rounded-rectangle blocks"),
    ]
    for i, (view, title) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        add_mesh(ax, mesh, view, title)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUT.resolve())


if __name__ == "__main__":
    main()
