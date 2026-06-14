import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps_model_fix"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


MESH_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("model/fixed_unnamed_object_3/needle_structured_tail_reconstruction.ply")
OUT_PATH = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else MESH_PATH.with_name(MESH_PATH.stem + "_material_render.png")
)


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) * 0.5
    radius = (maxs - mins).max() * 0.52
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def face_colors(mesh, light_dir):
    vcols = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
    fcols = vcols[mesh.faces].mean(axis=1)
    normals = np.asarray(mesh.face_normals, dtype=np.float32)
    light_dir = light_dir / np.linalg.norm(light_dir)
    diffuse = np.clip(normals @ light_dir, 0.0, 1.0)
    shade = 0.34 + 0.66 * diffuse[:, None]
    spec = np.power(np.clip(normals @ light_dir, 0.0, 1.0), 24.0)[:, None] * 0.22
    rgba = np.clip(fcols * shade + spec, 0.0, 1.0)
    alpha = np.ones((len(rgba), 1), dtype=np.float32)
    return np.hstack([rgba, alpha])


def render_view(ax, mesh, elev, azim, title):
    triangles = mesh.vertices[mesh.faces]
    colors = face_colors(mesh, np.array([0.35, -0.45, 0.82], dtype=np.float32))
    coll = Poly3DCollection(
        triangles,
        facecolors=colors,
        edgecolors=(0.02, 0.02, 0.02, 0.08),
        linewidths=0.08,
        antialiased=True,
    )
    ax.add_collection3d(coll)
    set_axes_equal(ax, mesh.vertices)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_axis_off()
    ax.set_facecolor((0.96, 0.96, 0.94))


def main():
    mesh = trimesh.load_mesh(MESH_PATH, process=False)
    fig = plt.figure(figsize=(14, 7), dpi=180)
    fig.patch.set_facecolor((0.96, 0.96, 0.94))

    views = [
        (18, -82, "profile"),
        (78, -88, "top"),
        (8, 6, "tail detail"),
    ]
    for i, (elev, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        render_view(ax, mesh, elev, azim, title)
        if title == "tail detail":
            # Zoom toward the structured tail.
            verts = mesh.vertices
            x_cut = np.percentile(verts[:, 0], 76)
            tail = verts[verts[:, 0] > x_cut]
            set_axes_equal(ax, tail)

    plt.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
