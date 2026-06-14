import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps_model_fix"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402


def main() -> None:
    mesh_path = Path("model/fixed_unnamed_object_3/needle_structured_tail_reconstruction.ply")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = mesh.vertices
    colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0

    c = vertices.mean(axis=0)
    cov = np.cov((vertices - c).T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, np.argmax(vals)]
    if axis[0] < 0:
        axis = -axis
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, axis)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)

    rel = vertices - c
    t = rel @ axis
    a = rel @ u
    b = rel @ v

    fig, axs = plt.subplots(3, 1, figsize=(11, 6), dpi=160)
    views = [
        ("side/profile", t, a),
        ("top/profile", t, b),
        ("tail end view", a, b),
    ]
    for ax, (title, x, y) in zip(axs, views):
        ax.scatter(x, y, s=4, c=colors, linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    out = mesh_path.with_name("needle_structured_tail_aligned_preview.png")
    fig.savefig(out, bbox_inches="tight")
    print(out.resolve())


if __name__ == "__main__":
    main()
