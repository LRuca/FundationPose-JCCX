import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps_model_fix"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402


def main() -> None:
    mesh_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("model/fixed_unnamed_object_3/cleaned_smoothed.ply")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    points = mesh.vertices
    if len(points) > 8000:
        rng = np.random.default_rng(7)
        points = points[rng.choice(len(points), 8000, replace=False)]

    fig = plt.figure(figsize=(10, 4), dpi=160)
    views = [("front", (0, 1)), ("side", (1, 2)), ("top", (0, 2))]
    for i, (name, axes) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i)
        ax.scatter(points[:, axes[0]], points[:, axes[1]], s=0.25, c=points[:, 2], cmap="viridis")
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    fig.tight_layout()
    out = mesh_path.with_name(mesh_path.stem + "_preview.png")
    fig.savefig(out, transparent=False, bbox_inches="tight")
    print(out.resolve())


if __name__ == "__main__":
    main()
