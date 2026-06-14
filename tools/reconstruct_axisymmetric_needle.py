import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps_model_fix"))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402


IN_MESH = Path("model/fixed_unnamed_object_3/object_without_flat_sheet_main.ply")
OUT_DIR = Path("model/fixed_unnamed_object_3")


def robust_axis(points: np.ndarray, rounds: int = 5):
    center = points.mean(axis=0)
    kept = np.ones(len(points), dtype=bool)
    axis = np.array([1.0, 0.0, 0.0])
    for _ in range(rounds):
        pts = points[kept]
        center = pts.mean(axis=0)
        cov = np.cov((pts - center).T)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, np.argmax(vals)]
        if axis[0] < 0:
            axis = -axis
        t = (points - center) @ axis
        radial = np.linalg.norm((points - center) - np.outer(t, axis), axis=1)
        # The scan contains background sheet remnants. Keep the slender core.
        cutoff = np.percentile(radial[kept], 72)
        kept = radial <= max(cutoff, 1e-6)
    return center, axis / np.linalg.norm(axis), kept


def make_basis(axis: np.ndarray):
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, axis)) > 0.92:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)
    return u, v


def radius_profile(t: np.ndarray, radial: np.ndarray, length: float, sections: int):
    max_radius = min(np.percentile(radial, 68), length * 0.018)
    max_radius = max(max_radius, length * 0.006)

    x = np.linspace(-0.5, 0.5, sections)
    taper = np.ones_like(x)
    tip = 0.16
    left = x < -0.5 + tip
    right = x > 0.5 - tip
    taper[left] = np.sin((x[left] + 0.5) / tip * np.pi / 2.0)
    taper[right] = np.sin((0.5 - x[right]) / tip * np.pi / 2.0)

    # Very small nonzero end radius avoids degenerate triangles at the tips.
    radii = max_radius * np.clip(taper, 0.04, 1.0)
    return radii, max_radius


def build_needle(center: np.ndarray, axis: np.ndarray, length: float, radius_data: np.ndarray):
    sections = 96
    segments = 32
    u, v = make_basis(axis)
    x = np.linspace(-0.5, 0.5, sections)
    radii, max_radius = radius_profile(x, radius_data, length, sections)

    vertices = []
    for xi, radius in zip(x, radii):
        c = center + axis * (xi * length)
        for j in range(segments):
            a = 2.0 * np.pi * j / segments
            vertices.append(c + radius * (np.cos(a) * u + np.sin(a) * v))
    vertices = np.asarray(vertices)

    faces = []
    for i in range(sections - 1):
        base = i * segments
        nxt = (i + 1) * segments
        for j in range(segments):
            j2 = (j + 1) % segments
            faces.append([base + j, nxt + j, nxt + j2])
            faces.append([base + j, nxt + j2, base + j2])

    start_center = len(vertices)
    end_center = start_center + 1
    vertices = np.vstack(
        [
            vertices,
            center - axis * (0.5 * length),
            center + axis * (0.5 * length),
        ]
    )
    for j in range(segments):
        j2 = (j + 1) % segments
        faces.append([start_center, j2, j])
        last = (sections - 1) * segments
        faces.append([end_center, last + j, last + j2])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    return mesh, max_radius


def main() -> None:
    scan = trimesh.load_mesh(IN_MESH, process=False)
    points = scan.vertices
    center, axis, kept = robust_axis(points)
    t = (points - center) @ axis
    radial = np.linalg.norm((points - center) - np.outer(t, axis), axis=1)

    core_t = t[kept]
    length = float(np.percentile(core_t, 99.5) - np.percentile(core_t, 0.5))
    center = center + axis * float((np.percentile(core_t, 99.5) + np.percentile(core_t, 0.5)) * 0.5)
    core_radial = radial[kept]

    needle, radius = build_needle(center, axis, length, core_radial)
    needle.export(OUT_DIR / "needle_axisymmetric_reconstruction.ply")
    needle.export(OUT_DIR / "needle_axisymmetric_reconstruction.stl")
    needle.export(OUT_DIR / "needle_axisymmetric_reconstruction.obj")

    print(f"source_vertices={len(points)}")
    print(f"core_vertices={int(kept.sum())}")
    print(f"axis={axis.tolist()}")
    print(f"length={length:.8f}")
    print(f"max_radius={radius:.8f}")
    print(f"diameter={2 * radius:.8f}")
    print(f"vertices={len(needle.vertices)}")
    print(f"faces={len(needle.faces)}")
    print(f"watertight={needle.is_watertight}")
    print(f"out={str((OUT_DIR / 'needle_axisymmetric_reconstruction.stl').resolve())}")


if __name__ == "__main__":
    main()
