import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps_model_fix"))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402


OUT_DIR = Path("model/fixed_unnamed_object_3")


def ring(center_x, radius, sides, axis_center, axis, u, v):
    verts = []
    for j in range(sides):
        a = 2.0 * np.pi * j / sides
        verts.append(axis_center + axis * center_x + radius * (np.cos(a) * u + np.sin(a) * v))
    return verts


def make_lathe(profile, sides=48, axis_center=None, axis=None):
    axis_center = np.zeros(3) if axis_center is None else axis_center
    axis = np.array([1.0, 0.0, 0.0]) if axis is None else axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, axis)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)

    vertices = []
    ring_sides = []
    for x, radius, local_sides in profile:
        s = local_sides or sides
        ring_sides.append(s)
        vertices.extend(ring(x, radius, s, axis_center, axis, u, v))

    faces = []
    offset = np.cumsum([0] + ring_sides)
    for i in range(len(profile) - 1):
        s0 = ring_sides[i]
        s1 = ring_sides[i + 1]
        # Connect rings by sampling both at the max angular resolution. This lets
        # a hex nut connect cleanly to round collars.
        steps = max(s0, s1)
        for j in range(steps):
            a0 = int(np.floor(j * s0 / steps))
            a1 = int(np.floor((j + 1) * s0 / steps)) % s0
            b0 = int(np.floor(j * s1 / steps))
            b1 = int(np.floor((j + 1) * s1 / steps)) % s1
            faces.append([offset[i] + a0, offset[i + 1] + b0, offset[i + 1] + b1])
            faces.append([offset[i] + a0, offset[i + 1] + b1, offset[i] + a1])

    start_center = len(vertices)
    end_center = start_center + 1
    x0 = profile[0][0]
    x1 = profile[-1][0]
    vertices.append(axis_center + axis * x0)
    vertices.append(axis_center + axis * x1)
    for j in range(ring_sides[0]):
        faces.append([start_center, offset[0] + (j + 1) % ring_sides[0], offset[0] + j])
    last = len(profile) - 1
    for j in range(ring_sides[last]):
        faces.append([end_center, offset[last] + j, offset[last] + (j + 1) % ring_sides[last]])

    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)


def add_material_obj(mesh: trimesh.Trimesh, out_obj: Path) -> None:
    # Simple single-material OBJ for broad compatibility.
    mesh.export(out_obj)
    mtl = out_obj.with_suffix(".mtl")
    text = out_obj.read_text(encoding="utf-8", errors="ignore")
    if "mtllib" not in text[:200]:
        out_obj.write_text(f"mtllib {mtl.name}\nusemtl dark_metal\n" + text, encoding="utf-8")
    mtl.write_text(
        "newmtl dark_metal\n"
        "Ka 0.25 0.23 0.20\n"
        "Kd 0.38 0.36 0.32\n"
        "Ks 0.75 0.70 0.60\n"
        "Ns 80\n",
        encoding="utf-8",
    )


def apply_vertex_colors(mesh: trimesh.Trimesh, axis: np.ndarray, length: float) -> None:
    t = (mesh.vertices @ axis) / length + 0.5
    colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    colors[:] = [58, 54, 45, 255]  # dark steel needle shaft

    def paint(lo, hi, rgba):
        mask = (t >= lo) & (t <= hi)
        colors[mask] = rgba

    paint(0.705, 0.742, [28, 28, 26, 255])
    paint(0.756, 0.815, [205, 169, 96, 255])
    paint(0.830, 0.852, [62, 58, 50, 255])
    paint(0.865, 0.887, [185, 178, 164, 255])
    paint(0.900, 0.915, [54, 50, 44, 255])
    paint(0.928, 0.957, [35, 35, 33, 255])
    paint(0.972, 1.000, [78, 74, 68, 255])
    mesh.visual.vertex_colors = colors


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # File units follow the previous reconstruction: total length about 0.10.
    # The RGB frames show a long straight needle shaft plus a structured coaxial tail.
    length = 0.1008
    L = length
    rod_r = 0.00062

    # x runs from tip (-L/2) to structured tail (+L/2).
    # Each item is (x, radius, sides). sides=6 approximates the hex-looking nut.
    raw = [
        (0.000, 0.00010, 48),  # very fine pointed tip
        (0.018, rod_r, 48),
        (0.690, rod_r, 48),  # long straight shaft
        (0.705, 0.00105, 48),  # small black collar before the tail
        (0.725, 0.00128, 48),
        (0.742, 0.00110, 48),
        (0.756, 0.00172, 48),  # pale sleeve/nut region
        (0.815, 0.00185, 48),
        (0.830, 0.00120, 48),  # narrow waist
        (0.852, 0.00118, 48),
        (0.865, 0.00165, 48),  # bright ring
        (0.887, 0.00185, 48),
        (0.900, 0.00142, 48),
        (0.915, 0.00142, 48),
        (0.928, 0.00190, 48),  # dark rear ring
        (0.957, 0.00205, 48),
        (0.972, 0.00155, 48),
        (0.988, 0.00195, 48),  # short rear cap
        (1.000, 0.00165, 48),
    ]
    profile = [((t - 0.5) * L, r, s) for t, r, s in raw]

    mesh = make_lathe(profile)
    # Rotate the default x-axis model to match the previous scan/reconstruction axis roughly.
    target_axis = np.array([0.999703580791627, -0.014713206881434883, 0.01939773429199522])
    rot = trimesh.geometry.align_vectors(np.array([1.0, 0.0, 0.0]), target_axis)
    mesh.apply_transform(rot)
    apply_vertex_colors(mesh, target_axis, L)

    base = OUT_DIR / "needle_structured_tail_reconstruction"
    mesh.export(base.with_suffix(".ply"))
    mesh.export(base.with_suffix(".stl"))
    mesh.export(base.with_suffix(".glb"))
    add_material_obj(mesh, base.with_suffix(".obj"))

    print(f"vertices={len(mesh.vertices)}")
    print(f"faces={len(mesh.faces)}")
    print(f"bounds={mesh.bounds.tolist()}")
    print(f"extents={mesh.extents.tolist()}")
    print(f"watertight={mesh.is_watertight}")
    print(f"out={base.with_suffix('.obj').resolve()}")


if __name__ == "__main__":
    main()
