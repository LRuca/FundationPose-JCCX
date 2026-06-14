import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps_model_fix"))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402


OUT_DIR = Path("model/fixed_unnamed_object_3")


def color_mesh(mesh: trimesh.Trimesh, rgba) -> trimesh.Trimesh:
    mesh.visual.vertex_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def orient_z_to_x(mesh: trimesh.Trimesh, x_center: float) -> trimesh.Trimesh:
    mesh.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0]))
    mesh.apply_translation([x_center, 0, 0])
    return mesh


def cylinder_between(x0, x1, radius, sections, rgba):
    mesh = trimesh.creation.cylinder(radius=radius, height=x1 - x0, sections=sections)
    orient_z_to_x(mesh, (x0 + x1) * 0.5)
    return color_mesh(mesh, rgba)


def sphere_at(x, radius, rgba):
    mesh = trimesh.creation.uv_sphere(radius=radius, count=[24, 12])
    mesh.apply_translation([x, 0, 0])
    return color_mesh(mesh, rgba)


def lathe_profile(profile, sections, rgba):
    vertices = []
    faces = []
    for x, radius in profile:
        for j in range(sections):
            a = 2 * np.pi * j / sections
            vertices.append([x, radius * np.cos(a), radius * np.sin(a)])
    for i in range(len(profile) - 1):
        for j in range(sections):
            j2 = (j + 1) % sections
            a = i * sections + j
            b = (i + 1) * sections + j
            c = (i + 1) * sections + j2
            d = i * sections + j2
            faces.append([a, b, c])
            faces.append([a, c, d])
    start = len(vertices)
    end = start + 1
    vertices.append([profile[0][0], 0, 0])
    vertices.append([profile[-1][0], 0, 0])
    for j in range(sections):
        faces.append([start, (j + 1) % sections, j])
        base = (len(profile) - 1) * sections
        faces.append([end, base + j, base + (j + 1) % sections])
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
    return color_mesh(mesh, rgba)


def engrave_flat_marks(hex_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # Add shallow dark rectangular patches on alternate hex faces to mimic printed/etched marks.
    # They are tiny raised plates, not boolean cuts, so the mesh remains robust.
    patches = []
    for y_sign in [-1, 1]:
        patch = trimesh.creation.box(extents=[0.010, 0.00008, 0.0010])
        patch.apply_translation([-0.004, y_sign * 0.00198, 0.00015])
        patches.append(color_mesh(patch, [55, 55, 52, 255]))
    return trimesh.util.concatenate([hex_mesh] + patches)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    L = 0.1008
    x_tip = -0.5 * L
    x_tail = 0.5 * L

    dark = [42, 42, 38, 255]
    black = [22, 22, 21, 255]
    steel = [170, 166, 154, 255]
    bright = [212, 207, 192, 255]
    warm = [146, 122, 74, 255]

    parts = []
    parts.append(lathe_profile([(x_tip, 0.00008), (x_tip + 0.004, 0.00042), (0.014, 0.00058)], 48, dark))
    parts.append(cylinder_between(0.013, 0.018, 0.00058, 48, dark))
    parts.append(sphere_at(0.0188, 0.00116, steel))
    parts.append(lathe_profile([(0.019, 0.00078), (0.022, 0.00138), (0.026, 0.00168)], 48, steel))

    # Main hexagonal sleeve visible in the new references.
    hex_body = cylinder_between(0.025, 0.0345, 0.00220, 6, steel)
    hex_body.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(30), [1, 0, 0]))
    parts.append(hex_body)

    # Short cylindrical neck and rings after the hex sleeve.
    parts.append(cylinder_between(0.0335, 0.0390, 0.00135, 48, steel))
    parts.append(cylinder_between(0.0382, 0.0402, 0.00182, 48, bright))
    parts.append(cylinder_between(0.0404, 0.0424, 0.00120, 48, black))
    parts.append(cylinder_between(0.0420, 0.0441, 0.00186, 48, steel))
    parts.append(cylinder_between(0.0442, 0.0461, 0.00125, 48, black))
    parts.append(cylinder_between(0.0457, 0.0477, 0.00170, 48, steel))

    # Rear button-like cap: thin stem, wide flat disk, bevel, shallow front groove.
    parts.append(cylinder_between(0.0477, 0.0490, 0.00110, 48, black))
    parts.append(cylinder_between(0.0486, 0.0499, 0.00168, 48, steel))
    parts.append(lathe_profile([(0.0496, 0.00170), (0.0506, 0.00265), (0.0527, 0.00265), (0.0536, 0.00205)], 64, bright))
    parts.append(cylinder_between(0.0528, 0.0535, 0.00120, 48, black))
    parts.append(cylinder_between(0.0532, 0.0540, 0.00175, 48, steel))

    mesh = trimesh.util.concatenate(parts)
    target_axis = np.array([0.999703580791627, -0.014713206881434883, 0.01939773429199522])
    mesh.apply_transform(trimesh.geometry.align_vectors([1, 0, 0], target_axis))
    mesh.process(validate=True)

    base = OUT_DIR / "needle_structured_tail_reconstruction_v2"
    mesh.export(base.with_suffix(".ply"))
    mesh.export(base.with_suffix(".stl"))
    mesh.export(base.with_suffix(".glb"))
    mesh.export(base.with_suffix(".obj"))

    print(f"vertices={len(mesh.vertices)}")
    print(f"faces={len(mesh.faces)}")
    print(f"components={len(mesh.split(only_watertight=False))}")
    print(f"watertight={mesh.is_watertight}")
    print(f"bounds={mesh.bounds.tolist()}")
    print(f"out={base.with_suffix('.glb').resolve()}")


if __name__ == "__main__":
    main()
