import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps_model_fix"))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402


OUT_DIR = Path("model/fixed_unnamed_object_3")


def color_mesh(mesh, rgba):
    mesh.visual.vertex_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def z_to_x(mesh, x_center):
    mesh.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0]))
    mesh.apply_translation([x_center, 0, 0])
    return mesh


def cyl(x0, x1, r, sections, color):
    mesh = trimesh.creation.cylinder(radius=r, height=x1 - x0, sections=sections)
    z_to_x(mesh, (x0 + x1) * 0.5)
    return color_mesh(mesh, color)


def box(x0, x1, y, z, color):
    mesh = trimesh.creation.box(extents=[x1 - x0, y, z])
    mesh.apply_translation([(x0 + x1) * 0.5, 0, 0])
    return color_mesh(mesh, color)


def rounded_square_prism(x0, x1, half, radius, color, samples=5, angle=0.0):
    pts = []
    centers = [
        (half - radius, half - radius, 0.0),
        (-half + radius, half - radius, np.pi / 2),
        (-half + radius, -half + radius, np.pi),
        (half - radius, -half + radius, 3 * np.pi / 2),
    ]
    for cy, cz, start in centers:
        for i in range(samples + 1):
            a = start + i * (np.pi / 2) / samples
            pts.append([cy + radius * np.cos(a), cz + radius * np.sin(a)])
    pts = np.asarray(pts)

    vertices = []
    for x in [x0, x1]:
        for y, z in pts:
            vertices.append([x, y, z])

    n = len(pts)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, n + i, n + j])
        faces.append([i, n + j, j])
    start_center = len(vertices)
    end_center = start_center + 1
    vertices.append([x0, 0, 0])
    vertices.append([x1, 0, 0])
    for i in range(n):
        j = (i + 1) % n
        faces.append([start_center, j, i])
        faces.append([end_center, n + i, n + j])

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
    if angle:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, [1, 0, 0]))
    return color_mesh(mesh, color)


def sphere(x, r, color):
    mesh = trimesh.creation.uv_sphere(radius=r, count=[24, 12])
    mesh.apply_translation([x, 0, 0])
    return color_mesh(mesh, color)


def lathe(profile, sections, color):
    vertices = []
    faces = []
    for x, r in profile:
        for j in range(sections):
            a = 2 * np.pi * j / sections
            vertices.append([x, r * np.cos(a), r * np.sin(a)])
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
    return color_mesh(mesh, color)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The new reference shows a long straight needle and a compact, mostly square/hex tail.
    L = 0.1008
    x_tip = -0.5 * L

    dark = [48, 49, 46, 255]
    black = [28, 28, 26, 255]
    steel = [166, 165, 154, 255]
    bright = [216, 213, 202, 255]
    shadow = [88, 88, 82, 255]

    parts = []

    # Long rod: slightly rounded closed tip, constant diameter shaft.
    parts.append(lathe([(x_tip, 0.00012), (x_tip + 0.0022, 0.00043), (0.020, 0.00055)], 48, dark))
    parts.append(cyl(0.018, 0.0235, 0.00055, 48, dark))

    # Small bead just before the tail, clearly visible in the reference.
    parts.append(sphere(0.0242, 0.00104, bright))
    parts.append(cyl(0.0245, 0.0260, 0.00070, 48, shadow))

    # Tapered cone into the main square/hex sleeve.
    parts.append(lathe([(0.0258, 0.00075), (0.0292, 0.00155), (0.0320, 0.00185)], 48, steel))

    # Main faceted body: closer to a square/hex block than a round cylinder.
    # Use 6 sides but keep it short and compact.
    hex_body = cyl(0.0312, 0.0412, 0.00205, 6, steel)
    hex_body.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(30), [1, 0, 0]))
    parts.append(hex_body)

    # Short dark cylindrical waist after the faceted block.
    parts.append(cyl(0.0403, 0.0445, 0.00132, 48, shadow))

    # Two narrow rings and a dark groove, as seen just above the terminal cap.
    parts.append(cyl(0.0435, 0.0453, 0.00172, 48, bright))
    parts.append(cyl(0.0451, 0.0474, 0.00118, 48, black))
    parts.append(cyl(0.0470, 0.0488, 0.00166, 48, steel))

    # Final structure: keep the recessed groove before the bottom cap round.
    # Only the two raised blocks are rounded rectangles. Their sides are rotated
    # to align with the faceted sleeve, and their radial size matches the sleeve
    # flat-to-flat scale rather than forming an oversized square cap.
    cap_angle = np.deg2rad(30)
    cap_thick = 0.00048
    cap_half = 0.00178
    cap_radius = 0.00046
    parts.append(rounded_square_prism(0.04795, 0.04795 + cap_thick, cap_half, cap_radius, bright, samples=12, angle=cap_angle))
    parts.append(cyl(0.04855, 0.04918, 0.00108, 48, black))
    parts.append(rounded_square_prism(0.04928, 0.04928 + cap_thick, cap_half, cap_radius, bright, samples=12, angle=cap_angle))

    # Dark flat face inset on the final raised block, matching its orientation.
    parts.append(rounded_square_prism(0.04972, 0.04982, 0.00135, 0.00034, shadow, samples=12, angle=cap_angle))

    mesh = trimesh.util.concatenate(parts)
    target_axis = np.array([0.999703580791627, -0.014713206881434883, 0.01939773429199522])
    mesh.apply_transform(trimesh.geometry.align_vectors([1, 0, 0], target_axis))
    mesh.process(validate=True)

    base = OUT_DIR / "needle_structured_tail_reconstruction_v3"
    mesh.export(base.with_suffix(".ply"))
    mesh.export(base.with_suffix(".stl"))
    mesh.export(base.with_suffix(".glb"))
    mesh.export(base.with_suffix(".obj"))

    print(f"vertices={len(mesh.vertices)}")
    print(f"faces={len(mesh.faces)}")
    print(f"components={len(mesh.split(only_watertight=False))}")
    print(f"bounds={mesh.bounds.tolist()}")
    print(f"out={base.with_suffix('.glb').resolve()}")


if __name__ == "__main__":
    main()
