import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps_model_fix"))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402


SRC = Path(r"model\未命名对象 3_raw_ObjectMaskOn(1).usdz")
OUT_DIR = Path("model") / "fixed_unnamed_object_3"


def load_usdz_mesh(path: Path):
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Cannot open USDZ: {path}")

    for prim in stage.Traverse():
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if not np.all(counts == 3):
            raise RuntimeError("This script expects a triangulated scan mesh.")
        faces = indices.reshape((-1, 3))

        uv = None
        uv_indices = None
        st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if st:
            uv = np.asarray(st.Get(), dtype=np.float64)
            authored = st.GetIndicesAttr().HasAuthoredValue()
            if authored:
                uv_indices = np.asarray(st.GetIndices(), dtype=np.int64).reshape((-1, 3))
            else:
                uv_indices = np.arange(len(indices), dtype=np.int64).reshape((-1, 3))
        return points, faces, uv, uv_indices
    raise RuntimeError("No UsdGeom.Mesh found in USDZ.")


def extract_texture(src: Path, out_dir: Path) -> Path | None:
    with zipfile.ZipFile(src) as zf:
        pngs = [name for name in zf.namelist() if name.lower().endswith(".png")]
        if not pngs:
            return None
        out = out_dir / "texture.png"
        with zf.open(pngs[0]) as inp, out.open("wb") as dst:
            shutil.copyfileobj(inp, dst)
        return out


def connected_component_mask(vertices: np.ndarray, faces: np.ndarray):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(faces)))
    sizes = np.array([len(c) for c in components], dtype=np.int64)
    order = np.argsort(sizes)[::-1]

    # Keep the main body and any substantial attached fragments; drop tiny scan speckles.
    min_faces = max(50, int(len(faces) * 0.002))
    keep = np.zeros(len(faces), dtype=bool)
    for idx in order:
        if sizes[idx] >= min_faces:
            keep[np.asarray(components[idx], dtype=np.int64)] = True
    return keep, sizes[order]


def smooth_vertices(vertices: np.ndarray, faces: np.ndarray, iterations: int = 4, alpha: float = 0.28):
    neighbors = [set() for _ in range(len(vertices))]
    for tri in faces:
        a, b, c = map(int, tri)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    out = vertices.copy()
    counts = np.array([len(n) for n in neighbors], dtype=np.int64)
    movable = counts > 2
    for _ in range(iterations):
        avg = out.copy()
        for i, ns in enumerate(neighbors):
            if movable[i]:
                avg[i] = out[list(ns)].mean(axis=0)
        out[movable] = (1.0 - alpha) * out[movable] + alpha * avg[movable]
    return out


def compact_mesh(vertices, faces):
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return vertices[used], remap[faces], used


def write_obj(path: Path, vertices, faces, uv, uv_indices, texture_name: str | None):
    mtl_name = path.with_suffix(".mtl").name
    with path.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("o fixed_unnamed_object_3\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        if uv is not None:
            for item in uv:
                f.write(f"vt {item[0]:.8f} {1.0 - item[1]:.8f}\n")
        f.write("usemtl scan_texture\n")
        for face_idx, tri in enumerate(faces):
            if uv is not None and uv_indices is not None:
                parts = [f"{int(v) + 1}/{int(t) + 1}" for v, t in zip(tri, uv_indices[face_idx])]
            else:
                parts = [str(int(v) + 1) for v in tri]
            f.write("f " + " ".join(parts) + "\n")

    with path.with_suffix(".mtl").open("w", encoding="utf-8") as f:
        f.write("newmtl scan_texture\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        if texture_name:
            f.write(f"map_Kd {texture_name}\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vertices, faces, uv, uv_indices = load_usdz_mesh(SRC)
    keep, sizes = connected_component_mask(vertices, faces)

    faces_kept = faces[keep]
    uv_indices_kept = uv_indices[keep] if uv_indices is not None else None
    compact_vertices, compact_faces, used = compact_mesh(vertices, faces_kept)
    smoothed_vertices = smooth_vertices(compact_vertices, compact_faces)

    # Remap face indices after compaction.
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    compact_faces = remap[faces_kept]

    texture = extract_texture(SRC, OUT_DIR)
    write_obj(
        OUT_DIR / "textured_cleaned.obj",
        smoothed_vertices,
        compact_faces,
        uv,
        uv_indices_kept,
        texture.name if texture else None,
    )

    mesh = trimesh.Trimesh(vertices=smoothed_vertices, faces=compact_faces, process=True)
    mesh.export(OUT_DIR / "cleaned_smoothed.ply")
    mesh.export(OUT_DIR / "cleaned_smoothed.stl")

    normals = mesh.face_normals
    non_planar = np.abs(normals[:, 1]) < 0.92
    non_planar_mesh = mesh.submesh([np.flatnonzero(non_planar)], append=True, repair=True)
    if len(non_planar_mesh.faces) > 0:
        non_planar_mesh.remove_unreferenced_vertices()
        non_planar_mesh.export(OUT_DIR / "object_without_flat_sheet.ply")
        non_planar_mesh.export(OUT_DIR / "object_without_flat_sheet.obj")
        parts = non_planar_mesh.split(only_watertight=False)
        if parts:
            main_part = max(parts, key=lambda item: len(item.faces))
            main_part.export(OUT_DIR / "object_without_flat_sheet_main.ply")
            main_part.export(OUT_DIR / "object_without_flat_sheet_main.obj")

    print(f"original_vertices={len(vertices)}")
    print(f"original_faces={len(faces)}")
    print(f"components_largest_first={sizes[:12].tolist()}")
    print(f"kept_faces={int(keep.sum())}")
    print(f"kept_vertices={len(smoothed_vertices)}")
    print(f"watertight={mesh.is_watertight}")
    print(f"euler_number={mesh.euler_number}")
    print(f"non_planar_faces={int(non_planar.sum())}")
    print(f"out_dir={OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
