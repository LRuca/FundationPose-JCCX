import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps_model_fix"))

from pxr import Usd, UsdGeom  # noqa: E402


def main() -> None:
    src = Path(r"model\未命名对象 3_raw_ObjectMaskOn(1).usdz")
    print(f"source={src.resolve()}")
    with zipfile.ZipFile(src) as zf:
        for info in zf.infolist():
            print(f"zip {info.filename} {info.file_size} bytes")

    stage = Usd.Stage.Open(str(src))
    if stage is None:
        raise RuntimeError("USD stage could not be opened")

    for prim in stage.Traverse():
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        print(f"mesh={prim.GetPath()}")
        print(f"points={len(points) if points else 0}")
        print(f"faces={len(counts) if counts else 0}")
        print(f"face_vertex_indices={len(indices) if indices else 0}")
        print(f"subdivisionScheme={mesh.GetSubdivisionSchemeAttr().Get()}")
        for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            value = pv.Get()
            try:
                n = len(value)
            except TypeError:
                n = "scalar"
            print(
                f"primvar={pv.GetName()} interpolation={pv.GetInterpolation()} "
                f"indices={pv.GetIndicesAttr().HasAuthoredValue()} values={n}"
            )


if __name__ == "__main__":
    main()
