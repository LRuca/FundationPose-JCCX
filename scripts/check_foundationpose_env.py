import importlib

mods = [
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "cv2",
    "open3d",
    "trimesh",
    "pyrender",
    "pybind11",
    "nvdiffrast.torch",
    "pytorch3d",
    "common",
    "mycpp",
]

for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:
        print(f"{name}: IMPORT FAILED: {type(exc).__name__}: {exc}")

import torch

print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
