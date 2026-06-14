import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SRC_DIR = Path("model/未命名对象 3_export/images_heic")
OUT_DIR = Path("model/fixed_unnamed_object_3/rgb_review")


def convert_one(src: Path, dst: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-map",
        "0:48",
        "-vf",
        "transpose=2,scale=360:-1",
        str(dst),
    ]
    return subprocess.run(cmd, check=False).returncode == 0


def make_sheet(images: list[Path], out: Path, cols: int = 5) -> None:
    thumbs = []
    for p in images:
        im = Image.open(p).convert("RGB")
        canvas = Image.new("RGB", (im.width, im.height + 24), "white")
        canvas.paste(im, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, im.height + 4), p.stem.replace("thumb_", ""), fill=(0, 0, 0))
        thumbs.append(canvas)

    rows = (len(thumbs) + cols - 1) // cols
    w = cols * thumbs[0].width
    h = rows * thumbs[0].height
    sheet = Image.new("RGB", (w, h), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * im.width, (i // cols) * im.height))
    sheet.save(out)


def crop_tail_candidates(images: list[Path], out: Path) -> None:
    crops = []
    for p in images:
        im = Image.open(p).convert("RGB")
        # In these frames the needle is near the center; crop the lower-center area
        # where the structured tail appears in most views.
        w, h = im.size
        crop = im.crop((int(w * 0.32), int(h * 0.38), int(w * 0.68), int(h * 0.88)))
        crop = crop.resize((crop.width * 2, crop.height * 2))
        canvas = Image.new("RGB", (crop.width, crop.height + 24), "white")
        canvas.paste(crop, (0, 0))
        ImageDraw.Draw(canvas).text((6, crop.height + 4), p.stem.replace("thumb_", ""), fill=(0, 0, 0))
        crops.append(canvas)

    cols = 5
    rows = (len(crops) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * crops[0].width, rows * crops[0].height), "white")
    for i, im in enumerate(crops):
        sheet.paste(im, ((i % cols) * im.width, (i // cols) * im.height))
    sheet.save(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    heics = [p for p in sorted(SRC_DIR.glob("*.HEIC")) if not p.name.startswith("._")]
    # Evenly sample the capture arc, with enough views to see the tail shape.
    idxs = sorted(set(round(i * (len(heics) - 1) / 19) for i in range(20)))
    thumbs = []
    for idx in idxs:
        src = heics[idx]
        dst = OUT_DIR / f"thumb_{src.stem}.png"
        if convert_one(src, dst):
            thumbs.append(dst)
    make_sheet(thumbs, OUT_DIR / "rgb_contact_sheet.png")
    crop_tail_candidates(thumbs, OUT_DIR / "tail_center_crops.png")
    print((OUT_DIR / "rgb_contact_sheet.png").resolve())
    print((OUT_DIR / "tail_center_crops.png").resolve())


if __name__ == "__main__":
    main()
