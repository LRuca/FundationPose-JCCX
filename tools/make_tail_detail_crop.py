from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    src = Path("model/fixed_unnamed_object_3/rgb_review/full_try_complex.png")
    out = Path("model/fixed_unnamed_object_3/rgb_review/tail_detail_00070.png")
    im = Image.open(src).convert("RGB")
    # Crop around the structured tail in frame 00070 after full-frame conversion.
    crop = im.crop((470, 330, 700, 455))
    crop = crop.resize((crop.width * 4, crop.height * 4))
    draw = ImageDraw.Draw(crop)
    draw.rectangle((6, 6, crop.width - 7, crop.height - 7), outline=(255, 220, 0), width=4)
    crop.save(out)
    print(out.resolve())


if __name__ == "__main__":
    main()
