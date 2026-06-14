from pathlib import Path

from PIL import Image, ImageDraw


SRC = Path("model/image_implement")
OUT = Path("model/fixed_unnamed_object_3/reference_image_implement_sheet.png")


def main() -> None:
    files = sorted(SRC.glob("*.jpg"))
    thumbs = []
    for p in files:
        im = Image.open(p).convert("RGB")
        im.thumbnail((420, 300), Image.LANCZOS)
        canvas = Image.new("RGB", (420, 330), "white")
        canvas.paste(im, ((420 - im.width) // 2, 0))
        ImageDraw.Draw(canvas).text((8, 306), p.name[:12], fill=(0, 0, 0))
        thumbs.append(canvas)

    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 420, rows * 330), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 420, (i // cols) * 330))
    sheet.save(OUT)
    print(OUT.resolve())


if __name__ == "__main__":
    main()
