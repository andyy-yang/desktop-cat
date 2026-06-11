"""Generate the OverlayCat icon source set: white cat silhouette on a dark circle.

Usage: python -B scripts/make_icon.py OUTPUT_ICONSET_DIR

Writes the icon_NxN.png / icon_NxN@2x.png set that `iconutil -c icns` expects.
Pure Pillow drawing; no files are read.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

MASTER_SIZE = 1024
CIRCLE_COLOR = (43, 43, 51, 255)
CAT_COLOR = (255, 255, 255, 255)
ICONSET_SIZES = (16, 32, 128, 256, 512)


def draw_master() -> Image.Image:
    img = Image.new("RGBA", (MASTER_SIZE, MASTER_SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((64, 64, 960, 960), fill=CIRCLE_COLOR)
    # ears (triangles overlapping the head top)
    d.polygon([(350, 500), (372, 290), (530, 400)], fill=CAT_COLOR)
    d.polygon([(674, 500), (652, 290), (494, 400)], fill=CAT_COLOR)
    # head
    d.ellipse((322, 380, 702, 760), fill=CAT_COLOR)
    # eyes and nose punched back in the circle colour
    d.ellipse((430, 510, 478, 590), fill=CIRCLE_COLOR)
    d.ellipse((546, 510, 594, 590), fill=CIRCLE_COLOR)
    d.polygon([(488, 642), (536, 642), (512, 682)], fill=CIRCLE_COLOR)
    return img


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_icon.py OUTPUT_ICONSET_DIR")
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    master = draw_master()
    for size in ICONSET_SIZES:
        master.resize((size, size), Image.Resampling.LANCZOS).save(
            out_dir / f"icon_{size}x{size}.png")
        master.resize((size * 2, size * 2), Image.Resampling.LANCZOS).save(
            out_dir / f"icon_{size}x{size}@2x.png")
    print(f"wrote iconset to {out_dir}")


if __name__ == "__main__":
    main()
