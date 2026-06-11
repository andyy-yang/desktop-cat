"""QC contact sheet for a Clip Package: 12 evenly-spaced frames composited over
a light (#f0f0f0) and a dark (#202020) checkerstrip — fur halos that survive
matting show against one or the other — plus the raw alpha channels of the
first/middle/last frames.

Usage: python -B pipeline/qc_clip.py --clip clips/sleep_curl [--out work/qc]
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LIGHT = (240, 240, 240)
LIGHT_ALT = (212, 212, 212)
DARK = (32, 32, 32)
DARK_ALT = (58, 58, 58)
SHEET_BG = (110, 110, 110)
LABEL_FG = (255, 255, 255)
SAMPLES = 12
THUMB_W = 240
CHECKER = 16
MARGIN = 8
LABEL_H = 14


def checkerstrip(size: tuple[int, int], color_a: tuple, color_b: tuple) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx // CHECKER + yy // CHECKER) % 2).astype(bool)
    pixels = np.empty((h, w, 3), dtype=np.uint8)
    pixels[~mask] = color_a
    pixels[mask] = color_b
    return Image.fromarray(pixels)


def build_sheet(frames: dict[int, Image.Image], sample_indices: list[int],
                alpha_indices: list[int]) -> Image.Image:
    src_w, src_h = next(iter(frames.values())).size
    th = max(int(round(src_h * THUMB_W / src_w)), 1)
    by_index = {i: f.resize((THUMB_W, th), Image.LANCZOS) for i, f in frames.items()}

    light_bg = checkerstrip((THUMB_W, th), LIGHT, LIGHT_ALT)
    dark_bg = checkerstrip((THUMB_W, th), DARK, DARK_ALT)

    cell_w, cell_h = THUMB_W + MARGIN, th + LABEL_H + MARGIN
    cols = len(sample_indices)
    sheet = Image.new("RGB", (MARGIN + cols * cell_w, MARGIN + 3 * cell_h), SHEET_BG)
    draw = ImageDraw.Draw(sheet)

    def place(cell: Image.Image, col: int, row: int, label: str) -> None:
        x, y = MARGIN + col * cell_w, MARGIN + row * cell_h
        sheet.paste(cell, (x, y))
        draw.text((x, y + th + 2), label, fill=LABEL_FG)

    for col, idx in enumerate(sample_indices):
        over_light = light_bg.copy()
        over_light.paste(by_index[idx], (0, 0), by_index[idx])
        over_dark = dark_bg.copy()
        over_dark.paste(by_index[idx], (0, 0), by_index[idx])
        place(over_light, col, 0, f"frame {idx:04d} / light")
        place(over_dark, col, 1, f"frame {idx:04d} / dark")

    for col, idx in enumerate(alpha_indices):
        alpha = by_index[idx].split()[3].convert("RGB")
        place(alpha, col, 2, f"alpha {idx:04d}")
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="render a QC contact sheet for a clip package")
    parser.add_argument("--clip", type=Path, required=True, help="clip package dir")
    parser.add_argument("--out", type=Path, default=Path("work/qc"), help="output dir")
    args = parser.parse_args()

    manifest_path = args.clip / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest.json in {args.clip}")
    manifest = json.loads(manifest_path.read_text())
    frame_paths = sorted((args.clip / "frames").glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"no PNG frames in {args.clip / 'frames'}")
    if len(frame_paths) != manifest["frameCount"]:
        raise SystemExit(f"{args.clip}: frameCount {manifest['frameCount']} != "
                         f"{len(frame_paths)} PNGs on disk")

    last = len(frame_paths) - 1
    sample_indices = sorted({int(round(i)) for i in np.linspace(0, last, SAMPLES)})
    alpha_indices = sorted({0, last // 2, last})
    load_indices = sorted(set(sample_indices) | set(alpha_indices))
    frames = {i: Image.open(frame_paths[i]).convert("RGBA") for i in load_indices}

    sheet = build_sheet(frames, sample_indices, alpha_indices)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{manifest['name']}_sheet.png"
    sheet.save(out_path)
    print(f"wrote {out_path} ({sheet.size[0]}x{sheet.size[1]}px, "
          f"{len(sample_indices)} sampled frames)")


main()
