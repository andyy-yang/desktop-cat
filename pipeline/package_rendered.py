"""Package pre-rendered RGBA frames (Blender bakes, generated clips) into a
Clip Package: union-bbox crop, optional mirror, manifest with anchor.
No matting — input already has alpha.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

CATEGORIES = ["sleep", "idle", "groom", "play", "walk", "reaction"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--loop", default="pingpong", choices=["pingpong", "forward", "once"])
    parser.add_argument("--mirror", action="store_true",
                        help="flip horizontally (e.g. walk right -> walk left)")
    parser.add_argument("--target-width", type=int, default=0,
                        help="resize cat to this width in px (0 = keep)")
    args = parser.parse_args()

    paths = sorted(args.frames.glob("*.png"))
    if not paths:
        raise SystemExit(f"no frames in {args.frames}")
    frames = [np.asarray(Image.open(p).convert("RGBA")) for p in paths]

    union = np.zeros(frames[0].shape[:2], dtype=bool)
    for f in frames:
        union |= f[..., 3] > 16
    ys, xs = np.where(union)
    pad = 8
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, union.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, union.shape[0])

    crop_w = x1 - x0
    scale = (args.target_width / crop_w) if args.target_width else 1.0
    out_size = (int(crop_w * scale), int((y1 - y0) * scale))

    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for n, f in enumerate(frames):
        img = Image.fromarray(f[y0:y1, x0:x1])
        if scale != 1.0:
            img = img.resize(out_size, Image.LANCZOS)
        if args.mirror:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(frames_dir / f"{n:04d}.png")

    anchor_x = ((xs.min() + xs.max()) / 2 - x0) * scale
    if args.mirror:
        anchor_x = out_size[0] - anchor_x
    manifest = {
        "name": args.name,
        "category": args.category,
        "fps": args.fps,
        "frameCount": len(frames),
        "loop": args.loop,
        "pixelSize": list(out_size),
        "displayScale": 2.0,
        "model": "blender_bake",
        "sourceVideo": str(args.frames),
        "anchor": [int(round(anchor_x)), min(int(round((ys.max() + 1 - y0) * scale)), out_size[1])],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"packaged {args.out}: {len(frames)} frames @ {out_size}px mirror={args.mirror}")


main()
