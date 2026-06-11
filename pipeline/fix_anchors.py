"""Recompute Clip Package anchors from solid alpha (CONTRACTS.md section 1).

The anchor is the ground-contact point in pixels: the bottom-center of the
bounding box of the union of alpha > 64 across all frames (x = rounded bbox
center, y = one past the lowest opaque row, matching the v2 production pass).

Usage: python -B pipeline/fix_anchors.py --clips clips_incoming [--only name ...]
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ALPHA_SOLID = 64


def solid_union_anchor(clip_dir: Path, frame_count: int) -> list[int]:
    frame_paths = sorted((clip_dir / "frames").glob("*.png"))
    if len(frame_paths) != frame_count:
        raise SystemExit(f"{clip_dir}: frameCount {frame_count} != "
                         f"{len(frame_paths)} PNGs on disk")
    union = None
    for path in frame_paths:
        alpha = np.asarray(Image.open(path).convert("RGBA"))[:, :, 3] > ALPHA_SOLID
        union = alpha if union is None else (union | alpha)
    ys, xs = np.where(union)
    if ys.size == 0:
        raise SystemExit(f"{clip_dir}: no pixels above alpha {ALPHA_SOLID} in any frame")
    return [round((int(xs.min()) + int(xs.max())) / 2), int(ys.max()) + 1]


def fix_clip(clip_dir: Path) -> None:
    manifest_path = clip_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    anchor = solid_union_anchor(clip_dir, manifest["frameCount"])
    old = manifest.get("anchor")
    manifest["anchor"] = anchor
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{manifest['name']}: anchor {old} -> {anchor} "
          f"(pixelSize {manifest['pixelSize']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="recompute anchors from solid alpha union")
    parser.add_argument("--clips", type=Path, required=True, help="clips root directory")
    parser.add_argument("--only", nargs="*", default=None,
                        help="clip names to process (default: every clip dir with a manifest)")
    args = parser.parse_args()

    if not args.clips.is_dir():
        raise SystemExit(f"clips directory not found: {args.clips}")
    if args.only:
        clip_dirs = [args.clips / name for name in args.only]
    else:
        clip_dirs = sorted(d for d in args.clips.iterdir()
                           if (d / "manifest.json").is_file())
    if not clip_dirs:
        raise SystemExit(f"no clip packages under {args.clips}")
    for clip_dir in clip_dirs:
        if not (clip_dir / "manifest.json").is_file():
            raise SystemExit(f"no manifest.json under {clip_dir}")
        fix_clip(clip_dir)


main()
