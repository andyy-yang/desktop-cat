"""Repair single-frame toy/wand strobes in a Clip Package: detect frames whose
dark-opaque-pixel count is an outlier, then replace each outlier frame's alpha
with the elementwise AND (per-pixel minimum) of its two neighbors' alphas,
applied to the frame's own RGB. Timing is preserved; the strobing silhouette
extension dies because the contaminating object is absent from both neighbors.

Where the contaminating object crosses IN FRONT of the cat (inside the
neighbor-AND silhouette), alpha replacement cannot remove it, so those RGB
pixels — dark here but dark in neither neighbor, which preserves eyes/nose —
are additionally inpainted from the mean of the two neighbors' RGB.

Usage: python -B pipeline/fix_strobe_frames.py --clip clips_incoming/<name>
Refuses to fix outliers at clip boundaries or with outlier neighbors (no
silent fallback) and exits 1 if outliers remain after the fix.
"""

import sys

sys.dont_write_bytecode = True

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DARK_ALPHA_THRESHOLD = 64
DARK_RGB_MAX = 100
DARK_OUTLIER_FACTOR = 3.0
DARK_OUTLIER_MIN_DELTA = 1000.0
INPAINT_DILATE_PX = 5


def dark_opaque_count(rgba: np.ndarray) -> int:
    alpha = rgba[..., 3]
    rgb_max = rgba[..., :3].max(axis=2)
    return int(np.count_nonzero((alpha > DARK_ALPHA_THRESHOLD) & (rgb_max < DARK_RGB_MAX)))


def find_outliers(counts: list[int]) -> tuple[list[int], float]:
    median = float(np.median(counts))
    floor = max(DARK_OUTLIER_FACTOR * median, median + DARK_OUTLIER_MIN_DELTA)
    return [i for i, c in enumerate(counts) if c > floor], floor


def main() -> None:
    parser = argparse.ArgumentParser(description="kill single-frame dark-object strobes")
    parser.add_argument("--clip", type=Path, required=True, help="clip package dir")
    args = parser.parse_args()

    frame_paths = sorted((args.clip / "frames").glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"no PNG frames in {args.clip / 'frames'}")
    frames = [np.asarray(Image.open(p).convert("RGBA")).copy() for p in frame_paths]
    counts = [dark_opaque_count(f) for f in frames]
    outliers, floor = find_outliers(counts)
    print(f"dark-opaque counts: median {np.median(counts):.0f}, outlier floor {floor:.0f}")
    if not outliers:
        raise SystemExit("no outlier frames found; nothing to fix")

    for i in outliers:
        if i == 0 or i == len(frames) - 1:
            raise SystemExit(f"frame {i:04d} is an outlier at the clip boundary; "
                             f"cannot AND two neighbors")
        if i - 1 in outliers or i + 1 in outliers:
            raise SystemExit(f"frame {i:04d} has an outlier neighbor; AND of neighbors "
                             f"would propagate contamination")
        print(f"  frame {i:04d}: {counts[i]} dark-opaque px -> replacing alpha with "
              f"min(alpha[{i - 1:04d}], alpha[{i + 1:04d}])")
        prev, nxt = frames[i - 1], frames[i + 1]
        repaired = frames[i].copy()
        repaired[..., 3] = np.minimum(prev[..., 3], nxt[..., 3])

        # Inpaint contamination crossing in front of the cat: pixels dark-opaque
        # here but dark in NEITHER neighbor (true features like eyes/nose are
        # dark in both) get the mean of the neighbors' RGB.
        dark_own = (repaired[..., 3] > DARK_ALPHA_THRESHOLD) & \
                   (repaired[..., :3].max(axis=2) < DARK_RGB_MAX)
        dark_prev = prev[..., :3].max(axis=2) < DARK_RGB_MAX
        dark_next = nxt[..., :3].max(axis=2) < DARK_RGB_MAX
        contaminated = dark_own & ~(dark_prev & dark_next)
        kernel = np.ones((2 * INPAINT_DILATE_PX + 1, 2 * INPAINT_DILATE_PX + 1), np.uint8)
        contaminated = cv2.dilate(contaminated.astype(np.uint8), kernel).astype(bool)
        neighbor_rgb = ((prev[..., :3].astype(np.uint16) +
                         nxt[..., :3].astype(np.uint16)) // 2).astype(np.uint8)
        repaired[..., :3][contaminated] = neighbor_rgb[contaminated]
        print(f"  frame {i:04d}: inpainted {int(np.count_nonzero(contaminated))} px "
              f"of in-silhouette contamination from neighbor RGB")

        frames[i] = repaired
        Image.fromarray(repaired).save(frame_paths[i])

    new_counts = [dark_opaque_count(f) for f in frames]
    remaining, new_floor = find_outliers(new_counts)
    for i in outliers:
        print(f"  frame {i:04d}: dark-opaque px {counts[i]} -> {new_counts[i]}")
    if remaining:
        raise SystemExit(f"FAIL: outliers remain after fix: {remaining} "
                         f"(floor {new_floor:.0f})")
    print(f"PASS: {len(outliers)} strobe frame(s) repaired, no outliers remain")


main()
