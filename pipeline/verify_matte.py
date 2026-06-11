"""Numeric QC gate for a Clip Package: fails on empty frames, alpha touching any
frame border, and per-frame alpha-area spikes; optionally on dark-opaque-pixel
outliers (toy/wand contamination in non-play clips).

Usage: python -B pipeline/verify_matte.py --clip clips_incoming/<name> [--check-dark]
Exit code 0 = all enabled checks pass, 1 = at least one failure.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

EMPTY_ALPHA_THRESHOLD = 64
BORDER_ALPHA_THRESHOLD = 32
AREA_RATIO_LOW = 0.65
AREA_RATIO_HIGH = 1.35
DARK_ALPHA_THRESHOLD = 64
DARK_RGB_MAX = 100
DARK_OUTLIER_FACTOR = 3.0


def frame_stats(rgba: np.ndarray) -> tuple[int, int, int]:
    alpha = rgba[..., 3]
    area = int(np.count_nonzero(alpha > EMPTY_ALPHA_THRESHOLD))
    border = np.concatenate([alpha[0, :], alpha[-1, :], alpha[1:-1, 0], alpha[1:-1, -1]])
    border_contact = int(np.count_nonzero(border >= BORDER_ALPHA_THRESHOLD))
    rgb_max = rgba[..., :3].max(axis=2)
    dark_opaque = int(np.count_nonzero((alpha > DARK_ALPHA_THRESHOLD) & (rgb_max < DARK_RGB_MAX)))
    return area, border_contact, dark_opaque


def main() -> None:
    parser = argparse.ArgumentParser(description="numeric QC gate for a clip package")
    parser.add_argument("--clip", type=Path, required=True, help="clip package dir")
    parser.add_argument("--check-dark", action="store_true",
                        help="fail on dark-opaque-pixel outliers (use for non-play clips)")
    parser.add_argument("--no-check-area", action="store_true",
                        help="report area stats but do not fail on spikes (clips whose "
                             "motion genuinely changes silhouette area, e.g. rear-ups)")
    args = parser.parse_args()

    manifest = json.loads((args.clip / "manifest.json").read_text())
    frame_paths = sorted((args.clip / "frames").glob("*.png"))
    if len(frame_paths) != manifest["frameCount"]:
        raise SystemExit(f"FAIL frame count: manifest says {manifest['frameCount']}, "
                         f"{len(frame_paths)} PNGs on disk")

    areas, borders, darks = [], [], []
    for path in frame_paths:
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        area, border_contact, dark_opaque = frame_stats(rgba)
        areas.append(area)
        borders.append(border_contact)
        darks.append(dark_opaque)

    area_median = float(np.median(areas))
    dark_median = float(np.median(darks))
    dark_outlier_floor = max(DARK_OUTLIER_FACTOR * dark_median, dark_median + 1000.0)

    failures: list[str] = []
    for i, (area, border_contact, dark_opaque) in enumerate(zip(areas, borders, darks)):
        if area == 0:
            failures.append(f"frame {i:04d}: EMPTY (0 px above alpha {EMPTY_ALPHA_THRESHOLD})")
        if border_contact > 0:
            failures.append(f"frame {i:04d}: BORDER CONTACT "
                            f"({border_contact} border px >= alpha {BORDER_ALPHA_THRESHOLD})")
        ratio = area / area_median if area_median > 0 else 0.0
        if not args.no_check_area and area > 0 and not AREA_RATIO_LOW <= ratio <= AREA_RATIO_HIGH:
            failures.append(f"frame {i:04d}: AREA SPIKE ({area} px, {ratio:.2f}x median "
                            f"{area_median:.0f})")
        if args.check_dark and dark_opaque > dark_outlier_floor:
            failures.append(f"frame {i:04d}: DARK-OPAQUE OUTLIER ({dark_opaque} px, median "
                            f"{dark_median:.0f}, floor {dark_outlier_floor:.0f})")

    print(f"{manifest['name']}: {len(frame_paths)} frames @ {manifest['pixelSize']}")
    print(f"  area px: min {min(areas)}  median {area_median:.0f}  max {max(areas)}  "
          f"(ratio range {min(areas) / area_median:.2f}-{max(areas) / area_median:.2f})")
    print(f"  border-contact px: max {max(borders)} across all frames")
    print(f"  dark-opaque px: min {min(darks)}  median {dark_median:.0f}  max {max(darks)}"
          f"{'  [enforced]' if args.check_dark else '  [report only]'}")
    if failures:
        print(f"FAIL ({len(failures)}):")
        for line in failures:
            print(f"  {line}")
        raise SystemExit(1)
    print("PASS: no empty frames, no border contact"
          + (" (area check skipped)" if args.no_check_area else ", no area spikes")
          + (", no dark-opaque outliers" if args.check_dark else ""))


main()
