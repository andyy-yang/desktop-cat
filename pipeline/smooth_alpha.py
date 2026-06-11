"""Motion-gated temporal alpha smoothing for a Clip Package, IN PLACE: fast cat
motion makes BiRefNet's per-frame mattes inconsistent, so edges shimmer frame
to frame. For each interior frame t a per-pixel motion mask (max-channel RGB
diff against EITHER neighbor above --threshold) gates a temporal median: where
the pixel is static, alpha_t = median(alpha_{t-1}, alpha_t, alpha_{t+1}); where
it moves, alpha_t is kept. Afterwards single-pixel holes in the binary
(alpha > 24) mask are filled with a 3x3 grayscale close restricted to those
hole pixels. RGB, frame count and frame sizes are untouched.

Originals are backed up to <backup-root>/<clip_name>/ first; the script refuses
to run if that backup already exists (restore or remove it explicitly to
re-run). Reports mean per-pixel |alpha_t - alpha_{t+1}| over static pixels
before and after, both over the full frame and restricted to pixels carrying
alpha in either frame of the pair.

Usage: python -B pipeline/smooth_alpha.py --clip clips/<name> [--threshold 28]
       [--backup-root work/backup_smooth]
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MOTION_THRESHOLD = 28
BINARY_ALPHA_THRESHOLD = 24


def median3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))


def motion_mask(cur: np.ndarray, prev: np.ndarray, nxt: np.ndarray,
                threshold: int) -> np.ndarray:
    d_prev = cv2.absdiff(cur[..., :3], prev[..., :3]).max(axis=2)
    d_next = cv2.absdiff(cur[..., :3], nxt[..., :3]).max(axis=2)
    return (d_prev > threshold) | (d_next > threshold)


def smooth_alphas(frames: list[np.ndarray], threshold: int) -> list[np.ndarray]:
    out = [f.copy() for f in frames]
    for t in range(1, len(frames) - 1):
        prev, cur, nxt = frames[t - 1], frames[t], frames[t + 1]
        moving = motion_mask(cur, prev, nxt, threshold)
        med = median3(prev[..., 3], cur[..., 3], nxt[..., 3])
        out[t][..., 3] = np.where(moving, cur[..., 3], med)
    return out


def fill_single_pixel_holes(rgba: np.ndarray) -> int:
    alpha = rgba[..., 3]
    binary = (alpha > BINARY_ALPHA_THRESHOLD).astype(np.uint8)
    ring = np.ones((3, 3), np.uint8)
    ring[1, 1] = 0
    neighbor_count = cv2.filter2D(binary, -1, ring, borderType=cv2.BORDER_CONSTANT)
    holes = (binary == 0) & (neighbor_count == 8)
    n = int(np.count_nonzero(holes))
    if n:
        closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        alpha[holes] = closed[holes]
    return n


def static_flicker(frames: list[np.ndarray], threshold: int) -> tuple[float, float]:
    """Mean per-pixel |alpha_t - alpha_{t+1}| over static-RGB pixels across all
    consecutive pairs: (full-frame mean, mean restricted to pixels with alpha
    in either frame of the pair)."""
    total = total_px = active_total = active_px = 0
    for t in range(len(frames) - 1):
        a, b = frames[t], frames[t + 1]
        static = cv2.absdiff(a[..., :3], b[..., :3]).max(axis=2) <= threshold
        d = cv2.absdiff(a[..., 3], b[..., 3]).astype(np.int64)
        total += int(d[static].sum())
        total_px += int(np.count_nonzero(static))
        active = static & ((a[..., 3] > 0) | (b[..., 3] > 0))
        active_total += int(d[active].sum())
        active_px += int(np.count_nonzero(active))
    return total / total_px, active_total / active_px


def backup_clip(clip_dir: Path, backup_root: Path, frame_paths: list[Path]) -> Path:
    backup_dir = backup_root / clip_dir.name
    if backup_dir.exists():
        raise SystemExit(f"backup {backup_dir} already exists; restore or remove it "
                         f"before re-running (refusing to overwrite originals' backup)")
    (backup_dir / "frames").mkdir(parents=True)
    shutil.copy2(clip_dir / "manifest.json", backup_dir / "manifest.json")
    for p in frame_paths:
        shutil.copy2(p, backup_dir / "frames" / p.name)
    return backup_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="motion-gated temporal alpha smoothing, in place")
    parser.add_argument("--clip", type=Path, required=True, help="clip package dir")
    parser.add_argument("--threshold", type=int, default=MOTION_THRESHOLD,
                        help="max-channel RGB diff above which a pixel counts as moving")
    parser.add_argument("--backup-root", type=Path, default=Path("work/backup_smooth"),
                        help="originals are copied to <backup-root>/<clip_name>/")
    args = parser.parse_args()

    manifest = json.loads((args.clip / "manifest.json").read_text())
    frame_paths = sorted((args.clip / "frames").glob("*.png"))
    if len(frame_paths) != manifest["frameCount"]:
        raise SystemExit(f"frame count mismatch: manifest says {manifest['frameCount']}, "
                         f"{len(frame_paths)} PNGs on disk")
    if len(frame_paths) < 3:
        raise SystemExit("need at least 3 frames for temporal smoothing")

    frames = [np.asarray(Image.open(p).convert("RGBA")).copy() for p in frame_paths]
    sizes = {f.shape for f in frames}
    if len(sizes) != 1:
        raise SystemExit(f"inconsistent frame sizes: {sorted(sizes)}")

    before_full, before_active = static_flicker(frames, args.threshold)
    smoothed = smooth_alphas(frames, args.threshold)
    holes_filled = sum(fill_single_pixel_holes(f) for f in smoothed)
    after_full, after_active = static_flicker(smoothed, args.threshold)

    for orig, new in zip(frames, smoothed):
        if new.shape != orig.shape:
            raise SystemExit("frame size changed during smoothing (bug)")
        if not np.array_equal(new[..., :3], orig[..., :3]):
            raise SystemExit("RGB channels changed during smoothing (bug)")

    backup_dir = backup_clip(args.clip, args.backup_root, frame_paths)
    for path, frame in zip(frame_paths, smoothed):
        Image.fromarray(frame, "RGBA").save(path)

    changed = sum(int(not np.array_equal(n[..., 3], o[..., 3]))
                  for n, o in zip(smoothed, frames))
    print(f"{manifest['name']}: {len(frames)} frames, threshold {args.threshold}, "
          f"backup {backup_dir}")
    print(f"  alpha changed in {changed} frames; {holes_filled} single-pixel holes filled")
    print(f"  static flicker (full frame):   {before_full:.4f} -> {after_full:.4f} "
          f"({100.0 * (1.0 - after_full / before_full):.1f}% drop)")
    print(f"  static flicker (active px):    {before_active:.4f} -> {after_active:.4f} "
          f"({100.0 * (1.0 - after_active / before_active):.1f}% drop)")


main()
