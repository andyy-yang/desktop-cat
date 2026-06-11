"""Rank source/ videos as candidates for behavior clips.

Scores every video on: white-cat presence (the target cat is the only large
near-white object in this dataset), background darkness (dark backgrounds matte
best for white fur), and motion level (sleep loops want stillness). Writes a
ranked JSON report. Pure analysis — no files are modified.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

SAMPLE_FRAMES = 9
ANALYSIS_WIDTH = 480

# Near-white in HSV: low saturation, high value. The white cat reads ~S<70, V>140
# even in dim indoor light; walls/bedding are excluded later by the largest-component
# size + the background-darkness score rather than by threshold tuning.
WHITE_S_MAX = 70
WHITE_V_MIN = 140


@dataclass
class VideoScore:
    path: str
    duration_s: float
    frames_sampled: int
    white_blob_frac: float      # largest near-white component, fraction of frame (cat size proxy)
    white_presence: float       # fraction of sampled frames where the blob exceeds 2% of frame
    bg_darkness: float          # 1 - mean V of non-blob pixels / 255 (higher = darker bg = better)
    motion: float               # mean abs diff between consecutive samples on the blob's bbox
    sleep_spike_score: float    # composite for the Phase-0 sleeping-loop pick


def largest_white_component(frame_bgr: np.ndarray) -> tuple[float, np.ndarray]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] < WHITE_S_MAX) & (hsv[..., 2] > WHITE_V_MIN)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0.0, np.zeros_like(mask)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == idx).astype(np.uint8)
    frac = float(stats[idx, cv2.CC_STAT_AREA]) / mask.size
    return frac, blob


def score_video(path: Path) -> VideoScore | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total < SAMPLE_FRAMES:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, SAMPLE_FRAMES, dtype=int)
    frames, blob_fracs, bg_vals = [], [], []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        scale = ANALYSIS_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frac, blob = largest_white_component(frame)
        hsv_v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
        bg = hsv_v[blob == 0]
        frames.append((frame, blob))
        blob_fracs.append(frac)
        bg_vals.append(float(bg.mean()) if bg.size else 255.0)
    cap.release()
    if len(frames) < 3:
        return None

    diffs = []
    for (f1, b1), (f2, b2) in zip(frames, frames[1:]):
        union = ((b1 | b2) > 0)
        if union.sum() < 100:
            continue
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.int16)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.int16)
        diffs.append(float(np.abs(g1 - g2)[union].mean()))

    blob_frac = float(np.median(blob_fracs))
    presence = float(np.mean([f > 0.02 for f in blob_fracs]))
    bg_darkness = 1.0 - float(np.median(bg_vals)) / 255.0
    motion = float(np.median(diffs)) if diffs else 99.0

    # Sleep-spike composite: big visible cat, dark background, low motion.
    size_term = min(blob_frac / 0.15, 1.0)
    still_term = max(0.0, 1.0 - motion / 25.0)
    score = presence * (0.4 * size_term + 0.35 * bg_darkness + 0.25 * still_term)

    return VideoScore(
        path=path.name,
        duration_s=round(total / fps, 1),
        frames_sampled=len(frames),
        white_blob_frac=round(blob_frac, 4),
        white_presence=round(presence, 2),
        bg_darkness=round(bg_darkness, 3),
        motion=round(motion, 2),
        sleep_spike_score=round(score, 4),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    videos = sorted(p for p in args.source.iterdir() if p.suffix.lower() == ".mov")
    print(f"scoring {len(videos)} videos...")
    scores = []
    for i, path in enumerate(videos, 1):
        s = score_video(path)
        if s:
            scores.append(s)
        if i % 20 == 0:
            print(f"  {i}/{len(videos)}")

    scores.sort(key=lambda s: s.sleep_spike_score, reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([asdict(s) for s in scores], indent=1))
    print(f"\nwrote {args.out}  ({len(scores)} scored)")
    print("\ntop 12 sleep-spike candidates:")
    print(f"{'video':28s} {'dur':>6s} {'blob%':>6s} {'pres':>5s} {'bgDark':>6s} {'motion':>6s} {'score':>6s}")
    for s in scores[:12]:
        print(f"{s.path:28s} {s.duration_s:6.1f} {s.white_blob_frac*100:6.1f} "
              f"{s.white_presence:5.2f} {s.bg_darkness:6.3f} {s.motion:6.2f} {s.sleep_spike_score:6.4f}")


main()
