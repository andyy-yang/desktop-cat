"""Behavior-oriented triage: classify each video's dominant behavior signature.

Signals per video (sampled at ~3 fps): white-blob centroid drift (locomotion),
in-place motion energy (grooming/play vs sleep), blob aspect ratio (sitting
upright vs lying), plus the bake-off-proven background-darkness score.
Writes a JSON catalog ranked per category for visual confirmation.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

ANALYSIS_WIDTH = 480
SAMPLE_FPS = 3.0
WHITE_S_MAX = 70
WHITE_V_MIN = 140


@dataclass
class BehaviorScore:
    path: str
    duration_s: float
    blob_frac: float
    bg_darkness: float
    presence: float
    drift_px_s: float       # centroid speed in analysis-px/second (locomotion signal)
    motion_in_place: float  # mean abs-diff inside blob bbox after centroid alignment
    aspect: float           # blob height / width (tall = sitting upright)
    category: str


def white_blob(frame_bgr: np.ndarray):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] < WHITE_S_MAX) & (hsv[..., 2] > WHITE_V_MIN)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    frac = float(stats[idx, cv2.CC_STAT_AREA]) / mask.size
    w = float(stats[idx, cv2.CC_STAT_WIDTH])
    h = float(stats[idx, cv2.CC_STAT_HEIGHT])
    return frac, centroids[idx], (h / w if w > 0 else 0.0), (labels == idx)


def categorize(drift: float, motion: float, aspect: float) -> str:
    if drift > 18.0:
        return "walk"
    if motion < 2.2:
        return "sleep"
    if motion > 6.0:
        return "play_or_groom"
    return "idle_sit" if aspect > 0.95 else "loaf_idle"


def score_video(path: Path) -> BehaviorScore | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(fps / SAMPLE_FPS), 1)
    indices = list(range(0, total, step))
    if len(indices) < 4:
        cap.release()
        return None

    obs = []
    bg_vals = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        scale = ANALYSIS_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        found = white_blob(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if found is None:
            obs.append(None)
            continue
        frac, centroid, aspect, blob = found
        bg = gray[~blob]
        bg_vals.append(float(bg.mean()) if bg.size else 255.0)
        obs.append({"frac": frac, "c": centroid, "aspect": aspect, "gray": gray, "blob": blob})
    cap.release()

    valid = [o for o in obs if o and o["frac"] > 0.02]
    if len(valid) < 4:
        return None

    dt = step / fps
    drifts = []
    motions = []
    pairs = [(a, b) for a, b in zip(obs, obs[1:]) if a and b and a["frac"] > 0.02 and b["frac"] > 0.02]
    for a, b in pairs:
        drifts.append(float(np.hypot(*(np.array(b["c"]) - np.array(a["c"])))) / dt)
        # align by centroid shift, then measure residual motion inside the union blob
        shift = np.array(b["c"]) - np.array(a["c"])
        m = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
        b_aligned = cv2.warpAffine(b["gray"], m, (b["gray"].shape[1], b["gray"].shape[0]))
        union = a["blob"] | b["blob"]
        if union.sum() > 100:
            motions.append(float(np.abs(a["gray"].astype(np.int16)
                                        - b_aligned.astype(np.int16))[union].mean()))

    drift = float(np.median(drifts)) if drifts else 0.0
    motion = float(np.median(motions)) if motions else 0.0
    aspect = float(np.median([o["aspect"] for o in valid]))
    blob_frac = float(np.median([o["frac"] for o in valid]))

    return BehaviorScore(
        path=path.name,
        duration_s=round(total / fps, 1),
        blob_frac=round(blob_frac, 4),
        bg_darkness=round(1.0 - float(np.median(bg_vals)) / 255.0, 3),
        presence=round(len(valid) / len(indices), 2),
        drift_px_s=round(drift, 1),
        motion_in_place=round(motion, 2),
        aspect=round(aspect, 2),
        category=categorize(drift, motion, aspect),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    videos = sorted(p for p in args.source.iterdir() if p.suffix.lower() == ".mov")
    print(f"scoring {len(videos)} videos for behavior signatures...")
    scores = []
    for i, path in enumerate(videos, 1):
        s = score_video(path)
        if s:
            scores.append(s)
        if i % 20 == 0:
            print(f"  {i}/{len(videos)}")

    args.out.write_text(json.dumps([asdict(s) for s in scores], indent=1))
    by_cat: dict[str, list[BehaviorScore]] = {}
    for s in scores:
        by_cat.setdefault(s.category, []).append(s)
    print(f"\nwrote {args.out} ({len(scores)} scored)\n")
    for cat, items in sorted(by_cat.items()):
        items.sort(key=lambda s: (s.presence, s.bg_darkness, s.blob_frac), reverse=True)
        print(f"== {cat} ({len(items)}) — top 6 by presence/bgDark/size:")
        for s in items[:6]:
            print(f"   {s.path:24s} dur={s.duration_s:5.1f}s blob={s.blob_frac*100:4.1f}% "
                  f"bgDark={s.bg_darkness:.2f} drift={s.drift_px_s:5.1f} "
                  f"motion={s.motion_in_place:4.1f} aspect={s.aspect:.2f}")


main()
