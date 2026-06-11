"""Per-second scan of a video: motion energy + hand-presence heuristic, to choose
clean Clip Package segments (still windows for sleep/idle, active for groom,
hand-free for everything).
"""

import sys

sys.dont_write_bytecode = True

import argparse
from pathlib import Path

import cv2
import numpy as np

ANALYSIS_WIDTH = 320


def skin_fraction(frame_bgr: np.ndarray) -> float:
    """Skin-tone pixels (hands) — distinguishable from the white cat by saturation."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    skin = ((h < 25) | (h > 170)) & (s > 60) & (s < 190) & (v > 80)
    return float(skin.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"{args.video.name}: {total} frames @ {fps:.0f}fps = {total/fps:.1f}s")
    print(f"{'sec':>4s} {'frames':>11s} {'motion':>7s} {'skin%':>6s}")
    prev = None
    second = 0
    diffs: list[float] = []
    skins: list[float] = []
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scale = ANALYSIS_WIDTH / frame.shape[1]
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev is not None:
            diffs.append(float(np.abs(gray - prev).mean()))
        skins.append(skin_fraction(small))
        prev = gray
        frame_i += 1
        if frame_i >= (second + 1) * fps:
            f0, f1 = int(second * fps), frame_i
            print(f"{second:4d} {f0:5d}-{f1:5d} {np.mean(diffs):7.2f} {np.mean(skins)*100:6.2f}")
            diffs, skins = [], []
            second += 1
    cap.release()


main()
