"""Full-corpus scan: BiRefNet subject masks on sampled frames of every video,
merged with behavior signatures, to find ALL clip-worthy segments (fully in
frame, single white cat, behavior-classified). Writes work/corpus_quality.json.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

SAMPLES = 5
MARGIN_OK_PX = 6      # at 320px analysis width
MIN_SUBJECT_FRAC = 0.02


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--behaviors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    behaviors = {b["path"]: b for b in json.loads(args.behaviors.read_text())}
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True).to("mps").float().eval()
    tf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results = []
    videos = sorted(args.source.glob("*.MOV")) + sorted(args.source.glob("*.mov"))
    for vi, path in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 10:
            cap.release()
            continue
        margins, fracs, whiteness = [], [], []
        for i in np.linspace(total * 0.1, total * 0.9, SAMPLES, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                pred = model(tf(img).unsqueeze(0).to("mps"))[-1].sigmoid().cpu()[0, 0].numpy()
            small_w = 320
            small_h = int(small_w * frame.shape[0] / frame.shape[1])
            mask = cv2.resize(pred, (small_w, small_h)) > 0.25
            ys, xs = np.where(mask)
            if len(xs) < 50:
                continue
            margins.append(int(min(xs.min(), ys.min(),
                                    small_w - 1 - xs.max(), small_h - 1 - ys.max())))
            fracs.append(float(mask.mean()))
            # is the masked subject the WHITE cat (vs the tabby/objects)?
            small = cv2.resize(frame, (small_w, small_h))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            sat = hsv[..., 1][mask]
            val = hsv[..., 2][mask]
            whiteness.append(float(((sat < 70) & (val > 140)).mean()))
        cap.release()
        if not margins:
            continue
        rec = {
            "path": path.name,
            "margin_px": float(np.median(margins)),
            "subject_frac": round(float(np.median(fracs)), 4),
            "subject_whiteness": round(float(np.median(whiteness)), 3),
            "full_frame": bool(np.median(margins) >= MARGIN_OK_PX
                               and np.median(fracs) > MIN_SUBJECT_FRAC),
        }
        rec.update({k: behaviors[path.name][k]
                    for k in ("duration_s", "bg_darkness", "drift_px_s",
                              "motion_in_place", "aspect", "category")
                    if path.name in behaviors})
        results.append(rec)
        if vi % 10 == 0:
            print(f"{vi}/{len(videos)}")

    args.out.write_text(json.dumps(results, indent=1))
    usable = [r for r in results if r["full_frame"] and r["subject_whiteness"] > 0.5]
    print(f"\nwrote {args.out}: {len(results)} scanned, {len(usable)} fully-in-frame white-cat videos")
    for r in sorted(usable, key=lambda r: -r.get("drift_px_s", 0))[:20]:
        print(f"  {r['path']:24s} cat={r['category'] if 'category' in r else '?':14s} "
              f"drift={r.get('drift_px_s', 0):5.1f} motion={r.get('motion_in_place', 0):4.1f} "
              f"margin={r['margin_px']:4.0f} white={r['subject_whiteness']:.2f} "
              f"dur={r.get('duration_s', 0):5.1f}s")


main()
