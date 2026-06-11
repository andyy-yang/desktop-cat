"""Matte a frame sequence into a Clip Package like matte_clip.py, but when a
frame mattes empty at the standard 1024px BiRefNet input (motion blur can do
this), retry that single frame at 1536px input before packaging. Emits the
identical Clip Package contract; the union bbox and anchor include the retried
frame. Exits 1 if any frame is still empty after the 1536px retry (no silent
fallback).

Usage mirrors matte_clip.py:
python -B pipeline/matte_clip_retry.py --frames work/factory2/<name> \
    --model birefnet_general --out clips_incoming/<name> --name <name> \
    --source-fps 12 --target-fps 12 --category <category>
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

DEVICE = "mps"
ALPHA_NOISE_FLOOR = 8
EMPTY_ALPHA_THRESHOLD = 64
BASE_SIZE = 1024
RETRY_SIZE = 1536
CATEGORIES = ["sleep", "idle", "groom", "play", "walk", "reaction"]


def load_model(name: str):
    repo = {"birefnet_general": "ZhengPeng7/BiRefNet",
            "birefnet_matting": "ZhengPeng7/BiRefNet-matting"}[name]
    return AutoModelForImageSegmentation.from_pretrained(
        repo, trust_remote_code=True).to(DEVICE).float().eval()


def predict_alpha(model, image: Image.Image, size: int) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds = model(tensor)[-1].sigmoid().cpu()
    alpha = transforms.ToPILImage()(preds[0].squeeze()).resize(image.size, Image.BILINEAR)
    return np.asarray(alpha).copy()


def keep_largest_component(alpha: np.ndarray) -> np.ndarray:
    binary = (alpha > 32).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return alpha
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = (labels == idx)
    region = cv2.dilate(keep.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    out = alpha.copy()
    out[~region] = 0
    return out


def matte_frame(model, image: Image.Image, size: int) -> np.ndarray:
    alpha = predict_alpha(model, image, size)
    alpha[alpha < ALPHA_NOISE_FLOOR] = 0
    return keep_largest_component(alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--model", required=True,
                        choices=["birefnet_general", "birefnet_matting"])
    parser.add_argument("--out", type=Path, required=True, help="clip package dir")
    parser.add_argument("--name", required=True, help="clip name in manifest")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=12.0)
    parser.add_argument("--target-width", type=int, default=840)
    parser.add_argument("--loop", default="pingpong", choices=["pingpong", "forward", "once"])
    parser.add_argument("--category", default=None, choices=CATEGORIES)
    args = parser.parse_args()

    frame_paths = sorted(args.frames.glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"no source frames in {args.frames}")
    step = args.source_fps / args.target_fps
    indices = sorted({int(round(i * step)) for i in range(int(len(frame_paths) / step))})
    indices = [i for i in indices if i < len(frame_paths)]
    print(f"matting {len(indices)}/{len(frame_paths)} frames with {args.model} "
          f"(retry-at-{RETRY_SIZE} on empty)...")

    model = load_model(args.model)
    mattes: list[tuple[np.ndarray, np.ndarray]] = []
    retried: list[int] = []
    for n, i in enumerate(indices):
        image = Image.open(frame_paths[i]).convert("RGB")
        alpha = matte_frame(model, image, BASE_SIZE)
        if np.count_nonzero(alpha > EMPTY_ALPHA_THRESHOLD) == 0:
            print(f"  frame {n:04d} ({frame_paths[i].name}) empty at {BASE_SIZE}px, "
                  f"retrying at {RETRY_SIZE}px...")
            alpha = matte_frame(model, image, RETRY_SIZE)
            if np.count_nonzero(alpha > EMPTY_ALPHA_THRESHOLD) == 0:
                raise SystemExit(f"FAIL: frame {n:04d} still empty at {RETRY_SIZE}px input")
            retried.append(n)
            print(f"  frame {n:04d} recovered: "
                  f"{np.count_nonzero(alpha > EMPTY_ALPHA_THRESHOLD)} px")
        mattes.append((np.asarray(image), alpha))
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(indices)}")

    union = np.zeros_like(mattes[0][1], dtype=bool)
    for _, alpha in mattes:
        union |= alpha > 16
    ys, xs = np.where(union)
    pad = 30
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, union.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, union.shape[0])
    crop_w = x1 - x0
    scale = min(args.target_width / crop_w, 1.0)
    out_size = (int(crop_w * scale), int((y1 - y0) * scale))

    anchor = [
        min(int(round(((int(xs.min()) + int(xs.max())) / 2 - x0) * scale)), out_size[0]),
        min(int(round((int(ys.max()) + 1 - y0) * scale)), out_size[1]),
    ]

    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for n, (rgb, alpha) in enumerate(mattes):
        rgba = np.dstack([rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]])
        img = Image.fromarray(rgba).resize(out_size, Image.LANCZOS)
        img.save(frames_dir / f"{n:04d}.png")

    manifest = {
        "name": args.name,
        "fps": args.target_fps,
        "frameCount": len(mattes),
        "loop": args.loop,
        "pixelSize": list(out_size),
        "displayScale": 2.0,
        "model": args.model,
        "sourceVideo": str(args.frames),
        "anchor": anchor,
    }
    if args.category is not None:
        manifest["category"] = args.category
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"clip package written: {args.out}  ({len(mattes)} frames @ {out_size}px, "
          f"{len(retried)} frame(s) recovered at {RETRY_SIZE}px: {retried})")


main()
