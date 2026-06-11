"""Matte a full frame sequence with the chosen model and build a Clip Package:
cropped, downscaled RGBA PNG frames at the target fps + manifest.json.

The Clip Package is the producer/runtime contract: any future producer (Blender
bakes, generated clips) emits this same format.
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
from transformers import AutoModel, AutoModelForImageSegmentation

DEVICE = "mps"
ALPHA_NOISE_FLOOR = 8
CATEGORIES = ["sleep", "idle", "groom", "play", "walk", "reaction"]


def parse_segment(segment: str, total: int) -> tuple[int, int]:
    start_str, sep, end_str = segment.partition(":")
    if not sep or not start_str.isdigit() or not end_str.isdigit():
        raise SystemExit(f"--segment must be START:END frame indices, got '{segment}'")
    start, end = int(start_str), int(end_str)
    if not 0 <= start < end <= total:
        raise SystemExit(f"--segment {segment} out of range for {total} source frames")
    return start, end


def load_model(name: str):
    if name == "ben2":
        return AutoModel.from_pretrained(
            "PramaLLC/BEN2", trust_remote_code=True).to(DEVICE).float().eval()
    repo = {"birefnet_general": "ZhengPeng7/BiRefNet",
            "birefnet_matting": "ZhengPeng7/BiRefNet-matting"}[name]
    return AutoModelForImageSegmentation.from_pretrained(
        repo, trust_remote_code=True).to(DEVICE).float().eval()


def predict_alpha(model_name: str, model, image: Image.Image) -> np.ndarray:
    if model_name == "ben2":
        fg = model.inference(image)
        return np.asarray(fg)[..., 3].copy()
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
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
    # Retain soft edges by zeroing only pixels outside a dilated component region.
    region = cv2.dilate(keep.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    out = alpha.copy()
    out[~region] = 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--model", required=True,
                        choices=["birefnet_general", "birefnet_matting", "ben2"])
    parser.add_argument("--out", type=Path, required=True, help="clip package dir")
    parser.add_argument("--name", required=True, help="clip name in manifest")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=12.0)
    parser.add_argument("--target-width", type=int, default=840,
                        help="output cat width in px (2x of display points)")
    parser.add_argument("--loop", default="pingpong", choices=["pingpong", "forward", "once"])
    parser.add_argument("--segment", default=None,
                        help="START:END source frame index range (before fps subsampling)")
    parser.add_argument("--category", default=None, choices=CATEGORIES,
                        help="written into the manifest when given")
    parser.add_argument("--anchor-mode", default="bottom-center", choices=["bottom-center"],
                        help="how the manifest anchor is derived from the union bbox")
    args = parser.parse_args()

    frame_paths = sorted(args.frames.glob("*.png"))
    if args.segment is not None:
        start, end = parse_segment(args.segment, len(frame_paths))
        frame_paths = frame_paths[start:end]
    if not frame_paths:
        raise SystemExit(f"no source frames selected from {args.frames}")
    step = args.source_fps / args.target_fps
    indices = sorted({int(round(i * step)) for i in range(int(len(frame_paths) / step))})
    indices = [i for i in indices if i < len(frame_paths)]
    if not indices:
        raise SystemExit(f"fps subsampling selected 0 of {len(frame_paths)} frames; "
                         f"segment too short for source-fps/target-fps ratio")
    print(f"matting {len(indices)}/{len(frame_paths)} frames with {args.model}...")

    model = load_model(args.model)
    mattes: list[tuple[np.ndarray, np.ndarray]] = []
    for n, i in enumerate(indices, 1):
        image = Image.open(frame_paths[i]).convert("RGB")
        alpha = predict_alpha(args.model, model, image)
        alpha[alpha < ALPHA_NOISE_FLOOR] = 0
        alpha = keep_largest_component(alpha)
        mattes.append((np.asarray(image), alpha))
        if n % 20 == 0:
            print(f"  {n}/{len(indices)}")

    # Union bbox across all frames, padded, so the cat never clips at the edge.
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

    # bottom-center of the union bbox in output px: the ground-contact point.
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
    print(f"clip package written: {args.out}  ({len(mattes)} frames @ {out_size}px)")


main()
