"""Matting bake-off: run candidate matting models on representative frames and
save alpha mattes + composites for visual fur-edge comparison.

Models: BiRefNet (general), BiRefNet-matting, BEN2. Each gets identical inputs;
outputs land in work/spike/bakeoff/<model>/ as bbox-cropped composites over a
light background (the worst case for white fur) plus the raw alpha.
"""

import sys

sys.dont_write_bytecode = True

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoModelForImageSegmentation

DEVICE = "mps"
FRAME_INDICES = [1, 40, 80, 120, 160, 200, 240, 270]
LIGHT_BG = (240, 240, 240)


def birefnet_alpha(model, image: Image.Image) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds = model(tensor)[-1].sigmoid().cpu()
    alpha = transforms.ToPILImage()(preds[0].squeeze()).resize(image.size, Image.BILINEAR)
    return np.asarray(alpha)


def ben2_alpha(model, image: Image.Image) -> np.ndarray:
    fg = model.inference(image)  # RGBA foreground per BEN2 model card
    if fg.mode != "RGBA":
        raise RuntimeError(f"BEN2 returned mode {fg.mode}, expected RGBA")
    return np.asarray(fg)[..., 3]


def save_outputs(out_dir: Path, name: str, frame: np.ndarray, alpha: np.ndarray) -> None:
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        raise RuntimeError(f"{name}: empty matte")
    pad = 24
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, frame.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, frame.shape[0])
    crop = frame[y0:y1, x0:x1].astype(np.float32)
    a = (alpha[y0:y1, x0:x1].astype(np.float32) / 255.0)[..., None]
    comp = (crop * a + np.array(LIGHT_BG, dtype=np.float32) * (1 - a)).astype(np.uint8)
    Image.fromarray(comp).save(out_dir / f"{name}_comp.png")
    Image.fromarray(alpha[y0:y1, x0:x1]).save(out_dir / f"{name}_alpha.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame_paths = [args.frames / f"{i:04d}.png" for i in FRAME_INDICES]
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    arrays = [np.asarray(im) for im in images]

    runners = {
        "birefnet_general": lambda: AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True).to(DEVICE).float().eval(),
        "birefnet_matting": lambda: AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet-matting", trust_remote_code=True).to(DEVICE).float().eval(),
        "ben2": lambda: AutoModel.from_pretrained(
            "PramaLLC/BEN2", trust_remote_code=True).to(DEVICE).float().eval(),
    }

    for model_name, loader in runners.items():
        out_dir = args.out / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"loading {model_name}...")
        model = loader()
        for idx, image, arr in zip(FRAME_INDICES, images, arrays):
            if model_name == "ben2":
                alpha = ben2_alpha(model, image)
            else:
                alpha = birefnet_alpha(model, image)
            save_outputs(out_dir, f"f{idx:04d}", arr, alpha)
            print(f"  {model_name} frame {idx} done")
        del model
        torch.mps.empty_cache()

    print("bake-off complete:", args.out)


main()
