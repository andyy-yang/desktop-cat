"""Cross-clip scale normalization for Clip Packages (CONTRACTS.md section 1).

Applies a per-clip scale factor from a JSON spec {"<clip_name>": <factor>, ...},
resizing every frame (LANCZOS) and scaling manifest pixelSize + anchor. Records
provenance in a "normalizedFrom" manifest field. Factors > 1.25 are refused:
significant upscaling degrades quality — rescale the reference group down instead
so all factors stay <= 1.0 where possible.

Usage: python -B pipeline/normalize_scale.py --clips clips --spec work/scale_spec.json
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

MAX_FACTOR = 1.25


def validate_spec(clips_root: Path, spec: dict) -> list[str]:
    errors = []
    for name, factor in spec.items():
        clip_dir = clips_root / name
        if not (clip_dir / "manifest.json").is_file():
            errors.append(f"{name}: no manifest.json under {clip_dir}")
            continue
        if not isinstance(factor, (int, float)) or factor <= 0:
            errors.append(f"{name}: factor must be a positive number, got {factor!r}")
        elif factor > MAX_FACTOR:
            errors.append(f"{name}: factor {factor} > {MAX_FACTOR} (no significant "
                          f"upscaling; scale the reference group down instead)")
        manifest = json.loads((clip_dir / "manifest.json").read_text())
        if "normalizedFrom" in manifest:
            errors.append(f"{name}: already normalized (normalizedFrom present); "
                          f"refusing to compound scaling")
    return errors


def normalize_clip(clip_dir: Path, factor: float) -> dict:
    manifest_path = clip_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    src_w, src_h = manifest["pixelSize"]
    dst_w, dst_h = max(1, round(src_w * factor)), max(1, round(src_h * factor))
    src_anchor = manifest.get("anchor", [src_w // 2, src_h])
    dst_anchor = [min(round(src_anchor[0] * factor), dst_w),
                  min(round(src_anchor[1] * factor), dst_h)]

    frames_dir = clip_dir / "frames"
    frame_paths = sorted(frames_dir.glob("*.png"))
    if len(frame_paths) != manifest["frameCount"]:
        raise SystemExit(f"{clip_dir}: frameCount {manifest['frameCount']} != "
                         f"{len(frame_paths)} PNGs on disk")

    tmp_dir = clip_dir / "frames_norm_tmp"
    if tmp_dir.exists():
        raise SystemExit(f"{clip_dir}: stale {tmp_dir.name}/ from an aborted run; "
                         f"remove it and retry")
    tmp_dir.mkdir()
    for path in frame_paths:
        frame = Image.open(path).convert("RGBA")
        if frame.size != (src_w, src_h):
            shutil.rmtree(tmp_dir)
            raise SystemExit(f"{path}: frame size {frame.size} != manifest "
                             f"pixelSize ({src_w}, {src_h})")
        frame.resize((dst_w, dst_h), Image.LANCZOS).save(tmp_dir / path.name)

    # all frames written; swap directories, then commit the manifest
    old_dir = clip_dir / "frames_old_tmp"
    frames_dir.rename(old_dir)
    tmp_dir.rename(frames_dir)
    shutil.rmtree(old_dir)

    manifest["normalizedFrom"] = {"pixelSize": [src_w, src_h],
                                  "anchor": list(src_anchor),
                                  "factor": factor}
    manifest["pixelSize"] = [dst_w, dst_h]
    manifest["anchor"] = dst_anchor
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="normalize clip scales per a JSON factor spec")
    parser.add_argument("--clips", type=Path, required=True, help="clips root directory")
    parser.add_argument("--spec", type=Path, required=True,
                        help="JSON file mapping clip name -> scale factor")
    args = parser.parse_args()

    if not args.clips.is_dir():
        raise SystemExit(f"clips directory not found: {args.clips}")
    spec = json.loads(args.spec.read_text())
    if not isinstance(spec, dict) or not spec:
        raise SystemExit(f"{args.spec}: spec must be a non-empty JSON object")

    errors = validate_spec(args.clips, spec)
    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(f"{len(errors)} spec error(s); nothing modified")

    for name, factor in spec.items():
        manifest = normalize_clip(args.clips / name, factor)
        print(f"{name}: x{factor} -> pixelSize {manifest['pixelSize']} "
              f"anchor {manifest['anchor']} ({manifest['frameCount']} frames)")


main()
