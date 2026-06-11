"""Scan clip packages under a clips root and write <clips>/index.json
(CONTRACTS.md section 1). Every subdirectory must be a valid Clip Package;
validation failures are collected and reported per clip, and the index is
only written when the whole tree is clean.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

REQUIRED_FIELDS = ("name", "category", "fps", "frameCount", "loop", "pixelSize", "displayScale")
CATEGORIES = ("sleep", "idle", "groom", "play", "walk", "reaction")
LOOPS = ("pingpong", "forward", "once")


def validate_clip(clip_dir: Path) -> tuple[dict | None, list[str]]:
    manifest_path = clip_dir / "manifest.json"
    if not manifest_path.is_file():
        return None, [f"{clip_dir}: missing manifest.json"]
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return None, [f"{clip_dir}: manifest.json is not valid JSON ({exc})"]

    errors = [f"{clip_dir}: manifest missing required field '{field}'"
              for field in REQUIRED_FIELDS if field not in manifest]
    if manifest.get("name") is not None and manifest["name"] != clip_dir.name:
        errors.append(f"{clip_dir}: manifest name '{manifest['name']}' != directory name "
                      f"'{clip_dir.name}'")
    if "category" in manifest and manifest["category"] not in CATEGORIES:
        errors.append(f"{clip_dir}: category '{manifest['category']}' not one of {CATEGORIES}")
    if "loop" in manifest and manifest["loop"] not in LOOPS:
        errors.append(f"{clip_dir}: loop '{manifest['loop']}' not one of {LOOPS}")

    frames_dir = clip_dir / "frames"
    if not frames_dir.is_dir():
        errors.append(f"{clip_dir}: missing frames/ directory")
    elif isinstance(manifest.get("frameCount"), int):
        count = manifest["frameCount"]
        missing = [f"{i:04d}.png" for i in range(count)
                   if not (frames_dir / f"{i:04d}.png").is_file()]
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            errors.append(f"{clip_dir}: {len(missing)} missing frame file(s): {shown}")
        on_disk = len(list(frames_dir.glob("*.png")))
        if on_disk != count:
            errors.append(f"{clip_dir}: frameCount {count} != {on_disk} PNGs on disk")

    return manifest, errors


def build_index(clips_root: Path) -> dict:
    clip_dirs = sorted(d for d in clips_root.iterdir()
                       if d.is_dir() and not d.name.startswith("."))
    entries: list[dict] = []
    all_errors: list[str] = []
    for clip_dir in clip_dirs:
        manifest, errors = validate_clip(clip_dir)
        all_errors.extend(errors)
        if manifest is not None and not errors:
            entries.append({"name": manifest["name"],
                            "category": manifest["category"],
                            "dir": clip_dir.as_posix()})
    if all_errors:
        for error in all_errors:
            print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(f"{len(all_errors)} validation error(s); index not written")
    if not entries:
        raise SystemExit(f"no clip packages found under {clips_root}")
    return {"clips": entries}


def main() -> None:
    parser = argparse.ArgumentParser(description="build clips/index.json from clip manifests")
    parser.add_argument("--clips", type=Path, required=True, help="clips root directory")
    args = parser.parse_args()

    if not args.clips.is_dir():
        raise SystemExit(f"clips directory not found: {args.clips}")
    index = build_index(args.clips)
    index_path = args.clips / "index.json"
    index_path.write_text(json.dumps(index, indent=1) + "\n")
    print(f"wrote {index_path} ({len(index['clips'])} clip(s))")


main()
