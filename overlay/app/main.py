"""Overlay Cat entry point.

python -B -m overlay.app.main --clips clips [--state PATH] [--settings PATH] [--smoke]

--smoke constructs everything (window, renderer, monitor, status item, first
clip), pumps the run loop for 2 seconds, and exits 0 without NSApp.run().

Inside a py2app bundle (detected by Contents/Resources/clips above
sys.executable) --clips defaults to the bundled clips; outside a bundle it is
required.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from Foundation import NSDate, NSRunLoop

from .orchestrator import DEFAULT_SETTINGS_PATH, AppOrchestrator

SMOKE_PUMP_SECONDS = 2.0


@dataclass(frozen=True)
class SmokePlayClip:
    clip: str
    loop: str
    min_seconds: float
    motion: str | None = None


class SmokeBrain:
    """Brain-shaped stand-in used only by --smoke for tests that need no real brain."""

    def __init__(self, clip_name: str):
        self._clip_name = clip_name
        self._issued = False

    def tick(self, events):
        if self._issued:
            return None
        self._issued = True
        return SmokePlayClip(self._clip_name, "pingpong", 5.0)

    def shutdown(self):
        pass


def _smoke_brain_factory(index: dict, state_path: Path, clock, rng) -> SmokeBrain:
    return SmokeBrain(index["clips"][0]["name"])


def _smoke_index_path(clips_dir: Path) -> Path:
    index_path = clips_dir / "index.json"
    if index_path.exists():
        return index_path
    entries = []
    for manifest_path in sorted(clips_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if "category" not in manifest:
            raise RuntimeError(f"{manifest_path} has no 'category' (required by "
                               f"CONTRACTS.md); run pipeline/build_index.py instead")
        entries.append({
            "name": manifest["name"],
            "category": manifest["category"],
            "dir": str(manifest_path.parent.resolve()),
        })
    if not entries:
        raise RuntimeError(f"no clip manifests under {clips_dir}")
    smoke_dir = Path(tempfile.mkdtemp(prefix="overlaycat_smoke_"))
    generated = smoke_dir / "index.json"
    generated.write_text(json.dumps({"clips": entries}, indent=1))
    return generated


def _run_smoke(clips_dir: Path, state_path: Path, settings_path: Path) -> None:
    orchestrator = AppOrchestrator(
        _smoke_index_path(clips_dir), state_path, settings_path=settings_path,
        brain_factory=_smoke_brain_factory)
    orchestrator.start()
    NSRunLoop.currentRunLoop().runUntilDate_(
        NSDate.dateWithTimeIntervalSinceNow_(SMOKE_PUMP_SECONDS))
    state = orchestrator.renderer.debug_state()
    frame = orchestrator.renderer.window_frame()
    orchestrator.shutdown()
    if state["clip"] is None:
        raise RuntimeError(f"smoke: no clip playing after run loop pump: {state}")
    if state["frames_rendered"] == 0:
        raise RuntimeError(f"smoke: frame timer never advanced: {state}")
    print(f"smoke ok: {state} window_frame={frame}")


def _bundle_clips_dir() -> Path | None:
    """Contents/Resources/clips of the .app bundle hosting sys.executable.

    Walks up from sys.executable; returns None when not inside a bundle (or
    the bundle ships no clips). No env vars — pure path derivation.
    """
    for parent in Path(sys.executable).resolve().parents:
        candidate = parent / "Contents" / "Resources" / "clips"
        if candidate.is_dir():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay Cat desktop pet")
    parser.add_argument(
        "--clips", type=Path, default=None,
        help="clip library directory; defaults to the bundled "
             "Contents/Resources/clips inside an .app, required otherwise")
    parser.add_argument(
        "--state", type=Path,
        default=Path.home() / "Library" / "Application Support" / "OverlayCat"
        / "state.json")
    parser.add_argument(
        "--settings", type=Path, default=DEFAULT_SETTINGS_PATH,
        help="user settings JSON (cat size); created on first Size change")
    parser.add_argument(
        "--smoke", action="store_true",
        help="construct everything, pump the run loop 2 s, then exit 0")
    args = parser.parse_args()
    clips_dir = args.clips if args.clips is not None else _bundle_clips_dir()
    if clips_dir is None:
        parser.error("--clips is required outside an .app bundle (no "
                     "Contents/Resources/clips found above sys.executable)")
    if args.smoke:
        _run_smoke(clips_dir, args.state, args.settings)
        return
    args.state.parent.mkdir(parents=True, exist_ok=True)
    orchestrator = AppOrchestrator(clips_dir / "index.json", args.state,
                                   settings_path=args.settings)
    orchestrator.run()


if __name__ == "__main__":
    main()
