"""Live verification of walk locomotion without a human at the machine.

Launches the production object graph in-process with a stub brain that
immediately commands walk_right_somali with motion='right', then:
  1. plants the cat near the left of the visible frame and pumps 4 s, asserting
     window x increases monotonically at ~60 pt/s while y stays on the floor
     line (visibleFrame bottom adjusted for the clip anchor),
  2. forces the window next to the right screen edge, pumps, and asserts the
     runtime swapped to walk_left_somali (no brain round trip) with x now
     decreasing,
  3. reports a JSON verdict.

Run: .venv/bin/python -B -m tests.integration.verify_walk --clips clips_3d --out work/verify_walk
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from AppKit import NSApplication, NSApplicationActivationPolicyAccessory, NSScreen
from Foundation import NSDate, NSRunLoop

from overlay.app.locomotion import EDGE_MARGIN_PT, WALK_SPEED_PT_S
from overlay.app.orchestrator import AppOrchestrator

WALK_CLIP = "walk_right_somali"
OPPOSITE_CLIP = "walk_left_somali"
WALK_SECONDS = 4.0
SAMPLE_INTERVAL_S = 0.25
SPEED_TOLERANCE = 0.25          # accept 45..75 pt/s around the 60 pt/s default
FLOOR_TOLERANCE_PT = 0.5
EDGE_APPROACH_PT = 20.0


@dataclass(frozen=True)
class WalkPlayClip:
    clip: str
    loop: str
    min_seconds: float
    motion: str | None = None


class WalkBrain:
    """Brain-shaped stub: one walk command on the first tick, then silence."""

    def __init__(self):
        self._issued = False

    def tick(self, events):
        if self._issued:
            return None
        self._issued = True
        return WalkPlayClip(WALK_CLIP, "forward", 3600.0, "right")

    def shutdown(self):
        pass


def walk_brain_factory(index: dict, state_path: Path, clock, rng) -> WalkBrain:
    names = {entry["name"] for entry in index["clips"]}
    missing = {WALK_CLIP, OPPOSITE_CLIP} - names
    if missing:
        raise RuntimeError(f"clip index lacks walk pair members: {sorted(missing)}")
    return WalkBrain()


def pump(seconds: float) -> None:
    NSRunLoop.currentRunLoop().runUntilDate_(
        NSDate.dateWithTimeIntervalSinceNow_(seconds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, default=Path("clips_3d"))
    parser.add_argument("--out", type=Path, default=Path("work/verify_walk"))
    parser.add_argument("--state", type=Path,
                        default=Path("work/verify_walk/state.json"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    orch = AppOrchestrator(args.clips / "index.json", args.state,
                           brain_factory=walk_brain_factory)
    orch.start()
    pump(0.5)

    renderer = orch.renderer
    report: dict = {"checks": {}}
    report["checks"]["walkClipPlaying"] = {
        "clip": renderer.current_clip(),
        "locomotionActive": orch.locomotion.active(),
        "ok": renderer.current_clip() == WALK_CLIP and orch.locomotion.active(),
    }

    visible = NSScreen.mainScreen().visibleFrame()
    floor_y = orch.locomotion.floor_window_y()

    # 1. monotonic rightward walk at ~WALK_SPEED_PT_S along the floor line
    renderer.set_frame_origin(visible.origin.x + 60.0, floor_y)
    pump(0.1)
    samples = []
    for _ in range(int(WALK_SECONDS / SAMPLE_INTERVAL_S)):
        pump(SAMPLE_INTERVAL_S)
        fx, fy, _, _ = renderer.window_frame()
        samples.append((time.monotonic(), fx, fy))
    xs = [s[1] for s in samples]
    ys = [s[2] for s in samples]
    monotonic = all(b > a for a, b in zip(xs, xs[1:]))
    speed = (xs[-1] - xs[0]) / (samples[-1][0] - samples[0][0])
    floor_dev = max(abs(y - floor_y) for y in ys)
    speed_lo = WALK_SPEED_PT_S * (1.0 - SPEED_TOLERANCE)
    speed_hi = WALK_SPEED_PT_S * (1.0 + SPEED_TOLERANCE)
    report["checks"]["rightwardWalk"] = {
        "samples": len(samples),
        "xStart": round(xs[0], 1),
        "xEnd": round(xs[-1], 1),
        "monotonicIncrease": monotonic,
        "speedPtS": round(speed, 2),
        "floorY": round(floor_y, 2),
        "maxFloorDeviationPt": round(floor_dev, 3),
        "ok": (monotonic and speed_lo <= speed <= speed_hi
               and floor_dev <= FLOOR_TOLERANCE_PT),
    }

    # 2. force the leading edge near the right screen edge -> swap + reverse
    _, _, fw, _ = renderer.window_frame()
    max_x = visible.origin.x + visible.size.width
    renderer.set_frame_origin(max_x - fw - EDGE_MARGIN_PT - EDGE_APPROACH_PT, floor_y)
    pump(1.5)  # EDGE_APPROACH_PT at 60 pt/s + margin to swap and walk back
    swapped_clip = renderer.current_clip()
    swapped_direction = orch.locomotion.direction()
    x_after_swap_1 = renderer.window_frame()[0]
    pump(0.5)
    x_after_swap_2 = renderer.window_frame()[0]
    report["checks"]["edgeSwap"] = {
        "clipAfterEdge": swapped_clip,
        "directionAfterEdge": swapped_direction,
        "xAfterSwap": round(x_after_swap_1, 1),
        "xHalfSecondLater": round(x_after_swap_2, 1),
        "ok": (swapped_clip == OPPOSITE_CLIP and swapped_direction == "left"
               and x_after_swap_2 < x_after_swap_1),
    }

    orch.shutdown()

    report["allOk"] = all(c["ok"] for c in report["checks"].values())
    (args.out / "walk_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    raise SystemExit(0 if report["allOk"] else 1)


main()
