"""Memory gate: plays every clip through the production object graph and
asserts the process RSS stays under the budget after cycling all of them.

Run: .venv/bin/python -B -m tests.integration.measure_memory --clips clips
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
import subprocess
from pathlib import Path

from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
from Foundation import NSDate, NSRunLoop

from overlay.app.orchestrator import AppOrchestrator

RSS_BUDGET_MB = 300.0
PUMP_SECONDS_PER_CLIP = 1.0


class InertBrain:
    """Never issues commands; the driver owns renderer.play directly."""

    def tick(self, events):
        return None

    def shutdown(self):
        pass


def inert_brain_factory(index, state_path, clock, rng):
    return InertBrain()


def pump(seconds: float) -> None:
    NSRunLoop.currentRunLoop().runUntilDate_(
        NSDate.dateWithTimeIntervalSinceNow_(seconds))


def rss_mb() -> float:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True, text=True, check=True)
    return int(out.stdout.strip()) / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--state", type=Path,
                        default=Path("work/measure_memory_state.json"))
    parser.add_argument("--budget-mb", type=float, default=RSS_BUDGET_MB)
    args = parser.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    orch = AppOrchestrator(args.clips / "index.json", args.state,
                           brain_factory=inert_brain_factory)
    orch.start()
    pump(1.0)

    renderer = orch.renderer
    readings = {"startup": round(rss_mb(), 1)}
    for name in sorted(renderer.available_clips()):
        renderer.play(name, "pingpong")
        pump(PUMP_SECONDS_PER_CLIP)
        readings[name] = round(rss_mb(), 1)

    final = rss_mb()
    readings["final"] = round(final, 1)
    orch.shutdown()

    verdict = {
        "readingsMB": readings,
        "budgetMB": args.budget_mb,
        "ok": final < args.budget_mb,
    }
    print(json.dumps(verdict, indent=1))
    raise SystemExit(0 if verdict["ok"] else 1)


main()
