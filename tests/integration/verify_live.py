"""Live end-to-end verification of the overlay app without a human at the machine.

Launches the production object graph in-process, pumps the run loop, then:
  1. asserts frames advance at the clip's fps,
  2. warps the REAL cursor (CGWarpMouseCursorPosition, no TCC permission) onto a
     transparent corner and onto an opaque cat pixel, asserting the
     InteractionMonitor flips panel.ignoresMouseEvents both ways,
  3. forces a clip switch and asserts the crossfade path executes,
  4. captures the app's OWN window via CGWindowListCreateImage (own-process
     windows are exempt from Screen Recording TCC) and saves it for visual QC,
  5. restores the cursor and reports a JSON verdict.

Run: .venv/bin/python -B tests/integration/verify_live.py --clips clips --out work/verify
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from AppKit import NSApplication, NSApplicationActivationPolicyAccessory, NSEvent
from Foundation import NSDate, NSRunLoop
from PIL import Image
from Quartz import (
    CGRectNull,
    CGWarpMouseCursorPosition,
    CGWindowListCreateImage,
    CGImageGetWidth,
    CGImageGetHeight,
    CGImageGetDataProvider,
    CGDataProviderCopyData,
    CGImageGetBytesPerRow,
    kCGWindowImageBoundsIgnoreFraming,
    kCGWindowListOptionIncludingWindow,
)

from overlay.app.orchestrator import AppOrchestrator


def pump(seconds: float) -> None:
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(seconds))


def window_top_left_to_global(panel, x_pt: float, y_pt_top: float):
    """Convert window-local top-left coords to CG global (top-left origin) coords."""
    f = panel.frame()
    screen_h = panel.screen().frame().size.height
    gx = f.origin.x + x_pt
    gy = screen_h - (f.origin.y + f.size.height) + y_pt_top
    return gx, gy


def capture_own_window(panel, out_path: Path) -> dict:
    image = CGWindowListCreateImage(
        CGRectNull, kCGWindowListOptionIncludingWindow, panel.windowNumber(),
        kCGWindowImageBoundsIgnoreFraming)
    if image is None:
        return {"captured": False}
    w, h = CGImageGetWidth(image), CGImageGetHeight(image)
    data = CGDataProviderCopyData(CGImageGetDataProvider(image))
    stride = CGImageGetBytesPerRow(image)
    arr = np.frombuffer(data, dtype=np.uint8)[: h * stride].reshape(h, stride // 4, 4)[:, :w]
    rgba = arr[..., [2, 1, 0, 3]]  # BGRA -> RGBA
    Image.fromarray(rgba, "RGBA").save(out_path)
    alpha = rgba[..., 3]
    return {
        "captured": True,
        "size": [int(w), int(h)],
        "opaqueFraction": round(float((alpha > 24).mean()), 4),
        "file": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("work/verify/state.json"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    orch = AppOrchestrator(args.clips / "index.json", args.state)
    orch.start()
    pump(1.5)

    renderer = orch.renderer
    panel = orch.window.panel
    report: dict = {"checks": {}}
    original_mouse = NSEvent.mouseLocation()

    # 1. frames advance
    f0 = renderer.debug_state()["frames_rendered"]
    pump(1.0)
    f1 = renderer.debug_state()["frames_rendered"]
    report["checks"]["framesAdvance"] = {
        "framesIn1s": f1 - f0,
        "ok": 8 <= (f1 - f0) <= 16,  # 12fps nominal
    }

    # 2. hit-toggle loop with the real cursor
    decoded = renderer._decoded_clip(renderer.current_clip())
    frame_mask = decoded.mask_at(renderer.debug_state()["frame_index"])
    h_px, w_px = frame_mask.shape
    scale = 2.0
    ys, xs = np.where(frame_mask)
    # nearest opaque pixel to the centroid (the centroid itself can fall off-cat)
    centroid = (xs.mean(), ys.mean())
    nearest = np.argmin((xs - centroid[0]) ** 2 + (ys - centroid[1]) ** 2)
    cat_px = (float(xs[nearest]), float(ys[nearest]))
    cat_pt = (cat_px[0] / scale, cat_px[1] / scale)

    corner_pt = None
    for cx, cy in [(3, 3), (w_px / scale - 4, 3), (3, h_px / scale - 4)]:
        if not frame_mask[int(cy * scale), int(cx * scale)]:
            corner_pt = (cx, cy)
            break

    gx, gy = window_top_left_to_global(panel, *corner_pt)
    CGWarpMouseCursorPosition((gx, gy))
    pump(2.2)  # slow tick (1 Hz) must escalate to fast polling first
    ignores_on_corner = bool(panel.ignoresMouseEvents())

    gx, gy = window_top_left_to_global(panel, *cat_pt)
    CGWarpMouseCursorPosition((gx, gy))
    pump(2.2)
    ignores_on_cat = bool(panel.ignoresMouseEvents())

    # geometry probe, independent of timer timing: does the renderer itself see
    # an opaque pixel at the warped cursor position?
    loc = NSEvent.mouseLocation()
    f = panel.frame()
    probe_alpha = renderer.current_alpha_at(loc.x - f.origin.x, loc.y - f.origin.y)

    report["checks"]["hitToggle"] = {
        "ignoresOnTransparentCorner": ignores_on_corner,
        "ignoresOnCatPixel": ignores_on_cat,
        "rendererAlphaAtCursor": int(probe_alpha),
        "mouseLocation": [round(loc.x, 1), round(loc.y, 1)],
        "windowFrame": [round(f.origin.x, 1), round(f.origin.y, 1),
                        round(f.size.width, 1), round(f.size.height, 1)],
        "ok": ignores_on_corner is True and ignores_on_cat is False,
    }

    # 3. forced clip switch exercises crossfade + anchor-preserving resize
    clip_names = list(renderer.available_clips().keys())
    other = next((c for c in clip_names if c != renderer.current_clip()), None)
    switch_ok = False
    if other:
        before = renderer.current_clip()
        renderer.play(other, "pingpong")
        pump(0.6)
        switch_ok = renderer.current_clip() == other != before
    report["checks"]["clipSwitch"] = {"to": other, "ok": switch_ok}

    # 4. own-window capture (TCC-exempt for own process)
    report["checks"]["windowCapture"] = capture_own_window(
        panel, args.out / "own_window.png")

    CGWarpMouseCursorPosition((original_mouse.x,
                               panel.screen().frame().size.height - original_mouse.y))
    orch.shutdown()

    report["allOk"] = all(c.get("ok", c.get("captured", False))
                          for c in report["checks"].values())
    (args.out / "live_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    raise SystemExit(0 if report["allOk"] else 1)


main()
