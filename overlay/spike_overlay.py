"""Phase-0 overlay spike: play an RGBA frame loop in a transparent, borderless,
always-on-top panel where ONLY the cat's pixels are interactive.

Proves on macOS 13.4.1: transparent NSPanel over full-screen Spaces, CALayer
frame playback, explicit per-pixel hit-testing via alpha sampling +
ignoresMouseEvents toggling (never the undocumented automatic pass-through),
drag-the-cat, click reaction, and idle CPU behavior.

Usage: python -B spike_overlay.py --clip <dir-with-rgba-pngs> --fps 12 [--duration 60]
"""

import sys

sys.dont_write_bytecode = True

import argparse
import time
from pathlib import Path

import numpy as np
import objc
from PIL import Image as PILImage
from AppKit import (
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSColor,
    NSEvent,
    NSImage,
    NSMakeRect,
    NSMenu,
    NSMenuItem,
    NSPanel,
    NSScreen,
    NSStatusBar,
    NSView,
    NSVariableStatusItemLength,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSObject, NSRunLoop, NSRunLoopCommonModes, NSTimer
from Quartz import CABasicAnimation, CALayer

DISPLAY_SCALE = 2.0  # assets are authored at 2x for the retina display


class ClipFrames:
    """Loads an RGBA PNG sequence once; owns pixel data and per-frame alpha."""

    def __init__(self, clip_dir: Path):
        paths = sorted(clip_dir.glob("*.png"))
        if not paths:
            raise RuntimeError(f"no PNG frames in {clip_dir}")
        self.images = [NSImage.alloc().initWithContentsOfFile_(str(p)) for p in paths]
        self.alphas = [np.asarray(PILImage.open(p))[..., 3] for p in paths]
        h, w = self.alphas[0].shape
        self.pixel_size = (w, h)
        self.count = len(paths)

    def alpha_at(self, frame_index: int, x_px: float, y_px_top: float) -> int:
        a = self.alphas[frame_index]
        xi, yi = int(x_px), int(y_px_top)
        if 0 <= yi < a.shape[0] and 0 <= xi < a.shape[1]:
            return int(a[yi, xi])
        return 0


class CatView(NSView):
    """Layer-hosting view; forwards mouse events to the controller."""

    def initWithFrame_controller_(self, frame, controller):
        self = objc.super(CatView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._controller = controller
        return self

    def acceptsFirstMouse_(self, event):  # noqa: N802
        return True

    def mouseDown_(self, event):  # noqa: N802
        self._controller.onMouseDown_(event)

    def mouseDragged_(self, event):  # noqa: N802
        self._controller.onMouseDragged_(event)

    def mouseUp_(self, event):  # noqa: N802
        self._controller.onMouseUp_(event)


class OverlayController(NSObject):
    def initWithClip_fps_duration_(self, clip: ClipFrames, fps: float, duration: float):
        self = objc.super(OverlayController, self).init()
        if self is None:
            return None
        self.clip = clip
        self.fps = fps
        self.duration = duration
        self.frame_index = 0
        self.direction = 1  # ping-pong looping: no visible seam on a breathing loop
        self.drag_offset = None
        self.dragged = False
        self.started = time.monotonic()
        self.poll_hits = {"interactive": False}
        return self

    # -- window/layer setup ------------------------------------------------
    def buildWindow(self):
        w_px, h_px = self.clip.pixel_size
        w_pt, h_pt = w_px / DISPLAY_SCALE, h_px / DISPLAY_SCALE
        screen = NSScreen.mainScreen().visibleFrame()
        origin_x = screen.origin.x + screen.size.width * 0.62
        origin_y = screen.origin.y + 0.0  # sit on the desktop floor (above Dock)

        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(origin_x, origin_y, w_pt, h_pt), style, NSBackingStoreBuffered, False)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(False)
        panel.setLevel_(25)  # NSStatusWindowLevel: above normal windows
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary)
        panel.setIgnoresMouseEvents_(True)  # we own hit-testing from here on
        panel.setMovableByWindowBackground_(False)

        view = CatView.alloc().initWithFrame_controller_(
            NSMakeRect(0, 0, w_pt, h_pt), self)
        view.setWantsLayer_(True)
        layer = CALayer.layer()
        layer.setFrame_(((0, 0), (w_pt, h_pt)))
        layer.setContentsGravity_("resize")
        layer.setContentsScale_(DISPLAY_SCALE)
        view.layer().addSublayer_(layer)
        self.cat_layer = layer
        panel.setContentView_(view)
        panel.orderFrontRegardless()
        self.panel = panel
        self.showFrame_(0)

    def showFrame_(self, idx):
        self.cat_layer.setContents_(self.clip.images[idx])

    # -- timers ------------------------------------------------------------
    def startTimers(self):
        self.frame_timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / self.fps, self, b"tickFrame:", None, True)
        self.hit_timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / 15.0, self, b"tickHitTest:", None, True)
        run_loop = NSRunLoop.currentRunLoop()
        run_loop.addTimer_forMode_(self.frame_timer, NSRunLoopCommonModes)
        run_loop.addTimer_forMode_(self.hit_timer, NSRunLoopCommonModes)

    def tickFrame_(self, timer):
        nxt = self.frame_index + self.direction
        if nxt >= self.clip.count or nxt < 0:
            self.direction *= -1
            nxt = self.frame_index + self.direction
        self.frame_index = nxt
        self.showFrame_(nxt)
        if self.duration and time.monotonic() - self.started > self.duration:
            NSApp.terminate_(None)

    def tickHitTest_(self, timer):
        if self.drag_offset is not None:
            return  # stay interactive during a drag
        loc = NSEvent.mouseLocation()
        f = self.panel.frame()
        inside = (f.origin.x <= loc.x <= f.origin.x + f.size.width
                  and f.origin.y <= loc.y <= f.origin.y + f.size.height)
        interactive = False
        if inside:
            x_pt = loc.x - f.origin.x
            y_pt_bottom = loc.y - f.origin.y
            x_px = x_pt * DISPLAY_SCALE
            y_px_top = (f.size.height - y_pt_bottom) * DISPLAY_SCALE
            interactive = self.clip.alpha_at(self.frame_index, x_px, y_px_top) > 24
        if interactive != self.poll_hits["interactive"]:
            self.poll_hits["interactive"] = interactive
            self.panel.setIgnoresMouseEvents_(not interactive)

    # -- interactions --------------------------------------------------------
    def onMouseDown_(self, event):
        loc = NSEvent.mouseLocation()
        f = self.panel.frame()
        self.drag_offset = (loc.x - f.origin.x, loc.y - f.origin.y)
        self.dragged = False

    def onMouseDragged_(self, event):
        if self.drag_offset is None:
            return
        self.dragged = True
        loc = NSEvent.mouseLocation()
        self.panel.setFrameOrigin_((loc.x - self.drag_offset[0], loc.y - self.drag_offset[1]))

    def onMouseUp_(self, event):
        if not self.dragged:
            self.boop()
        self.drag_offset = None

    def boop(self):
        anim = CABasicAnimation.animationWithKeyPath_("transform.scale")
        anim.setFromValue_(1.0)
        anim.setToValue_(1.05)
        anim.setDuration_(0.09)
        anim.setAutoreverses_(True)
        self.cat_layer.addAnimation_forKey_(anim, "boop")

    # -- status item ---------------------------------------------------------
    def buildStatusItem(self):
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength)
        self.status_item.button().setTitle_("🐱")
        menu = NSMenu.alloc().init()
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit Overlay Cat", b"terminate:", "q")
        menu.addItem_(quit_item)
        self.status_item.setMenu_(menu)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="auto-quit after N seconds (0 = run until quit)")
    args = parser.parse_args()

    clip = ClipFrames(args.clip)
    print(f"loaded {clip.count} frames @ {clip.pixel_size}px "
          f"-> window {clip.pixel_size[0]/DISPLAY_SCALE:.0f}x{clip.pixel_size[1]/DISPLAY_SCALE:.0f}pt")

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    controller = OverlayController.alloc().initWithClip_fps_duration_(
        clip, args.fps, args.duration)
    controller.buildWindow()
    controller.buildStatusItem()
    controller.startTimers()
    print("overlay running — cat pixels are clickable/draggable; 🐱 menu bar item to quit")
    app.run()


main()
