"""Transparent always-on-top panel hosting two crossfade layers; resizes per
clip while keeping the manifest anchor point fixed on screen."""

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Quartz import CALayer, CATransaction

PANEL_LEVEL = 25  # NSStatusWindowLevel, proven in the spike
DEFAULT_SCREEN_X_FRACTION = 0.62
SCREEN_FIT_FRACTION = 0.95  # scaled clip may cover at most this much screen


def clamp_user_scale(user_scale: float, native_w_pt: float, native_h_pt: float,
                     visible_w_pt: float, visible_h_pt: float) -> float:
    """Largest scale <= user_scale at which the clip (native point size
    native_w_pt x native_h_pt) fits within SCREEN_FIT_FRACTION of the screen's
    visible frame in BOTH dimensions; aspect is preserved because a single
    factor scales both axes."""
    if native_w_pt <= 0.0 or native_h_pt <= 0.0:
        raise ValueError(
            f"native clip size must be positive, got {native_w_pt}x{native_h_pt}")
    limit = min(visible_w_pt * SCREEN_FIT_FRACTION / native_w_pt,
                visible_h_pt * SCREEN_FIT_FRACTION / native_h_pt)
    return min(user_scale, limit)


class CatContentView(NSView):
    """Layer-hosting content view; forwards mouse events to a Python handler."""

    def initWithFrame_(self, frame):
        self = objc.super(CatContentView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._handler = None
        return self

    def setMouseHandler_(self, handler):
        self._handler = handler

    def acceptsFirstMouse_(self, event):  # noqa: N802
        return True

    def mouseDown_(self, event):  # noqa: N802
        if self._handler is None:
            raise RuntimeError("mouse event arrived before a handler was wired")
        self._handler.on_mouse_down(event)

    def mouseDragged_(self, event):  # noqa: N802
        if self._handler is None:
            raise RuntimeError("mouse event arrived before a handler was wired")
        self._handler.on_mouse_dragged(event)

    def mouseUp_(self, event):  # noqa: N802
        if self._handler is None:
            raise RuntimeError("mouse event arrived before a handler was wired")
        self._handler.on_mouse_up(event)


class OverlayWindow:
    """Owns the NSPanel, the content view, and the two sprite layers. Sizing
    and anchor bookkeeping live here; layer contents/opacity belong to the
    renderer. ignoresMouseEvents starts True and is ours from the first toggle.

    user_scale multiplies every clip's on-screen point size (1.0 = the
    manifest's native displayScale mapping). resize_to_clip clamps the
    EFFECTIVE scale so the clip never exceeds SCREEN_FIT_FRACTION of the
    screen's visible frame; the window is the single source of truth for the
    effective scale — effective_scale()/anchor_offset() (and therefore alpha
    hit-testing and locomotion) always reflect the clamped value, while
    user_scale() keeps reporting the requested one."""

    def __init__(self, user_scale: float = 1.0):
        if user_scale <= 0.0:
            raise ValueError(f"user_scale must be positive, got {user_scale}")
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0.0, 0.0, 2.0, 2.0), style, NSBackingStoreBuffered, False)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(False)
        panel.setLevel_(PANEL_LEVEL)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary)
        panel.setIgnoresMouseEvents_(True)
        panel.setMovableByWindowBackground_(False)

        view = CatContentView.alloc().initWithFrame_(NSMakeRect(0.0, 0.0, 2.0, 2.0))
        view.setWantsLayer_(True)
        layer_a = CALayer.layer()
        layer_b = CALayer.layer()
        for layer in (layer_a, layer_b):
            layer.setFrame_(((0.0, 0.0), (2.0, 2.0)))
            layer.setContentsGravity_("resize")
            view.layer().addSublayer_(layer)
        layer_b.setOpacity_(0.0)
        panel.setContentView_(view)

        self.panel = panel
        self.layer_a = layer_a
        self.layer_b = layer_b
        self._view = view
        self._anchor_offset_pt = None
        self._user_scale = user_scale
        self._effective_user_scale = user_scale
        self._clip_manifest = None
        # float zoom pivot held across a slider drag — re-reading AppKit's
        # integral-rounded frame each tick accumulates ~0.5pt/step drift
        self._zoom_center = None

    def user_scale(self) -> float:
        """The REQUESTED scale (what the slider set), clamp ignored."""
        return self._user_scale

    def set_user_scale(self, value: float) -> None:
        """Re-applies the current clip's geometry immediately, zooming around
        the window CENTER (user expectation for a size control — the old
        feet-anchor zoom read as the cat sliding downward during slider drags);
        anchor bookkeeping stays consistent for subsequent clip switches."""
        if value <= 0.0:
            raise ValueError(f"user_scale must be positive, got {value}")
        self._user_scale = value
        if self._clip_manifest is not None:
            if self._zoom_center is None:
                frame = self.panel.frame()
                self._zoom_center = (frame.origin.x + frame.size.width / 2.0,
                                     frame.origin.y + frame.size.height / 2.0)
            center = self._zoom_center
            self.resize_to_clip(self._clip_manifest)
            self._zoom_center = center  # resize_to_clip invalidates it
            new = self.panel.frame()
            self.panel.setFrameOrigin_((center[0] - new.size.width / 2.0,
                                        center[1] - new.size.height / 2.0))
        else:
            self._effective_user_scale = value

    def scale_clamped(self) -> bool:
        """True when the current clip's effective scale was clamped below the
        requested user scale to fit the screen (False before any clip)."""
        return self._effective_user_scale != self._user_scale

    def effective_scale(self, manifest: dict) -> float:
        """Pixels per point for this clip under the current EFFECTIVE user
        scale (screen-fit clamp applied by resize_to_clip)."""
        return manifest["displayScale"] / self._effective_user_scale

    def _anchor_offset(self, manifest: dict) -> tuple[float, float]:
        # anchor is in pixel coords (top-left origin); offset is window points
        # from the bottom-left corner.
        w_px, h_px = manifest["pixelSize"]
        scale = self.effective_scale(manifest)
        anchor = manifest.get("anchor", [w_px / 2.0, float(h_px)])
        return anchor[0] / scale, (h_px - anchor[1]) / scale

    def resize_to_clip(self, manifest: dict) -> None:
        self._zoom_center = None  # clip switch: geometry changes, pivot is stale
        w_px, h_px = manifest["pixelSize"]
        display_scale = manifest["displayScale"]
        visible = self._visible_frame()
        self._effective_user_scale = clamp_user_scale(
            self._user_scale, w_px / display_scale, h_px / display_scale,
            visible.size.width, visible.size.height)
        scale = self.effective_scale(manifest)
        w_pt, h_pt = w_px / scale, h_px / scale
        offset = self._anchor_offset(manifest)
        if self._anchor_offset_pt is None:
            anchor_x = visible.origin.x + visible.size.width * DEFAULT_SCREEN_X_FRACTION
            anchor_y = visible.origin.y  # desktop floor, above the Dock
        else:
            frame = self.panel.frame()
            anchor_x = frame.origin.x + self._anchor_offset_pt[0]
            anchor_y = frame.origin.y + self._anchor_offset_pt[1]
        self._anchor_offset_pt = offset
        self._clip_manifest = manifest
        self.panel.setFrame_display_(
            NSMakeRect(anchor_x - offset[0], anchor_y - offset[1], w_pt, h_pt), True)
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        for layer in (self.layer_a, self.layer_b):
            layer.setFrame_(((0.0, 0.0), (w_pt, h_pt)))
            layer.setContentsScale_(scale)
        CATransaction.commit()

    def _visible_frame(self):
        screen = NSScreen.mainScreen()
        if screen is None:
            raise RuntimeError("no screen available for clip sizing")
        return screen.visibleFrame()

    def anchor_offset(self) -> tuple[float, float]:
        """Current clip's anchor as points from the window's bottom-left."""
        if self._anchor_offset_pt is None:
            raise RuntimeError("anchor offset queried before the first clip resize")
        return self._anchor_offset_pt

    def set_mouse_handler(self, handler) -> None:
        self._view.setMouseHandler_(handler)

    def frame(self) -> tuple[float, float, float, float]:
        f = self.panel.frame()
        return (f.origin.x, f.origin.y, f.size.width, f.size.height)

    def set_frame_origin(self, x: float, y: float) -> None:
        self._zoom_center = None  # cat moved: next zoom pivots on the new spot
        self.panel.setFrameOrigin_((x, y))

    def set_ignores_mouse_events(self, flag: bool) -> None:
        self.panel.setIgnoresMouseEvents_(flag)

    def order_front(self) -> None:
        self.panel.orderFrontRegardless()
