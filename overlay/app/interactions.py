"""Cursor proximity polling, alpha-driven interactivity toggling, drag,
click/double-click, and stroke ('pet') detection. Emits BrainEvent via
on_event; a single click is debounced CLICK_DEBOUNCE_S so a double-click
emits only 'double_click', never 'click' as well."""

import math
import time
from collections import deque
from typing import Callable

from AppKit import NSEvent

from ..brain.commands import BrainEvent
from .renderer import Renderer
from .runtime_support import schedule_timer

ALPHA_THRESHOLD = 24
PROXIMITY_MARGIN_PT = 48.0
SLOW_INTERVAL_S = 1.0
FAST_INTERVAL_S = 1.0 / 15.0
PET_MIN_PATH_PT = 120.0
PET_WINDOW_S = 0.7
PET_THROTTLE_S = 2.0
CLICK_DEBOUNCE_S = 0.25  # single-click emission delay; cancelled by clickCount 2


class InteractionMonitor:
    """1 Hz proximity check escalating to 15 Hz polling near the window.
    Requires the concrete SpriteRenderer surface (set_interactive in addition
    to the Renderer protocol). Wire as the window's mouse handler."""

    def __init__(self, renderer: Renderer, on_event: Callable[[BrainEvent], None]):
        self._renderer = renderer
        self._on_event = on_event
        self._slow_timer = None
        self._fast_timer = None
        self._interactive = False
        self._drag_offset = None
        self._dragged = False
        self._stroke = deque()  # (monotonic time, path-length increment)
        self._last_stroke_point = None
        self._last_pet_at = float("-inf")
        self._click_timer = None  # pending debounced single-click emission

    def start(self) -> None:
        if self._slow_timer is None:
            self._slow_timer = schedule_timer(SLOW_INTERVAL_S, self._tick_slow)

    def pause(self) -> None:
        if self._slow_timer is not None:
            self._slow_timer.invalidate()
            self._slow_timer = None
        self._stop_fast()
        self._cancel_pending_click()
        self._set_interactive(False)
        self._drag_offset = None
        self._dragged = False
        self._reset_stroke()

    def resume(self) -> None:
        self.start()

    def is_dragging(self) -> bool:
        """True from the first dragged movement until mouse-up (locomotion
        suspends while this holds)."""
        return self._dragged

    # -- mouse handler (called by CatContentView) ---------------------------
    def on_mouse_down(self, event) -> None:
        loc = NSEvent.mouseLocation()
        fx, fy, _, _ = self._renderer.window_frame()
        self._drag_offset = (loc.x - fx, loc.y - fy)
        self._dragged = False

    def on_mouse_dragged(self, event) -> None:
        if self._drag_offset is None:
            return
        if not self._dragged:
            self._dragged = True
            self._emit("drag_start")
        loc = NSEvent.mouseLocation()
        self._renderer.set_frame_origin(
            loc.x - self._drag_offset[0], loc.y - self._drag_offset[1])

    def on_mouse_up(self, event) -> None:
        if self._drag_offset is None:
            return
        if self._dragged:
            self._emit("drag_end")
        else:
            self._renderer.boop()
            if event.clickCount() == 2:
                # a double-click replaces the pending single click entirely
                self._cancel_pending_click()
                self._emit("double_click")
            elif event.clickCount() == 1:
                self._cancel_pending_click()
                self._click_timer = schedule_timer(
                    CLICK_DEBOUNCE_S, self._fire_pending_click, repeats=False)
        self._drag_offset = None
        self._dragged = False

    def _fire_pending_click(self, timer) -> None:
        self._click_timer = None
        self._emit("click")

    def _cancel_pending_click(self) -> None:
        if self._click_timer is not None:
            self._click_timer.invalidate()
            self._click_timer = None

    # -- polling -------------------------------------------------------------
    def _tick_slow(self, timer) -> None:
        if self._fast_timer is not None:
            return
        if self._near_window(NSEvent.mouseLocation()):
            self._fast_timer = schedule_timer(FAST_INTERVAL_S, self._tick_fast)

    def _tick_fast(self, timer) -> None:
        if self._drag_offset is not None:
            return  # stay interactive for the whole drag
        loc = NSEvent.mouseLocation()
        if not self._near_window(loc):
            self._stop_fast()
            self._set_interactive(False)
            self._reset_stroke()
            return
        fx, fy, fw, fh = self._renderer.window_frame()
        inside = fx <= loc.x <= fx + fw and fy <= loc.y <= fy + fh
        over_cat = False
        if inside:
            alpha = self._renderer.current_alpha_at(loc.x - fx, loc.y - fy)
            over_cat = alpha > ALPHA_THRESHOLD
        self._set_interactive(over_cat)
        now = time.monotonic()
        if over_cat and self._last_stroke_point is not None:
            dx = loc.x - self._last_stroke_point[0]
            dy = loc.y - self._last_stroke_point[1]
            self._stroke.append((now, math.hypot(dx, dy)))
        self._last_stroke_point = (loc.x, loc.y) if over_cat else None
        while self._stroke and now - self._stroke[0][0] > PET_WINDOW_S:
            self._stroke.popleft()
        if (sum(d for _, d in self._stroke) >= PET_MIN_PATH_PT
                and now - self._last_pet_at >= PET_THROTTLE_S):
            self._last_pet_at = now
            self._reset_stroke()
            self._emit("pet")

    def _near_window(self, loc) -> bool:
        fx, fy, fw, fh = self._renderer.window_frame()
        return (fx - PROXIMITY_MARGIN_PT <= loc.x <= fx + fw + PROXIMITY_MARGIN_PT
                and fy - PROXIMITY_MARGIN_PT <= loc.y <= fy + fh + PROXIMITY_MARGIN_PT)

    def _stop_fast(self) -> None:
        if self._fast_timer is not None:
            self._fast_timer.invalidate()
            self._fast_timer = None

    def _set_interactive(self, interactive: bool) -> None:
        if interactive == self._interactive:
            return
        self._interactive = interactive
        self._renderer.set_interactive(interactive)

    def _reset_stroke(self) -> None:
        self._stroke.clear()
        self._last_stroke_point = None

    def _emit(self, kind: str) -> None:
        self._on_event(BrainEvent(kind, time.monotonic()))
