"""Horizontal walk locomotion: moves the overlay window along the screen's
visible-frame floor on its own 30 Hz NSTimer, reversing velocity at screen
edges and notifying the orchestrator so it can swap to the opposite-direction
walk clip. Movement is suspended while a drag is active and resumes from the
drop position."""

import time
from typing import Callable

from AppKit import NSScreen

from .renderer import Renderer
from .runtime_support import schedule_timer

WALK_SPEED_PT_S = 60.0          # default; manifests may override via "walkSpeedPtS"
TICK_INTERVAL_S = 1.0 / 30.0
EDGE_MARGIN_PT = 8.0
MAX_TICK_DT_S = 0.25            # run-loop stalls advance the cat at most this far


class Locomotion:
    """Owns walk velocity, the floor line, and edge detection. The orchestrator
    starts/stops it per brain command; on_edge_swap(new_direction) fires after
    the velocity has been reversed at a screen edge so the orchestrator can play
    the opposite-direction clip of the same pair (no brain round trip)."""

    def __init__(self, renderer: Renderer, on_edge_swap: Callable[[str], None],
                 drag_active: Callable[[], bool]):
        self._renderer = renderer
        self._on_edge_swap = on_edge_swap
        self._drag_active = drag_active
        self._timer = None
        self._velocity = 0.0
        self._direction = None
        self._last_tick = None
        self._paused = False

    def start(self, direction: str, speed_pt_s: float) -> None:
        if direction not in ("left", "right"):
            raise ValueError(f"unknown walk direction {direction!r}")
        if speed_pt_s <= 0.0:
            raise ValueError(f"walk speed must be positive, got {speed_pt_s}")
        self._direction = direction
        self._velocity = speed_pt_s if direction == "right" else -speed_pt_s
        self._last_tick = None
        if self._timer is None and not self._paused:
            self._timer = schedule_timer(TICK_INTERVAL_S, self._tick)

    def stop(self) -> None:
        self._direction = None
        self._velocity = 0.0
        self._last_tick = None
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None

    def pause(self) -> None:
        """App-level pause (occlusion / screen sleep / user pause); the walk
        state survives and resume() continues it."""
        self._paused = True
        self._last_tick = None
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None

    def resume(self) -> None:
        self._paused = False
        if self._direction is not None and self._timer is None:
            self._timer = schedule_timer(TICK_INTERVAL_S, self._tick)

    def active(self) -> bool:
        return self._direction is not None

    def direction(self) -> str | None:
        return self._direction

    def floor_window_y(self) -> float:
        """Window y that puts the clip's anchor on the visible-frame floor."""
        return self._visible_frame().origin.y - self._renderer.anchor_offset()[1]

    def _visible_frame(self):
        screen = NSScreen.mainScreen()
        if screen is None:
            raise RuntimeError("no screen available for locomotion")
        return screen.visibleFrame()

    def _tick(self, timer) -> None:
        if self._drag_active():
            self._last_tick = None  # resume from the drop position after land
            return
        now = time.monotonic()
        if self._last_tick is None:
            self._last_tick = now
            return
        dt = min(now - self._last_tick, MAX_TICK_DT_S)
        self._last_tick = now
        fx, _, fw, _ = self._renderer.window_frame()
        x = fx + self._velocity * dt
        visible = self._visible_frame()
        min_x = visible.origin.x
        max_x = visible.origin.x + visible.size.width
        if self._velocity > 0.0 and x + fw >= max_x - EDGE_MARGIN_PT:
            x = max_x - EDGE_MARGIN_PT - fw
            self._reverse()
        elif self._velocity < 0.0 and x <= min_x + EDGE_MARGIN_PT:
            x = min_x + EDGE_MARGIN_PT
            self._reverse()
        self._renderer.set_frame_origin(x, self.floor_window_y())

    def _reverse(self) -> None:
        self._velocity = -self._velocity
        self._direction = "left" if self._direction == "right" else "right"
        self._on_edge_swap(self._direction)
