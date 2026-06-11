"""Composition root: loads the clip index and manifests, wires window /
renderer / interactions / status item / brain, owns pause state and the
occlusion + screen-sleep handling."""

import json
import random
import time
from pathlib import Path
from typing import Callable

from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSWindowDidChangeOcclusionStateNotification,
    NSWindowOcclusionStateVisible,
    NSWorkspace,
    NSWorkspaceScreensDidSleepNotification,
    NSWorkspaceScreensDidWakeNotification,
)
from Foundation import NSNotificationCenter

from ..brain.activities import build_catalog
from ..brain.commands import BrainEvent
from ..brain.facade import Brain
from ..brain.store import PersistenceStore
from .interactions import InteractionMonitor
from .locomotion import WALK_SPEED_PT_S, Locomotion
from .renderer import ClipLoader, SpriteRenderer
from .runtime_support import CallbackTarget, schedule_timer
from .settings import SettingsStore
from .status_item import StatusItemController
from .window import OverlayWindow

BRAIN_TICK_INTERVAL_S = 1.0
BRAIN_TICK_TOLERANCE_S = 0.3
DEFAULT_USER_SCALE = 1.0
DEFAULT_SETTINGS_PATH = (Path.home() / "Library" / "Application Support"
                         / "OverlayCat" / "settings.json")


class RuntimeClock:
    def now(self) -> float:
        return time.monotonic()

    def wall(self) -> float:
        return time.time()


class AppOrchestrator:
    """brain_factory(index: dict, state_path: Path, clock, rng) returns a
    Brain-shaped object; injected by tests/smoke, omitted in production
    (real overlay.brain wiring)."""

    def __init__(self, clips_index_path: Path, state_path: Path,
                 settings_path: Path | None = None,
                 brain_factory: Callable | None = None):
        self._clips_index_path = Path(clips_index_path)
        self._state_path = Path(state_path)
        self._settings_store = SettingsStore(
            settings_path if settings_path is not None else DEFAULT_SETTINGS_PATH)
        self._brain_factory = brain_factory
        self._pending_events = []
        self._user_paused = False
        self._occluded = False
        self._screen_asleep = False
        self._running = False
        self._brain_timer = None
        self._observer_targets = []
        self._app = None
        self._brain = None
        self._walk_loop = None
        self.window = None
        self.renderer = None
        self.monitor = None
        self.locomotion = None
        self.status_item = None

    def start(self) -> None:
        index = json.loads(self._clips_index_path.read_text())
        manifests, clip_dirs = self._load_manifests(index)
        user_scale = self._load_user_scale()
        self._app = NSApplication.sharedApplication()
        self._app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        self.window = OverlayWindow(user_scale=user_scale)
        loader = ClipLoader(clip_dirs, manifests)
        self.renderer = SpriteRenderer(self.window, manifests, loader)
        clock = RuntimeClock()
        rng = random.Random()
        if self._brain_factory is None:
            catalog = build_catalog(index, manifests)
            store = PersistenceStore(self._state_path)
            self._brain = Brain(clock, rng, catalog, store)
        else:
            self._brain = self._brain_factory(index, self._state_path, clock, rng)
        self.monitor = InteractionMonitor(self.renderer, self._on_brain_event)
        self.window.set_mouse_handler(self.monitor)
        self.locomotion = Locomotion(
            self.renderer, self._on_edge_swap, self.monitor.is_dragging)
        self.status_item = StatusItemController(
            self.toggle_pause, self.quit, self.set_user_scale, user_scale)
        self.status_item.build()
        self._register_observers()
        self.monitor.start()
        self._brain_timer = schedule_timer(
            BRAIN_TICK_INTERVAL_S, lambda timer: self._tick_brain(),
            tolerance=BRAIN_TICK_TOLERANCE_S)
        self._running = True
        self._tick_brain()
        self.window.order_front()

    def run(self) -> None:
        self.start()
        self._app.run()

    def shutdown(self) -> None:
        self._running = False
        if self._brain_timer is not None:
            self._brain_timer.invalidate()
            self._brain_timer = None
        if self.locomotion is not None:
            self.locomotion.stop()
        if self.monitor is not None:
            self.monitor.pause()
        if self.renderer is not None:
            self.renderer.pause()
        center = NSNotificationCenter.defaultCenter()
        workspace_center = NSWorkspace.sharedWorkspace().notificationCenter()
        for target in self._observer_targets:
            center.removeObserver_(target)
            workspace_center.removeObserver_(target)
        self._observer_targets = []
        if self._brain is not None:
            self._brain.shutdown()
            self._brain = None

    def quit(self) -> None:
        self.shutdown()
        self._app.terminate_(None)

    def toggle_pause(self) -> bool:
        self._user_paused = not self._user_paused
        self._apply_pause_state()
        return self._user_paused

    # -- user scale ------------------------------------------------------------
    def set_user_scale(self, value: float) -> bool:
        """Size-slider handler (fires continuously during drag): re-applies
        geometry now, persists the REQUESTED scale, and returns True when the
        window clamped the effective scale to fit the screen."""
        self.window.set_user_scale(value)
        self._settings_store.save({"user_scale": value})
        return self.window.scale_clamped()

    def _load_user_scale(self) -> float:
        settings = self._settings_store.load()
        if settings is None:
            return DEFAULT_USER_SCALE
        user_scale = settings["user_scale"]  # loud KeyError on a malformed file
        if (isinstance(user_scale, bool) or not isinstance(user_scale, (int, float))
                or user_scale <= 0):
            raise RuntimeError(
                f"settings user_scale must be a positive number, got {user_scale!r}")
        return float(user_scale)

    # -- brain wiring --------------------------------------------------------
    def _on_brain_event(self, event) -> None:
        self._pending_events.append(event)
        self._tick_brain()  # contract: tick immediately after events

    def _tick_brain(self) -> None:
        if not self._running:
            return
        events = self._pending_events
        self._pending_events = []
        command = self._brain.tick(events)
        if command is None:
            return
        self.renderer.play(command.clip, command.loop)
        # a different-sized clip can change the screen-fit clamp state
        self.status_item.set_clamped(self.window.scale_clamped())
        # PlayClip.motion is contract-optional with default None; Brain-shaped
        # seam objects without the field simply never walk
        motion = getattr(command, "motion", None)
        if motion is None:
            self.locomotion.stop()
        else:
            self._walk_loop = command.loop
            self.locomotion.start(motion, self._walk_speed(command.clip))

    def _walk_speed(self, clip: str) -> float:
        manifest = self.renderer.available_clips()[clip]
        return float(manifest.get("walkSpeedPtS", WALK_SPEED_PT_S))

    def _on_edge_swap(self, direction: str) -> None:
        current = self.renderer.current_clip()
        src, dst = (("walk_left", "walk_right") if direction == "right"
                    else ("walk_right", "walk_left"))
        if src not in current:
            raise RuntimeError(
                f"walking clip {current!r} has no {src!r} substring to swap")
        opposite = current.replace(src, dst)
        if opposite not in self.renderer.available_clips():
            raise RuntimeError(
                f"opposite walk clip {opposite!r} for {current!r} is not in the index")
        self.renderer.play(opposite, self._walk_loop)
        self.status_item.set_clamped(self.window.scale_clamped())

    # -- pause / occlusion / screen sleep -------------------------------------
    def _apply_pause_state(self) -> None:
        should_run = not (self._user_paused or self._occluded or self._screen_asleep)
        if should_run == self._running:
            return
        if should_run:
            self._running = True
            self.renderer.resume()
            self.monitor.resume()
            self.locomotion.resume()
            self._brain_timer = schedule_timer(
                BRAIN_TICK_INTERVAL_S, lambda timer: self._tick_brain(),
                tolerance=BRAIN_TICK_TOLERANCE_S)
            # brain fast-forwards its needs internally on "wake"
            self._on_brain_event(BrainEvent("wake", time.monotonic()))
        else:
            self._running = False
            if self._brain_timer is not None:
                self._brain_timer.invalidate()
                self._brain_timer = None
            self.renderer.pause()
            self.monitor.pause()
            self.locomotion.pause()

    def _register_observers(self) -> None:
        occlusion_target = CallbackTarget.alloc().initWithCallback_(
            self._on_occlusion_changed)
        NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            occlusion_target, b"fire:",
            NSWindowDidChangeOcclusionStateNotification, self.window.panel)
        workspace_center = NSWorkspace.sharedWorkspace().notificationCenter()
        sleep_target = CallbackTarget.alloc().initWithCallback_(self._on_screens_sleep)
        wake_target = CallbackTarget.alloc().initWithCallback_(self._on_screens_wake)
        workspace_center.addObserver_selector_name_object_(
            sleep_target, b"fire:", NSWorkspaceScreensDidSleepNotification, None)
        workspace_center.addObserver_selector_name_object_(
            wake_target, b"fire:", NSWorkspaceScreensDidWakeNotification, None)
        # notification centers hold observers weakly; keep them alive here
        self._observer_targets = [occlusion_target, sleep_target, wake_target]

    def _on_occlusion_changed(self, notification) -> None:
        state = self.window.panel.occlusionState()
        self._occluded = not bool(state & NSWindowOcclusionStateVisible)
        self._apply_pause_state()

    def _on_screens_sleep(self, notification) -> None:
        self._screen_asleep = True
        self._apply_pause_state()

    def _on_screens_wake(self, notification) -> None:
        self._screen_asleep = False
        self._apply_pause_state()

    # -- index / manifests -----------------------------------------------------
    def _load_manifests(self, index: dict) -> tuple[dict, dict]:
        entries = index["clips"]
        if not entries:
            raise RuntimeError(f"clip index {self._clips_index_path} is empty")
        # index "dir" entries like "clips/sleep_curl" are relative to the
        # directory that contains the clips folder
        root = self._clips_index_path.resolve().parent.parent
        manifests = {}
        clip_dirs = {}
        for entry in entries:
            raw_dir = Path(entry["dir"])
            clip_dir = raw_dir if raw_dir.is_absolute() else root / raw_dir
            manifest = json.loads((clip_dir / "manifest.json").read_text())
            name = entry["name"]
            if manifest["name"] != name:
                raise RuntimeError(
                    f"index name {name!r} != manifest name {manifest['name']!r} "
                    f"in {clip_dir}")
            manifests[name] = manifest
            clip_dirs[name] = clip_dir
        return manifests, clip_dirs
