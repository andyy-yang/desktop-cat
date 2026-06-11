"""CALayer sprite renderer: two layers crossfading 200 ms between clips, a
per-clip NSTimer at the clip's native fps. Frames are NOT kept decoded: each
tick creates the displayed frame's NSImage from its PNG path and the CALayer's
retention is the only thing keeping it (and its ~5 MB decode cache) alive.
Per-clip state in memory is just frame paths + packbits alpha masks, current
clip only. Crossfade needs no decoded previous clip: the outgoing CALayer
retains the single NSImage it is already displaying."""

from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import numpy as np
from AppKit import NSImage
from PIL import Image as PILImage
from Quartz import CABasicAnimation, CATransaction

from .runtime_support import schedule_timer
from .window import OverlayWindow

CROSSFADE_SECONDS = 0.2
# A stretched outgoing pose reads as a deformed second cat, so the 200 ms fade
# only runs between clips whose point sizes agree within 15% per dimension;
# larger mismatches hard-cut (outgoing opacity 0 immediately).
CROSSFADE_MAX_SIZE_RATIO = 1.15
DECODED_CLIP_CAPACITY = 1  # current only; previous survives as one CALayer image
# Decoded-frame cache: a clip's frames are kept as NSImages ONLY if the whole
# clip fits this budget (partial caching is useless under ping-pong cycling —
# LRU evicts exactly the frame needed next). Steady-state loops then decode 0
# frames/tick; oversized clips fall back to per-tick decode (CPU for memory).
FRAME_CACHE_BUDGET_BYTES = 200 * 1024 * 1024
LOOP_MODES = ("pingpong", "forward", "once")
BOOP_SCALE = 1.05
BOOP_SECONDS = 0.09
ALPHA_THRESHOLD = 24  # must equal interactions.ALPHA_THRESHOLD (import would cycle)


def crossfade_allowed(outgoing: dict, incoming: dict) -> bool:
    """True when both manifests' point sizes agree within
    CROSSFADE_MAX_SIZE_RATIO in BOTH dimensions (user_scale multiplies both
    clips identically, so the manifest-only ratio is exact)."""
    for axis in (0, 1):
        a = outgoing["pixelSize"][axis] / outgoing["displayScale"]
        b = incoming["pixelSize"][axis] / incoming["displayScale"]
        if max(a, b) / min(a, b) > CROSSFADE_MAX_SIZE_RATIO + 1e-9:
            return False
    return True


class Renderer(Protocol):
    def available_clips(self) -> dict[str, dict]: ...
    def play(self, clip: str, loop: str) -> None: ...
    def current_clip(self) -> str | None: ...
    def current_alpha_at(self, x_pt: float, y_pt_bottom: float) -> int: ...
    def window_frame(self) -> tuple[float, float, float, float]: ...
    def set_frame_origin(self, x: float, y: float) -> None: ...
    def anchor_offset(self) -> tuple[float, float]: ...
    def boop(self) -> None: ...


class DecodedClip:
    """Dumb holder for one clip's frame paths and per-frame packbits masks
    (alpha > ALPHA_THRESHOLD, MSB-first along rows)."""

    def __init__(self, manifest: dict, frame_paths: list, masks: list):
        self.manifest = manifest
        self.frame_paths = frame_paths
        self.masks = masks
        self.count = len(frame_paths)
        self.width, self.height = manifest["pixelSize"]

    def alpha_at(self, frame_index: int, x_px: float, y_px_top: float) -> int:
        xi, yi = int(x_px), int(y_px_top)
        if 0 <= yi < self.height and 0 <= xi < self.width:
            byte = self.masks[frame_index][yi, xi >> 3]
            if (byte >> (7 - (xi & 7))) & 1:
                return 255
        return 0

    def mask_at(self, frame_index: int) -> np.ndarray:
        return np.unpackbits(
            self.masks[frame_index], axis=1, count=self.width).astype(bool)

    def release(self) -> None:
        self.frame_paths = None
        self.masks = None
        self.count = 0


class ClipLoader:
    """Builds a clip's packed alpha masks; frames stay on disk as paths."""

    def __init__(self, clip_dirs: dict[str, Path], manifests: dict[str, dict]):
        self._clip_dirs = clip_dirs
        self._manifests = manifests

    def load(self, name: str) -> DecodedClip:
        manifest = self._manifests[name]
        frames_dir = self._clip_dirs[name] / "frames"
        paths = sorted(frames_dir.glob("*.png"))
        if not paths:
            raise RuntimeError(f"no PNG frames in {frames_dir}")
        if len(paths) != manifest["frameCount"]:
            raise RuntimeError(
                f"{name}: manifest frameCount {manifest['frameCount']} != "
                f"{len(paths)} files in {frames_dir}")
        w_px, h_px = manifest["pixelSize"]
        masks = []
        for path in paths:
            pixels = np.asarray(PILImage.open(path))
            if pixels.ndim != 3 or pixels.shape[2] != 4:
                raise RuntimeError(f"{path} is not RGBA")
            if pixels.shape[:2] != (h_px, w_px):
                raise RuntimeError(
                    f"{name}: frame size {pixels.shape[1::-1]} != manifest "
                    f"pixelSize {(w_px, h_px)} in {path}")
            masks.append(np.packbits(pixels[..., 3] > ALPHA_THRESHOLD, axis=1))
            del pixels
        return DecodedClip(manifest, paths, masks)


class SpriteRenderer:
    def __init__(self, window: OverlayWindow, manifests: dict[str, dict],
                 loader: ClipLoader):
        self._window = window
        self._manifests = manifests
        self._loader = loader
        self._decoded = OrderedDict()
        self._front = window.layer_a
        self._back = window.layer_b
        self._current = None
        self._clip = None
        self._loop = None
        self._frame_index = 0
        self._direction = 1
        self._finished = False
        self._paused = False
        self._timer = None
        self._frames_rendered = 0
        self._frame_cache: dict[int, NSImage] = {}

    def available_clips(self) -> dict[str, dict]:
        return dict(self._manifests)

    def play(self, clip: str, loop: str) -> None:
        if loop not in LOOP_MODES:
            raise ValueError(f"unknown loop mode {loop!r}")
        if clip not in self._manifests:
            raise KeyError(f"unknown clip {clip!r}")
        if clip == self._current:
            if loop != self._loop:
                self._loop = loop
                self._finished = False
                self._restart_timer()
            return
        # release the outgoing clip BEFORE loading the incoming one; the front
        # layer keeps the last displayed NSImage alive for the crossfade
        outgoing_manifest = self._clip.manifest if self._clip is not None else None
        self._clip = None
        self._current = None
        self._frame_cache.clear()
        decoded = self._decoded_clip(clip)
        self._window.resize_to_clip(decoded.manifest)
        incoming, outgoing = self._back, self._front
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        incoming.setContents_(self._frame_image(decoded, 0))
        CATransaction.commit()
        fade = (outgoing_manifest is None
                or crossfade_allowed(outgoing_manifest, decoded.manifest))
        CATransaction.begin()
        if fade:
            CATransaction.setAnimationDuration_(CROSSFADE_SECONDS)
        else:
            CATransaction.setDisableActions_(True)
        incoming.setOpacity_(1.0)
        outgoing.setOpacity_(0.0)
        CATransaction.commit()
        self._front, self._back = incoming, outgoing
        self._current = clip
        self._clip = decoded
        self._loop = loop
        self._frame_index = 0
        self._direction = 1
        self._finished = False
        self._restart_timer()

    def current_alpha_at(self, x_pt: float, y_pt_bottom: float) -> int:
        if self._clip is None:
            return 0
        manifest = self._clip.manifest
        scale = self._window.effective_scale(manifest)  # user_scale + fit clamp
        h_px = manifest["pixelSize"][1]
        return self._clip.alpha_at(
            self._frame_index, x_pt * scale, h_px - y_pt_bottom * scale)

    def window_frame(self) -> tuple[float, float, float, float]:
        return self._window.frame()

    def set_frame_origin(self, x: float, y: float) -> None:
        self._window.set_frame_origin(x, y)

    def anchor_offset(self) -> tuple[float, float]:
        return self._window.anchor_offset()

    def boop(self) -> None:
        anim = CABasicAnimation.animationWithKeyPath_("transform.scale")
        anim.setFromValue_(1.0)
        anim.setToValue_(BOOP_SCALE)
        anim.setDuration_(BOOP_SECONDS)
        anim.setAutoreverses_(True)
        self._front.addAnimation_forKey_(anim, "boop")

    def set_interactive(self, interactive: bool) -> None:
        self._window.set_ignores_mouse_events(not interactive)

    def pause(self) -> None:
        self._paused = True
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None

    def resume(self) -> None:
        self._paused = False
        if self._clip is not None:
            self._restart_timer()

    def current_clip(self) -> str | None:
        return self._current

    def debug_state(self) -> dict:
        return {
            "clip": self._current,
            "loop": self._loop,
            "frame_index": self._frame_index,
            "frames_rendered": self._frames_rendered,
            "decoded": list(self._decoded.keys()),
        }

    def _frame_image(self, clip: DecodedClip, frame_index: int) -> NSImage:
        cached = self._frame_cache.get(frame_index)
        if cached is not None:
            return cached
        path = clip.frame_paths[frame_index]
        image = NSImage.alloc().initWithContentsOfFile_(str(path))
        if image is None:
            raise RuntimeError(f"NSImage failed to load {path}")
        if clip.width * clip.height * 4 * clip.count <= FRAME_CACHE_BUDGET_BYTES:
            self._frame_cache[frame_index] = image
        return image

    def _decoded_clip(self, name: str) -> DecodedClip:
        if name in self._decoded:
            self._decoded.move_to_end(name)
            return self._decoded[name]
        while len(self._decoded) >= DECODED_CLIP_CAPACITY:
            _, evicted = self._decoded.popitem(last=False)
            evicted.release()
            del evicted
        decoded = self._loader.load(name)
        self._decoded[name] = decoded
        return decoded

    def _restart_timer(self) -> None:
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        if self._paused or self._finished:
            return
        self._timer = schedule_timer(
            1.0 / self._clip.manifest["fps"], self._tick_frame)

    def _tick_frame(self, timer) -> None:
        clip = self._clip
        if clip is None or clip.count <= 1:
            return
        nxt = self._frame_index + self._direction
        if self._loop == "pingpong":
            if nxt >= clip.count or nxt < 0:
                self._direction *= -1
                nxt = self._frame_index + self._direction
        elif self._loop == "forward":
            if nxt >= clip.count:
                nxt = 0
        else:  # once: hold the last frame, stop ticking
            if nxt >= clip.count:
                self._finished = True
                self._timer.invalidate()
                self._timer = None
                return
        self._frame_index = nxt
        self._frames_rendered += 1
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        self._front.setContents_(self._frame_image(clip, nxt))
        CATransaction.commit()
