"""In-process checks for the runtime features: SettingsStore round-trip,
user_scale geometry (window size, anchor, current_alpha_at agreement), the
Size slider's logarithmic mapping, the screen-fit clamp, crossfade size
gating, and the click/double-click debounce. The NSPanel is constructed but
never ordered front — nothing becomes visible."""

import json

import numpy as np
import pytest
from AppKit import NSScreen
from Foundation import NSDate, NSRunLoop
from PIL import Image

from overlay.app.interactions import CLICK_DEBOUNCE_S, InteractionMonitor
from overlay.app.orchestrator import AppOrchestrator
from overlay.app.renderer import ClipLoader, SpriteRenderer, crossfade_allowed
from overlay.app.settings import SettingsStore
from overlay.app.status_item import (
    SCALE_MAX,
    SCALE_MIN,
    SLIDER_WIDTH_PT,
    StatusItemController,
    format_scale_label,
    scale_to_slider,
    slider_to_scale,
)
from overlay.app.window import SCREEN_FIT_FRACTION, OverlayWindow, clamp_user_scale

# 200x120 px at displayScale 2 -> 100x60 pt; left half opaque, right half clear
CLIP_W_PX, CLIP_H_PX = 200, 120


def _write_clip(root, name, w_px=CLIP_W_PX, h_px=CLIP_H_PX):
    clip_dir = root / name
    (clip_dir / "frames").mkdir(parents=True)
    rgba = np.zeros((h_px, w_px, 4), dtype=np.uint8)
    rgba[:, : w_px // 2] = (255, 0, 0, 255)
    for i in range(2):
        Image.fromarray(rgba, "RGBA").save(clip_dir / "frames" / f"{i:04d}.png")
    manifest = {
        "name": name, "category": "idle", "fps": 12.0, "frameCount": 2,
        "loop": "pingpong", "pixelSize": [w_px, h_px], "displayScale": 2.0,
        "anchor": [w_px / 2.0, float(h_px)],
    }
    (clip_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest, clip_dir


def _renderer(tmp_path, clips, user_scale=1.0):
    manifests, dirs = {}, {}
    for name in clips:
        manifests[name], dirs[name] = _write_clip(tmp_path, name)
    window = OverlayWindow(user_scale=user_scale)
    renderer = SpriteRenderer(window, manifests, ClipLoader(dirs, manifests))
    return window, renderer


def _pump(seconds):
    NSRunLoop.currentRunLoop().runUntilDate_(
        NSDate.dateWithTimeIntervalSinceNow_(seconds))


# -- settings -----------------------------------------------------------------

def test_settings_missing_returns_none(tmp_path):
    assert SettingsStore(tmp_path / "absent.json").load() is None


def test_settings_round_trip(tmp_path):
    store = SettingsStore(tmp_path / "nested" / "settings.json")
    store.save({"user_scale": 0.8})
    assert store.load() == {"user_scale": 0.8}
    assert SettingsStore(tmp_path / "nested" / "settings.json").load() == {
        "user_scale": 0.8}


def test_settings_persists_non_preset_floats(tmp_path):
    # the continuous slider produces arbitrary floats, not preset steps
    settings = tmp_path / "settings.json"
    SettingsStore(settings).save({"user_scale": 2.4137})
    assert SettingsStore(settings).load() == {"user_scale": 2.4137}
    orch = AppOrchestrator(tmp_path / "index.json", tmp_path / "state.json",
                           settings_path=settings)
    assert orch._load_user_scale() == 2.4137
    for value in (0.1, 10.0, 7.77):
        SettingsStore(settings).save({"user_scale": value})
        assert orch._load_user_scale() == value


def test_orchestrator_rejects_bad_user_scale(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"user_scale": -1.0}))
    orch = AppOrchestrator(tmp_path / "index.json", tmp_path / "state.json",
                           settings_path=settings)
    with pytest.raises(RuntimeError):
        orch._load_user_scale()
    settings.write_text(json.dumps({"user_scale": True}))
    with pytest.raises(RuntimeError):
        orch._load_user_scale()


# -- user_scale geometry --------------------------------------------------------

def test_scaled_geometry_and_alpha_agree(tmp_path):
    window, renderer = _renderer(tmp_path, ["clip_a"])
    renderer.play("clip_a", "pingpong")
    fx, fy, fw, fh = renderer.window_frame()
    assert (fw, fh) == (100.0, 60.0)
    assert window.anchor_offset() == (50.0, 0.0)
    # left half opaque, right half transparent at native scale
    assert renderer.current_alpha_at(10.0, 10.0) == 255
    assert renderer.current_alpha_at(60.0, 10.0) == 0
    center_before = (fx + fw / 2.0, fy + fh / 2.0)

    window.set_user_scale(0.6)  # re-applies geometry immediately
    fx2, fy2, fw2, fh2 = renderer.window_frame()
    assert (fw2, fh2) == (60.0, 36.0)
    ox, oy = window.anchor_offset()
    assert (ox, oy) == (30.0, 0.0)
    # user zoom is CENTER-preserving (feet-anchor zoom read as downward drift)
    center_after = (fx2 + fw2 / 2.0, fy2 + fh2 / 2.0)
    assert center_after == pytest.approx(center_before, abs=1.0)
    # the same clip pixels answer at scaled window points
    assert renderer.current_alpha_at(10.0 * 0.6, 10.0 * 0.6) == 255
    assert renderer.current_alpha_at(60.0 * 0.6, 10.0 * 0.6) == 0


def test_initial_user_scale_applies(tmp_path):
    window, renderer = _renderer(tmp_path, ["clip_a"], user_scale=1.4)
    renderer.play("clip_a", "pingpong")
    _, _, fw, fh = renderer.window_frame()
    assert (fw, fh) == pytest.approx((140.0, 84.0))
    assert window.anchor_offset() == pytest.approx((70.0, 0.0))


def test_user_scale_rejects_non_positive():
    with pytest.raises(ValueError):
        OverlayWindow(user_scale=0.0)


# -- size slider: logarithmic mapping ---------------------------------------------

def test_slider_midpoint_is_100_percent():
    assert slider_to_scale(0.5) == pytest.approx(1.0, rel=0.01)


def test_slider_endpoints_are_10_and_1000_percent():
    assert slider_to_scale(0.0) == pytest.approx(SCALE_MIN)
    assert slider_to_scale(1.0) == pytest.approx(SCALE_MAX)


def test_slider_mapping_round_trips():
    for v in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        assert scale_to_slider(slider_to_scale(v)) == pytest.approx(v)
    for scale in (0.1, 0.6, 1.0, 2.4137, 10.0):
        assert slider_to_scale(scale_to_slider(scale)) == pytest.approx(scale)


def test_scale_to_slider_rejects_non_positive():
    with pytest.raises(ValueError):
        scale_to_slider(0.0)


def test_format_scale_label_shows_requested_percent_and_max():
    assert format_scale_label(1.0, False) == "100%"
    assert format_scale_label(0.1, False) == "10%"
    assert format_scale_label(2.5, True) == "250% (max)"
    assert format_scale_label(10.0, True) == "1000% (max)"


def test_size_view_hosts_continuous_slider_at_log_position():
    controller = StatusItemController(
        lambda: False, lambda: None, lambda value: False, 1.0)
    item = controller._build_size_item()  # no NSStatusBar touched
    assert item.view() is not None
    slider = controller._slider
    assert slider.frame().size.width == SLIDER_WIDTH_PT
    assert slider.isContinuous()  # action fires during drag (live resize)
    assert slider.doubleValue() == pytest.approx(0.5)  # 100% sits mid-slider
    assert controller._scale_label.stringValue() == "100%"


def test_slider_action_applies_scale_and_marks_clamped():
    applied = []

    def select(value):
        applied.append(value)
        return value > 5.0  # pretend the window clamps only huge requests

    controller = StatusItemController(lambda: False, lambda: None, select, 1.0)
    controller._build_size_item()
    controller._slider.setDoubleValue_(1.0)
    controller._slider_changed(controller._slider)
    assert applied == [pytest.approx(SCALE_MAX)]
    assert controller._scale_label.stringValue() == "1000% (max)"
    controller._slider.setDoubleValue_(0.0)
    controller._slider_changed(controller._slider)
    assert applied[-1] == pytest.approx(SCALE_MIN)
    assert controller._scale_label.stringValue() == "10%"


# -- screen-fit clamp ---------------------------------------------------------------

def test_clamp_user_scale_math():
    # 10x on a 400 pt clip against a 1000 pt-tall screen
    assert clamp_user_scale(10.0, 400.0, 400.0, 1600.0, 1000.0) == pytest.approx(
        0.95 * 1000.0 / 400.0)  # height is the binding dimension -> 2.375
    # width-bound case
    assert clamp_user_scale(10.0, 800.0, 100.0, 1600.0, 1000.0) == pytest.approx(
        0.95 * 1600.0 / 800.0)
    # fits already: unchanged (exact float equality, not approx)
    assert clamp_user_scale(2.0, 400.0, 300.0, 1600.0, 1000.0) == 2.0
    with pytest.raises(ValueError):
        clamp_user_scale(1.0, 0.0, 100.0, 1600.0, 1000.0)


def test_window_clamps_effective_scale_and_alpha_agrees(tmp_path):
    # 100x60 pt native clip at an absurd 20000% -> clamps on ANY real screen
    window, renderer = _renderer(tmp_path, ["clip_a"], user_scale=200.0)
    renderer.play("clip_a", "pingpong")
    visible = NSScreen.mainScreen().visibleFrame()
    effective = clamp_user_scale(200.0, 100.0, 60.0,
                                 visible.size.width, visible.size.height)
    assert effective < 200.0
    assert window.scale_clamped()
    assert window.user_scale() == 200.0  # requested scale survives for the label
    _, _, fw, fh = renderer.window_frame()
    # AppKit rounds panel frames to integral points; the scale math is exact
    assert (fw, fh) == pytest.approx((100.0 * effective, 60.0 * effective), abs=1.0)
    assert fw <= SCREEN_FIT_FRACTION * visible.size.width + 1.0
    assert fh <= SCREEN_FIT_FRACTION * visible.size.height + 1.0
    # anchor math uses the effective scale (locomotion floor reads this)
    assert window.anchor_offset() == pytest.approx((50.0 * effective, 0.0))
    # alpha hit-testing agrees with the effective scale: left quarter of the
    # window maps onto the opaque half, right quarter onto the transparent one
    assert renderer.current_alpha_at(fw * 0.25, fh * 0.5) == 255
    assert renderer.current_alpha_at(fw * 0.75, fh * 0.5) == 0


def test_unclamped_scale_reports_not_clamped(tmp_path):
    window, renderer = _renderer(tmp_path, ["clip_a"], user_scale=200.0)
    renderer.play("clip_a", "pingpong")
    assert window.scale_clamped()
    window.set_user_scale(1.0)  # back inside the screen budget
    assert not window.scale_clamped()
    _, _, fw, fh = renderer.window_frame()
    assert (fw, fh) == pytest.approx((100.0, 60.0))
    assert renderer.current_alpha_at(10.0, 10.0) == 255
    assert renderer.current_alpha_at(60.0, 10.0) == 0


# -- crossfade gating -----------------------------------------------------------

def _manifest(w_px, h_px, display_scale=2.0):
    return {"pixelSize": [w_px, h_px], "displayScale": display_scale}


def test_crossfade_same_size_allowed():
    assert crossfade_allowed(_manifest(200, 120), _manifest(200, 120))


def test_crossfade_within_15pct_allowed():
    assert crossfade_allowed(_manifest(200, 120), _manifest(220, 130))
    assert crossfade_allowed(_manifest(200, 120), _manifest(230, 120))  # 15% exact


def test_crossfade_blocked_on_either_dimension():
    assert not crossfade_allowed(_manifest(200, 120), _manifest(250, 120))
    assert not crossfade_allowed(_manifest(200, 120), _manifest(200, 150))
    assert not crossfade_allowed(_manifest(250, 120), _manifest(200, 120))


def test_crossfade_compares_points_not_pixels():
    # double the pixels AND the displayScale -> identical point size
    assert crossfade_allowed(_manifest(200, 120, 2.0), _manifest(400, 240, 4.0))


def test_switch_between_mismatched_clips_hard_cuts(tmp_path):
    manifests, dirs = {}, {}
    manifests["small"], dirs["small"] = _write_clip(tmp_path, "small", 200, 120)
    manifests["large"], dirs["large"] = _write_clip(tmp_path, "large", 300, 120)
    window = OverlayWindow()
    renderer = SpriteRenderer(window, manifests, ClipLoader(dirs, manifests))
    renderer.play("small", "pingpong")
    renderer.play("large", "pingpong")  # >15% wider: hard-cut path
    assert renderer.current_clip() == "large"
    assert renderer._back.opacity() == 0.0
    assert renderer._front.opacity() == 1.0
    _, _, fw, fh = renderer.window_frame()
    assert (fw, fh) == (150.0, 60.0)


# -- double-click debounce --------------------------------------------------------

class FakeRenderer:
    def __init__(self):
        self.boops = 0

    def window_frame(self):
        return (0.0, 0.0, 10.0, 10.0)

    def boop(self):
        self.boops += 1


class FakeEvent:
    def __init__(self, count):
        self._count = count

    def clickCount(self):  # noqa: N802
        return self._count


def _click(monitor, count):
    monitor.on_mouse_down(FakeEvent(count))
    monitor.on_mouse_up(FakeEvent(count))


def test_single_click_emits_after_debounce():
    events = []
    renderer = FakeRenderer()
    monitor = InteractionMonitor(renderer, events.append)
    _click(monitor, 1)
    assert events == []  # debounced, not yet emitted
    assert renderer.boops == 1  # visual feedback is immediate
    _pump(CLICK_DEBOUNCE_S + 0.15)
    assert [e.kind for e in events] == ["click"]


def test_double_click_emits_only_double():
    events = []
    monitor = InteractionMonitor(FakeRenderer(), events.append)
    _click(monitor, 1)
    _click(monitor, 2)
    assert [e.kind for e in events] == ["double_click"]
    _pump(CLICK_DEBOUNCE_S + 0.15)
    assert [e.kind for e in events] == ["double_click"]  # single never fires


def test_pause_cancels_pending_click():
    events = []
    monitor = InteractionMonitor(FakeRenderer(), events.append)
    _click(monitor, 1)
    monitor.pause()
    _pump(CLICK_DEBOUNCE_S + 0.15)
    assert events == []
