"""Menu bar status item: Pause/Resume toggle, a continuous Size slider
(10%..1000%, logarithmic so 100% sits mid-slider), and Quit. The slider lives
in a custom NSView hosted by an NSMenuItem via setView_ (public AppKit); its
action fires continuously during drag so the orchestrator can resize the cat
live."""

import math
from typing import Callable

from AppKit import (
    NSFont,
    NSMakeRect,
    NSMenu,
    NSMenuItem,
    NSSlider,
    NSStatusBar,
    NSTextField,
    NSVariableStatusItemLength,
    NSView,
)

from .runtime_support import CallbackTarget

SCALE_MIN = 0.10            # 10%
SCALE_MAX = 10.0            # 1000%
SLIDER_WIDTH_PT = 220.0
SLIDER_HEIGHT_PT = 21.0
LABEL_WIDTH_PT = 96.0       # fits "1000% (max)"
SIZE_VIEW_PADDING_PT = 14.0
SIZE_VIEW_GAP_PT = 8.0
SIZE_VIEW_HEIGHT_PT = 28.0
LABEL_FONT_SIZE_PT = 12.0


def slider_to_scale(value: float) -> float:
    """Logarithmic mapping: slider 0..1 -> scale SCALE_MIN..SCALE_MAX with
    1.0 (100%) exactly at the midpoint. Linear over a 100x range is unusable."""
    return SCALE_MIN * (SCALE_MAX / SCALE_MIN) ** value


def scale_to_slider(scale: float) -> float:
    """Inverse of slider_to_scale (scale must be positive)."""
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    return math.log(scale / SCALE_MIN) / math.log(SCALE_MAX / SCALE_MIN)


def format_scale_label(scale: float, clamped: bool) -> str:
    """Live label text: the REQUESTED percentage, '(max)' when the window
    clamped the effective scale to fit the screen."""
    text = f"{round(scale * 100)}%"
    return f"{text} (max)" if clamped else text


class StatusItemController:
    """on_toggle_pause() -> bool returns the new user-paused state; on_quit()
    terminates the app; on_select_scale(value) -> bool applies + persists a
    scale and returns True when the effective scale was clamped to fit the
    screen (label bookkeeping stays here)."""

    def __init__(self, on_toggle_pause: Callable[[], bool], on_quit: Callable[[], None],
                 on_select_scale: Callable[[float], bool], current_scale: float):
        self._on_toggle_pause = on_toggle_pause
        self._on_quit = on_quit
        self._on_select_scale = on_select_scale
        self._current_scale = current_scale
        self._clamped = False
        # menu item / control targets are weak references; keep them alive here
        self._pause_target = CallbackTarget.alloc().initWithCallback_(self._toggle)
        self._quit_target = CallbackTarget.alloc().initWithCallback_(self._quit)
        self._slider_target = CallbackTarget.alloc().initWithCallback_(
            self._slider_changed)
        self._status_item = None
        self._pause_item = None
        self._slider = None
        self._scale_label = None

    def build(self) -> None:
        self._status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength)
        self._status_item.button().setTitle_("🐱")
        menu = NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)
        pause_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Pause", b"fire:", "")
        pause_item.setTarget_(self._pause_target)
        menu.addItem_(pause_item)
        header = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Size", None, "")
        header.setEnabled_(False)
        menu.addItem_(header)
        menu.addItem_(self._build_size_item())
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit Overlay Cat", b"fire:", "q")
        quit_item.setTarget_(self._quit_target)
        menu.addItem_(quit_item)
        self._status_item.setMenu_(menu)
        self._pause_item = pause_item

    def set_clamped(self, clamped: bool) -> None:
        """Orchestrator pushes the window's clamp state after clip changes so
        an open menu's label stays honest."""
        self._clamped = clamped
        self._update_label()

    def _build_size_item(self) -> NSMenuItem:
        size_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "", None, "")
        size_item.setView_(self._build_size_view())
        return size_item

    def _build_size_view(self) -> NSView:
        width = (SIZE_VIEW_PADDING_PT + SLIDER_WIDTH_PT + SIZE_VIEW_GAP_PT
                 + LABEL_WIDTH_PT + SIZE_VIEW_PADDING_PT)
        container = NSView.alloc().initWithFrame_(
            NSMakeRect(0.0, 0.0, width, SIZE_VIEW_HEIGHT_PT))
        slider = NSSlider.alloc().initWithFrame_(NSMakeRect(
            SIZE_VIEW_PADDING_PT, (SIZE_VIEW_HEIGHT_PT - SLIDER_HEIGHT_PT) / 2.0,
            SLIDER_WIDTH_PT, SLIDER_HEIGHT_PT))
        slider.setMinValue_(0.0)
        slider.setMaxValue_(1.0)
        slider.setDoubleValue_(scale_to_slider(self._current_scale))
        slider.setContinuous_(True)  # action fires during drag -> live resize
        slider.setTarget_(self._slider_target)
        slider.setAction_(b"fire:")
        label = NSTextField.alloc().initWithFrame_(NSMakeRect(
            SIZE_VIEW_PADDING_PT + SLIDER_WIDTH_PT + SIZE_VIEW_GAP_PT,
            (SIZE_VIEW_HEIGHT_PT - SLIDER_HEIGHT_PT) / 2.0,
            LABEL_WIDTH_PT, SLIDER_HEIGHT_PT))
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setFont_(NSFont.systemFontOfSize_(LABEL_FONT_SIZE_PT))
        container.addSubview_(slider)
        container.addSubview_(label)
        self._slider = slider
        self._scale_label = label
        self._update_label()
        return container

    def _slider_changed(self, sender) -> None:
        scale = slider_to_scale(sender.doubleValue())
        self._current_scale = scale
        self._clamped = self._on_select_scale(scale)
        self._update_label()

    def _update_label(self) -> None:
        self._scale_label.setStringValue_(
            format_scale_label(self._current_scale, self._clamped))

    def _toggle(self, sender) -> None:
        paused = self._on_toggle_pause()
        self._pause_item.setTitle_("Resume" if paused else "Pause")

    def _quit(self, sender) -> None:
        self._on_quit()
