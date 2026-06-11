"""NSObject bridge for selector-based AppKit callbacks and timer scheduling."""

import objc
from Foundation import NSObject, NSRunLoop, NSRunLoopCommonModes, NSTimer


class CallbackTarget(NSObject):
    """Routes a selector invocation (timer, notification, menu action) to a
    Python callable; the ObjC sender is passed through unchanged."""

    def initWithCallback_(self, callback):
        self = objc.super(CallbackTarget, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def fire_(self, sender):  # noqa: N802
        self._callback(sender)


def schedule_timer(interval_s: float, callback, tolerance: float | None = None,
                   repeats: bool = True) -> NSTimer:
    target = CallbackTarget.alloc().initWithCallback_(callback)
    timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
        interval_s, target, b"fire:", None, repeats)
    if tolerance is not None:
        timer.setTolerance_(tolerance)
    NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
    return timer
