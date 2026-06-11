from dataclasses import dataclass


@dataclass(frozen=True)
class PlayClip:
    clip: str
    loop: str            # "pingpong" | "forward" | "once"
    min_seconds: float
    motion: str | None = None   # "left" | "right" while walking, None otherwise


@dataclass(frozen=True)
class BrainEvent:
    kind: str            # "pet" | "click" | "double_click" | "drag_start"
                         # | "drag_end" | "wake"; double_click is the user's
                         # deterministic scene shuffle ("show me a different clip")
    at: float            # monotonic seconds
