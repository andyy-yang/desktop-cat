import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...          # monotonic seconds

    def wall(self) -> float: ...         # unix epoch seconds


class SystemClock:
    def now(self) -> float:
        return time.monotonic()

    def wall(self) -> float:
        return time.time()
