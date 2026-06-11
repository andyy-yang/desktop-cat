import time

from overlay.brain.activities import Catalog, build_catalog

DEFAULT_FPS = 12.0
DEFAULT_FRAME_COUNT = 30  # 2.5 s at 12 fps


def wall_at(hour: int, minute: int = 0) -> float:
    """Unix time whose LOCAL clock reads hour:minute on a fixed date."""
    return time.mktime((2026, 6, 10, hour, minute, 0, 0, 0, -1))


def make_index(categories) -> dict:
    clips = []
    for category in categories:
        if category == "walk":
            names = ("walk_left", "walk_right")  # direction lives in the name
        else:
            names = (f"{category}_a", f"{category}_b")
        for name in names:
            clips.append({
                "name": name,
                "category": category,
                "dir": f"clips/{name}",
            })
    return {"clips": clips}


def make_manifests(index: dict, fps: float = DEFAULT_FPS,
                   frame_count: int = DEFAULT_FRAME_COUNT,
                   frame_counts: dict[str, int] | None = None) -> dict:
    manifests = {}
    for entry in index["clips"]:
        name = entry["name"]
        count = frame_count
        if frame_counts is not None and name in frame_counts:
            count = frame_counts[name]
        manifests[name] = {
            "name": name,
            "category": entry["category"],
            "fps": fps,
            "frameCount": count,
        }
    return manifests


def make_catalog(categories, fps: float = DEFAULT_FPS,
                 frame_count: int = DEFAULT_FRAME_COUNT,
                 frame_counts: dict[str, int] | None = None) -> Catalog:
    index = make_index(categories)
    return build_catalog(index, make_manifests(index, fps, frame_count, frame_counts))


class FakeClock:
    def __init__(self, start_wall: float, start_now: float = 1000.0):
        self._now = start_now
        self._wall = start_wall

    def now(self) -> float:
        return self._now

    def wall(self) -> float:
        return self._wall

    def advance(self, seconds: float) -> None:
        self._now += seconds
        self._wall += seconds
