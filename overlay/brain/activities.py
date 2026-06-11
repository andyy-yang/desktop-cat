from dataclasses import dataclass


@dataclass(frozen=True)
class Activity:
    name: str
    clips: tuple[str, ...]
    category: str
    utility_need: str
    invert: bool         # True: utility = need value; False: utility = 1 - need
    min_s: float
    max_s: float
    motions: tuple[str | None, ...]  # parallel to clips; walk direction or None


@dataclass(frozen=True)
class Catalog:
    activities: tuple[Activity, ...]
    durations: dict[str, float]      # every indexed clip -> frameCount / fps seconds
    reaction_clips: tuple[str, ...]  # reaction-category clips, never scheduled


KNOWN_CATEGORIES = ("sleep", "idle", "groom", "play", "walk", "reaction")

LOOP_BY_CATEGORY = {
    "sleep": "pingpong",
    "idle": "pingpong",
    "groom": "pingpong",
    "play": "pingpong",
    "walk": "forward",
    "reaction": "once",
}

# (utility_need, invert, min_s, max_s); "reaction" never becomes a schedulable
# activity — the facade plays it from Catalog.reaction_clips on click. Order
# here fixes catalog order.
_ACTIVITY_SPECS = {
    "sleep": ("energy", False, 240.0, 1080.0),
    "idle": ("energy", True, 60.0, 240.0),
    "groom": ("cleanliness", False, 30.0, 120.0),
    "play": ("playfulness", True, 20.0, 90.0),
    "walk": ("energy", True, 20.0, 60.0),
}


def walk_direction(name: str) -> str | None:
    left = "walk_left" in name
    right = "walk_right" in name
    if left and right:
        raise ValueError(f"clip name {name!r} contains both walk directions")
    if left:
        return "left"
    if right:
        return "right"
    return None


def _clip_duration(name: str, manifests: dict[str, dict]) -> float:
    if name not in manifests:
        raise ValueError(f"clip {name!r} is in the index but has no manifest")
    manifest = manifests[name]
    fps = manifest["fps"]
    frame_count = manifest["frameCount"]
    if fps <= 0 or frame_count <= 0:
        raise ValueError(
            f"clip {name!r} has non-positive fps/frameCount: {fps}/{frame_count}")
    return frame_count / fps


def build_catalog(index: dict, manifests: dict[str, dict]) -> Catalog:
    by_category: dict[str, list[str]] = {}
    durations: dict[str, float] = {}
    reactions: list[str] = []
    for entry in index["clips"]:
        category = entry["category"]
        if category not in KNOWN_CATEGORIES:
            raise ValueError(f"unknown clip category: {category!r}")
        name = entry["name"]
        durations[name] = _clip_duration(name, manifests)
        if category == "walk" and walk_direction(name) is None:
            raise ValueError(
                f"walk clip {name!r} has neither 'walk_left' nor 'walk_right' "
                f"in its name; it cannot be scheduled")
        if category == "reaction":
            reactions.append(name)
            continue
        by_category.setdefault(category, []).append(name)

    activities = []
    for category, (utility_need, invert, min_s, max_s) in _ACTIVITY_SPECS.items():
        clips = by_category.get(category)
        if not clips:
            continue
        ordered = tuple(sorted(clips))
        motions = tuple(
            walk_direction(clip) if category == "walk" else None for clip in ordered)
        activities.append(Activity(
            name=category,
            clips=ordered,
            category=category,
            utility_need=utility_need,
            invert=invert,
            min_s=min_s,
            max_s=max_s,
            motions=motions,
        ))
    return Catalog(
        activities=tuple(activities),
        durations=durations,
        reaction_clips=tuple(sorted(reactions)),
    )
