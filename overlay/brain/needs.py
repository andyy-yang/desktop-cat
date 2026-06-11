"""Need state and drift: five scalars in [0, 1] evolved by exponential
approach toward per-mode targets, modulated by the local wall clock."""

import math
import random
import time

NEED_NAMES = ("energy", "hunger", "playfulness", "affection", "cleanliness")

DEFAULT_NEEDS = {
    "energy": 0.9,
    "hunger": 0.3,
    "playfulness": 0.4,
    "affection": 0.6,
    "cleanliness": 0.8,
}

# energy: full -> 0.25 (sleep-enter threshold) over ~3h awake
AWAKE_ENERGY_RATE = math.log(4.0) / (3.0 * 3600.0)
# energy: ~0 -> 0.95 over ~100min of sleep
SLEEP_ENERGY_RATE = math.log(20.0) / (100.0 * 60.0)
PLAY_ENERGY_FACTOR = 1.5
NIGHT_SLEEP_PRESSURE = 2.8           # within the contract's x2-3 band
NIGHT_NAP_FACTOR = 0.6               # night sleep refills slower: consolidated naps
NIGHT_START_H = 23.0
NIGHT_END_H = 8.0
HUNGER_RATE = 1.0 / 14400.0
MEAL_HOURS = (8.0, 18.0)
MEAL_WINDOW_H = 0.75
MEAL_MULTIPLIER = 3.0
PLAYFULNESS_RISE_RATE = 1.0 / 7200.0
PLAYFULNESS_BURN_RATE = 1.0 / 180.0
AFFECTION_BASELINE = 0.4
AFFECTION_RATE = 1.0 / 10800.0
CLEANLINESS_DECAY_RATE = 1.0 / 21600.0
GROOM_RATE = 1.0 / 240.0
FAST_FORWARD_CAP_S = 12.0 * 3600.0

EVENT_EFFECTS = {
    "pet": ("affection", 0.15),
    "click": ("playfulness", 0.05),
    "double_click": ("playfulness", 0.05),
    "drag_start": ("affection", -0.08),
    "drag_end": None,
    "wake": None,
}


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _approach(value: float, target: float, rate: float, dt: float) -> float:
    return value + (target - value) * (1.0 - math.exp(-rate * dt))


def _local_hour(wall: float) -> float:
    t = time.localtime(wall)
    return t.tm_hour + t.tm_min / 60.0 + t.tm_sec / 3600.0


class NeedsModel:
    def __init__(self, rng: random.Random, initial: dict[str, float] | None = None):
        self.rng = rng
        needs = dict(DEFAULT_NEEDS)
        if initial is not None:
            unknown = sorted(set(initial) - set(NEED_NAMES))
            if unknown:
                raise ValueError(f"unknown needs: {unknown}")
            for name, value in initial.items():
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"need {name}={value} outside [0, 1]")
                needs[name] = float(value)
        self.needs = needs
        self.mode = "idle"

    def set_mode(self, category: str) -> None:
        self.mode = category

    def tick(self, dt: float, wall: float) -> None:
        if dt < 0.0:
            raise ValueError(f"negative dt: {dt}")
        if dt == 0.0:
            return
        hour = _local_hour(wall)
        night = hour >= NIGHT_START_H or hour < NIGHT_END_H
        mealtime = any(abs(hour - meal) <= MEAL_WINDOW_H for meal in MEAL_HOURS)
        self._drift(
            dt,
            sleeping=self.mode == "sleep",
            energy_factor=(NIGHT_SLEEP_PRESSURE if night else 1.0)
            * (PLAY_ENERGY_FACTOR if self.mode == "play" else 1.0),
            refill_factor=NIGHT_NAP_FACTOR if night else 1.0,
            hunger_factor=MEAL_MULTIPLIER if mealtime else 1.0,
            playing=self.mode == "play",
            grooming=self.mode == "groom",
        )

    def fast_forward(self, elapsed: float) -> None:
        # unobserved gap: neutral awake drift, no circadian/meal modulation
        elapsed = min(max(elapsed, 0.0), FAST_FORWARD_CAP_S)
        if elapsed == 0.0:
            return
        self._drift(elapsed, sleeping=False, energy_factor=1.0, refill_factor=1.0,
                    hunger_factor=1.0, playing=False, grooming=False)

    def apply_event(self, kind: str) -> None:
        if kind not in EVENT_EFFECTS:
            raise ValueError(f"unknown event kind: {kind!r}")
        effect = EVENT_EFFECTS[kind]
        if effect is None:
            return
        name, delta = effect
        self.needs[name] = _clamp(self.needs[name] + delta)

    def to_dict(self) -> dict:
        return {"needs": dict(self.needs)}

    @classmethod
    def from_dict(cls, d: dict, rng: random.Random) -> "NeedsModel":
        return cls(rng, initial=d["needs"])

    def _drift(self, dt: float, *, sleeping: bool, energy_factor: float,
               refill_factor: float, hunger_factor: float, playing: bool,
               grooming: bool) -> None:
        n = self.needs
        if sleeping:
            n["energy"] = _approach(n["energy"], 1.0, SLEEP_ENERGY_RATE * refill_factor, dt)
        else:
            n["energy"] = _approach(n["energy"], 0.0, AWAKE_ENERGY_RATE * energy_factor, dt)
        n["hunger"] = _approach(n["hunger"], 1.0, HUNGER_RATE * hunger_factor, dt)
        if playing:
            n["playfulness"] = _approach(n["playfulness"], 0.0, PLAYFULNESS_BURN_RATE, dt)
        else:
            n["playfulness"] = _approach(n["playfulness"], 1.0, PLAYFULNESS_RISE_RATE, dt)
        n["affection"] = _approach(n["affection"], AFFECTION_BASELINE, AFFECTION_RATE, dt)
        if grooming:
            n["cleanliness"] = _approach(n["cleanliness"], 1.0, GROOM_RATE, dt)
        else:
            n["cleanliness"] = _approach(n["cleanliness"], 0.0, CLEANLINESS_DECAY_RATE, dt)
        for name in NEED_NAMES:
            n[name] = _clamp(n[name])
