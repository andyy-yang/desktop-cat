"""Softmax activity selection with recency penalty, cooldowns, sleep
hysteresis, post-sleep grooming bias and min-duration stickiness.
Deterministic given the injected rng."""

import math
import random

from .activities import Activity

SLEEP_ENTER_ENERGY = 0.25
SLEEP_EXIT_ENERGY = 0.85
RECENCY_PENALTY = 0.6
RECENCY_HALF_LIFE_S = 600.0
DESELECT_COOLDOWN_S = 45.0
WAKE_SLEEP_COOLDOWN_S = 120.0
POST_SLEEP_GROOM_S = 120.0
POST_SLEEP_GROOM_FACTOR = 2.0


class ActivitySelector:
    def __init__(self, catalog: list[Activity], rng: random.Random,
                 temperature: float = 0.6):
        if not catalog:
            raise ValueError("catalog is empty")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self._catalog = list(catalog)
        self._rng = rng
        self._temperature = temperature
        self._current: Activity | None = None
        self._current_since = 0.0
        self._cooldown_until: dict[str, float] = {}
        self._deselected_at: dict[str, float] = {}
        self._released = False
        self._groom_bias_until = 0.0

    def release(self) -> None:
        # facade-approved preemption: lifts min-duration stickiness once
        self._released = True

    def force(self, activity: Activity, wall: float) -> None:
        # facade shuffle (double-click): adopt the given activity as current
        # with the exact deselection bookkeeping choose() applies on a switch;
        # forcing the current activity again is a no-op (its min-duration
        # window keeps its original start). Any pending release is cleared:
        # the forced activity starts with full stickiness, like a normal pick.
        self._released = False
        current = self._current
        if activity is current:
            return
        if current is not None:
            self._deselected_at[current.name] = wall
            self._cooldown_until[current.name] = wall + DESELECT_COOLDOWN_S
        self._current = activity
        self._current_since = wall

    def wake(self, wall: float) -> None:
        # click-wake: leave sleep now and keep it off the menu briefly so the
        # still-low energy cannot re-enter it on the very next choose
        for activity in self._catalog:
            if activity.category == "sleep":
                self._cooldown_until[activity.name] = wall + WAKE_SLEEP_COOLDOWN_S
                self._deselected_at[activity.name] = wall
        self._current = None
        self._released = True
        self._groom_bias_until = wall + POST_SLEEP_GROOM_S

    def choose(self, needs: dict[str, float], wall: float) -> Activity:
        current = self._current
        released = self._released
        self._released = False  # consumed even by the early holds: no stale carry-over
        energy = needs["energy"]
        if (current is not None
                and current.category == "sleep" and energy <= SLEEP_EXIT_ENERGY):
            return current
        if (current is not None and not released
                and wall - self._current_since < current.min_s):
            return current

        candidates = []
        for activity in self._catalog:
            if activity is current:
                if activity.category == "sleep":
                    # past exit threshold: force the wake-up; cats stretch
                    # and groom right after waking
                    self._groom_bias_until = wall + POST_SLEEP_GROOM_S
                    continue
                candidates.append(activity)
                continue
            if activity.category == "sleep" and energy >= SLEEP_ENTER_ENERGY:
                continue
            if self._cooldown_until.get(activity.name, 0.0) > wall:
                continue
            candidates.append(activity)
        if not candidates:
            # everything gated at once (e.g. wake-cooldown sleep + freshly
            # deselected alternatives): re-admit the whole catalog
            candidates = list(self._catalog)

        chosen = self._sample(candidates, needs, wall)
        if chosen is not current:
            if current is not None:
                self._deselected_at[current.name] = wall
                self._cooldown_until[current.name] = wall + DESELECT_COOLDOWN_S
            self._current = chosen
            self._current_since = wall
        return chosen

    def _utility(self, activity: Activity, needs: dict[str, float], wall: float) -> float:
        value = needs[activity.utility_need]
        utility = value if activity.invert else 1.0 - value
        if activity.category == "groom" and wall < self._groom_bias_until:
            utility *= POST_SLEEP_GROOM_FACTOR
        deselected = self._deselected_at.get(activity.name)
        if deselected is not None:
            age = max(wall - deselected, 0.0)
            utility -= RECENCY_PENALTY * 2.0 ** (-age / RECENCY_HALF_LIFE_S)
        return utility

    def _sample(self, candidates: list[Activity], needs: dict[str, float],
                wall: float) -> Activity:
        weights = [math.exp(self._utility(a, needs, wall) / self._temperature)
                   for a in candidates]
        pick = self._rng.random() * sum(weights)
        acc = 0.0
        for activity, weight in zip(candidates, weights):
            acc += weight
            if pick <= acc:
                return activity
        return candidates[-1]  # float-summation edge only
