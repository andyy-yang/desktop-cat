"""24h behavior simulation: the cat must be calm (few activity switches)
and circadian (sleeping roughly twice as much at night as by day)."""

import random

import pytest

from overlay.brain.facade import Brain
from overlay.brain.needs import NIGHT_END_H, NIGHT_START_H
from overlay.brain.store import PersistenceStore

from .util import FakeClock, make_catalog, wall_at

TICK_S = 5.0
SIM_HOURS = 24
START_HOUR = 12.0


def _is_night(hour: float) -> bool:
    return hour >= NIGHT_START_H or hour < NIGHT_END_H


def _simulate(seed: int, tmp_path):
    catalog = make_catalog(("sleep", "idle", "groom", "play"))
    store = PersistenceStore(tmp_path / f"state_{seed}.json")
    clock = FakeClock(wall_at(int(START_HOUR)))
    brain = Brain(clock, random.Random(seed), catalog, store)

    switches = 0
    category = None
    night_sleep = night_total = day_sleep = day_total = 0
    elapsed = 0.0
    while elapsed < SIM_HOURS * 3600.0:
        cmd = brain.tick([])
        if cmd is not None:
            switches += 1
            category = cmd.clip.rsplit("_", 1)[0]
        hour = (START_HOUR + elapsed / 3600.0) % 24.0
        if _is_night(hour):
            night_total += 1
            night_sleep += category == "sleep"
        else:
            day_total += 1
            day_sleep += category == "sleep"
        clock.advance(TICK_S)
        elapsed += TICK_S
    return switches, night_sleep / night_total, day_sleep / day_total


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_24h_switch_count_and_circadian_sleep(seed, tmp_path):
    switches, night_fraction, day_fraction = _simulate(seed, tmp_path)
    assert 100 <= switches <= 500
    assert day_fraction > 0.0
    assert 1.5 <= night_fraction / day_fraction <= 3.5
