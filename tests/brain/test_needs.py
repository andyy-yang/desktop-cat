import math
import random

import pytest

from overlay.brain.needs import (
    AWAKE_ENERGY_RATE,
    DEFAULT_NEEDS,
    NEED_NAMES,
    NeedsModel,
)

from .util import wall_at

NOON = wall_at(12)


def make(initial=None, seed=0):
    return NeedsModel(random.Random(seed), initial=initial)


def test_defaults_are_copied_and_in_range():
    m = make()
    assert m.needs == DEFAULT_NEEDS
    m.needs["energy"] = 0.0
    assert DEFAULT_NEEDS["energy"] != 0.0
    for name in NEED_NAMES:
        assert 0.0 <= make().needs[name] <= 1.0


def test_unknown_initial_key_raises():
    with pytest.raises(ValueError):
        make({"boredom": 0.5})


def test_out_of_range_initial_raises():
    with pytest.raises(ValueError):
        make({"energy": 1.5})
    with pytest.raises(ValueError):
        make({"energy": -0.1})


def test_energy_drains_to_quarter_over_three_hours_awake():
    m = make({"energy": 1.0})
    m.tick(3.0 * 3600.0, NOON)
    assert m.needs["energy"] == pytest.approx(0.25, abs=1e-6)


def test_drift_is_step_size_independent():
    one_shot = make({"energy": 1.0})
    one_shot.tick(10800.0, NOON)
    stepped = make({"energy": 1.0})
    for i in range(360):
        stepped.tick(30.0, NOON + i * 30.0)
    assert stepped.needs["energy"] == pytest.approx(one_shot.needs["energy"], abs=1e-9)


def test_sleep_refills_energy_in_hundred_minutes():
    m = make({"energy": 0.0})
    m.set_mode("sleep")
    m.tick(100.0 * 60.0, NOON)
    assert m.needs["energy"] == pytest.approx(0.95, abs=1e-6)
    assert m.needs["energy"] > 0.85


def test_night_sleep_refills_slower_than_day():
    night = make({"energy": 0.0})
    night.set_mode("sleep")
    night.tick(3600.0, wall_at(2))
    day = make({"energy": 0.0})
    day.set_mode("sleep")
    day.tick(3600.0, NOON)
    assert night.needs["energy"] < day.needs["energy"]
    # consolidation factor 0.6 on the exponential refill rate
    ratio = math.log(1.0 - night.needs["energy"]) / math.log(1.0 - day.needs["energy"])
    assert ratio == pytest.approx(0.6, abs=1e-9)


def test_night_multiplies_sleep_pressure_by_two_to_three():
    night = make({"energy": 1.0})
    night.tick(3600.0, wall_at(2))
    day = make({"energy": 1.0})
    day.tick(3600.0, NOON)
    assert night.needs["energy"] < day.needs["energy"]
    ratio = math.log(night.needs["energy"]) / math.log(day.needs["energy"])
    assert 2.0 <= ratio <= 3.0


def test_night_window_boundaries():
    at_23 = make({"energy": 1.0})
    at_23.tick(60.0, wall_at(23))
    at_22 = make({"energy": 1.0})
    at_22.tick(60.0, wall_at(22))
    at_8 = make({"energy": 1.0})
    at_8.tick(60.0, wall_at(8))
    at_7 = make({"energy": 1.0})
    at_7.tick(60.0, wall_at(7))
    assert at_23.needs["energy"] < at_22.needs["energy"]   # 23:00 is night
    assert at_7.needs["energy"] < at_8.needs["energy"]     # 08:00 is day again
    assert at_8.needs["energy"] == pytest.approx(at_22.needs["energy"], abs=1e-9)


def test_hunger_bumps_near_breakfast_and_dinner():
    morning = make({"hunger": 0.0})
    morning.tick(1800.0, wall_at(8))
    evening = make({"hunger": 0.0})
    evening.tick(1800.0, wall_at(18))
    noon = make({"hunger": 0.0})
    noon.tick(1800.0, NOON)
    outside = make({"hunger": 0.0})
    outside.tick(1800.0, wall_at(9, 30))
    assert morning.needs["hunger"] > noon.needs["hunger"]
    assert evening.needs["hunger"] > noon.needs["hunger"]
    assert outside.needs["hunger"] == pytest.approx(noon.needs["hunger"], abs=1e-9)


def test_fast_forward_matches_neutral_awake_drift():
    m = make({"energy": 1.0})
    m.fast_forward(3.0 * 3600.0)
    assert m.needs["energy"] == pytest.approx(0.25, abs=1e-6)


def test_fast_forward_clamped_at_twelve_hours():
    capped = make()
    capped.fast_forward(12.0 * 3600.0)
    way_over = make()
    way_over.fast_forward(400.0 * 3600.0)
    for name in NEED_NAMES:
        assert way_over.needs[name] == pytest.approx(capped.needs[name], abs=1e-12)


def test_fast_forward_zero_and_negative_are_noops():
    m = make()
    before = dict(m.needs)
    m.fast_forward(0.0)
    assert m.needs == before
    m.fast_forward(-100.0)
    assert m.needs == before


def test_fast_forward_ignores_mode():
    asleep = make({"energy": 0.5})
    asleep.set_mode("sleep")
    asleep.fast_forward(3600.0)
    awake = make({"energy": 0.5})
    awake.fast_forward(3600.0)
    assert asleep.needs["energy"] == pytest.approx(awake.needs["energy"], abs=1e-12)
    assert asleep.needs["energy"] < 0.5


def test_apply_event_pet_raises_affection():
    m = make({"affection": 0.5})
    m.apply_event("pet")
    assert m.needs["affection"] == pytest.approx(0.65)


def test_apply_event_click_raises_playfulness_slightly():
    m = make({"playfulness": 0.5})
    m.apply_event("click")
    assert m.needs["playfulness"] == pytest.approx(0.55)


def test_apply_event_double_click_raises_playfulness_slightly():
    m = make({"playfulness": 0.5})
    m.apply_event("double_click")
    assert m.needs["playfulness"] == pytest.approx(0.55)


def test_apply_event_drag_lowers_affection():
    m = make({"affection": 0.5})
    m.apply_event("drag_start")
    assert m.needs["affection"] == pytest.approx(0.42)
    before = dict(m.needs)
    m.apply_event("drag_end")
    assert m.needs == before


def test_apply_event_wake_is_noop_on_needs():
    m = make()
    before = dict(m.needs)
    m.apply_event("wake")
    assert m.needs == before


def test_apply_event_clamps_at_bounds():
    m = make({"affection": 0.95})
    m.apply_event("pet")
    m.apply_event("pet")
    assert m.needs["affection"] == 1.0
    low = make({"affection": 0.05})
    low.apply_event("drag_start")
    assert low.needs["affection"] == 0.0


def test_apply_event_unknown_kind_raises():
    m = make()
    with pytest.raises(ValueError):
        m.apply_event("teleport")


def test_tick_negative_dt_raises():
    m = make()
    with pytest.raises(ValueError):
        m.tick(-1.0, NOON)


def test_all_needs_stay_in_unit_interval_under_long_ticks():
    m = make()
    for mode in ("idle", "play", "groom", "sleep"):
        m.set_mode(mode)
        m.tick(6.0 * 3600.0, NOON)
        for name in NEED_NAMES:
            assert 0.0 <= m.needs[name] <= 1.0


def test_round_trip_dict():
    m = make({"energy": 0.33, "hunger": 0.7})
    m.tick(500.0, NOON)
    d = m.to_dict()
    restored = NeedsModel.from_dict(d, random.Random(99))
    assert restored.needs == m.needs


def test_to_dict_is_detached_copy():
    m = make()
    d = m.to_dict()
    d["needs"]["energy"] = -42.0
    assert m.needs["energy"] != -42.0


def test_from_dict_missing_needs_key_raises():
    with pytest.raises(KeyError):
        NeedsModel.from_dict({"wall": 0.0}, random.Random(0))


def test_awake_rate_constant_matches_three_hour_tuning():
    # full -> sleep-enter threshold (0.25) over 3h
    assert math.exp(-AWAKE_ENERGY_RATE * 3 * 3600) == pytest.approx(0.25, abs=1e-12)
