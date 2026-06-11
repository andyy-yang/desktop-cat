import random

import pytest

from overlay.brain.selector import ActivitySelector

from .util import make_catalog, wall_at

T0 = wall_at(12)

FULL = ("sleep", "idle", "groom", "play")


def needs_with(**overrides):
    base = {"energy": 0.5, "hunger": 0.3, "playfulness": 0.5,
            "affection": 0.5, "cleanliness": 0.5}
    base.update(overrides)
    return base


def catalog_of(categories):
    return list(make_catalog(categories).activities)


def test_empty_catalog_raises():
    with pytest.raises(ValueError):
        ActivitySelector([], random.Random(0))


def test_nonpositive_temperature_raises():
    with pytest.raises(ValueError):
        ActivitySelector(catalog_of(FULL), random.Random(0), temperature=0.0)


def test_missing_need_key_raises():
    selector = ActivitySelector(catalog_of(FULL), random.Random(0))
    with pytest.raises(KeyError):
        selector.choose({"energy": 0.5}, T0)


def test_sleepy_brain_picks_sleep_over_80_percent():
    rng = random.Random(7)
    sleepy = needs_with(energy=0.05, playfulness=0.3, cleanliness=0.9)
    hits = 0
    trials = 200
    for _ in range(trials):
        selector = ActivitySelector(catalog_of(FULL), rng, temperature=0.1)
        if selector.choose(sleepy, T0).name == "sleep":
            hits += 1
    assert hits / trials > 0.8


def test_sleep_never_entered_at_moderate_energy():
    rng = random.Random(11)
    moderate = needs_with(energy=0.3)
    for _ in range(100):
        selector = ActivitySelector(catalog_of(FULL), rng, temperature=2.0)
        assert selector.choose(moderate, T0).name != "sleep"


def test_sleep_hysteresis_holds_until_085_then_exits():
    selector = ActivitySelector(catalog_of(FULL), random.Random(3), temperature=0.05)
    assert selector.choose(needs_with(energy=0.02, playfulness=0.0,
                                      cleanliness=1.0), T0).name == "sleep"
    # energy recovered past the 0.25 entry gate but below the 0.85 exit gate
    assert selector.choose(needs_with(energy=0.5), T0 + 5000.0).name == "sleep"
    assert selector.choose(needs_with(energy=0.84), T0 + 6000.0).name == "sleep"
    assert selector.choose(needs_with(energy=0.9), T0 + 7000.0).name != "sleep"


def test_min_duration_keeps_current_activity():
    selector = ActivitySelector(catalog_of(("idle", "play")), random.Random(5),
                                temperature=0.05)
    idle_needs = needs_with(energy=1.0, playfulness=0.0)
    play_needs = needs_with(energy=0.3, playfulness=1.0)
    assert selector.choose(idle_needs, T0).name == "idle"
    # idle.min_s is 60: at +30s even an overwhelming play urge cannot switch
    assert selector.choose(play_needs, T0 + 30.0).name == "idle"
    assert selector.choose(play_needs, T0 + 61.0).name == "play"


def test_release_lifts_min_duration_once():
    selector = ActivitySelector(catalog_of(("idle", "play")), random.Random(5),
                                temperature=0.05)
    idle_needs = needs_with(energy=1.0, playfulness=0.0)
    play_needs = needs_with(energy=0.3, playfulness=1.0)
    assert selector.choose(idle_needs, T0).name == "idle"
    selector.release()
    assert selector.choose(play_needs, T0 + 10.0).name == "play"


def test_wake_blocks_sleep_reentry_until_cooldown_expires():
    selector = ActivitySelector(catalog_of(("sleep", "idle")), random.Random(9),
                                temperature=0.05)
    sleepy = needs_with(energy=0.02, playfulness=0.0, cleanliness=1.0)
    assert selector.choose(sleepy, T0).name == "sleep"
    selector.wake(T0 + 30.0)
    assert selector.choose(sleepy, T0 + 30.0).name == "idle"
    # cooldown (120s) still active
    selector.release()
    assert selector.choose(sleepy, T0 + 100.0).name == "idle"
    # cooldown expired, still exhausted -> back to sleep
    assert selector.choose(sleepy, T0 + 200.0).name == "sleep"


def test_cooldown_blocks_immediate_reselection():
    selector = ActivitySelector(catalog_of(("idle", "play")), random.Random(13),
                                temperature=0.05)
    play_needs = needs_with(energy=0.0, playfulness=1.0)
    idle_needs = needs_with(energy=1.0, playfulness=0.0)
    assert selector.choose(play_needs, T0).name == "play"
    assert selector.choose(idle_needs, T0 + 21.0).name == "idle"
    # play deselected at +21s; its 45s cooldown holds even with a huge urge
    selector.release()
    assert selector.choose(play_needs, T0 + 60.0).name == "idle"
    selector.release()
    assert selector.choose(play_needs, T0 + 70.0).name == "play"


def _play_pick_rate_after_deselection(gap_s: float, trials: int = 300,
                                      seed: int = 42) -> float:
    rng = random.Random(seed)
    favor_play = needs_with(energy=0.0, playfulness=1.0)
    favor_idle = needs_with(energy=1.0, playfulness=0.0)
    equal = needs_with(energy=0.6, playfulness=0.6)
    picks = 0
    valid = 0
    for _ in range(trials):
        selector = ActivitySelector(catalog_of(("idle", "play")), rng,
                                    temperature=0.3)
        if selector.choose(favor_play, T0).name != "play":
            continue
        t1 = T0 + 21.0
        if selector.choose(favor_idle, t1).name != "idle":
            continue
        valid += 1
        if selector.choose(equal, t1 + gap_s).name == "play":
            picks += 1
    assert valid > trials // 2
    return picks / valid


def test_recency_penalty_reduces_immediate_repeats():
    # past idle's 60s min duration and play's 45s cooldown: penalty still strong
    immediate = _play_pick_rate_after_deselection(61.0)
    # an hour later the penalty has decayed through ~6 half-lives
    late = _play_pick_rate_after_deselection(3600.0)
    assert immediate < 0.30
    assert late > 0.35
    assert immediate < late - 0.15


def test_post_sleep_groom_bias_applies_at_natural_wake():
    selector = ActivitySelector(catalog_of(("sleep", "idle", "groom")),
                                random.Random(3), temperature=0.05)
    sleepy = needs_with(energy=0.02, playfulness=0.0, cleanliness=1.0)
    assert selector.choose(sleepy, T0).name == "sleep"
    # unbiased: groom 0.6 < idle 0.9; doubled: groom 1.2 > idle 0.9
    rested = needs_with(energy=0.9, cleanliness=0.4)
    assert selector.choose(rested, T0 + 5000.0).name == "groom"


def test_no_groom_bias_without_preceding_sleep():
    selector = ActivitySelector(catalog_of(("sleep", "idle", "groom")),
                                random.Random(3), temperature=0.05)
    rested = needs_with(energy=0.9, cleanliness=0.4)
    assert selector.choose(rested, T0).name == "idle"


def test_post_sleep_groom_bias_expires_after_two_minutes():
    selector = ActivitySelector(catalog_of(("sleep", "idle", "groom")),
                                random.Random(3), temperature=0.05)
    sleepy = needs_with(energy=0.02, playfulness=0.0, cleanliness=1.0)
    assert selector.choose(sleepy, T0).name == "sleep"
    rested = needs_with(energy=0.9, cleanliness=0.4)
    assert selector.choose(rested, T0 + 5000.0).name == "groom"
    # bias window (120s) over; undoubled groom 0.6 loses to idle 0.9 again
    assert selector.choose(rested, T0 + 5200.0).name == "idle"


def test_click_wake_sets_groom_bias():
    selector = ActivitySelector(catalog_of(("sleep", "idle", "groom")),
                                random.Random(3), temperature=0.02)
    sleepy = needs_with(energy=0.02, playfulness=0.0, cleanliness=1.0)
    assert selector.choose(sleepy, T0).name == "sleep"
    selector.wake(T0 + 30.0)
    # unbiased: groom 0.3 < idle 0.4; doubled: groom 0.6 > idle 0.4
    groggy = needs_with(energy=0.4, cleanliness=0.7)
    assert selector.choose(groggy, T0 + 40.0).name == "groom"
    assert selector.choose(groggy, T0 + 200.0).name == "idle"


def by_name(catalog, name):
    return next(a for a in catalog if a.name == name)


def test_force_adopts_activity_and_cools_down_the_one_left():
    catalog = catalog_of(("idle", "play"))
    selector = ActivitySelector(catalog, random.Random(5), temperature=0.05)
    idle_needs = needs_with(energy=1.0, playfulness=0.0)
    assert selector.choose(idle_needs, T0).name == "idle"
    selector.force(by_name(catalog, "play"), T0 + 10.0)
    # play is now current: min-duration stickiness (20s) holds like a normal pick
    assert selector.choose(idle_needs, T0 + 15.0).name == "play"
    # past play's min duration, but idle's 45s deselect cooldown still holds
    assert selector.choose(idle_needs, T0 + 31.0).name == "play"
    # cooldown expired: the overwhelming idle urge wins again
    assert selector.choose(idle_needs, T0 + 70.0).name == "idle"


def test_force_same_activity_is_noop():
    catalog = catalog_of(("idle", "play"))
    selector = ActivitySelector(catalog, random.Random(5), temperature=0.05)
    idle_needs = needs_with(energy=1.0, playfulness=0.0)
    assert selector.choose(idle_needs, T0).name == "idle"
    selector.force(by_name(catalog, "idle"), T0 + 30.0)
    # current_since was NOT reset: idle's original 60s window has elapsed at
    # +61s, so the play urge can take over (a reset would hold until +90s)
    play_needs = needs_with(energy=0.3, playfulness=1.0)
    assert selector.choose(play_needs, T0 + 61.0).name == "play"


def test_force_clears_pending_release():
    catalog = catalog_of(("idle", "groom", "play"))
    selector = ActivitySelector(catalog, random.Random(5), temperature=0.05)
    clean = needs_with(energy=1.0, playfulness=0.0, cleanliness=1.0)
    assert selector.choose(clean, T0).name == "idle"
    selector.release()
    selector.force(by_name(catalog, "play"), T0 + 5.0)
    # the pre-force release must not leak into the next choose: play's
    # min-duration stickiness holds against an overwhelming groom urge
    dirty = needs_with(energy=1.0, playfulness=0.0, cleanliness=0.0)
    assert selector.choose(dirty, T0 + 10.0).name == "play"


def test_equal_utilities_split_roughly_evenly():
    rng = random.Random(21)
    equal = needs_with(energy=0.5, playfulness=0.5)
    idle_picks = 0
    trials = 400
    for _ in range(trials):
        selector = ActivitySelector(catalog_of(("idle", "play")), rng)
        if selector.choose(equal, T0).name == "idle":
            idle_picks += 1
    assert 0.4 < idle_picks / trials < 0.6


def test_choose_is_deterministic_given_seed():
    def run(seed):
        rng = random.Random(seed)
        selector = ActivitySelector(catalog_of(FULL), rng)
        out = []
        wall = T0
        for i in range(50):
            wall += 60.0
            out.append(selector.choose(needs_with(energy=0.5 + 0.004 * i), wall).name)
        return out

    assert run(123) == run(123)
