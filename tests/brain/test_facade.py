import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

from overlay.brain.activities import LOOP_BY_CATEGORY, build_catalog, walk_direction
from overlay.brain.commands import BrainEvent, PlayClip
from overlay.brain.facade import REACTION_COOLDOWN_S, Brain
from overlay.brain.needs import AWAKE_ENERGY_RATE
from overlay.brain.selector import WAKE_SLEEP_COOLDOWN_S
from overlay.brain.store import PersistenceStore

from .util import FakeClock, make_catalog, make_manifests, wall_at

REPO_ROOT = Path(__file__).resolve().parents[2]
NOON = wall_at(12)

SLEEPY_NEEDS = {"energy": 0.02, "hunger": 0.3, "playfulness": 0.0,
                "affection": 0.5, "cleanliness": 1.0}
# seed whose first softmax draw selects sleep under SLEEPY_NEEDS (P ~ 0.83);
# pinned to make the sleep-entry fixture deterministic
SLEEPY_SEED = 1
REACTION_S = 2.5  # util manifest defaults: 30 frames at 12 fps


def make_brain(tmp_path, categories=("sleep", "idle", "groom", "play"), seed=1,
               initial_needs=None, state_name="state.json", catalog=None):
    if catalog is None:
        catalog = make_catalog(categories)
    store = PersistenceStore(tmp_path / state_name)
    clock = FakeClock(NOON)
    if initial_needs is not None:
        store.save({"version": 1, "wall": clock.wall(),
                    "needs": {"needs": initial_needs}, "activity": None})
    brain = Brain(clock, random.Random(seed), catalog, store)
    return brain, clock, store


def make_sleeping_brain(tmp_path):
    brain, clock, store = make_brain(tmp_path, categories=("sleep", "idle"),
                                     seed=SLEEPY_SEED, initial_needs=SLEEPY_NEEDS)
    cmd = brain.tick([])
    assert cmd is not None and cmd.clip.startswith("sleep_")
    return brain, clock, store


def test_first_tick_emits_playclip(tmp_path):
    brain, clock, store = make_brain(tmp_path)
    cmd = brain.tick([])
    assert isinstance(cmd, PlayClip)
    category = cmd.clip.rsplit("_", 1)[0]
    assert category in ("sleep", "idle", "groom", "play")
    assert cmd.loop in ("pingpong", "forward", "once")
    assert cmd.motion is None
    by_name = {a.name: a for a in
               make_catalog(("sleep", "idle", "groom", "play")).activities}
    activity = by_name[category]
    assert activity.min_s <= cmd.min_seconds <= activity.max_s


def test_no_command_while_activity_continues(tmp_path):
    brain, clock, store = make_brain(tmp_path)
    assert brain.tick([]) is not None
    for _ in range(4):  # all activity min durations are >= 10s
        clock.advance(1.0)
        assert brain.tick([]) is None


def test_commands_only_on_change_and_gaps_respect_duration(tmp_path):
    brain, clock, store = make_brain(tmp_path, categories=("idle", "groom", "play"),
                                     seed=2)
    commands = []
    for t in range(1200):
        cmd = brain.tick([])
        if cmd is not None:
            commands.append((t, cmd))
        clock.advance(1.0)
    assert len(commands) >= 3  # the cat does change its mind over 20 minutes
    for (t_prev, cmd_prev), (t_next, _) in zip(commands, commands[1:]):
        assert t_next - t_prev >= cmd_prev.min_seconds - 1e-9


def activity_of(clip: str) -> str:
    return clip.rsplit("_", 1)[0]


def test_drag_and_pet_never_change_activity(tmp_path):
    kinds = ("pet", "drag_start", "drag_end")
    for seed in range(10):
        brain, clock, store = make_brain(tmp_path, seed=seed,
                                         state_name=f"s{seed}.json")
        first = brain.tick([])
        assert first is not None
        step = first.min_seconds / 8.0  # 6 events stay inside the deadline
        for kind in kinds * 2:
            clock.advance(step)
            assert brain.tick([BrainEvent(kind=kind, at=clock.now())]) is None


def test_pet_updates_needs_without_command(tmp_path):
    brain, clock, store = make_brain(tmp_path)
    assert brain.tick([]) is not None
    clock.advance(6.0)
    affection_before = brain._needs.needs["affection"]
    assert brain.tick([BrainEvent(kind="pet", at=clock.now())]) is None
    assert brain._needs.needs["affection"] > affection_before


def test_urgent_needs_do_not_let_events_preempt(tmp_path):
    # the old 5 s preempt gate is gone: even with a huge play urge a drag,
    # pet or cooldown-free reactionless click never re-rolls the activity
    brain, clock, store = make_brain(tmp_path, categories=("idle", "play"), seed=1,
                                     initial_needs={"energy": 0.9, "playfulness": 0.0})
    cmd = brain.tick([])
    assert cmd is not None and cmd.clip.startswith("idle_")
    brain._needs.needs["playfulness"] = 1.0
    brain._needs.needs["energy"] = 0.1
    clock.advance(6.0)
    for kind in ("click", "pet", "drag_start", "drag_end"):
        assert brain.tick([BrainEvent(kind=kind, at=clock.now())]) is None
        clock.advance(1.0)


def test_click_wakes_sleeping_cat(tmp_path):
    brain, clock, store = make_sleeping_brain(tmp_path)
    clock.advance(2.0)
    cmd = brain.tick([BrainEvent(kind="click", at=clock.now())])
    assert cmd is not None and cmd.clip.startswith("idle_")
    assert brain._needs.mode == "idle"


def test_pet_does_not_wake_sleeping_cat(tmp_path):
    brain, clock, store = make_sleeping_brain(tmp_path)
    clock.advance(6.0)
    assert brain.tick([BrainEvent(kind="pet", at=clock.now())]) is None
    assert brain._needs.mode == "sleep"


def test_drag_does_not_wake_sleeping_cat(tmp_path):
    brain, clock, store = make_sleeping_brain(tmp_path)
    clock.advance(6.0)
    assert brain.tick([BrainEvent(kind="drag_start", at=clock.now())]) is None
    clock.advance(1.0)
    assert brain.tick([BrainEvent(kind="drag_end", at=clock.now())]) is None
    assert brain._needs.mode == "sleep"


def test_sleep_refills_energy_then_exits_above_085(tmp_path):
    brain, clock, store = make_sleeping_brain(tmp_path)
    exit_cmd = None
    for _ in range(200):  # up to 6000s in 30s steps
        clock.advance(30.0)
        cmd = brain.tick([])
        if cmd is not None:
            exit_cmd = cmd
            break
    assert exit_cmd is not None, "cat never woke up"
    assert exit_cmd.clip.startswith("idle_")
    assert brain._needs.needs["energy"] > 0.85


def test_wake_event_fast_forwards_needs(tmp_path):
    brain, clock, store = make_brain(tmp_path)
    brain.tick([])
    energy_before = brain._needs.needs["energy"]
    clock.advance(7200.0)
    brain.tick([BrainEvent(kind="wake", at=clock.now())])
    expected = energy_before * math.exp(-AWAKE_ENERGY_RATE * 7200.0)
    assert brain._needs.needs["energy"] == pytest.approx(expected, rel=1e-9)


def test_shutdown_persists_and_restart_fast_forwards(tmp_path):
    brain, clock, store = make_brain(tmp_path)
    brain.tick([])
    clock.advance(100.0)
    brain.tick([])
    saved_energy = brain._needs.needs["energy"]
    brain.shutdown()

    raw = store.load()
    assert raw["version"] == 1
    assert raw["wall"] == clock.wall()
    assert raw["needs"]["needs"]["energy"] == saved_energy

    clock2 = FakeClock(clock.wall() + 3600.0)
    catalog = make_catalog(("sleep", "idle", "groom", "play"))
    brain2 = Brain(clock2, random.Random(99), catalog, store)
    expected = saved_energy * math.exp(-AWAKE_ENERGY_RATE * 3600.0)
    assert brain2._needs.needs["energy"] == pytest.approx(expected, rel=1e-9)


def test_restart_gap_clamped_at_12h(tmp_path):
    catalog = make_catalog(("sleep", "idle"))
    needs = {"energy": 1.0, "hunger": 0.0, "playfulness": 0.0,
             "affection": 1.0, "cleanliness": 1.0}
    state = {"version": 1, "wall": NOON, "needs": {"needs": needs}, "activity": None}

    store_a = PersistenceStore(tmp_path / "a.json")
    store_a.save(state)
    brain_a = Brain(FakeClock(NOON + 12 * 3600.0), random.Random(0), catalog, store_a)

    store_b = PersistenceStore(tmp_path / "b.json")
    store_b.save(state)
    brain_b = Brain(FakeClock(NOON + 24 * 3600.0), random.Random(0), catalog, store_b)

    for name in needs:
        assert brain_b._needs.needs[name] == pytest.approx(
            brain_a._needs.needs[name], abs=1e-12)


def test_brain_is_deterministic_given_clock_and_seed(tmp_path):
    def run(tag):
        brain, clock, store = make_brain(tmp_path, seed=77, state_name=f"{tag}.json")
        commands = []
        for t in range(300):
            events = []
            if t == 50:
                events.append(BrainEvent(kind="click", at=clock.now()))
            if t == 120:
                events.append(BrainEvent(kind="pet", at=clock.now()))
            if t == 200:
                events.append(BrainEvent(kind="double_click", at=clock.now()))
            commands.append(brain.tick(events))
            clock.advance(1.0)
        return commands, dict(brain._needs.needs)

    commands_1, needs_1 = run("run1")
    commands_2, needs_2 = run("run2")
    assert commands_1 == commands_2
    assert needs_1 == needs_2


def test_unknown_event_kind_raises(tmp_path):
    brain, clock, store = make_brain(tmp_path)
    brain.tick([])
    clock.advance(1.0)
    with pytest.raises(ValueError):
        brain.tick([BrainEvent(kind="meow", at=clock.now())])


# -- walking ------------------------------------------------------------------

def test_walk_emits_motion_and_matching_clip(tmp_path):
    brain, clock, store = make_brain(tmp_path, categories=("walk",))
    cmd = brain.tick([])
    assert cmd is not None
    assert cmd.loop == "forward"
    assert cmd.motion in ("left", "right")
    assert cmd.clip == f"walk_{cmd.motion}"


def test_walk_direction_varies_with_rng(tmp_path):
    motions = set()
    for seed in range(12):
        brain, clock, store = make_brain(tmp_path, categories=("walk",), seed=seed,
                                         state_name=f"w{seed}.json")
        motions.add(brain.tick([]).motion)
    assert motions == {"left", "right"}


def test_non_walk_commands_have_no_motion(tmp_path):
    brain, clock, store = make_brain(tmp_path, categories=("idle", "groom", "play"),
                                     seed=2)
    for t in range(1200):
        cmd = brain.tick([])
        if cmd is not None:
            assert cmd.motion is None
        clock.advance(1.0)


# -- reactions ------------------------------------------------------------------

def make_reactive_brain(tmp_path, categories=("idle", "reaction"), seed=1,
                        state_name="state.json"):
    brain, clock, store = make_brain(tmp_path, categories=categories, seed=seed,
                                     state_name=state_name)
    first = brain.tick([])
    assert first is not None
    return brain, clock, store, first


def test_click_triggers_reaction_with_clip_duration(tmp_path):
    brain, clock, store, first = make_reactive_brain(tmp_path)
    assert first.clip.startswith("idle_")
    clock.advance(1.0)  # before the 5 s preempt gate: reaction fires regardless
    cmd = brain.tick([BrainEvent(kind="click", at=clock.now())])
    assert cmd is not None
    assert cmd.clip.startswith("reaction_")
    assert cmd.loop == "once"
    assert cmd.min_seconds == pytest.approx(REACTION_S)
    assert cmd.motion is None


def test_reaction_resumes_exact_previous_clip(tmp_path):
    brain, clock, store, first = make_reactive_brain(tmp_path)
    clock.advance(1.0)
    reaction = brain.tick([BrainEvent(kind="click", at=clock.now())])
    assert reaction.clip.startswith("reaction_")
    clock.advance(1.0)
    assert brain.tick([]) is None  # 1.0 s into a 2.5 s reaction
    clock.advance(1.0)
    assert brain.tick([]) is None  # 2.0 s: still running
    clock.advance(1.0)
    resume = brain.tick([])        # 3.0 s: first tick after completion
    assert resume is not None
    assert resume.clip == first.clip
    assert resume.loop == first.loop
    assert resume.motion is None
    # the original activity deadline kept running through the reaction
    assert resume.min_seconds == pytest.approx(first.min_seconds - 4.0)


def test_click_during_reaction_emits_nothing(tmp_path):
    brain, clock, store, first = make_reactive_brain(tmp_path)
    clock.advance(1.0)
    assert brain.tick([BrainEvent(kind="click",
                                  at=clock.now())]).clip.startswith("reaction_")
    clock.advance(1.0)
    assert brain.tick([BrainEvent(kind="click", at=clock.now())]) is None
    clock.advance(2.0)
    assert brain.tick([]).clip == first.clip


def test_reaction_cooldown_blocks_then_expires(tmp_path):
    brain, clock, store, first = make_reactive_brain(tmp_path)
    clock.advance(1.0)
    assert brain.tick([BrainEvent(kind="click",
                                  at=clock.now())]).clip.startswith("reaction_")
    clock.advance(3.0)
    assert brain.tick([]).clip == first.clip  # resume
    clock.advance(1.0)
    # 4 s since the reaction: cooldown holds, click keeps the old behavior
    assert brain.tick([BrainEvent(kind="click", at=clock.now())]) is None
    clock.advance(REACTION_COOLDOWN_S - 4.0)
    cmd = brain.tick([BrainEvent(kind="click", at=clock.now())])
    assert cmd is not None and cmd.clip.startswith("reaction_")


def test_click_on_sleeping_cat_wakes_instead_of_reaction(tmp_path):
    brain, clock, store = make_brain(tmp_path,
                                     categories=("sleep", "idle", "reaction"),
                                     seed=SLEEPY_SEED, initial_needs=SLEEPY_NEEDS)
    cmd = brain.tick([])
    assert cmd is not None and cmd.clip.startswith("sleep_")
    clock.advance(2.0)
    cmd = brain.tick([BrainEvent(kind="click", at=clock.now())])
    assert cmd is not None and cmd.clip.startswith("idle_")
    assert brain._needs.mode == "idle"


def test_reaction_during_walk_resumes_walk_with_motion(tmp_path):
    brain, clock, store = make_brain(tmp_path, categories=("walk", "reaction"))
    first = brain.tick([])
    assert first.motion in ("left", "right")
    clock.advance(1.0)
    reaction = brain.tick([BrainEvent(kind="click", at=clock.now())])
    assert reaction.clip.startswith("reaction_")
    assert reaction.motion is None
    clock.advance(3.0)
    resume = brain.tick([])
    assert resume.clip == first.clip
    assert resume.motion == first.motion
    assert resume.loop == "forward"


def test_no_reaction_without_reaction_clips(tmp_path):
    brain, clock, store = make_brain(tmp_path, categories=("idle",))
    assert brain.tick([]) is not None
    clock.advance(1.0)
    assert brain.tick([BrainEvent(kind="click", at=clock.now())]) is None


# -- double-click (scene shuffle) -------------------------------------------------

def catalog_from(entries):
    """Catalog from explicit (name, category) pairs — for single-clip cases."""
    index = {"clips": [{"name": name, "category": category, "dir": f"clips/{name}"}
                       for name, category in entries]}
    return build_catalog(index, make_manifests(index))


def test_double_click_always_changes_clip(tmp_path):
    for seed in range(10):
        brain, clock, store = make_brain(tmp_path, seed=seed,
                                         state_name=f"d{seed}.json")
        current = brain.tick([])
        assert current is not None
        for _ in range(5):  # repeated shuffles beat gates and cooldowns too
            clock.advance(2.0)  # well inside any min duration
            cmd = brain.tick([BrainEvent(kind="double_click", at=clock.now())])
            assert cmd is not None
            assert cmd.clip != current.clip  # the scene ALWAYS visibly changes
            current = cmd


def test_double_click_command_matches_activity_loop_duration_motion(tmp_path):
    categories = ("sleep", "idle", "groom", "play", "walk")
    catalog = make_catalog(categories)
    by_name = {a.name: a for a in catalog.activities}
    brain, clock, store = make_brain(tmp_path, seed=5, catalog=catalog)
    prev = brain.tick([])
    assert prev is not None
    for _ in range(40):
        clock.advance(2.0)
        cmd = brain.tick([BrainEvent(kind="double_click", at=clock.now())])
        assert cmd is not None and cmd.clip != prev.clip
        activity = by_name[activity_of(cmd.clip)]
        assert cmd.loop == LOOP_BY_CATEGORY[activity.category]
        assert activity.min_s <= cmd.min_seconds <= activity.max_s
        if activity.category == "walk":
            assert cmd.motion == walk_direction(cmd.clip)
        else:
            assert cmd.motion is None
        prev = cmd


def test_double_click_distribution_covers_every_eligible_clip(tmp_path):
    # energy starts at 0.9 (far above the 0.25 sleep-entry gate) and every
    # shuffle puts the just-left activity on its deselect cooldown: sleep
    # clips appearing proves the gates are bypassed
    categories = ("sleep", "idle", "groom", "play", "reaction")
    catalog = make_catalog(categories)
    brain, clock, store = make_brain(tmp_path, seed=3, catalog=catalog)
    prev = brain.tick([])
    assert prev is not None
    seen = set()
    for _ in range(200):
        clock.advance(1.0)
        cmd = brain.tick([BrainEvent(kind="double_click", at=clock.now())])
        assert cmd is not None
        assert cmd.clip != prev.clip
        seen.add(cmd.clip)
        prev = cmd
    eligible = {clip for a in catalog.activities for clip in a.clips}
    assert not any(clip.startswith("reaction_") for clip in eligible)
    assert seen == eligible  # every schedulable clip appears, reactions never


def test_double_click_shuffles_within_single_activity(tmp_path):
    brain, clock, store = make_brain(tmp_path, categories=("idle",))
    current = brain.tick([])
    assert current is not None
    for _ in range(3):  # two idle clips: the shuffle must alternate them
        clock.advance(2.0)
        cmd = brain.tick([BrainEvent(kind="double_click", at=clock.now())])
        assert cmd is not None
        assert activity_of(cmd.clip) == "idle"
        assert cmd.clip != current.clip
        current = cmd


def test_double_click_with_single_clip_is_noop(tmp_path):
    catalog = catalog_from([("idle_a", "idle")])
    brain, clock, store = make_brain(tmp_path, catalog=catalog)
    assert brain.tick([]) is not None
    clock.advance(2.0)
    assert brain.tick([BrainEvent(kind="double_click", at=clock.now())]) is None


def test_double_click_after_shuffle_respects_new_deadline(tmp_path):
    brain, clock, store = make_brain(tmp_path)
    assert brain.tick([]) is not None
    clock.advance(2.0)
    cmd = brain.tick([BrainEvent(kind="double_click", at=clock.now())])
    assert cmd is not None
    for _ in range(15):  # below every activity's min duration (>= 20 s)
        clock.advance(1.0)
        assert brain.tick([]) is None


def test_double_click_shuffles_sleeping_cat_with_wake_bookkeeping(tmp_path):
    brain, clock, store = make_sleeping_brain(tmp_path)
    sleeping_clip = brain._current_clip
    clock.advance(2.0)
    cmd = brain.tick([BrainEvent(kind="double_click", at=clock.now())])
    assert cmd is not None
    assert cmd.clip != sleeping_clip
    assert brain._needs.mode == activity_of(cmd.clip)
    # wake bookkeeping ran: sleep re-entry sits on the wake cooldown, so the
    # next natural reconsideration cannot insta-return to it
    assert brain._selector._cooldown_until["sleep"] == pytest.approx(
        clock.wall() + WAKE_SLEEP_COOLDOWN_S)


def test_double_click_cuts_reaction_short_and_changes_clip(tmp_path):
    brain, clock, store = make_brain(tmp_path,
                                     categories=("idle", "play", "reaction"), seed=1)
    first = brain.tick([])
    assert first is not None
    clock.advance(1.0)
    reaction = brain.tick([BrainEvent(kind="click", at=clock.now())])
    assert reaction.clip.startswith("reaction_")
    clock.advance(1.0)  # mid-reaction
    cmd = brain.tick([BrainEvent(kind="double_click", at=clock.now())])
    assert cmd is not None
    assert not cmd.clip.startswith("reaction_")
    assert cmd.clip != first.clip


def test_double_click_during_reaction_shuffles_within_single_activity(tmp_path):
    brain, clock, store, first = make_reactive_brain(tmp_path)  # idle_a/b + reaction
    clock.advance(1.0)
    reaction = brain.tick([BrainEvent(kind="click", at=clock.now())])
    assert reaction.clip.startswith("reaction_")
    clock.advance(1.0)
    cmd = brain.tick([BrainEvent(kind="double_click", at=clock.now())])
    assert cmd is not None
    assert activity_of(cmd.clip) == "idle"
    assert cmd.clip != first.clip  # the OTHER idle clip, not a resume


def test_double_click_during_reaction_single_clip_resumes(tmp_path):
    catalog = catalog_from([("idle_a", "idle"), ("reaction_a", "reaction")])
    brain, clock, store = make_brain(tmp_path, catalog=catalog)
    first = brain.tick([])
    assert first is not None and first.clip == "idle_a"
    clock.advance(1.0)
    reaction = brain.tick([BrainEvent(kind="click", at=clock.now())])
    assert reaction.clip == "reaction_a"
    clock.advance(1.0)
    cmd = brain.tick([BrainEvent(kind="double_click", at=clock.now())])
    assert cmd is not None
    assert cmd.clip == "idle_a"  # no other schedulable clip: resume in place
    # the original activity deadline kept running through the reaction
    assert cmd.min_seconds == pytest.approx(first.min_seconds - 2.0)


def test_double_click_beats_click_in_same_batch(tmp_path):
    brain, clock, store = make_brain(tmp_path,
                                     categories=("idle", "play", "reaction"), seed=1)
    first = brain.tick([])
    assert first is not None
    clock.advance(1.0)
    now = clock.now()
    cmd = brain.tick([BrainEvent(kind="click", at=now),
                      BrainEvent(kind="double_click", at=now)])
    assert cmd is not None
    assert not cmd.clip.startswith("reaction_")
    assert cmd.clip != first.clip


def test_brain_package_imports_no_appkit():
    code = ("import sys\n"
            "import overlay.brain\n"
            "banned = [m for m in sys.modules if m.split('.')[0] in "
            "('AppKit', 'Foundation', 'objc', 'Quartz', 'Cocoa')]\n"
            "raise SystemExit(1 if banned else 0)\n")
    proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO_ROOT)
    assert proc.returncode == 0
