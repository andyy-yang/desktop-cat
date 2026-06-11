import dataclasses

import pytest

from overlay.brain.commands import BrainEvent, PlayClip


def test_playclip_fields_and_equality():
    a = PlayClip(clip="sleep_a", loop="pingpong", min_seconds=42.0)
    b = PlayClip(clip="sleep_a", loop="pingpong", min_seconds=42.0)
    assert a == b
    assert a.clip == "sleep_a"
    assert a.loop == "pingpong"
    assert a.min_seconds == 42.0


def test_playclip_motion_defaults_to_none():
    cmd = PlayClip(clip="idle_a", loop="pingpong", min_seconds=10.0)
    assert cmd.motion is None


def test_playclip_motion_distinguishes_commands():
    left = PlayClip(clip="walk_left", loop="forward", min_seconds=30.0, motion="left")
    right = PlayClip(clip="walk_left", loop="forward", min_seconds=30.0, motion="right")
    assert left.motion == "left"
    assert left != right


def test_playclip_frozen():
    cmd = PlayClip(clip="idle_a", loop="forward", min_seconds=10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cmd.clip = "other"


def test_brainevent_fields_and_frozen():
    ev = BrainEvent(kind="pet", at=12.5)
    assert ev.kind == "pet"
    assert ev.at == 12.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.kind = "click"
