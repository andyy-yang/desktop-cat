import pytest

from overlay.brain.activities import (
    LOOP_BY_CATEGORY,
    build_catalog,
    walk_direction,
)

from .util import make_catalog, make_index, make_manifests


def test_full_index_builds_expected_activities():
    catalog = make_catalog(("sleep", "idle", "groom", "play", "walk", "reaction"))
    assert [a.name for a in catalog.activities] == \
        ["sleep", "idle", "groom", "play", "walk"]
    by_name = {a.name: a for a in catalog.activities}
    assert "reaction" not in by_name  # never schedulable

    sleep = by_name["sleep"]
    assert sleep.utility_need == "energy" and sleep.invert is False
    idle = by_name["idle"]
    assert idle.utility_need == "energy" and idle.invert is True
    groom = by_name["groom"]
    assert groom.utility_need == "cleanliness" and groom.invert is False
    play = by_name["play"]
    assert play.utility_need == "playfulness" and play.invert is True

    for activity in catalog.activities:
        assert activity.category == activity.name
        assert 0.0 < activity.min_s < activity.max_s
        assert activity.clips == tuple(sorted(activity.clips))
        assert len(activity.clips) == 2
        assert len(activity.motions) == len(activity.clips)


def test_only_present_categories_are_built():
    catalog = make_catalog(("idle", "play"))
    assert [a.name for a in catalog.activities] == ["idle", "play"]


def test_walk_only_included_when_clips_exist():
    assert "walk" not in [a.name for a in make_catalog(("idle",)).activities]
    assert "walk" in [a.name for a in make_catalog(("idle", "walk")).activities]


def test_unknown_category_raises():
    index = {"clips": [{"name": "x", "category": "zoomies", "dir": "clips/x"}]}
    with pytest.raises(ValueError):
        build_catalog(index, make_manifests(index))


def test_empty_index_builds_empty_catalog():
    catalog = build_catalog({"clips": []}, {})
    assert catalog.activities == ()
    assert catalog.durations == {}
    assert catalog.reaction_clips == ()


def test_malformed_index_raises():
    with pytest.raises(KeyError):
        build_catalog({}, {})
    index = {"clips": [{"name": "x", "dir": "clips/x"}]}
    with pytest.raises(KeyError):
        build_catalog(index, make_manifests({"clips": []}))


def test_loop_map_covers_all_categories():
    for category in ("sleep", "idle", "groom", "play", "walk", "reaction"):
        assert LOOP_BY_CATEGORY[category] in ("pingpong", "forward", "once")
    assert LOOP_BY_CATEGORY["walk"] == "forward"
    assert LOOP_BY_CATEGORY["reaction"] == "once"


def test_clips_grouped_per_category():
    index = {"clips": [
        {"name": "nap_long", "category": "sleep", "dir": "d1"},
        {"name": "nap_curl", "category": "sleep", "dir": "d2"},
        {"name": "loaf", "category": "idle", "dir": "d3"},
    ]}
    catalog = build_catalog(index, make_manifests(index))
    by_name = {a.name: a for a in catalog.activities}
    assert by_name["sleep"].clips == ("nap_curl", "nap_long")
    assert by_name["idle"].clips == ("loaf",)


# -- walk direction derivation ------------------------------------------------

def test_walk_direction_from_name_substring():
    assert walk_direction("walk_left_somali") == "left"
    assert walk_direction("walk_right_somali") == "right"
    assert walk_direction("somali_walk_left") == "left"
    assert walk_direction("loaf_front") is None
    assert walk_direction("walk_sideways") is None


def test_walk_direction_ambiguous_name_raises():
    with pytest.raises(ValueError):
        walk_direction("walk_left_walk_right")


def test_walk_activity_stores_paired_motions():
    catalog = make_catalog(("walk",))
    walk = catalog.activities[0]
    assert walk.clips == ("walk_left", "walk_right")
    assert walk.motions == ("left", "right")


def test_directionless_walk_clip_raises_at_catalog_build():
    index = {"clips": [{"name": "walk_amble", "category": "walk", "dir": "d"}]}
    with pytest.raises(ValueError, match="walk_amble"):
        build_catalog(index, make_manifests(index))


def test_non_walk_clips_have_no_motion_even_with_direction_in_name():
    index = {"clips": [
        {"name": "idle_walk_left_pose", "category": "idle", "dir": "d"},
    ]}
    catalog = build_catalog(index, make_manifests(index))
    assert catalog.activities[0].motions == (None,)


# -- durations and reaction clips ----------------------------------------------

def test_durations_computed_from_fps_and_frame_count():
    index = make_index(("idle", "reaction"))
    manifests = make_manifests(index, fps=12.0, frame_count=30,
                               frame_counts={"reaction_a": 29})
    catalog = build_catalog(index, manifests)
    assert catalog.durations["idle_a"] == pytest.approx(2.5)
    assert catalog.durations["reaction_a"] == pytest.approx(29 / 12.0)
    assert set(catalog.durations) == {"idle_a", "idle_b", "reaction_a", "reaction_b"}


def test_reaction_clips_collected_sorted_and_not_scheduled():
    catalog = make_catalog(("idle", "reaction"))
    assert catalog.reaction_clips == ("reaction_a", "reaction_b")
    assert [a.name for a in catalog.activities] == ["idle"]


def test_missing_manifest_raises():
    index = make_index(("idle",))
    manifests = make_manifests(index)
    del manifests["idle_b"]
    with pytest.raises(ValueError, match="idle_b"):
        build_catalog(index, manifests)


def test_non_positive_fps_or_frame_count_raises():
    index = make_index(("idle",))
    bad_fps = make_manifests(index)
    bad_fps["idle_a"]["fps"] = 0.0
    with pytest.raises(ValueError):
        build_catalog(index, bad_fps)
    bad_count = make_manifests(index)
    bad_count["idle_a"]["frameCount"] = 0
    with pytest.raises(ValueError):
        build_catalog(index, bad_count)
