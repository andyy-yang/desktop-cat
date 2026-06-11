import json

import pytest

from overlay.brain.store import PersistenceStore


def test_load_missing_returns_none(tmp_path):
    store = PersistenceStore(tmp_path / "absent.json")
    assert store.load() is None


def test_save_load_round_trip(tmp_path):
    store = PersistenceStore(tmp_path / "state.json")
    data = {"version": 1, "wall": 123.5, "needs": {"needs": {"energy": 0.25}},
            "activity": None, "nested": {"list": [1, 2, 3]}}
    store.save(data)
    assert store.load() == data


def test_save_creates_parent_dirs(tmp_path):
    store = PersistenceStore(tmp_path / "deep" / "nested" / "state.json")
    store.save({"x": 1})
    assert store.load() == {"x": 1}


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    store = PersistenceStore(path)
    with pytest.raises(json.JSONDecodeError):
        store.load()


def test_store_is_dumb_no_mutation(tmp_path):
    store = PersistenceStore(tmp_path / "state.json")
    data = {"unknown_key": "kept verbatim", "wall": -5}
    store.save(data)
    assert store.load() == data
