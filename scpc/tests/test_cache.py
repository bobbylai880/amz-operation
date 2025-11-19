"""Tests for the in-process feature cache."""
from __future__ import annotations

from scpc.cache import FeatureCache, SceneCacheKey


def test_scene_cache_loader_invoked_once() -> None:
    cache = FeatureCache()
    key = SceneCacheKey(scene_id="SCN", iso_weeks=("2025-W01", "2025-W02"))
    calls: list[int] = []

    def loader() -> dict[str, object]:
        calls.append(1)
        return {"value": 42}

    first = cache.get_scene(key, loader)
    second = cache.get_scene(key, loader)
    assert first is second
    assert len(calls) == 1


def test_scene_cache_includes_end_week_dimension() -> None:
    cache = FeatureCache()
    key_a = SceneCacheKey(scene_id="SCN", iso_weeks=("2024-W30",), end_week="2024-W40")
    key_b = SceneCacheKey(scene_id="SCN", iso_weeks=("2024-W30",), end_week="2024-W41")

    loader_calls = {"count": 0}

    def loader() -> dict[str, object]:
        loader_calls["count"] += 1
        return {"value": loader_calls["count"]}

    cache.get_scene(key_a, loader)
    cache.get_scene(key_b, loader)
    assert loader_calls["count"] == 2
