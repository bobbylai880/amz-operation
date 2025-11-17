"""Simple in-memory caches to reuse scene and competition features."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, TypeVar, cast


T = TypeVar("T")


@dataclass(frozen=True)
class SceneCacheKey:
    """Uniquely identifies a cached scene feature bundle."""

    scene_id: str
    iso_weeks: Tuple[str, ...]
    end_week: str | None = None


@dataclass(frozen=True)
class CompetitionCacheKey:
    """Uniquely identifies a cached competition feature bundle."""

    parent_id: str
    iso_weeks: Tuple[str, ...]
    scene_id: str | None = None
    end_week: str | None = None


@dataclass(slots=True)
class FeatureCache:
    """In-memory cache for scene and competition features.

    The cache is intentionally lightweight and process-local.  Callers provide a
    ``loader`` callable that computes the desired features when the cache does
    not contain an entry for the requested key.  Subsequent lookups reuse the
    cached value, avoiding redundant database reads and computations when the
    weekly job is invoked multiple times for the same scene/parent pair.
    """

    _scene: Dict[SceneCacheKey, object] = field(default_factory=dict)
    _competition: Dict[CompetitionCacheKey, object] = field(default_factory=dict)

    def get_scene(self, key: SceneCacheKey, loader: Callable[[], T]) -> T:
        """Return scene features for ``key`` computing them if absent."""

        if key not in self._scene:
            self._scene[key] = loader()
        return cast(T, self._scene[key])

    def has_scene(self, key: SceneCacheKey) -> bool:
        """Return ``True`` when ``key`` already has cached scene features."""

        return key in self._scene

    def get_competition(self, key: CompetitionCacheKey, loader: Callable[[], T]) -> T:
        """Return competition features for ``key`` computing them if absent."""

        if key not in self._competition:
            self._competition[key] = loader()
        return cast(T, self._competition[key])

    def has_competition(self, key: CompetitionCacheKey) -> bool:
        """Return ``True`` when ``key`` already has cached competition features."""

        return key in self._competition

    def invalidate_scene(self, key: SceneCacheKey) -> None:
        """Remove a scene cache entry if it exists."""

        self._scene.pop(key, None)

    def invalidate_competition(self, key: CompetitionCacheKey) -> None:
        """Remove a competition cache entry if it exists."""

        self._competition.pop(key, None)

    def clear(self) -> None:
        """Drop all cached artefacts."""

        self._scene.clear()
        self._competition.clear()


GLOBAL_CACHE = FeatureCache()


__all__ = [
    "CompetitionCacheKey",
    "FeatureCache",
    "GLOBAL_CACHE",
    "SceneCacheKey",
]
