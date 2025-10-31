"""Scene level volume and momentum features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(slots=True)
class SceneFeatureSet:
    """Aggregated metrics describing scene momentum."""

    vol_current: int
    vol_previous: int
    vol_last_year: int
    wow: float
    yoy: float
    wow_sa: float
    slope8: float


def compute_scene_features(records: Iterable[Mapping[str, object]]) -> SceneFeatureSet:
    """Aggregate keyword metrics into a consolidated feature bundle."""

    vol_current = 0
    vol_previous = 0
    vol_last_year = 0
    wow = 0.0
    yoy = 0.0
    wow_sa = 0.0
    slope8 = 0.0
    for record in records:
        vol_current += int(record.get("vol_current", 0) or 0)
        vol_previous += int(record.get("vol_previous", 0) or 0)
        vol_last_year += int(record.get("vol_last_year", 0) or 0)
        wow += float(record.get("wow", 0.0) or 0.0)
        yoy += float(record.get("yoy", 0.0) or 0.0)
        wow_sa += float(record.get("wow_sa", 0.0) or 0.0)
        slope8 += float(record.get("slope8", 0.0) or 0.0)
    return SceneFeatureSet(
        vol_current=vol_current,
        vol_previous=vol_previous,
        vol_last_year=vol_last_year,
        wow=wow,
        yoy=yoy,
        wow_sa=wow_sa,
        slope8=slope8,
    )


__all__ = ["SceneFeatureSet", "compute_scene_features"]
