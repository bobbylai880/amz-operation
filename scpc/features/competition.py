"""Competition level feature helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(slots=True)
class CompetitionScore:
    """Summarises competitive intensity for a parent."""

    intensity: float
    threats: list[str]
    opportunities: list[str]


def score_competition(pairs: Iterable[Mapping[str, object]]) -> CompetitionScore:
    """Assign a simple score to the current competitive landscape."""

    intensity = 0.0
    threats: list[str] = []
    opportunities: list[str] = []
    for pair in pairs:
        price_gap = float(pair.get("price_gap", 0.0) or 0.0)
        rank_diff = float(pair.get("rank_diff", 0.0) or 0.0)
        if price_gap < 0:
            threats.append(f"价格劣势:{pair.get('comp_child')}")
            intensity += abs(price_gap)
        if rank_diff > 0:
            opportunities.append(f"排名优势:{pair.get('our_child')}")
            intensity -= rank_diff * 0.1
    return CompetitionScore(
        intensity=max(0.0, intensity),
        threats=threats,
        opportunities=opportunities,
    )


__all__ = ["CompetitionScore", "score_competition"]
