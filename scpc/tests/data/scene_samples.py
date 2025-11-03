"""Sample datasets used to simulate Doris snapshots for Scene tests."""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Tuple

import pandas as pd


BASE_WEEK = date(2024, 1, 7)  # Sunday, matches PRD requirement


def _week_series(num_weeks: int) -> List[date]:
    return [BASE_WEEK + timedelta(weeks=i) for i in range(num_weeks)]


def build_keyword_facts_with_gaps() -> Tuple[pd.DataFrame, List[date]]:
    """Construct raw keyword×周 facts with short/long gaps.

    The layout mirrors ``bi_amz_vw_kw_week`` and allows ``clean_keyword_panel``
    to exercise interpolation, gap detection and smoothing branches.
    """

    weeks = _week_series(5)
    facts = pd.DataFrame(
        {
            "marketplace_id": ["US"] * 6,
            "keyword_norm": ["shower caddy"] * 4 + ["shower bag"] * 2,
            "year": [2024, 2024, 2024, 2024, 2024, 2024],
            "week_num": [1, 3, 4, 5, 1, 5],
            "startDate": [weeks[0], weeks[2], weeks[3], weeks[4], weeks[0], weeks[4]],
            "vol": [420, 610, 890, 760, 130, 150],
        }
    )
    return facts, weeks


def build_weight_table() -> pd.DataFrame:
    """Return keyword weights as stored in ``bi_amz_vw_scene_keyword``."""

    return pd.DataFrame(
        {
            "keyword_norm": ["shower caddy", "shower bag"],
            "weight": [1.0, 0.6],
        }
    )


def build_clean_panel_sample(num_weeks: int = 10) -> pd.DataFrame:
    """Generate a cleaned keyword panel resembling ``bi_amz_scene_kw_week_clean``."""

    weeks = _week_series(num_weeks)
    records = []
    for idx, start in enumerate(weeks):
        year, week_num, _ = start.isocalendar()
        records.append(
            {
                "keyword_norm": "shower caddy",
                "start_date": start,
                "year": year,
                "week_num": week_num,
                "vol_s": 400 + idx * 40,
                "gap_flag": 0,
                "winsor_low": 200.0,
                "winsor_high": 1200.0,
                "z": 0.0,
            }
        )
        records.append(
            {
                "keyword_norm": "shower bag",
                "start_date": start,
                "year": year,
                "week_num": week_num,
                "vol_s": 150 + idx * 12,
                "gap_flag": 0,
                "winsor_low": 80.0,
                "winsor_high": 500.0,
                "z": 0.0,
            }
        )
    return pd.DataFrame(records)


def build_coverage_table(num_weeks: int = 10) -> pd.DataFrame:
    """Provide scene×周 coverage snapshots."""

    weeks = _week_series(num_weeks)
    return pd.DataFrame(
        {
            "scene": ["浴室架"] * num_weeks,
            "marketplace_id": ["US"] * num_weeks,
            "year": [start.isocalendar()[0] for start in weeks],
            "week_num": [start.isocalendar()[1] for start in weeks],
            "start_date": weeks,
            "coverage": [0.92] * num_weeks,
        }
    )


def build_raw_facts_for_drivers() -> pd.DataFrame:
    """Raw weekly snapshots for driver attribution tests."""

    week1, week2 = BASE_WEEK, BASE_WEEK + timedelta(weeks=1)
    return pd.DataFrame(
        {
            "marketplace_id": ["US", "US", "US", "US"],
            "keyword_norm": ["shower caddy", "shower bag", "shower caddy", "shower bag"],
            "year": [2024, 2024, 2024, 2024],
            "week_num": [1, 1, 2, 2],
            "startDate": [week1, week1, week2, week2],
            "vol": [420, 150, 520, 120],
            "rank": [8, 18, 6, 25],
            "clickShare": [0.13, 0.07, 0.16, 0.05],
            "conversionShare": [0.028, 0.014, 0.031, 0.010],
        }
    )


def build_features_for_drivers() -> pd.DataFrame:
    """Scene feature aggregates used to normalise contributions."""

    week1, week2 = BASE_WEEK, BASE_WEEK + timedelta(weeks=1)
    return pd.DataFrame(
        {
            "scene": ["浴室架", "浴室架"],
            "marketplace_id": ["US", "US"],
            "year": [2024, 2024],
            "week_num": [1, 2],
            "start_date": [week1, week2],
            "VOL": [570, 652],
        }
    )
