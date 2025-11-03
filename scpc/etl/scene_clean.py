"""Keyword level cleaning for the Scene AI pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd


def _iso_to_sunday(year: int, week: int) -> date:
    """Return the Sunday date for the given ISO ``year``/``week``."""

    monday = date.fromisocalendar(year, week, 1)
    return monday - timedelta(days=1)


@dataclass(slots=True)
class CleanResult:
    """Container holding the cleaned keyword panel."""

    data: pd.DataFrame
    week_index: list[date]


def _ensure_start_date(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "startDate" in frame.columns:
        frame["start_date"] = pd.to_datetime(frame["startDate"]).dt.date
    else:
        frame["start_date"] = pd.NaT
    missing_mask = frame["start_date"].isna()
    if missing_mask.any():
        frame.loc[missing_mask, "start_date"] = frame.loc[missing_mask, ["year", "week_num"]].apply(
            lambda row: _iso_to_sunday(int(row["year"]), int(row["week_num"])), axis=1
        )
    return frame


def _run_winsor(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Winsorise the series at the P1/P99 boundaries."""

    valid = series.dropna()
    if valid.empty:
        return series, float("nan"), float("nan")
    lower = float(np.nanpercentile(valid, 1))
    upper = float(np.nanpercentile(valid, 99))
    clipped = series.clip(lower, upper)
    return clipped, lower, upper


def _robust_z(series: pd.Series) -> pd.Series:
    """Return a robust z-score based on the median absolute deviation."""

    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(series)), index=series.index)
    median = float(valid.median())
    mad = float((valid - median).abs().median())
    if mad == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - median) / (1.4826 * mad)


def _gap_flags(series: pd.Series) -> pd.Series:
    """Identify gaps longer than two consecutive weeks."""

    flags = pd.Series(np.zeros(len(series), dtype=int), index=series.index)
    is_missing = series.isna()
    if not is_missing.any():
        return flags
    groups = []
    start = None
    for idx, missing in enumerate(is_missing):
        if missing and start is None:
            start = idx
        elif not missing and start is not None:
            groups.append((start, idx - 1))
            start = None
    if start is not None:
        groups.append((start, len(series) - 1))
    for left, right in groups:
        if right - left + 1 > 2:
            flags.iloc[left : right + 1] = 1
    return flags


def clean_keyword_panel(
    keyword_facts: pd.DataFrame,
    *,
    week_index: Iterable[date] | None = None,
) -> CleanResult:
    """Return smoothed keyword time series suitable for aggregation."""

    frame = _ensure_start_date(keyword_facts)
    if frame.empty:
        empty = pd.DataFrame(
            columns=[
                "keyword_norm",
                "start_date",
                "vol_s",
                "gap_flag",
                "winsor_low",
                "winsor_high",
                "z",
                "year",
                "week_num",
            ]
        )
        return CleanResult(data=empty, week_index=[])
    if week_index is None:
        candidates = frame["start_date"].unique().tolist()
    else:
        candidates = list(week_index)
    week_index = sorted({pd.Timestamp(value).date() for value in candidates})
    frame["start_date"] = pd.to_datetime(frame["start_date"]).dt.date
    frame = frame.sort_values(["keyword_norm", "start_date"])  # type: ignore[list-item]
    cleaned_rows: list[pd.DataFrame] = []
    for keyword, group in frame.groupby("keyword_norm"):
        group = group.set_index("start_date")
        series = group["vol"].groupby(level=0).mean().reindex(week_index)
        gap_flag = _gap_flags(series)
        interpolated = series.interpolate(method="linear", limit=2, limit_direction="both")
        interpolated[gap_flag.astype(bool)] = np.nan
        winsorised, winsor_low, winsor_high = _run_winsor(interpolated)
        z_scores = _robust_z(winsorised)
        smoothed = winsorised.rolling(window=3, min_periods=1).mean()
        smoothed[gap_flag.astype(bool)] = np.nan
        result = pd.DataFrame(
            {
                "keyword_norm": keyword,
                "start_date": week_index,
                "vol_s": smoothed.values,
                "gap_flag": gap_flag.values,
                "winsor_low": winsor_low,
                "winsor_high": winsor_high,
                "z": z_scores.values,
            }
        )
        cleaned_rows.append(result)
    cleaned = pd.concat(cleaned_rows, ignore_index=True)
    cleaned["start_date"] = pd.to_datetime(cleaned["start_date"]).dt.date
    cleaned["year"] = cleaned["start_date"].apply(lambda d: d.isocalendar()[0])
    cleaned["week_num"] = cleaned["start_date"].apply(lambda d: d.isocalendar()[1])
    cleaned = cleaned.sort_values(["keyword_norm", "start_date"]).reset_index(drop=True)
    return CleanResult(data=cleaned, week_index=list(week_index))


__all__ = ["CleanResult", "clean_keyword_panel"]
