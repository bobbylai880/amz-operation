"""Scene level aggregation and feature engineering."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureResult:
    """Container holding scene level weekly features."""

    data: pd.DataFrame


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float)
    denom = denominator.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        output = result / denom
    output[(denom == 0) | denom.isna()] = np.nan
    return output


def _rolling_slope(series: pd.Series, window: int = 8) -> pd.Series:
    values = series.to_numpy(dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    for idx in range(len(values)):
        left = max(0, idx - window + 1)
        segment = values[left : idx + 1]
        mask = ~np.isnan(segment)
        if mask.sum() < 2:
            continue
        x = np.arange(mask.sum(), dtype=float)
        y = segment[mask]
        slope = np.polyfit(x, y, 1)[0]
        mean = np.nanmean(y)
        if mean != 0:
            result[idx] = slope / mean
    return pd.Series(result, index=series.index)


def _seasonal_index(weeks: pd.DataFrame) -> pd.Series:
    if len(weeks) < 26:
        return pd.Series(np.ones(len(weeks)), index=weeks.index)
    vol = weeks["VOL"].replace(0, np.nan)
    if vol.dropna().empty:
        return pd.Series(np.ones(len(weeks)), index=weeks.index)
    seasonal = weeks.groupby("week_num")[["VOL"]].transform("mean")
    overall_mean = vol.mean()
    factors = seasonal["VOL"] / overall_mean if overall_mean else 1.0
    mean_factor = factors.mean()
    if mean_factor and not np.isnan(mean_factor):
        factors = factors / mean_factor
    else:
        factors = pd.Series(np.ones(len(weeks)), index=weeks.index)
    return factors


def _strength_bucket(row: pd.Series) -> str:
    score = 0.0
    wow_sa = row.get("wow_sa", np.nan)
    yoy = row.get("yoy", np.nan)
    slope8 = row.get("slope8", np.nan)
    for value, weights in ((wow_sa, (0.02, 0.08, 1.0, 1.5)), (yoy, (0.05, 0.15, 1.0, 1.5))):
        if np.isnan(value):
            continue
        low, high, base, bonus = weights
        if value >= high:
            score += base + bonus
        elif value >= low:
            score += base
    if not np.isnan(slope8):
        if slope8 >= 0.08:
            score += 1.0
        elif slope8 >= 0.02:
            score += 0.5
    if score >= 4.0:
        return "S5"
    if score >= 3.0:
        return "S4"
    if score >= 2.0:
        return "S3"
    if score >= 1.0:
        return "S2"
    return "S1"


def _forecast_row(row: pd.Series) -> tuple[float, float, float]:
    wow_sa = row.get("wow_sa", 0.0)
    slope8 = row.get("slope8", 0.0)
    volatility = row.get("volatility_8w", 0.0)
    base = 1.0 + 0.5 * (wow_sa if not np.isnan(wow_sa) else 0.0) + 0.3 * (slope8 if not np.isnan(slope8) else 0.0)
    base = max(0.0, base)
    spread = 0.05 + (volatility if not np.isnan(volatility) else 0.0)
    p50 = base
    p10 = max(0.0, p50 - spread)
    p90 = p50 + spread
    return p10, p50, p90


def compute_scene_features(
    clean_panel: pd.DataFrame,
    keyword_weights: pd.DataFrame,
    coverage: pd.DataFrame | None = None,
    *,
    scene: str,
    marketplace_id: str,
) -> FeatureResult:
    """Aggregate keyword metrics into weekly scene level features."""

    if clean_panel.empty:
        columns = [
            "scene",
            "marketplace_id",
            "year",
            "week_num",
            "start_date",
            "VOL",
            "wow",
            "yoy",
            "season",
            "wow_sa",
            "slope8",
            "breadth_wow_pos",
            "breadth_yoy_pos",
            "HHI_kw",
            "volatility_8w",
            "coverage",
            "new_kw_share",
            "strength_bucket",
            "forecast_p10",
            "forecast_p50",
            "forecast_p90",
            "confidence",
        ]
        return FeatureResult(data=pd.DataFrame(columns=columns))

    weights = keyword_weights[["keyword_norm", "weight"]].drop_duplicates().copy()
    weights["weight"] = weights["weight"].fillna(1.0)
    panel = clean_panel.merge(weights, on="keyword_norm", how="left")
    panel["weight"] = panel["weight"].fillna(1.0)
    panel["vol_s"] = panel["vol_s"].astype(float)
    panel["has_vol"] = panel["vol_s"].notna()
    panel["weighted_vol"] = panel["vol_s"].fillna(0.0) * panel["weight"]
    panel = panel.sort_values(["keyword_norm", "start_date"])
    panel["vol_prev"] = panel.groupby("keyword_norm")["vol_s"].shift(1)
    panel["vol_yoy"] = panel.groupby("keyword_norm")["vol_s"].shift(52)
    panel["first_seen"] = panel.groupby("keyword_norm")["start_date"].transform("min")
    panel["start_date_dt"] = pd.to_datetime(panel["start_date"])
    panel["first_seen_dt"] = pd.to_datetime(panel["first_seen"])
    panel["wow_delta"] = panel["vol_s"] - panel["vol_prev"]
    panel["yoy_delta"] = panel["vol_s"] - panel["vol_yoy"]
    panel["wow_base"] = panel["vol_s"].notna() & panel["vol_prev"].notna()
    panel["yoy_base"] = panel["vol_s"].notna() & panel["vol_yoy"].notna()
    panel["wow_pos"] = panel["wow_base"] & (panel["wow_delta"] > 0)
    panel["yoy_pos"] = panel["yoy_base"] & (panel["yoy_delta"] > 0)
    days_since_intro = (panel["start_date_dt"] - panel["first_seen_dt"]).dt.days
    panel["weighted_new"] = np.where(
        (days_since_intro >= 0) & (days_since_intro <= 56),
        panel["weighted_vol"],
        0.0,
    )
    weekly = (
        panel.groupby(["year", "week_num", "start_date"], as_index=False)
        .agg(
            VOL=("weighted_vol", "sum"),
            keywords_with_data=("has_vol", "sum"),
            wow_pos=("wow_pos", "sum"),
            wow_base=("wow_base", "sum"),
            yoy_pos=("yoy_pos", "sum"),
            yoy_base=("yoy_base", "sum"),
            new_kw_volume=("weighted_new", "sum"),
        )
        .sort_values("start_date")
        .reset_index(drop=True)
    )
    weekly["VOL"] = weekly["VOL"].fillna(0.0)
    keyword_total = float(weights["keyword_norm"].nunique()) or 1.0
    weekly["breadth_wow_pos"] = _safe_div(weekly["wow_pos"], weekly["wow_base"])
    weekly["breadth_yoy_pos"] = _safe_div(weekly["yoy_pos"], weekly["yoy_base"])
    weekly["new_kw_share"] = _safe_div(weekly["new_kw_volume"], weekly["VOL"].replace(0, np.nan))
    weekly["coverage_fallback"] = weekly["keywords_with_data"] / keyword_total
    weekly["VOL_prev"] = weekly["VOL"].shift(1)
    weekly["VOL_yoy"] = weekly["VOL"].shift(52)
    weekly["wow"] = _safe_div(weekly["VOL"], weekly["VOL_prev"]) - 1
    weekly.loc[weekly["VOL_prev"] == 0, "wow"] = np.nan
    weekly["yoy"] = _safe_div(weekly["VOL"], weekly["VOL_yoy"]) - 1
    weekly.loc[weekly["VOL_yoy"] == 0, "yoy"] = np.nan
    seasonal = _seasonal_index(weekly)
    weekly["season"] = seasonal
    adjusted = weekly["VOL"] / weekly["season"].replace(0, np.nan)
    weekly["wow_sa"] = _safe_div(adjusted, adjusted.shift(1)) - 1
    weekly.loc[adjusted.shift(1) == 0, "wow_sa"] = np.nan
    weekly["slope8"] = _rolling_slope(weekly["VOL"], window=8)
    weekly["volatility_8w"] = weekly["wow"].rolling(window=8, min_periods=2).std()
    panel = panel.merge(weekly[["start_date", "VOL"]], on="start_date", how="left")
    panel["share"] = _safe_div(panel["weighted_vol"], panel["VOL"])
    panel["share_sq"] = panel["share"] ** 2
    hhi = panel.groupby("start_date")["share_sq"].sum().reset_index(name="HHI_kw")
    weekly = weekly.merge(hhi, on="start_date", how="left")
    if coverage is not None and not coverage.empty:
        cov = coverage.copy()
        cov["start_date"] = pd.to_datetime(cov["start_date"]).dt.date
        weekly = weekly.merge(
            cov[["year", "week_num", "start_date", "coverage"]],
            on=["year", "week_num", "start_date"],
            how="left",
        )
    else:
        weekly["coverage"] = np.nan
    weekly["coverage"] = weekly["coverage"].fillna(weekly["coverage_fallback"].clip(upper=1.0))
    sample_weeks = len(weekly)
    cap = 0.9 if sample_weeks >= 26 else 0.55
    stab = 1 - weekly["volatility_8w"].fillna(0).clip(upper=1.0)
    cons = np.where(keyword_total >= 10, 1.0, 0.7)
    weekly["confidence"] = (
        0.5 * weekly["coverage"].fillna(0)
        + 0.3 * stab
        + 0.2 * cons
    )
    weekly["confidence"] = weekly["confidence"].clip(lower=0.0, upper=cap)
    forecasts = weekly.apply(_forecast_row, axis=1, result_type="expand")
    weekly[["forecast_p10", "forecast_p50", "forecast_p90"]] = forecasts
    weekly["strength_bucket"] = weekly.apply(_strength_bucket, axis=1)
    weekly["scene"] = scene
    weekly["marketplace_id"] = marketplace_id
    weekly = weekly[
        [
            "scene",
            "marketplace_id",
            "year",
            "week_num",
            "start_date",
            "VOL",
            "wow",
            "yoy",
            "season",
            "wow_sa",
            "slope8",
            "breadth_wow_pos",
            "breadth_yoy_pos",
            "HHI_kw",
            "volatility_8w",
            "coverage",
            "new_kw_share",
            "strength_bucket",
            "forecast_p10",
            "forecast_p50",
            "forecast_p90",
            "confidence",
        ]
    ]
    weekly["start_date"] = pd.to_datetime(weekly["start_date"]).dt.date
    return FeatureResult(data=weekly)


__all__ = ["FeatureResult", "compute_scene_features"]
