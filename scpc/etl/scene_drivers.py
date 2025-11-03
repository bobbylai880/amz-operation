"""Driver contribution decomposition for the Scene AI pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DriversResult:
    """Container holding keyword driver rows ready for persistence."""

    data: pd.DataFrame


def _ensure_start_date(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "startDate" in result.columns:
        result["start_date"] = pd.to_datetime(result["startDate"]).dt.date
    else:
        result["start_date"] = pd.NaT
    missing = result["start_date"].isna()
    if missing.any():
        result.loc[missing, "start_date"] = result.loc[missing, ["year", "week_num"]].apply(
            lambda row: (date.fromisocalendar(int(row["year"]), int(row["week_num"]), 1) - timedelta(days=1)),
            axis=1,
        )
    return result


def _prepare_raw_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    frame = _ensure_start_date(raw)
    frame = frame.sort_values(["keyword_norm", "start_date"])
    for column in ("rank", "clickShare", "conversionShare"):
        frame[f"{column}_prev"] = frame.groupby("keyword_norm")[column].shift(1)
        frame[f"{column}_yoy"] = frame.groupby("keyword_norm")[column].shift(52)
    return frame


def compute_scene_drivers(
    clean_panel: pd.DataFrame,
    raw_facts: pd.DataFrame,
    scene_features: pd.DataFrame,
    keyword_weights: pd.DataFrame,
    *,
    scene: str,
    marketplace_id: str,
    topn: int = 10,
) -> DriversResult:
    """Return top contributors explaining WoW/YoY movements."""

    if clean_panel.empty or scene_features.empty:
        columns = [
            "scene",
            "marketplace_id",
            "year",
            "week_num",
            "start_date",
            "horizon",
            "direction",
            "keyword",
            "contrib",
            "vol_delta",
            "rank_delta",
            "clickShare_delta",
            "conversionShare_delta",
            "is_new_kw",
        ]
        return DriversResult(data=pd.DataFrame(columns=columns))

    weights = keyword_weights[["keyword_norm", "weight"]].drop_duplicates().copy()
    weights["weight"] = weights["weight"].fillna(1.0)
    panel = clean_panel.merge(weights, on="keyword_norm", how="left")
    panel["weight"] = panel["weight"].fillna(1.0)
    panel = panel.sort_values(["keyword_norm", "start_date"])
    panel["vol_prev"] = panel.groupby("keyword_norm")["vol_s"].shift(1)
    panel["vol_yoy"] = panel.groupby("keyword_norm")["vol_s"].shift(52)
    panel["first_seen"] = panel.groupby("keyword_norm")["start_date"].transform("min")
    panel["is_new_kw"] = (panel["start_date"] == panel["first_seen"]).astype(int)
    panel["vol_delta_wow"] = panel["vol_s"] - panel["vol_prev"]
    panel["vol_delta_yoy"] = panel["vol_s"] - panel["vol_yoy"]
    panel["weighted_vol"] = panel["vol_s"].fillna(0.0) * panel["weight"]
    panel["start_date"] = pd.to_datetime(panel["start_date"]).dt.date

    raw_metrics = _prepare_raw_metrics(raw_facts)
    raw_metrics["start_date"] = pd.to_datetime(raw_metrics["start_date"]).dt.date
    raw_lookup = raw_metrics.set_index(["keyword_norm", "start_date"]) if not raw_metrics.empty else None

    features = scene_features.sort_values("start_date").copy()
    features["start_date"] = pd.to_datetime(features["start_date"]).dt.date
    features["VOL_prev"] = features["VOL"].shift(1)
    features["VOL_yoy"] = features["VOL"].shift(52)
    vol_prev_map = features.set_index("start_date")["VOL_prev"].to_dict()
    vol_yoy_map = features.set_index("start_date")["VOL_yoy"].to_dict()

    records: list[dict[str, object]] = []
    epsilon = 1e-9
    for _, feat_row in features.iterrows():
        start_date = feat_row["start_date"]
        year = int(feat_row["year"])
        week_num = int(feat_row["week_num"])
        week_panel = panel[panel["start_date"] == start_date]
        if week_panel.empty:
            continue
        total_prev = vol_prev_map.get(start_date, np.nan)
        total_yoy = vol_yoy_map.get(start_date, np.nan)
        week_panel = week_panel.copy()
        week_panel["weighted_delta_wow"] = week_panel["weight"] * week_panel["vol_delta_wow"].fillna(0.0)
        week_panel["weighted_delta_yoy"] = week_panel["weight"] * week_panel["vol_delta_yoy"].fillna(0.0)

        def emit(horizon: str, direction: str, data: pd.Series) -> None:
            records.append(
                {
                    "scene": scene,
                    "marketplace_id": marketplace_id,
                    "year": year,
                    "week_num": week_num,
                    "start_date": start_date,
                    "horizon": horizon,
                    "direction": direction,
                    "keyword": data["keyword_norm"],
                    "contrib": float(data["contrib"]),
                    "vol_delta": float(data["vol_delta"]),
                    "rank_delta": data.get("rank_delta"),
                    "clickShare_delta": data.get("clickShare_delta"),
                    "conversionShare_delta": data.get("conversionShare_delta"),
                    "is_new_kw": int(data["is_new_kw"]),
                }
            )

        # WoW contributions
        denom_wow = total_prev if total_prev and not np.isnan(total_prev) else np.nan
        if not np.isnan(denom_wow) and abs(denom_wow) > epsilon:
            wow_candidates = week_panel.dropna(subset=["vol_delta_wow"]).copy()
            wow_candidates["contrib"] = wow_candidates["weighted_delta_wow"] / denom_wow
            wow_candidates["vol_delta"] = wow_candidates["vol_delta_wow"]
            wow_candidates = wow_candidates.merge(
                raw_metrics[[
                    "keyword_norm",
                    "start_date",
                    "rank",
                    "rank_prev",
                    "clickShare",
                    "clickShare_prev",
                    "conversionShare",
                    "conversionShare_prev",
                ]],
                on=["keyword_norm", "start_date"],
                how="left",
            ) if raw_lookup is not None else wow_candidates
            if raw_lookup is not None:
                wow_candidates["rank_delta"] = wow_candidates["rank"] - wow_candidates["rank_prev"]
                wow_candidates["clickShare_delta"] = wow_candidates["clickShare"] - wow_candidates["clickShare_prev"]
                wow_candidates["conversionShare_delta"] = (
                    wow_candidates["conversionShare"] - wow_candidates["conversionShare_prev"]
                )
            else:
                wow_candidates["rank_delta"] = np.nan
                wow_candidates["clickShare_delta"] = np.nan
                wow_candidates["conversionShare_delta"] = np.nan
            wow_pos = wow_candidates[wow_candidates["contrib"] > 0].nlargest(topn, "contrib")
            wow_neg = wow_candidates[wow_candidates["contrib"] < 0].nsmallest(topn, "contrib")
            for _, row in wow_pos.iterrows():
                emit("WoW", "pos", row)
            for _, row in wow_neg.iterrows():
                emit("WoW", "neg", row)

        # YoY contributions
        denom_yoy = total_yoy if total_yoy and not np.isnan(total_yoy) else np.nan
        if not np.isnan(denom_yoy) and abs(denom_yoy) > epsilon:
            yoy_candidates = week_panel.dropna(subset=["vol_delta_yoy"]).copy()
            yoy_candidates["contrib"] = yoy_candidates["weighted_delta_yoy"] / denom_yoy
            yoy_candidates["vol_delta"] = yoy_candidates["vol_delta_yoy"]
            yoy_candidates = yoy_candidates.merge(
                raw_metrics[[
                    "keyword_norm",
                    "start_date",
                    "rank",
                    "rank_yoy",
                    "clickShare",
                    "clickShare_yoy",
                    "conversionShare",
                    "conversionShare_yoy",
                ]],
                on=["keyword_norm", "start_date"],
                how="left",
            ) if raw_lookup is not None else yoy_candidates
            if raw_lookup is not None:
                yoy_candidates["rank_delta"] = yoy_candidates["rank"] - yoy_candidates["rank_yoy"]
                yoy_candidates["clickShare_delta"] = yoy_candidates["clickShare"] - yoy_candidates["clickShare_yoy"]
                yoy_candidates["conversionShare_delta"] = (
                    yoy_candidates["conversionShare"] - yoy_candidates["conversionShare_yoy"]
                )
            else:
                yoy_candidates["rank_delta"] = np.nan
                yoy_candidates["clickShare_delta"] = np.nan
                yoy_candidates["conversionShare_delta"] = np.nan
            yoy_pos = yoy_candidates[yoy_candidates["contrib"] > 0].nlargest(topn, "contrib")
            yoy_neg = yoy_candidates[yoy_candidates["contrib"] < 0].nsmallest(topn, "contrib")
            for _, row in yoy_pos.iterrows():
                emit("YoY", "pos", row)
            for _, row in yoy_neg.iterrows():
                emit("YoY", "neg", row)

    result_df = pd.DataFrame.from_records(records)
    return DriversResult(data=result_df)


__all__ = ["DriversResult", "compute_scene_drivers"]
