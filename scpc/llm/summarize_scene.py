"""Helpers to construct and execute scene level LLM summaries."""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import StatisticsError, median
from typing import Any, Mapping, Sequence

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from scpc.llm.deepseek_client import DeepSeekError, create_client_from_env
from scpc.settings import get_deepseek_settings


SYSTEM_PROMPT = """你是一位资深亚马逊跨境电商总监，负责场景级预测。\n严格遵守：\n1. 仅基于输入 JSON，禁止引入额外数据或假设。\n2. 引用任意时间必须同时包含 year、week_num、start_date（周日）。\n3. 输出必须使用中文，并严格匹配 response_schema。\n4. 在 analysis_summary 中，用不超过400字的中文总结场景与主要关键词未来趋势及驱动原因，并引用至少两个具体指标数值作为依据（例如最新周 vol、wow、yoy、关键词贡献或预测 pct_change）。\n5. 预测规则：\n   • 本年最近4周 WoW 中位数 = 短期趋势项 r。\n   • 去年“过去4周 + 未来4周”体量计算季节先验：s[h] = (LY_future_4w[h].vol / LY_pivot.vol) - 1。\n   • 组合增速：w[h] = α * r + (1-α) * s[h]，默认 α=0.6，可由 blend_weights 覆盖。\n   • 逐周滚动：pred[0] = 当前最后一周 vol；pred[h] = pred[h-1] * (1 + w[h])。\n   • 方向阈值：pct_change ≥ +1% → up；≤ −1% → down；其余 flat，pct_change 对“最后观测周”计算并保留1位小数。\n   • 关键词预测与场景同法、独立计算。\n6. 数据不足或缺行 → insufficient_data=true，并在 notes 说明原因。\n"""

OUTPUT_INSTRUCTIONS = [
    "仅返回一个 JSON 对象，UTF-8 编码，无额外解释或 Markdown。",
    "必须包含字段 scene_forecast、top_keywords_forecast、confidence、insufficient_data、analysis_summary；可选 notes（为空请输出 null）。",
    "scene_forecast.weeks 与 top_keywords_forecast[*].weeks 均需列出未来4周的 direction 与 pct_change。",
    "top_keywords_forecast 数量不超过 3。",
    "analysis_summary 需以中文总结场景与主要关键词的趋势及主因，长度不超过400字。",
]

OUTPUT_INSTRUCTIONS.append(
    "若提供 bounds.p10/p90，则第4周累计相对变化需落在该区间，越界时取边界值。"
)
OUTPUT_INSTRUCTIONS.append(
    "若 facts.forecast_guidance.scene.forecast_weeks 非空，则 scene_forecast 的 direction 与 pct_change 必须与之保持一致（允许按四舍五入保留一位小数）。"
)
OUTPUT_INSTRUCTIONS.append(
    "若 facts.forecast_guidance.keywords[*].forecast_weeks 非空，则对应 keyword 的 direction 与 pct_change 需与之保持一致（允许按四舍五入保留一位小数）。"
)
OUTPUT_INSTRUCTIONS.append(
    "analysis_summary 必须引用 facts.analysis_evidence 中至少两个具体指标值（如最新周 vol、wow、yoy、关键词贡献或预测 pct_change）作为结论依据。"
)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "scene.schema.json"

FEATURES_SQL = text(
    """
    SELECT scene, marketplace_id, year, week_num, start_date, VOL,
           wow, yoy, wow_sa, slope8, breadth_wow_pos, breadth_yoy_pos, HHI_kw,
           volatility_8w, coverage, new_kw_share, strength_bucket,
           forecast_p10, forecast_p50, forecast_p90, confidence
    FROM bi_amz_scene_features
    WHERE scene = :scene AND marketplace_id = :mk
    ORDER BY year, week_num
    """
)

DRIVERS_SQL = text(
    """
    SELECT scene, marketplace_id, year, week_num, start_date, horizon, direction,
           keyword, contrib, vol_delta, rank_delta, clickShare_delta,
           conversionShare_delta, is_new_kw
    FROM bi_amz_scene_drivers
    WHERE scene = :scene AND marketplace_id = :mk
      AND (year * 100 + week_num) = :yearweek
    """
)

KEYWORD_VOLUMES_SQL = text(
    """
    SELECT keyword_norm, year, week_num, startDate, vol
    FROM bi_amz_vw_kw_week
    WHERE marketplace_id = :mk
      AND keyword_norm IN (:keywords)
      AND startDate BETWEEN :start_min AND :start_max
    ORDER BY keyword_norm, year, week_num
    """
)

KEYWORD_VOLUMES_SQL_TEMPLATE = """
    SELECT keyword_norm, year, week_num, startDate, vol
    FROM bi_amz_vw_kw_week
    WHERE marketplace_id = :mk
      AND keyword_norm IN ({keywords})
      AND startDate BETWEEN :start_min AND :start_max
    ORDER BY keyword_norm, year, week_num
"""

@dataclass(slots=True)
class SceneSummaryPayload:
    """Container with prepared chat messages for DeepSeek."""

    messages: list[Mapping[str, str]]
    raw: Mapping[str, object]


class SceneSummarizationError(RuntimeError):
    """Raised when the LLM summary could not be produced."""

    def __init__(self, message: str, *, raw: str | None = None, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.raw = raw
        self.details = details or []


def _norm_kw(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)


def _to_volume(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _row_to_scene_record(row: pd.Series) -> dict[str, object]:
    start = row.get("start_date")
    if isinstance(start, pd.Timestamp):
        start = start.date()
    if not isinstance(start, date):
        raise ValueError("start_date must be datetime.date for scene features")
    vol = _to_volume(row.get("VOL"))
    iso = start.isocalendar()
    try:
        year = int(row.get("year"))
    except (TypeError, ValueError):
        year = int(iso[0])
    try:
        week = int(row.get("week_num"))
    except (TypeError, ValueError):
        week = int(iso[1])
    return {
        "year": year,
        "week_num": week,
        "start_date": start.isoformat(),
        "vol": vol,
    }

def _row_to_scene_record(row: pd.Series) -> dict[str, object]:
    start = row.get("start_date")
    if isinstance(start, pd.Timestamp):
        start = start.date()
    if not isinstance(start, date):
        raise ValueError("start_date must be datetime.date for scene features")
    vol = _to_volume(row.get("VOL"))
    iso = start.isocalendar()
    try:
        year = int(row.get("year"))
    except (TypeError, ValueError):
        year = int(iso[0])
    try:
        week = int(row.get("week_num"))
    except (TypeError, ValueError):
        week = int(iso[1])
    return {
        "year": year,
        "week_num": week,
        "start_date": start.isoformat(),
        "vol": vol,
    }

def _lookup_feature_row(features: pd.DataFrame, target: date) -> pd.Series | None:
    if features.empty:
        return None
    matches = features[features["start_date"] == target]
    if matches.empty:
        return None
    return matches.iloc[-1]


def _dates_to_scene_records(features: pd.DataFrame, dates: Sequence[date]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for target in dates:
        row = _lookup_feature_row(features, target)
        if row is not None:
            vol = _to_volume(row.get("VOL"))
            year = int(row.get("year"))
            week = int(row.get("week_num"))
        else:
            iso = target.isocalendar()
            year = int(iso[0])
            week = int(iso[1])
            vol = None
        records.append(
            {
                "year": year,
                "week_num": week,
                "start_date": target.isoformat(),
                "vol": vol,
            }
        )
    return records


@dataclass(slots=True)
class SceneWindows:
    recent: list[dict[str, object]]
    last_year_past: list[dict[str, object]]
    last_year_future: list[dict[str, object]]
    future_calendar: list[dict[str, object]]
    recent_dates: list[date]
    last_year_past_dates: list[date]
    last_year_future_dates: list[date]
    last_observed: dict[str, object] | None


def _derive_scene_windows(features: pd.DataFrame) -> SceneWindows:
    ordered = features.sort_values("start_date")
    recent_rows = ordered.tail(4)
    recent_dates = [row.start_date for row in recent_rows.itertuples(index=False)]
    recent = [_row_to_scene_record(row) for _, row in recent_rows.iterrows()]
    last_year_past_dates = [dt - timedelta(days=364) for dt in recent_dates]
    last_year_future_dates: list[date] = []
    if last_year_past_dates:
        pivot = last_year_past_dates[-1]
        last_year_future_dates = [pivot + timedelta(days=7 * offset) for offset in range(1, 5)]
    last_year_past = _dates_to_scene_records(ordered, last_year_past_dates)
    last_year_future = _dates_to_scene_records(ordered, last_year_future_dates)
    future_calendar: list[dict[str, object]] = []
    if recent_dates:
        anchor = recent_dates[-1]
        for offset in range(1, 5):
            target = anchor + timedelta(days=7 * offset)
            iso = target.isocalendar()
            future_calendar.append(
                {
                    "year": int(iso[0]),
                    "week_num": int(iso[1]),
                    "start_date": target.isoformat(),
                }
            )
    last_observed = recent[-1] if recent else None
    return SceneWindows(
        recent=recent,
        last_year_past=last_year_past,
        last_year_future=last_year_future,
        future_calendar=future_calendar,
        recent_dates=recent_dates,
        last_year_past_dates=last_year_past_dates,
        last_year_future_dates=last_year_future_dates,
        last_observed=last_observed,
    )


def _select_top_keywords(drivers: pd.DataFrame, limit: int) -> list[dict[str, object]]:
    if drivers.empty or limit <= 0:
        return []
    limit = min(limit, 3)
    subset = drivers.copy()
    subset["keyword"] = subset["keyword"].astype(str)
    subset["contrib"] = pd.to_numeric(subset["contrib"], errors="coerce")
    subset["abs_contrib"] = subset["contrib"].abs()
    subset = subset[subset["abs_contrib"] >= 0.01]
    subset = subset.sort_values("abs_contrib", ascending=False)
    seen: set[str] = set()
    results: list[dict[str, object]] = []
    for row in subset.itertuples(index=False):
        keyword = getattr(row, "keyword")
        if keyword in seen:
            continue
        contrib = _to_volume(getattr(row, "contrib", None))
        results.append({"keyword": keyword, "contrib": contrib})
        seen.add(keyword)
        if len(results) >= limit:
            break
    return results


def _fetch_keyword_volumes(
    conn, mk: str, keywords: Sequence[str], dates: Sequence[date]
) -> pd.DataFrame:
    unique_keywords = [kw for kw in dict.fromkeys(keywords) if kw]
    if not unique_keywords or not dates:
        return pd.DataFrame(columns=["keyword_norm", "year", "week_num", "startDate", "vol"])
    start_min = min(dates)
    start_max = max(dates)
    params = {
        "mk": mk,
        "keywords": unique_keywords,
        "start_min": start_min.isoformat(),
        "start_max": start_max.isoformat(),
    }
    try:
        stmt = KEYWORD_VOLUMES_SQL.bindparams(bindparam("keywords", expanding=True))
        frame = pd.read_sql_query(stmt, conn, params=params)
    except Exception:
        placeholders = ", ".join(f":kw_{i}" for i in range(len(unique_keywords)))
        sql = text(
            f"""
            SELECT keyword_norm, year, week_num, startDate, vol
            FROM bi_amz_vw_kw_week
            WHERE marketplace_id = :mk
              AND keyword_norm IN ({placeholders})
              AND startDate BETWEEN :start_min AND :start_max
            ORDER BY keyword_norm, year, week_num
            """
        )
        dyn_params = {
            "mk": mk,
            "start_min": start_min.isoformat(),
            "start_max": start_max.isoformat(),
            **{f"kw_{i}": v for i, v in enumerate(unique_keywords)},
        }
        frame = pd.read_sql_query(sql, conn, params=dyn_params)
    if frame.empty:
        return frame
    frame["startDate"] = pd.to_datetime(frame["startDate"]).dt.date
    return frame


def _keyword_records(data: pd.DataFrame, dates: Sequence[date]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for target in dates:
        iso = target.isocalendar()
        year = int(iso[0])
        week = int(iso[1])
        vol = None
        if not data.empty:
            matches = data[data["startDate"] == target]
            if not matches.empty:
                row = matches.iloc[-1]
                year_value = row.get("year")
                week_value = row.get("week_num")
                try:
                    year = int(year_value)
                except (TypeError, ValueError):
                    year = int(iso[0])
                try:
                    week = int(week_value)
                except (TypeError, ValueError):
                    week = int(iso[1])
                vol = _to_volume(row.get("vol"))
        records.append(
            {
                "year": year,
                "week_num": week,
                "start_date": target.isoformat(),
                "vol": vol,
            }
        )
    return records


def _prepare_keyword_payload(
    keywords: list[dict[str, object]],
    keyword_frame: pd.DataFrame,
    windows: SceneWindows,
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for entry in keywords:
        keyword = entry["keyword"]
        norm = _norm_kw(keyword)
        subset = (
            keyword_frame[keyword_frame["keyword_norm"] == norm]
            if not keyword_frame.empty
            else pd.DataFrame()
        )
        payload.append(
            {
                "keyword": keyword,
                "latest_contrib": entry.get("contrib"),
                "recent_4w": _keyword_records(subset, windows.recent_dates),
                "last_year_past_4w": _keyword_records(subset, windows.last_year_past_dates),
                "last_year_future_4w": _keyword_records(subset, windows.last_year_future_dates),
            }
        )
    return payload


def _count_missing(records: Sequence[Mapping[str, object]]) -> int:
    return sum(1 for record in records if record.get("vol") is None)


def _latest_volume(records: Sequence[Mapping[str, object]]) -> float | None:
    for record in reversed(records):
        vol = _to_volume(record.get("vol"))
        if vol is None:
            continue
        if math.isfinite(vol) and vol > 0:
            return float(vol)
    return None


def _week_over_week_changes(records: Sequence[Mapping[str, object]]) -> list[float]:
    changes: list[float] = []
    previous: float | None = None
    for record in records:
        current = _to_volume(record.get("vol"))
        if current is None or not math.isfinite(current):
            previous = current
            continue
        if previous is not None and math.isfinite(previous) and previous != 0:
            change = current / previous - 1.0
            if math.isfinite(change):
                changes.append(float(change))
        previous = current
    return changes


def _median_change(values: Sequence[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    try:
        return float(median(clean))
    except StatisticsError:
        return None


def _seasonal_prior(
    past_records: Sequence[Mapping[str, object]],
    future_records: Sequence[Mapping[str, object]],
) -> list[float | None]:
    pivot: float | None = None
    for record in reversed(past_records):
        vol = _to_volume(record.get("vol"))
        if vol is None or not math.isfinite(vol) or vol <= 0:
            continue
        pivot = float(vol)
        break
    priors: list[float | None] = []
    for record in future_records:
        vol = _to_volume(record.get("vol"))
        if pivot is None or vol is None or not math.isfinite(vol) or pivot <= 0:
            priors.append(None)
            continue
        priors.append(float(vol) / pivot - 1.0)
    return priors


def _classify_direction(pct_change: float | None, flat_band: float) -> str | None:
    if pct_change is None or not math.isfinite(pct_change):
        return None
    band = abs(float(flat_band))
    if pct_change >= band:
        return "up"
    if pct_change <= -band:
        return "down"
    return "flat"


def _clean_float(value: float | None, *, digits: int | None = None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if digits is not None:
        return round(float(value), digits)
    return float(value)


def _format_pct(value: float | None, *, digits: int = 2) -> str | None:
    if value is None or not math.isfinite(value):
        return None
    scaled = float(value) * 100.0
    if digits is not None:
        scaled = round(scaled, digits)
        if abs(scaled) < 0.5 * (10 ** -digits):
            scaled = 0.0
        return f"{scaled:.{digits}f}%"
    return f"{scaled}%"


def _apply_forecast_model(
    base_vol: float | None,
    future_calendar: Sequence[Mapping[str, object]],
    *,
    alpha: float,
    recent_median: float | None,
    seasonal_prior: Sequence[float | None],
    flat_band: float,
    bounds: Mapping[str, float] | None = None,
) -> Mapping[str, Any]:
    guidance: dict[str, Any] = {
        "base_vol": _clean_float(base_vol, digits=2),
        "recent_wow_median": _clean_float(recent_median, digits=4),
        "seasonal_prior": [_clean_float(value, digits=4) for value in seasonal_prior],
        "forecast_weeks": [],
        "fourth_week_pct_change": None,
        "fourth_week_pct_change_clamped": None,
        "insufficient": False,
    }
    if base_vol is None or not math.isfinite(base_vol) or base_vol <= 0:
        guidance["insufficient"] = True
        return guidance
    alpha = max(0.0, min(1.0, float(alpha)))
    flat_band = abs(float(flat_band))
    seasonal_list = list(seasonal_prior)
    if len(seasonal_list) < len(future_calendar):
        seasonal_list.extend([None] * (len(future_calendar) - len(seasonal_list)))
    recent_value = _clean_float(recent_median)
    base = float(base_vol)
    last = base
    forecasts: list[dict[str, Any]] = []
    for idx, week_meta in enumerate(future_calendar):
        seasonal_component = seasonal_list[idx] if idx < len(seasonal_list) else None
        seasonal_value = _clean_float(seasonal_component)
        r_value = recent_value if recent_value is not None else 0.0
        s_value = seasonal_value if seasonal_value is not None else 0.0
        weight = alpha * r_value + (1.0 - alpha) * s_value
        next_vol = last * (1.0 + weight)
        pct_change = None
        if base != 0:
            pct_change = (next_vol - base) / base
        direction = _classify_direction(pct_change, flat_band)
        pct_change_value = _clean_float(pct_change, digits=4)
        forecasts.append(
            {
                "week_index": idx + 1,
                "target_week": week_meta,
                "growth_rate": _clean_float(weight, digits=4),
                "projected_vol": _clean_float(next_vol, digits=2),
                "pct_change": _format_pct(pct_change_value),
                "pct_change_value": pct_change_value,
                "direction": direction,
            }
        )
        last = next_vol
    guidance["forecast_weeks"] = forecasts
    if forecasts:
        last_pct_value = forecasts[-1]["pct_change_value"]
        guidance["fourth_week_pct_change_value"] = last_pct_value
        guidance["fourth_week_pct_change"] = _format_pct(last_pct_value)
        if bounds:
            p10 = _clean_float(_to_volume(bounds.get("p10")))
            p90 = _clean_float(_to_volume(bounds.get("p90")))
            if (
                last_pct_value is not None
                and p10 is not None
                and p90 is not None
                and p10 <= p90
            ):
                clamped = max(p10, min(p90, last_pct_value))
                clamped_clean = _clean_float(clamped, digits=4)
                guidance["fourth_week_pct_change_clamped_value"] = clamped_clean
                guidance["fourth_week_pct_change_clamped"] = _format_pct(clamped_clean)
    return guidance


def _build_forecast_guidance(
    windows: SceneWindows,
    keyword_payload: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    thresholds: Mapping[str, float],
    bounds: Mapping[str, float],
) -> Mapping[str, Any]:
    flat_band = abs(float(thresholds.get("flat_band_pct", 0.01)))
    scene_recent_changes = _week_over_week_changes(windows.recent)
    scene_median = _median_change(scene_recent_changes)
    scene_seasonal = _seasonal_prior(windows.last_year_past, windows.last_year_future)
    scene_base = _latest_volume(windows.recent)
    scene_guidance = _apply_forecast_model(
        scene_base,
        windows.future_calendar,
        alpha=alpha,
        recent_median=scene_median,
        seasonal_prior=scene_seasonal,
        flat_band=flat_band,
        bounds=bounds,
    )
    keyword_guidance: list[Mapping[str, Any]] = []
    for entry in keyword_payload:
        recent_records = entry.get("recent_4w", [])
        ly_past_records = entry.get("last_year_past_4w", [])
        ly_future_records = entry.get("last_year_future_4w", [])
        kw_recent_changes = _week_over_week_changes(recent_records)
        kw_median = _median_change(kw_recent_changes)
        kw_seasonal = _seasonal_prior(ly_past_records, ly_future_records)
        kw_base = _latest_volume(recent_records)
        kw_guidance = _apply_forecast_model(
            kw_base,
            windows.future_calendar,
            alpha=alpha,
            recent_median=kw_median,
            seasonal_prior=kw_seasonal,
            flat_band=flat_band,
        )
        kw_guidance = dict(kw_guidance)
        kw_guidance["keyword"] = entry.get("keyword")
        kw_guidance["latest_contrib"] = _clean_float(
            _to_volume(entry.get("latest_contrib")), digits=4
        )
        keyword_guidance.append(kw_guidance)
    return {"scene": scene_guidance, "keywords": keyword_guidance}


def _build_analysis_evidence(
    windows: SceneWindows,
    scene_signals: Mapping[str, Any],
    keyword_payload: Sequence[Mapping[str, Any]],
    forecast_guidance: Mapping[str, Any],
) -> Mapping[str, Any]:
    latest_signals = scene_signals.get("latest", {}) if scene_signals else {}
    scene_guidance = forecast_guidance.get("scene", {})
    scene_forecast = scene_guidance.get("forecast_weeks", [])
    scene_evidence: dict[str, Any] = {}
    latest_vol = _latest_volume(windows.recent)
    if latest_vol is not None:
        scene_evidence["latest_vol"] = _clean_float(latest_vol, digits=2)
    for key in ("wow", "yoy", "wow_sa"):
        value = latest_signals.get(key)
        clean = _clean_float(_to_volume(value), digits=4)
        if clean is not None:
            scene_evidence[key] = clean
    median_value = scene_guidance.get("recent_wow_median")
    if median_value is not None:
        scene_evidence["recent_wow_median"] = median_value
    forecast_changes = [week.get("pct_change") for week in scene_forecast if week.get("pct_change") is not None]
    if forecast_changes:
        scene_evidence["forecast_pct_change"] = forecast_changes
    clamped = scene_guidance.get("fourth_week_pct_change_clamped")
    if clamped is not None:
        scene_evidence["fourth_week_pct_change_clamped"] = clamped
    keyword_guidance_map = {
        entry.get("keyword"): entry for entry in forecast_guidance.get("keywords", [])
    }
    keyword_evidence: list[Mapping[str, Any]] = []
    for entry in keyword_payload:
        keyword = entry.get("keyword")
        kw_guidance = keyword_guidance_map.get(keyword, {})
        kw_evidence: dict[str, Any] = {"keyword": keyword}
        kw_latest_vol = _latest_volume(entry.get("recent_4w", []))
        if kw_latest_vol is not None:
            kw_evidence["latest_vol"] = _clean_float(kw_latest_vol, digits=2)
        latest_contrib = kw_guidance.get("latest_contrib")
        if latest_contrib is not None:
            kw_evidence["latest_contrib"] = latest_contrib
        kw_median = kw_guidance.get("recent_wow_median")
        if kw_median is not None:
            kw_evidence["recent_wow_median"] = kw_median
        kw_forecast = kw_guidance.get("forecast_weeks", [])
        kw_changes = [week.get("pct_change") for week in kw_forecast if week.get("pct_change") is not None]
        if kw_changes:
            kw_evidence["forecast_pct_change"] = kw_changes
        keyword_evidence.append(kw_evidence)
    return {"scene": scene_evidence, "keywords": keyword_evidence}


def _compute_data_quality(
    windows: SceneWindows,
    keyword_payload: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    keyword_quality = [
        {
            "keyword": item.get("keyword"),
            "recent_4w_missing": _count_missing(item.get("recent_4w", [])),
            "last_year_past_4w_missing": _count_missing(item.get("last_year_past_4w", [])),
            "last_year_future_4w_missing": _count_missing(item.get("last_year_future_4w", [])),
        }
        for item in keyword_payload
    ]
    return {
        "scene": {
            "recent_4w_missing": _count_missing(windows.recent),
            "last_year_past_4w_missing": _count_missing(windows.last_year_past),
            "last_year_future_4w_missing": _count_missing(windows.last_year_future),
        },
        "keywords": keyword_quality,
    }


def _load_forecast_parameters() -> tuple[dict[str, float], dict[str, float]]:
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        if pd.isna(value):
            return default
        return value

    recent_weight = _env_float("SCENE_BLEND_WEIGHT_RECENT_WOW", 0.6)
    seasonal_weight = _env_float("SCENE_BLEND_WEIGHT_SEASONAL", 0.4)
    total = recent_weight + seasonal_weight
    if total <= 0:
        recent_weight, seasonal_weight = 0.6, 0.4
        total = 1.0
    blend_weights = {
        "recent_wow": recent_weight / total,
        "seasonal": seasonal_weight / total,
    }
    flat_band = abs(_env_float("SCENE_THRESHOLD_FLAT_BAND_PCT", 0.01))
    thresholds = {"flat_band_pct": flat_band}
    return blend_weights, thresholds


def _dynamic_alpha(latest: pd.Series) -> float:
    vol8 = float(latest.get("volatility_8w") or 0.0)
    vol8 = max(0.0, min(1.0, vol8))
    cov = float(latest.get("coverage") or 0.0)
    cov = max(0.0, min(1.0, cov))
    alpha = 0.4 + 0.35 * (1 - vol8) + 0.25 * cov
    return max(0.4, min(0.8, alpha))


def _build_scene_summary_payload(
    scene: str,
    marketplace_id: str,
    windows: SceneWindows,
    keyword_payload: list[dict[str, object]],
    *,
    blend_weights: Mapping[str, float],
    thresholds: Mapping[str, float],
    data_quality: Mapping[str, Any],
    bounds: Mapping[str, float],
    scene_signals: Mapping[str, Any],
    forecast_guidance: Mapping[str, Any],
    analysis_evidence: Mapping[str, Any],
    quality_notes: str | None = None,
) -> SceneSummaryPayload:
    facts = {
        "scene": scene,
        "marketplace_id": marketplace_id,
        "latest_observed_week": windows.last_observed,
        "scene_recent_4w": windows.recent,
        "scene_last_year_past_4w": windows.last_year_past,
        "scene_last_year_future_4w": windows.last_year_future,
        "future_weeks_calendar": windows.future_calendar,
        "top_keywords": keyword_payload,
        "blend_weights": blend_weights,
        "thresholds": thresholds,
        "data_quality": data_quality,
        "bounds": bounds,
        "scene_signals": scene_signals,
        "forecast_guidance": forecast_guidance,
        "analysis_evidence": analysis_evidence,
    }
    if quality_notes:
        facts["quality_notes"] = quality_notes
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
    ]
    return SceneSummaryPayload(messages=messages, raw=facts)


def _compose_user_content(
    payload: SceneSummaryPayload,
    schema: Mapping[str, Any],
    *,
    validation_errors: list[str] | None = None,
    previous_response: str | None = None,
) -> str:
    body = dict(payload.raw)
    body["response_schema"] = schema
    body["output_instructions"] = OUTPUT_INSTRUCTIONS
    if validation_errors:
        body["schema_errors"] = validation_errors
    if previous_response is not None:
        body["previous_response"] = previous_response
    return json.dumps(body, ensure_ascii=False)


def _resolve_schema_type(schema_type: Any, value: object) -> str | None:
    if schema_type is None:
        return None
    if isinstance(schema_type, list):
        for candidate in schema_type:
            python_types = _python_types(candidate)
            if python_types and isinstance(value, python_types):
                return candidate
        return schema_type[0]
    return schema_type


def _validate_schema(schema: Mapping[str, Any], payload: object) -> None:
    schema_type = _resolve_schema_type(schema.get("type"), payload)
    if schema_type is not None:
        expected_types = _python_types(schema_type)
        if expected_types and not isinstance(payload, expected_types):
            raise ValueError(
                f"Expected type {schema_type}, got {type(payload).__name__}"
            )
    enum = schema.get("enum")
    if enum is not None and payload not in enum:
        raise ValueError(f"Value {payload!r} not in enum {enum}")
    if schema_type == "object" and isinstance(payload, Mapping):
        required = schema.get("required", [])
        for key in required:
            if key not in payload:
                raise ValueError(f"Missing required key: {key}")
        properties: Mapping[str, Any] = schema.get("properties", {})
        for key, value in payload.items():
            expected = properties.get(key)
            if expected is not None:
                _validate_schema(expected, value)
        return
    if schema_type == "array" and isinstance(payload, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(payload) < int(min_items):
            raise ValueError(f"Array expected at least {min_items} items")
        max_items = schema.get("maxItems")
        if max_items is not None and len(payload) > int(max_items):
            raise ValueError(f"Array expected at most {max_items} items")
        items_schema = schema.get("items")
        if isinstance(items_schema, Mapping):
            for item in payload:
                _validate_schema(items_schema, item)
        return


def _python_types(name: str) -> tuple[type, ...]:
    mapping: dict[str, tuple[type, ...]] = {
        "object": (dict,),
        "array": (list,),
        "string": (str,),
        "number": (int, float),
        "integer": (int,),
        "boolean": (bool,),
        "null": (type(None),),
    }
    return mapping.get(name, (object,))


def _latest_yearweek(features: pd.DataFrame) -> int:
    values = features["year"].astype(int) * 100 + features["week_num"].astype(int)
    return int(values.max())


def _load_schema() -> Mapping[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def summarize_scene(*, engine: Engine, scene: str, mk: str, topn: int = 10) -> Mapping[str, Any]:
    """Fetch latest scene features/drivers and produce a schema-compliant summary."""

    if topn <= 0:
        topn = 10
    with engine.connect() as conn:
        features = pd.read_sql_query(FEATURES_SQL, conn, params={"scene": scene, "mk": mk})
        if features.empty:
            raise SceneSummarizationError("No scene_features rows available for summarisation")
        features["start_date"] = pd.to_datetime(features["start_date"]).dt.date
        features["VOL"] = pd.to_numeric(features["VOL"], errors="coerce")
        latest_row = features.sort_values("start_date").iloc[-1]
        yearweek = _latest_yearweek(features)
        drivers = pd.read_sql_query(
            DRIVERS_SQL,
            conn,
            params={"scene": scene, "mk": mk, "yearweek": yearweek},
        )
        windows = _derive_scene_windows(features)
        top_keywords = _select_top_keywords(drivers, topn)
        keyword_dates: list[date] = []
        keyword_dates.extend(windows.recent_dates)
        keyword_dates.extend(windows.last_year_past_dates)
        keyword_dates.extend(windows.last_year_future_dates)
        kw_norm_list = [_norm_kw(entry["keyword"]) for entry in top_keywords]
        keyword_frame = _fetch_keyword_volumes(
            conn,
            mk,
            kw_norm_list,
            keyword_dates,
        )
    keyword_payload = _prepare_keyword_payload(top_keywords, keyword_frame, windows)
    blend_weights, thresholds = _load_forecast_parameters()
    alpha = _dynamic_alpha(latest_row)
    blend_weights = {"recent_wow": round(alpha, 3), "seasonal": round(1 - alpha, 3)}
    conf = float(latest_row.get("confidence") or 0.0)
    cov = float(latest_row.get("coverage") or 0.0)
    flat_band = 0.02 if (conf < 0.6 or cov < 0.6) else 0.01
    thresholds = {"flat_band_pct": flat_band}
    bounds = {
        "p10": float(latest_row.get("forecast_p10") or 0.0),
        "p50": float(latest_row.get("forecast_p50") or 0.0),
        "p90": float(latest_row.get("forecast_p90") or 0.0),
    }
    scene_signals = {
        "latest": {
            "wow": latest_row.get("wow"),
            "yoy": latest_row.get("yoy"),
            "wow_sa": latest_row.get("wow_sa"),
            "slope8": latest_row.get("slope8"),
            "breadth_wow_pos": latest_row.get("breadth_wow_pos"),
            "breadth_yoy_pos": latest_row.get("breadth_yoy_pos"),
            "HHI_kw": latest_row.get("HHI_kw"),
            "volatility_8w": latest_row.get("volatility_8w"),
            "coverage": latest_row.get("coverage"),
            "new_kw_share": latest_row.get("new_kw_share"),
            "strength_bucket": latest_row.get("strength_bucket"),
        }
    }
    data_quality = _compute_data_quality(windows, keyword_payload)
    insufficient = (
        _count_missing(windows.recent) >= 2
        or _count_missing(windows.last_year_past) >= 2
        or _count_missing(windows.last_year_future) >= 2
    )
    quality_notes = "样本不足：某些4周窗口缺失≥2个数据点" if insufficient else None
    forecast_guidance = _build_forecast_guidance(
        windows,
        keyword_payload,
        alpha=alpha,
        thresholds=thresholds,
        bounds=bounds,
    )
    analysis_evidence = _build_analysis_evidence(
        windows,
        scene_signals,
        keyword_payload,
        forecast_guidance,
    )
    payload = _build_scene_summary_payload(
        scene,
        mk,
        windows,
        keyword_payload,
        blend_weights=blend_weights,
        thresholds=thresholds,
        data_quality=data_quality,
        bounds=bounds,
        scene_signals=scene_signals,
        forecast_guidance=forecast_guidance,
        analysis_evidence=analysis_evidence,
        quality_notes=quality_notes,
    )
    schema = _load_schema()
    settings = get_deepseek_settings()
    client = create_client_from_env(settings=settings)
    errors: list[str] = []
    repair_hints: list[str] = []
    previous_response: str | None = None
    last_raw: str | None = None
    failure_message = ""
    try:
        for attempt in range(1, 3):
            content = _compose_user_content(
                payload,
                schema,
                validation_errors=repair_hints if repair_hints else None,
                previous_response=previous_response,
            )
            try:
                response = client.generate(
                    prompt=SYSTEM_PROMPT,
                    facts=content,
                    model=settings.model,
                    temperature=0.1,
                    response_format="json_object",
                )
            except DeepSeekError as exc:
                raise SceneSummarizationError("DeepSeek request failed", details=[str(exc)]) from exc

            raw_content = response.content
            last_raw = raw_content
            try:
                parsed = json.loads(raw_content)
            except json.JSONDecodeError as exc:
                message = f"JSON decode error: {exc}"
                errors.append(f"attempt {attempt}: {message}")
                repair_hints.append(message)
                previous_response = raw_content
                failure_message = "LLM returned invalid JSON"
                if attempt == 1:
                    continue
                raise SceneSummarizationError(failure_message, raw=raw_content, details=errors) from exc
            try:
                _validate_schema(schema, parsed)
            except ValueError as exc:
                message = str(exc)
                errors.append(f"attempt {attempt}: {message}")
                repair_hints.append(message)
                previous_response = raw_content
                failure_message = "Scene summary failed schema validation"
                if attempt == 1:
                    continue
                raise SceneSummarizationError(failure_message, raw=raw_content, details=errors) from exc
            return parsed
    finally:
        client.close()
    if failure_message:
        raise SceneSummarizationError(failure_message, raw=last_raw, details=errors)
    raise SceneSummarizationError("Scene summary could not be produced", raw=last_raw, details=errors)


__all__ = [
    "SceneSummarizationError",
    "SceneSummaryPayload",
    "summarize_scene",
]
