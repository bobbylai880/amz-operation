"""Helpers to construct and execute scene level LLM summaries."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from scpc.llm.deepseek_client import DeepSeekError, create_client_from_env
from scpc.settings import get_deepseek_settings


SYSTEM_PROMPT = """你是一位资深亚马逊跨境电商总监，负责场景级预测。\n严格遵守：\n1. 仅基于输入 JSON，禁止引入额外数据或假设。\n2. 引用任意时间必须同时包含 year、week_num、start_date（周日）。\n3. 输出必须使用中文，并严格匹配 response_schema。\n4. 预测规则：\n   • 本年最近4周 WoW 中位数 = 短期趋势项 r。\n   • 去年“过去4周 + 未来4周”体量计算季节先验：s[h] = (LY_future_4w[h].vol / LY_pivot.vol) - 1。\n   • 组合增速：w[h] = α * r + (1-α) * s[h]，默认 α=0.6，可由 blend_weights 覆盖。\n   • 逐周滚动：pred[0] = 当前最后一周 vol；pred[h] = pred[h-1] * (1 + w[h])。\n   • 方向阈值：pct_change ≥ +1% → up；≤ −1% → down；其余 flat，pct_change 对“最后观测周”计算并保留1位小数。\n   • 关键词预测与场景同法、独立计算。\n5. 数据不足或缺行 → insufficient_data=true，并在 notes 说明原因。\n"""

OUTPUT_INSTRUCTIONS = [
    "仅返回一个 JSON 对象，UTF-8 编码，无额外解释或 Markdown。",
    "必须包含字段 scene_forecast、top_keywords_forecast、confidence、insufficient_data；可选 notes（为空请输出 null）。",
    "scene_forecast.weeks 与 top_keywords_forecast[*].weeks 均需列出未来4周的 direction 与 pct_change。",
    "top_keywords_forecast 数量不超过 3。",
]

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "scene.schema.json"

FEATURES_SQL = text(
    """
    SELECT scene, marketplace_id, year, week_num, start_date, VOL
    FROM bi_amz_scene_features
    WHERE scene = :scene AND marketplace_id = :mk
    ORDER BY year, week_num
    """
)

DRIVERS_SQL = text(
    """
    SELECT scene, marketplace_id, year, week_num, start_date, horizon, direction,
           keyword, contrib
    FROM bi_amz_scene_drivers
    WHERE scene = :scene AND marketplace_id = :mk AND (year * 100 + week_num) = :yearweek
    """
)

KEYWORD_VOLUMES_SQL = text(
    """
    SELECT keyword_norm, year, week_num, startDate, vol
    FROM bi_amz_vw_kw_week
    WHERE marketplace_id = :mk
      AND keyword_norm IN :keywords
      AND startDate BETWEEN :start_min AND :start_max
    ORDER BY keyword_norm, year, week_num
    """
).bindparams(bindparam("keywords", expanding=True))


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
    frame = pd.read_sql_query(
        KEYWORD_VOLUMES_SQL,
        conn,
        params={
            "mk": mk,
            "keywords": unique_keywords,
            "start_min": start_min.isoformat(),
            "start_max": start_max.isoformat(),
        },
    )
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
        subset = keyword_frame[keyword_frame["keyword_norm"] == keyword] if not keyword_frame.empty else pd.DataFrame()
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


def _build_scene_summary_payload(
    scene: str,
    marketplace_id: str,
    windows: SceneWindows,
    keyword_payload: list[dict[str, object]],
    *,
    blend_weights: Mapping[str, float],
    thresholds: Mapping[str, float],
    data_quality: Mapping[str, Any],
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
    }
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
        keyword_frame = _fetch_keyword_volumes(
            conn,
            mk,
            [entry["keyword"] for entry in top_keywords],
            keyword_dates,
        )
    keyword_payload = _prepare_keyword_payload(top_keywords, keyword_frame, windows)
    blend_weights, thresholds = _load_forecast_parameters()
    data_quality = _compute_data_quality(windows, keyword_payload)
    payload = _build_scene_summary_payload(
        scene,
        mk,
        windows,
        keyword_payload,
        blend_weights=blend_weights,
        thresholds=thresholds,
        data_quality=data_quality,
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
