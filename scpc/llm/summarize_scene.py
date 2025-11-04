"""Helpers to construct and execute scene level LLM summaries."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from scpc.llm.deepseek_client import DeepSeekError, create_client_from_env
from scpc.settings import get_deepseek_settings


SYSTEM_PROMPT = """你是一位资深亚马逊跨境电商总监，负责场景级大盘诊断。\n核心约束：\n1. 仅基于输入 JSON 判断，禁止自行加总或回归。\n2. 引用任意时间必须包含 year/week_num 与 start_date（周日）。\n3. 当 confidence < 0.6 或 coverage 偏低时，需要明确提示“谨慎判断/样本不足”。\n4. 输出需包含 3-5 条“下周动作清单”（含负责人/预算或资源/执行时间）。\n5. 结果必须严格按照 response_schema 返回 JSON 对象，禁止额外说明或 Markdown。\n"""

OUTPUT_INSTRUCTIONS = [
    "仅返回一个 JSON 对象，UTF-8，无额外解释或 Markdown。",
    "必须包含字段 status、drivers、insufficient_data；可选 notes，如无信息请输出 null。",
    "drivers 为对象数组，每项包含 keyword(string) 与 delta(number)。",
]

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "scene.schema.json"

FEATURES_SQL = text(
    """
    SELECT scene, marketplace_id, year, week_num, start_date, VOL, wow, yoy, wow_sa, slope8,
           breadth_wow_pos, breadth_yoy_pos, HHI_kw, volatility_8w, coverage,
           new_kw_share, strength_bucket, forecast_p10, forecast_p50, forecast_p90, confidence
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
    WHERE scene = :scene AND marketplace_id = :mk AND (year * 100 + week_num) = :yearweek
    """
)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "scene.schema.json"

FEATURES_SQL = text(
    """
    SELECT scene, marketplace_id, year, week_num, start_date, VOL, wow, yoy, wow_sa, slope8,
           breadth_wow_pos, breadth_yoy_pos, HHI_kw, volatility_8w, coverage,
           new_kw_share, strength_bucket, forecast_p10, forecast_p50, forecast_p90, confidence
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
    WHERE scene = :scene AND marketplace_id = :mk AND (year * 100 + week_num) = :yearweek
    """
)


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


def _json_ready(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date().isoformat()
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive fallback
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive fallback
            pass
    if pd.isna(value):
        return None
    return value


def _prepare_features(features: pd.DataFrame, limit: int = 8) -> list[dict[str, object]]:
    if features.empty:
        return []
    ordered = features.sort_values("start_date").tail(limit)
    payload: list[dict[str, object]] = []
    for _, row in ordered.iterrows():
        record = {key: _json_ready(row[key]) for key in row.index}
        payload.append(record)
    return payload


def _prepare_drivers(drivers: pd.DataFrame, topn: int) -> list[dict[str, object]]:
    if drivers.empty:
        return []
    drivers = drivers.copy()
    drivers["start_date"] = pd.to_datetime(drivers["start_date"]).dt.date
    result: list[pd.DataFrame] = []
    for (horizon, direction), subset in drivers.groupby(["horizon", "direction"], sort=False):
        ascending = direction == "neg"
        ordered = subset.sort_values("contrib", ascending=ascending).head(topn)
        result.append(ordered)
    if not result:
        return []
    limited = pd.concat(result, ignore_index=True)
    return [{key: _json_ready(row[key]) for key in limited.columns} for _, row in limited.iterrows()]


def _build_scene_summary_payload(
    scene: str,
    marketplace_id: str,
    features: pd.DataFrame,
    drivers: pd.DataFrame,
    *,
    topn: int,
) -> SceneSummaryPayload:
    facts = {
        "scene": scene,
        "marketplace_id": marketplace_id,
        "scene_features": _prepare_features(features),
        "scene_drivers": _prepare_drivers(drivers, topn),
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


def _validate_schema(schema: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    required = schema.get("required", [])
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing required key: {key}")
    properties: Mapping[str, Any] = schema.get("properties", {})
    for key, value in payload.items():
        expected = properties.get(key)
        if not expected:
            continue
        expected_type = expected.get("type")
        if expected_type is None:
            continue
        if isinstance(expected_type, list):
            types = tuple(_python_types(name) for name in expected_type)
            types = tuple(tp for group in types for tp in group)
        else:
            types = _python_types(expected_type)
        if types and not isinstance(value, types):
            raise ValueError(f"Field {key} expected {types}, got {type(value)!r}")


def _python_types(name: str) -> tuple[type, ...]:
    mapping: dict[str, tuple[type, ...]] = {
        "object": (dict,),
        "array": (list,),
        "string": (str,),
        "number": (int, float),
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
        yearweek = _latest_yearweek(features)
        drivers = pd.read_sql_query(
            DRIVERS_SQL,
            conn,
            params={"scene": scene, "mk": mk, "yearweek": yearweek},
        )
    payload = _build_scene_summary_payload(scene, mk, features, drivers, topn=topn)
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
