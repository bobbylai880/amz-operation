"""Helpers to construct LLM prompts for the scene summary agent."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping

import pandas as pd

SYSTEM_PROMPT = """你是一位资深亚马逊跨境电商总监，负责场景级大盘诊断。\n核心约束：\n1. 仅基于输入 JSON 判断，禁止自行加总或回归。\n2. 引用任意时间必须包含 year/week_num 与 start_date（周日）。\n3. 当 confidence < 0.6 或 coverage 偏低时，需要明确提示“谨慎判断/样本不足”。\n4. 输出需包含 3-5 条“下周动作清单”（含负责人/预算或资源/执行时间）。\n"""


@dataclass(slots=True)
class SceneSummaryPayload:
    """Container with prepared chat messages for DeepSeek."""

    messages: list[Mapping[str, str]]
    raw: Mapping[str, object]


def _json_ready(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
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


def _prepare_drivers(drivers: pd.DataFrame) -> list[dict[str, object]]:
    if drivers.empty:
        return []
    drivers = drivers.copy()
    drivers["start_date"] = pd.to_datetime(drivers["start_date"]).dt.date
    latest = drivers["start_date"].max()
    subset = drivers[drivers["start_date"] == latest].sort_values(["horizon", "direction", "contrib"], ascending=[True, True, False])
    result: list[dict[str, object]] = []
    for _, row in subset.iterrows():
        result.append({key: _json_ready(row[key]) for key in row.index})
    return result


def build_scene_summary_payload(
    scene: str,
    marketplace_id: str,
    features: pd.DataFrame,
    drivers: pd.DataFrame,
) -> SceneSummaryPayload:
    """Return chat messages ready for ``DeepSeekClient.generate``."""

    facts = {
        "scene": scene,
        "marketplace_id": marketplace_id,
        "scene_features": _prepare_features(features),
        "scene_drivers": _prepare_drivers(drivers),
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
    ]
    return SceneSummaryPayload(messages=messages, raw=facts)


__all__ = ["SceneSummaryPayload", "build_scene_summary_payload", "SYSTEM_PROMPT"]
