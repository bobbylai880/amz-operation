"""Configuration objects for the competition LLM orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass(slots=True)
class StageOneConfig:
    thresholds: Mapping[str, float]
    conf_min: float
    band_weight: float
    opp_weight: float
    severity_thresholds: Mapping[str, float]
    max_retries: int = 2
    rules_config_path: str | None = None
    enable_llm: bool = False


@dataclass(slots=True)
class StageTwoConfig:
    enabled: bool
    aggregate_per_asin: bool
    max_retries: int
    allowed_action_codes: Sequence[str]
    allowed_root_cause_codes: Sequence[str]
    trigger_status: Sequence[str]
    require_unfavorable_evidence: bool
    keyword_max_pairs_per_opp: int
    min_reviews_for_rating_priority: int
    rating_margin: float
    always_include_metrics: Mapping[str, tuple[str, ...]]


@dataclass(slots=True)
class LLMRuntimeConfig:
    model: str
    temperature: float
    top_p: float
    response_format: str
    max_retries: int
    timeout: float


@dataclass(slots=True)
class CompetitionLLMConfig:
    stage_1: StageOneConfig
    stage_2: StageTwoConfig
    llm: LLMRuntimeConfig


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Competition LLM config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("Competition LLM config must be a mapping")
    return data


def load_competition_llm_config(path: str | Path) -> CompetitionLLMConfig:
    """Read ``competition_llm.yaml`` and materialise dataclass wrappers."""

    resolved = Path(path)
    raw = _load_yaml(resolved)

    stage1_raw = raw.get("stage_1", {})
    stage2_raw = raw.get("stage_2", {})
    llm_raw = raw.get("llm", {})

    stage1_severity_raw = stage1_raw.get("severity_thresholds") or {}
    if not isinstance(stage1_severity_raw, Mapping):
        raise ValueError("stage_1.severity_thresholds must be a mapping when provided")
    severity_thresholds = {
        "high": float(stage1_severity_raw.get("high", 2.0)),
        "mid": float(stage1_severity_raw.get("mid", 1.0)),
    }

    rules_config_path = stage1_raw.get("rules_config_path")
    if isinstance(rules_config_path, str) and not rules_config_path.strip():
        rules_config_path = None

    stage1 = StageOneConfig(
        thresholds=stage1_raw.get("thresholds", {}),
        conf_min=float(stage1_raw.get("conf_min", 0.6)),
        band_weight=float(stage1_raw.get("band_weight", 0.5)),
        opp_weight=float(stage1_raw.get("opp_weight", 0.5)),
        severity_thresholds=severity_thresholds,
        max_retries=int(stage1_raw.get("max_retries", 2)),
        rules_config_path=rules_config_path,
        enable_llm=bool(stage1_raw.get("enable_llm", False)),
    )

    trigger_status_raw = stage2_raw.get("trigger_status")
    if trigger_status_raw is None:
        trigger_status = ("lag",)
    elif isinstance(trigger_status_raw, (list, tuple)):
        trigger_status = tuple(str(status).lower() for status in trigger_status_raw if str(status).strip())
    else:
        raise ValueError("stage_2.trigger_status must be a list of statuses when provided")
    if not trigger_status:
        trigger_status = ("lag",)

    keyword_section = stage2_raw.get("keyword")
    if not isinstance(keyword_section, Mapping):
        keyword_section = {}
    keyword_limit_raw = keyword_section.get("max_pairs_per_opp", 2)
    try:
        keyword_limit = int(keyword_limit_raw)
    except (TypeError, ValueError):
        keyword_limit = 2
    if keyword_limit < 1:
        keyword_limit = 2

    min_reviews_raw = stage2_raw.get("min_reviews_for_rating_priority", 200)
    try:
        min_reviews = int(min_reviews_raw)
    except (TypeError, ValueError):
        min_reviews = 200
    if min_reviews < 0:
        min_reviews = 0

    rating_margin_raw = stage2_raw.get("rating_margin", 0.05)
    try:
        rating_margin = float(rating_margin_raw)
    except (TypeError, ValueError):
        rating_margin = 0.05
    if rating_margin < 0:
        rating_margin = 0.0

    always_include_raw = stage2_raw.get("always_include_metrics")
    always_include: dict[str, tuple[str, ...]] = {}
    if isinstance(always_include_raw, Mapping):
        for lag_key, metrics_raw in always_include_raw.items():
            if not isinstance(lag_key, str):
                continue
            lag_type = lag_key.strip().lower()
            if not lag_type:
                continue
            metric_values: list[str] = []
            if isinstance(metrics_raw, Sequence) and not isinstance(metrics_raw, (str, bytes)):
                candidates = metrics_raw
            else:
                candidates = (metrics_raw,)
            for metric in candidates:
                if not isinstance(metric, str):
                    metric = str(metric) if metric is not None else ""
                metric_name = metric.strip().lower()
                if metric_name:
                    metric_values.append(metric_name)
            if metric_values:
                deduped = tuple(dict.fromkeys(metric_values))
                always_include[lag_type] = deduped

    stage2 = StageTwoConfig(
        enabled=bool(stage2_raw.get("enabled", True)),
        aggregate_per_asin=bool(stage2_raw.get("aggregate_per_asin", True)),
        max_retries=int(stage2_raw.get("max_retries", 2)),
        allowed_action_codes=tuple(stage2_raw.get("allowed_action_codes", [])),
        allowed_root_cause_codes=tuple(stage2_raw.get("allowed_root_cause_codes", [])),
        trigger_status=trigger_status,
        require_unfavorable_evidence=bool(stage2_raw.get("require_unfavorable_evidence", True)),
        keyword_max_pairs_per_opp=keyword_limit,
        min_reviews_for_rating_priority=min_reviews,
        rating_margin=rating_margin,
        always_include_metrics=always_include,
    )

    if not llm_raw.get("model"):
        raise ValueError("llm.model must be configured")

    llm = LLMRuntimeConfig(
        model=str(llm_raw["model"]),
        temperature=float(llm_raw.get("temperature", 0.1)),
        top_p=float(llm_raw.get("top_p", 0.9)),
        response_format=str(llm_raw.get("response_format", "json_object")),
        max_retries=int(llm_raw.get("max_retries", 2)),
        timeout=float(llm_raw.get("timeout", 30)),
    )

    return CompetitionLLMConfig(stage_1=stage1, stage_2=stage2, llm=llm)


__all__ = [
    "CompetitionLLMConfig",
    "LLMRuntimeConfig",
    "StageOneConfig",
    "StageTwoConfig",
    "load_competition_llm_config",
]
