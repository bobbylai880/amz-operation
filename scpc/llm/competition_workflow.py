"""Stage-1 and Stage-2 orchestration for the competition LLM workflow."""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import yaml
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from scpc.llm.orchestrator import LLMOrchestrator, LLMRunConfig, validate_schema
from scpc.prompts import load_prompt
from scpc.schemas import load_schema

from .competition_config import CompetitionLLMConfig

LOGGER = logging.getLogger(__name__)


def _ensure_stage2_context(
    machine_json: Mapping[str, Any], fallback_context: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Guarantee that the stage-2 machine JSON contains a context object."""

    context_payload = machine_json.get("context")
    if isinstance(context_payload, Mapping) and context_payload:
        # Return a shallow copy so downstream callers can freely mutate.
        return dict(machine_json)

    fallback = fallback_context if isinstance(fallback_context, Mapping) else None
    LOGGER.warning(
        "competition_llm.stage2_missing_context fallback_available=%s",
        bool(fallback),
    )
    result = dict(machine_json)
    result["context"] = dict(fallback or {})
    return result


_STAGE1_OVERVIEW_SQL_BASE = """
SELECT *
FROM vw_amz_comp_llm_overview
WHERE week = :week
"""

_STAGE1_TRAFFIC_SQL_BASE = """
SELECT *
FROM vw_amz_comp_llm_overview_traffic
WHERE week = :week
"""

_STAGE1_LATEST_WEEK_SQL_BASE = """
SELECT week, sunday
FROM vw_amz_comp_llm_overview
{marketplace_filter}
ORDER BY sunday DESC
LIMIT 1
"""

_STAGE1_KEY_FIELDS = (
    "scene_tag",
    "base_scene",
    "morphology",
    "marketplace_id",
    "week",
    "sunday",
    "my_asin",
    "opp_type",
)

_REQUIRED_CONTEXT_FIELDS = (
    "scene_tag",
    "base_scene",
    "morphology",
    "marketplace_id",
    "week",
    "sunday",
    "my_parent_asin",
    "my_asin",
    "opp_type",
)

_STAGE2_GROUP_FIELDS = (
    "scene_tag",
    "base_scene",
    "morphology",
    "marketplace_id",
    "week",
    "sunday",
    "my_parent_asin",
    "my_asin",
)

_PAGE_LAG_TYPES = {"price", "rank", "content", "social", "badge", "confidence"}
_TRAFFIC_LAG_TYPES = {"traffic_mix", "keyword"}

_LAG_TYPE_FILENAME_ALIASES: Mapping[str, str] = {
    "price": "pricing",
    "rank": "rank",
    "content": "content",
    "social": "social",
    "badge": "badge",
    "confidence": "confidence",
    "traffic_mix": "traffic_mix",
    "keyword": "keyword",
}

_ROOT_CAUSE_TO_LAG: Mapping[str, str] = {
    "pricing_misalignment": "price",
    "promo_gap": "price",
    "price_gap": "price",
    "ads_underinvestment": "traffic_mix",
    "traffic_mix_gap": "traffic_mix",
    "keyword_gap": "keyword",
    "content_quality": "content",
    "content_gap": "content",
    "assortment": "badge",
    "badge_gap": "badge",
    "rank_gap": "rank",
    "social_gap": "social",
}

_ENTITY_DETAIL_FIELDS: tuple[str, ...] = (
    "price_current",
    "price_list",
    "coupon_pct",
    "price_net",
    "rank_leaf",
    "rank_root",
    "rank_score",
    "image_cnt",
    "video_cnt",
    "bullet_cnt",
    "title_len",
    "aplus_flag",
    "content_score",
    "rating",
    "reviews",
    "social_proof",
    "badge_json",
    "brand",
)

_KEYWORD_LOOKBACK_DAYS = 7
_KEYWORD_PER_PAIR_LIMIT = 2

_RULES_CONFIG_DEFAULT_PATH = Path("configs/competition_lag_rules.yaml")
_RULES_CONFIG_SCHEMA_NAME = "competition_lag_rules.schema.json"
_DEFAULT_RULES_CONFIG = {
    "version": 1,
    "defaults": {"comparator": ">", "opp_type": "any", "weight": 1.0},
    "profiles": [
        {
            "name": "default",
            "description": "通用规则，适配大多数类目/场景",
            "is_active": True,
            "rules": [
                {
                    "lag_type": "price",
                    "metric": "price_gap_leader",
                    "opp_type": "leader",
                    "cmp": ">",
                    "threshold": 0.0,
                    "weight": 1.0,
                    "comment": "与 leader 的净价差（我-对手）> 0 表示更贵",
                },
                {
                    "lag_type": "price",
                    "metric": "price_index_med",
                    "opp_type": "median",
                    "cmp": ">",
                    "threshold": 1.05,
                    "weight": 1.0,
                    "comment": "价格指数 > 1.05 表示价格偏高",
                },
                {
                    "lag_type": "price",
                    "metric": "price_z",
                    "opp_type": "any",
                    "cmp": ">",
                    "threshold": 0.5,
                    "weight": 0.5,
                    "comment": "价格Z分偏高，辅助判定",
                },
                {
                    "lag_type": "rank",
                    "metric": "rank_pos_pct",
                    "opp_type": "any",
                    "cmp": ">",
                    "threshold": 0.60,
                    "weight": 1.0,
                    "comment": "排名百分位 > 0.60 视为排名弱",
                },
                {
                    "lag_type": "content",
                    "metric": "content_gap",
                    "opp_type": "leader",
                    "cmp": "<",
                    "threshold": -0.15,
                    "weight": 1.0,
                    "comment": "内容差 <-0.15 视为内容落后",
                },
                {
                    "lag_type": "social",
                    "metric": "social_gap",
                    "opp_type": "leader",
                    "cmp": "<",
                    "threshold": -0.10,
                    "weight": 1.0,
                    "comment": "社交口碑弱",
                },
                {
                    "lag_type": "badge",
                    "metric": "badge_delta_sum",
                    "opp_type": "any",
                    "cmp": "<",
                    "threshold": 0,
                    "weight": 1.0,
                    "comment": "权益徽章缺失",
                },
                {
                    "lag_type": "confidence",
                    "metric": "confidence",
                    "opp_type": "any",
                    "cmp": "<",
                    "threshold": 0.60,
                    "weight": 1.0,
                    "comment": "数据置信度低需要补齐证据",
                },
                {
                    "lag_type": "traffic_mix",
                    "metric": "ad_ratio_index_med",
                    "opp_type": "median",
                    "cmp": "<",
                    "threshold": 0.80,
                    "weight": 1.0,
                    "comment": "广告占比指数（我/中位）偏低",
                },
                {
                    "lag_type": "traffic_mix",
                    "metric": "ad_to_natural_gap",
                    "opp_type": "any",
                    "cmp": "<",
                    "threshold": -0.20,
                    "weight": 1.0,
                    "comment": "付费:自然差距偏低 → 结构弱于竞对",
                },
                {
                    "lag_type": "traffic_mix",
                    "metric": "sp_share_in_ad_gap",
                    "opp_type": "any",
                    "cmp": "<",
                    "threshold": -0.10,
                    "weight": 1.0,
                    "comment": "SP 在广告内部占比不足",
                },
                {
                    "lag_type": "keyword",
                    "metric": "kw_top3_share_gap",
                    "opp_type": "any",
                    "cmp": "<",
                    "threshold": -0.10,
                    "weight": 1.0,
                    "comment": "Top3 关键词份额差不足",
                },
                {
                    "lag_type": "keyword",
                    "metric": "kw_brand_share_gap",
                    "opp_type": "any",
                    "cmp": "<",
                    "threshold": -0.05,
                    "weight": 1.0,
                    "comment": "品牌词份额偏低",
                },
                {
                    "lag_type": "keyword",
                    "metric": "kw_competitor_share_gap",
                    "opp_type": "any",
                    "cmp": ">",
                    "threshold": 0.05,
                    "weight": 1.0,
                    "comment": "竞品词过高，可能吸错流量",
                },
            ],
        }
    ],
    "overrides": [
        {
            "name": "US-浴室袋-价格更敏感",
            "is_active": True,
            "when": {"marketplace_id": "US", "scene_tag": "浴室袋"},
            "rules": [
                {
                    "lag_type": "price",
                    "metric": "price_index_med",
                    "opp_type": "median",
                    "cmp": ">",
                    "threshold": 1.03,
                    "weight": 1.0,
                    "comment": "US/浴室袋价格敏感，阈值收紧",
                }
            ],
        }
    ],
}

_COMPARATOR_ALIASES = {
    "gt": ">",
    "gte": ">=",
    "ge": ">=",
    "lt": "<",
    "lte": "<=",
    "le": "<=",
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
}
_COMPARATOR_FUNCS = {
    ">": lambda value, threshold: value > threshold,
    "<": lambda value, threshold: value < threshold,
    ">=": lambda value, threshold: value >= threshold,
    "<=": lambda value, threshold: value <= threshold,
}
_MIRROR_MAP = {
    ">": "<",
    ">=": "<=",
    "<": ">",
    "<=": ">=",
}
_SEVERITY_ORDER = {"low": 1, "mid": 2, "medium": 2, "high": 3}


@dataclass(slots=True)
class LagRule:
    rule_name: str
    lag_type: str
    metric: str
    opp_type: str
    comparator: str
    threshold: float
    weight: float


@dataclass(slots=True)
class StageOneResult:
    context: Mapping[str, Any]
    summary: str
    dimensions: Sequence[Mapping[str, Any]]


@dataclass(slots=True)
class StageOneLLMResult(StageOneResult):
    """Structured Stage-1 output produced by the LLM round."""

    llm_metrics: Mapping[str, Any] | None = None
    prompt_path: Path | None = None

    def to_stage_one_result(self) -> "StageOneResult":
        """Drop any LLM-specific metadata and return the base result payload."""

        return StageOneResult(
            context=self.context,
            summary=self.summary,
            dimensions=self.dimensions,
        )


@dataclass(slots=True)
class StageTwoCandidate:
    context: Mapping[str, Any]
    dimension: Mapping[str, Any]


@dataclass(slots=True)
class StageTwoAggregateInput:
    context: Mapping[str, Any]
    items: Sequence[StageTwoCandidate]


@dataclass(slots=True)
class StageTwoAggregateResult:
    context: Mapping[str, Any]
    machine_json: Mapping[str, Any]
    human_markdown: str
    facts: Mapping[str, Any]
    storage_base_name: str
    prompt_path: Path | None = None


@dataclass(slots=True)
class CompetitionRunResult:
    week: str
    stage1_processed: int
    stage2_candidates: int
    stage2_processed: int
    storage_paths: Sequence[Path]


def _normalize_lag_type(value: object) -> str:
    mapping = {
        "pricing": "price",
        "price": "price",
        "ranking": "rank",
        "rank": "rank",
        "content": "content",
        "social": "social",
        "badge": "badge",
        "traffic": "traffic_mix",
        "traffic_mix": "traffic_mix",
        "keyword": "keyword",
        "confidence": "confidence",
    }
    raw = str(value or "").strip().lower()
    return mapping.get(raw, raw)


class CompetitionLLMOrchestrator:
    """Coordinate the Stage-1/Stage-2 competition workflow using a rule engine plus a single LLM round."""

    def __init__(
        self,
        *,
        engine: Engine,
        llm_orchestrator: LLMOrchestrator,
        config: CompetitionLLMConfig,
        storage_root: str | Path | None = None,
    ) -> None:
        self._engine = engine
        self._llm = llm_orchestrator
        self._config = config
        self._storage_root = Path(storage_root or "storage/competition_llm")
        self._stage1_prompt = load_prompt("competition.md")
        self._stage1_schema: Mapping[str, Any] = {
            "type": "object",
            "required": ["context", "summary", "dimensions"],
            "properties": {
                "context": {"type": "object"},
                "summary": {"type": "string"},
                "dimensions": {"type": "array", "items": {"type": "object"}},
                "metrics": {"type": ["object", "null"]},
            },
        }
        self._stage2_schema = load_schema("competition_stage2_aggregate.schema.json")
        self._stage2_prompt = load_prompt("competition_stage2_aggregate.md")
        self._current_marketplace_id: str | None = None
        self._brand_cache: dict[str, str | None] = {}
        self._brand_path_tokens: list[str] | None = None

    def run(
        self,
        week: str | None,
        *,
        marketplace_id: str | None = None,
        stages: Sequence[str] | None = None,
    ) -> CompetitionRunResult:
        """Execute Stage-1 and Stage-2 for the provided week."""

        self._current_marketplace_id = marketplace_id
        target_week = self._resolve_week(week, marketplace_id)
        raw_stage_request = {str(stage).lower() for stage in (stages or ("stage1", "stage2")) if stage}
        if not raw_stage_request:
            raw_stage_request = {"stage1", "stage2"}
        requested_stages = set(raw_stage_request)
        if "stage2" in requested_stages:
            requested_stages.add("stage1")
        stage2_requested = stages is None or "stage2" in raw_stage_request
        run_stage1 = "stage1" in requested_stages
        run_stage2 = "stage2" in requested_stages and self._config.stage_2.enabled

        LOGGER.info(
            "competition_llm.start week=%s marketplace_id=%s stages=%s",
            target_week,
            marketplace_id,
            sorted(requested_stages),
        )

        storage_paths: list[Path] = []
        stage1_outputs: Sequence[StageOneResult] = ()
        stage1_llm_requested = stages is None or "stage1" in raw_stage_request
        stage1_llm_enabled = bool(self._config.stage_1.enable_llm)
        use_stage1_llm = run_stage1 and stage1_llm_requested and stage1_llm_enabled
        if run_stage1:
            stage1_inputs = self._collect_stage1_inputs(target_week, marketplace_id)
            rule_results = self._execute_stage1_code(stage1_inputs)
            stage1_payloads: list[StageOneResult] = []
            if not use_stage1_llm:
                reason = "config_disabled"
                if stage1_llm_enabled:
                    reason = "stage2_only"
                LOGGER.info("competition_llm.stage1_llm_skipped reason=%s", reason)
            for index, rule_result in enumerate(rule_results):
                overview_rows: Sequence[Mapping[str, Any]] = ()
                traffic_rows: Sequence[Mapping[str, Any]] = ()
                if index < len(stage1_inputs):
                    _, overview_rows, traffic_rows = stage1_inputs[index]
                stage1_result: StageOneResult
                if use_stage1_llm:
                    stage1_result = self._apply_stage1_llm(rule_result, overview_rows, traffic_rows)
                else:
                    stage1_result = rule_result
                if isinstance(stage1_result, StageOneLLMResult) and stage1_result.prompt_path:
                    storage_paths.append(stage1_result.prompt_path)
                storage_paths.append(self._write_stage1_output(stage1_result))
                stage1_payloads.append(stage1_result)
            stage1_outputs = tuple(stage1_payloads)
        else:
            LOGGER.info("competition_llm.stage1_skipped")

        stage2_candidates: Sequence[StageTwoCandidate] = ()
        if run_stage2:
            stage2_candidates = self._prepare_stage2_candidates(stage1_outputs)

        stage2_outputs: Sequence[StageTwoAggregateResult] = ()
        if run_stage2 and stage2_candidates:
            if getattr(self._config.stage_2, "aggregate_per_asin", False):
                grouped = self._group_candidates_by_asin(stage2_candidates)
                stage2_outputs = self._execute_stage2_aggregate(grouped)
                for item in stage2_outputs:
                    if item.prompt_path:
                        storage_paths.append(item.prompt_path)
                    storage_paths.extend(self._write_stage2_output_aggregate(item))
            else:
                LOGGER.warning(
                    "competition_llm.stage2_aggregate_disabled candidate_count=%s",
                    len(stage2_candidates),
                )
        elif run_stage2:
            LOGGER.info(
                "competition_llm.stage2_skipped enabled=%s candidate_count=%s",
                self._config.stage_2.enabled,
                len(stage2_candidates),
            )
        elif stage2_requested:
            LOGGER.info(
                "competition_llm.stage2_skipped enabled=%s candidate_count=%s",
                self._config.stage_2.enabled,
                len(stage2_candidates),
            )
        else:
            LOGGER.info("competition_llm.stage2_not_requested")

        LOGGER.info(
            "competition_llm.end week=%s stage1=%s stage2_candidates=%s stage2=%s",
            target_week,
            len(stage1_outputs),
            len(stage2_candidates),
            len(stage2_outputs),
        )
        self._current_marketplace_id = None
        return CompetitionRunResult(
            week=target_week,
            stage1_processed=len(stage1_outputs),
            stage2_candidates=len(stage2_candidates),
            stage2_processed=len(stage2_outputs),
            storage_paths=tuple(storage_paths),
        )

    def _collect_stage1_inputs(
        self, week: str, marketplace_id: str | None
    ) -> Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]]:
        overview_rows = self._query_stage1_table(_STAGE1_OVERVIEW_SQL_BASE, week, marketplace_id)
        traffic_rows = self._query_stage1_table(_STAGE1_TRAFFIC_SQL_BASE, week, marketplace_id)

        grouped: dict[tuple[Any, ...], MutableMapping[str, Any]] = {}
        for row in overview_rows:
            key = self._build_stage1_key(row)
            bucket = grouped.setdefault(key, {"overview_rows": [], "traffic_rows": [], "context": None})
            bucket["overview_rows"].append(row)
            if bucket["context"] is None:
                bucket["context"] = self._build_context(row)
        for row in traffic_rows:
            key = self._build_stage1_key(row)
            bucket = grouped.setdefault(key, {"overview_rows": [], "traffic_rows": [], "context": None})
            bucket["traffic_rows"].append(row)
            if bucket["context"] is None:
                bucket["context"] = self._build_context(row)

        inputs: list[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]] = []
        for bucket in grouped.values():
            context = bucket.get("context")
            if not context:
                LOGGER.debug("competition_llm.skip_missing_context")
                continue
            overview = tuple(bucket.get("overview_rows", ()))
            traffic = tuple(bucket.get("traffic_rows", ()))
            if not overview and not traffic:
                LOGGER.debug("competition_llm.skip_empty_rows context=%s", context)
                continue
            inputs.append((context, overview, traffic))
        return tuple(inputs)

    def _execute_stage1_code(
        self,
        inputs: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]],
    ) -> Sequence[StageOneResult]:
        rules_config = self._load_rules_config()
        results: list[StageOneResult] = []
        has_applicable_rules = False
        for context, overview_rows, traffic_rows in inputs:
            self._validate_stage1_context(context)
            rules = self._load_active_rules(context, config_data=rules_config)
            opp_type = str(context.get("opp_type") or "").lower()
            applicable_rules = [rule for rule in rules if rule.opp_type in ("any", opp_type)]
            if not applicable_rules:
                LOGGER.debug("competition_llm.stage1_skip_no_rules opp_type=%s context=%s", opp_type, context)
                summary = "未检测到落后维度"
                results.append(StageOneResult(context=context, summary=summary, dimensions=()))
                continue
            has_applicable_rules = True

            lag_scores: defaultdict[str, float] = defaultdict(float)
            triggered_rules: defaultdict[str, list[str]] = defaultdict(list)
            lead_hits: defaultdict[str, list[str]] = defaultdict(list)
            confidence_tracker: defaultdict[str, float] = defaultdict(lambda: self._config.stage_1.conf_min)

            for channel, rows in (("page", overview_rows), ("traffic", traffic_rows)):
                if not rows:
                    continue
                confidence_field = "confidence" if channel == "page" else "t_confidence"
                for row in rows:
                    confidence_value = self._parse_confidence(row.get(confidence_field))
                    if confidence_value is None:
                        confidence_value = self._config.stage_1.conf_min
                    for rule in applicable_rules:
                        value = row.get(rule.metric)
                        numeric_value = _coerce_float(value)
                        if numeric_value is None:
                            continue
                        if _compare(numeric_value, rule.threshold, rule.comparator):
                            lag_scores[rule.lag_type] += rule.weight
                            confidence_tracker[rule.lag_type] = max(confidence_tracker[rule.lag_type], confidence_value)
                            if rule.rule_name:
                                triggered_rules[rule.lag_type].append(rule.rule_name)
                            else:
                                triggered_rules[rule.lag_type].append(f"{rule.metric}{rule.comparator}{rule.threshold}")
                        else:
                            mirror = _MIRROR_MAP.get(rule.comparator)
                            if mirror and _compare(numeric_value, rule.threshold, mirror):
                                lead_hits[rule.lag_type].append(rule.rule_name or rule.metric)
                            confidence_tracker[rule.lag_type] = max(confidence_tracker[rule.lag_type], confidence_value)

            dimensions: list[Mapping[str, Any]] = []
            summary_parts: list[str] = []
            for lag_type, score in sorted(lag_scores.items()):
                if score <= 0:
                    continue
                severity = self._map_severity(score)
                confidence_value = max(confidence_tracker.get(lag_type, 0.0), self._config.stage_1.conf_min)
                payload = {
                    "lag_type": lag_type,
                    "status": "lag",
                    "severity": severity,
                    "lag_score": round(score, 4),
                    "source_opp_type": context.get("opp_type"),
                    "source_confidence": round(confidence_value, 4),
                }
                if triggered_rules.get(lag_type):
                    payload["triggered_rules"] = triggered_rules[lag_type]
                dimensions.append(payload)
                LOGGER.info(
                    "competition_llm.stage1_lag lag_type=%s opp_type=%s lag_score=%.3f severity=%s rules=%s",
                    lag_type,
                    context.get("opp_type"),
                    score,
                    severity,
                    triggered_rules.get(lag_type, ()),
                )
                if triggered_rules.get(lag_type):
                    summary_parts.append(f"{lag_type}: {', '.join(triggered_rules[lag_type])}")
                else:
                    summary_parts.append(f"{lag_type}: lag_score={score:.2f}")

            if not dimensions:
                if lead_hits:
                    LOGGER.debug("competition_llm.stage1_lead_only opp_type=%s leads=%s", opp_type, dict(lead_hits))
                summary = "未检测到落后维度"
            else:
                summary = "；".join(summary_parts) if summary_parts else "已识别落后维度"

            results.append(
                StageOneResult(
                    context=context,
                    summary=summary,
                    dimensions=tuple(dimensions),
                )
            )
        if not has_applicable_rules:
            LOGGER.warning("competition_llm.stage1_no_rules active=0")
        return tuple(results)

    def _validate_stage1_context(self, context: Mapping[str, Any]) -> None:
        missing = [field for field in _REQUIRED_CONTEXT_FIELDS if context.get(field) is None]
        if missing:
            asin = context.get("my_asin") or "unknown"
            missing_fields = ",".join(missing)
            raise ValueError(
                f"Stage-1 context missing required fields [{missing_fields}] for ASIN {asin}"
            )

    def _apply_stage1_llm(
        self,
        rule_result: StageOneResult,
        overview_rows: Sequence[Mapping[str, Any]],
        traffic_rows: Sequence[Mapping[str, Any]],
    ) -> StageOneLLMResult:
        context = dict(rule_result.context)
        packets = self._load_stage1_packets(context)
        insights = self._load_stage1_insights(context)
        facts = self._build_stage1_llm_facts(
            context,
            overview_rows,
            traffic_rows,
            rule_result,
            packets,
            insights,
        )
        prompt_path: Path | None = None
        try:
            prompt_path = self._write_prompt_snapshot(
                stage="stage1",
                week=str(context.get("week") or "unknown"),
                base_name=self._build_stage1_storage_name(context),
                prompt=self._stage1_prompt,
                facts=facts,
            )
        except Exception as exc:  # pragma: no cover - snapshot errors should not block execution
            LOGGER.warning(
                "competition_llm.stage1_prompt_snapshot_failed asin=%s error=%s",
                context.get("my_asin"),
                exc,
            )
            prompt_path = None

        try:
            response = self._invoke_with_retries(
                facts,
                schema=self._stage1_schema,
                prompt=self._stage1_prompt,
                max_attempts=self._config.stage_1.max_retries,
            )
        except Exception as exc:
            LOGGER.warning(
                "competition_llm.stage1_llm_failed asin=%s error=%s",
                context.get("my_asin"),
                exc,
            )
            return StageOneLLMResult(
                context=context,
                summary=rule_result.summary,
                dimensions=rule_result.dimensions,
                llm_metrics=None,
                prompt_path=prompt_path,
            )

        try:
            validate_schema(self._stage1_schema, response)
        except Exception as exc:
            LOGGER.warning(
                "competition_llm.stage1_llm_invalid asin=%s error=%s",
                context.get("my_asin"),
                exc,
            )
            return StageOneLLMResult(
                context=context,
                summary=rule_result.summary,
                dimensions=rule_result.dimensions,
                llm_metrics=None,
                prompt_path=prompt_path,
            )

        llm_context_raw = response.get("context")
        if isinstance(llm_context_raw, Mapping):
            llm_context = dict(rule_result.context)
            llm_context.update({key: llm_context_raw.get(key) for key in llm_context_raw})
        else:
            llm_context = dict(rule_result.context)

        summary = response.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = rule_result.summary

        rule_dim_index: dict[str, Mapping[str, Any]] = {}
        for dim in rule_result.dimensions:
            if isinstance(dim, Mapping):
                key = _normalize_lag_type(dim.get("lag_type"))
                if key:
                    rule_dim_index[key] = dim

        dimensions_payload: list[Mapping[str, Any]] = []
        response_dimensions = response.get("dimensions")
        if isinstance(response_dimensions, Sequence):
            for dim in response_dimensions:
                if not isinstance(dim, Mapping):
                    continue
                dim_payload = dict(dim)
                original_lag_label = str(dim_payload.get("lag_type") or "").strip()
                lag_type = _normalize_lag_type(original_lag_label) or original_lag_label.lower()
                if lag_type:
                    dim_payload["lag_type_normalized"] = lag_type
                rule_dim = rule_dim_index.get(lag_type)
                if rule_dim:
                    for field in ("lag_score", "triggered_rules", "source_confidence", "source_opp_type"):
                        value = dim_payload.get(field)
                        if value in (None, "", ()):  # pragma: no branch - normalising falsy
                            rule_value = rule_dim.get(field)
                            if rule_value not in (None, "", ()):  # pragma: no branch - ensure signal present
                                dim_payload[field] = rule_value
                if "status" not in dim_payload or not dim_payload.get("status"):
                    dim_payload["status"] = "lag"
                dimensions_payload.append(dim_payload)

        if not dimensions_payload:
            dimensions_payload = [dict(dim) for dim in rule_result.dimensions]

        metrics_payload = response.get("metrics")
        if not isinstance(metrics_payload, Mapping):
            metrics_payload = None

        return StageOneLLMResult(
            context=llm_context,
            summary=summary,
            dimensions=tuple(dimensions_payload),
            llm_metrics=metrics_payload,
            prompt_path=prompt_path,
        )

    def _build_stage1_llm_facts(
        self,
        context: Mapping[str, Any],
        overview_rows: Sequence[Mapping[str, Any]],
        traffic_rows: Sequence[Mapping[str, Any]],
        rule_result: StageOneResult,
        packets: Sequence[Mapping[str, Any]],
        insights: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return {
            "context": context,
            "rule_summary": rule_result.summary,
            "rule_dimensions": [dict(dimension) for dimension in rule_result.dimensions],
            "overview_rows": [dict(row) for row in overview_rows],
            "traffic_rows": [dict(row) for row in traffic_rows],
            "lag_packets": list(packets),
            "lag_insights": list(insights),
            "output_language": "zh",
        }

    def _load_stage1_packets(self, context: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        sql = """
          SELECT lag_type, opp_type, evidence_json
          FROM bi_amz_comp_llm_packet
          WHERE scene_tag=:scene_tag AND base_scene=:base_scene
            AND COALESCE(morphology,'') = COALESCE(:morphology,'')
            AND marketplace_id=:marketplace_id
            AND week=:week AND sunday=:sunday
            AND my_asin=:my_asin
        """
        params = {
            "scene_tag": context.get("scene_tag"),
            "base_scene": context.get("base_scene"),
            "morphology": context.get("morphology"),
            "marketplace_id": context.get("marketplace_id"),
            "week": context.get("week"),
            "sunday": context.get("sunday"),
            "my_asin": context.get("my_asin"),
        }
        try:
            rows = self._fetch_all(sql, params)
        except SQLAlchemyError as exc:  # pragma: no cover - depends on optional table
            LOGGER.debug(
                "competition_llm.stage1_packets_unavailable asin=%s error=%s",
                context.get("my_asin"),
                exc,
            )
            return ()

        packets: list[Mapping[str, Any]] = []
        for row in rows:
            payload_raw = row.get("evidence_json")
            evidence_payload: Mapping[str, Any] | None = None
            if isinstance(payload_raw, str) and payload_raw.strip():
                try:
                    parsed = json.loads(payload_raw)
                except json.JSONDecodeError as exc:  # pragma: no cover - malformed fixtures
                    LOGGER.warning(
                        "competition_llm.stage1_packet_parse_error asin=%s error=%s",
                        context.get("my_asin"),
                        exc,
                    )
                else:
                    if isinstance(parsed, Mapping):
                        evidence_payload = parsed
            packets.append(
                {
                    "lag_type": _normalize_lag_type(row.get("lag_type")),
                    "opp_type": row.get("opp_type"),
                    "evidence": evidence_payload,
                }
            )
        return tuple(packets)

    def _load_stage1_insights(self, context: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        sql = """
          SELECT lag_type, opp_type, reason_code, severity, reason_detail, top_opp_asins_csv
          FROM bi_amz_comp_lag_insights
          WHERE scene_tag=:scene_tag AND base_scene=:base_scene
            AND COALESCE(morphology,'') = COALESCE(:morphology,'')
            AND marketplace_id=:marketplace_id
            AND week=:week AND sunday=:sunday
            AND my_asin=:my_asin
        """
        params = {
            "scene_tag": context.get("scene_tag"),
            "base_scene": context.get("base_scene"),
            "morphology": context.get("morphology"),
            "marketplace_id": context.get("marketplace_id"),
            "week": context.get("week"),
            "sunday": context.get("sunday"),
            "my_asin": context.get("my_asin"),
        }
        try:
            rows = self._fetch_all(sql, params)
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage1_insights_unavailable asin=%s error=%s",
                context.get("my_asin"),
                exc,
            )
            return ()

        insights: list[Mapping[str, Any]] = []
        for row in rows:
            insights.append(
                {
                    "lag_type": _normalize_lag_type(row.get("lag_type")),
                    "opp_type": row.get("opp_type"),
                    "reason_code": row.get("reason_code"),
                    "severity": row.get("severity"),
                    "reason_detail": row.get("reason_detail"),
                    "top_opp_asins_csv": row.get("top_opp_asins_csv"),
                }
            )
        return tuple(insights)

    def _write_stage1_output(self, result: StageOneResult) -> Path:
        week = str(result.context.get("week"))
        asin = result.context.get("my_asin", "unknown")
        opp_type = result.context.get("opp_type", "na")
        path = self._storage_root / week / "stage1" / f"{asin}_{opp_type}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "context": result.context,
            "summary": result.summary,
            "dimensions": result.dimensions,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        return path

    def _write_prompt_snapshot(
        self,
        *,
        stage: str,
        week: str,
        base_name: str,
        prompt: str,
        facts: Mapping[str, Any],
    ) -> Path:
        prompt_dir = self._storage_root / week / stage / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        path = prompt_dir / f"{base_name}.prompt.json"
        payload = {"prompt": prompt, "facts": facts}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        return path

    def _build_stage1_storage_name(self, context: Mapping[str, Any]) -> str:
        asin = str(context.get("my_asin") or "unknown")
        opp_type = str(context.get("opp_type") or "na")
        return f"{asin}_{opp_type}"

    def _prepare_stage2_candidates(self, stage1_results: Sequence[StageOneResult]) -> Sequence[StageTwoCandidate]:
        candidates: list[StageTwoCandidate] = []
        threshold = self._config.stage_1.conf_min
        allowed_statuses = tuple(
            status.lower() for status in getattr(self._config.stage_2, "trigger_status", ("lag",))
        ) or ("lag",)
        allowed_set = set(allowed_statuses)
        for result in stage1_results:
            for dimension in result.dimensions:
                status = str(dimension.get("status", "")).lower()
                if allowed_set and status not in allowed_set:
                    continue
                confidence = dimension.get("source_confidence")
                confidence_value = self._parse_confidence(confidence)
                if confidence_value is None:
                    confidence_value = threshold
                if confidence_value < threshold:
                    LOGGER.debug(
                        "competition_llm.stage2_skip_confidence threshold=%s confidence=%s context=%s dimension=%s",
                        threshold,
                        confidence,
                        result.context,
                        dimension,
                    )
                    continue
                candidates.append(StageTwoCandidate(context=result.context, dimension=dict(dimension)))
        return tuple(candidates)

    def _group_candidates_by_asin(self, candidates: Sequence[StageTwoCandidate]) -> Sequence[StageTwoAggregateInput]:
        grouped: dict[tuple[Any, ...], list[StageTwoCandidate]] = {}
        contexts: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for candidate in candidates:
            context = candidate.context
            base_context = {field: context.get(field) for field in _STAGE2_GROUP_FIELDS}
            base_context["asin_priority"] = context.get("asin_priority")
            key = tuple(base_context.get(field) for field in _STAGE2_GROUP_FIELDS)
            grouped.setdefault(key, []).append(candidate)
            contexts.setdefault(key, base_context)
        return tuple(
            StageTwoAggregateInput(context=contexts[key], items=tuple(items))
            for key, items in grouped.items()
        )

    def _execute_stage2_aggregate(
        self, groups: Sequence[StageTwoAggregateInput]
    ) -> Sequence[StageTwoAggregateResult]:
        outputs: list[StageTwoAggregateResult] = []
        for group in groups:
            facts = self._build_stage2_aggregate_facts(group)
            lag_items = tuple(facts.get("lag_items") or ())
            if not lag_items:
                LOGGER.info("competition_llm.stage2_aggregate_skip_empty context=%s", group.context)
                continue
            LOGGER.info(
                "competition_llm.stage2_aggregate_input asin=%s lag_items=%s top_opps=%s",
                facts.get("context", {}).get("my_asin"),
                len(lag_items),
                len(facts.get("top_opp_asins_csv", "").split(",")) if facts.get("top_opp_asins_csv") else 0,
            )
            context = facts.get("context") or {}
            week_label = str(context.get("week") or "unknown")
            base_name = self._build_stage2_storage_name(context)
            prompt_path = self._write_prompt_snapshot(
                stage="stage2",
                week=week_label,
                base_name=base_name,
                prompt=self._stage2_prompt,
                facts=facts,
            )
            llm_response = self._invoke_with_retries(
                facts,
                schema={"type": "object", "required": ["machine_json", "human_markdown"]},
                prompt=self._stage2_prompt,
                max_attempts=self._config.stage_2.max_retries,
            )
            machine_json = llm_response.get("machine_json")
            human_markdown = llm_response.get("human_markdown")
            if not isinstance(machine_json, Mapping):
                raise ValueError("Stage-2 machine_json missing or invalid")
            machine_json = dict(machine_json)
            if not isinstance(machine_json.get("context"), Mapping):
                machine_json["context"] = facts.get("context", {})
            machine_json = self._materialize_evidence(machine_json, facts)
            self._validate_stage2_machine_json(machine_json)
            if not isinstance(human_markdown, str):
                raise ValueError("Stage-2 human_markdown must be a string")
            outputs.append(
                StageTwoAggregateResult(
                    context=facts.get("context", {}),
                    machine_json=machine_json,
                    human_markdown=human_markdown,
                    facts=facts,
                    storage_base_name=base_name,
                    prompt_path=prompt_path,
                )
            )
        return tuple(outputs)

    def _build_stage2_storage_name(self, context: Mapping[str, Any]) -> str:
        asin = str(context.get("my_asin") or "unknown")
        return f"{asin}_ALL"

    def _build_stage2_aggregate_facts(self, group: StageTwoAggregateInput) -> Mapping[str, Any]:
        base_context = dict(group.context)
        lag_entries: dict[str, dict[str, Any]] = {}
        overall_top_map: dict[str, dict[str, Any]] = {}
        opp_types_seen: set[str] = set()
        evidence_cache: dict[tuple[Any, ...], Mapping[str, Any] | None] = {}
        first_round_item: Mapping[str, Any] | None = None

        for candidate in group.items:
            dimension = candidate.dimension
            lag_type = dimension.get("lag_type_normalized") or _normalize_lag_type(dimension.get("lag_type"))
            opp_type = str(candidate.context.get("opp_type") or dimension.get("source_opp_type") or "").lower()
            if not lag_type or not opp_type:
                LOGGER.debug("competition_llm.stage2_skip_invalid_dimension dimension=%s", dimension)
                continue
            display_label_raw = str(dimension.get("lag_type") or "").strip()
            if display_label_raw and display_label_raw.lower() != lag_type:
                display_label = display_label_raw
            else:
                display_label = _LAG_TYPE_FILENAME_ALIASES.get(lag_type, lag_type)
            severity_label = str(dimension.get("severity") or "low").lower()
            severity_rank = _SEVERITY_ORDER.get(severity_label, 1)
            opp_types_seen.add(opp_type)
            confidence_value = self._parse_confidence(dimension.get("source_confidence")) or self._config.stage_1.conf_min
            lag_score = _coerce_float(dimension.get("lag_score"))

            if first_round_item is None:
                first_round_item = {
                    "context": dict(candidate.context),
                    "lag_type": display_label,
                    "dimension": dict(dimension),
                }

            evidence_items: list[Mapping[str, Any]] = []
            if lag_type in _PAGE_LAG_TYPES:
                cache_key = _stage2_cache_key(candidate.context, lag_type, opp_type, "page")
                evidence_page = evidence_cache.get(cache_key)
                if evidence_page is None:
                    evidence_page = self._build_page_evidence(candidate.context, lag_type, opp_type)
                    evidence_cache[cache_key] = evidence_page
                if evidence_page:
                    evidence_items.append(evidence_page)
            if lag_type in _TRAFFIC_LAG_TYPES:
                cache_key = _stage2_cache_key(candidate.context, lag_type, opp_type, "traffic")
                evidence_traffic = evidence_cache.get(cache_key)
                if evidence_traffic is None:
                    evidence_traffic = self._build_traffic_evidence(candidate.context, lag_type, opp_type)
                    evidence_cache[cache_key] = evidence_traffic
                if evidence_traffic:
                    evidence_items.append(evidence_traffic)

            if not evidence_items:
                packet_evidence = self._load_cached_packet_evidence(candidate.context, lag_type, opp_type)
                if packet_evidence:
                    evidence_items.append(packet_evidence)

            if not evidence_items:
                LOGGER.warning(
                    "competition_llm.stage2_missing_evidence lag_type=%s opp_type=%s context=%s",
                    lag_type,
                    opp_type,
                    candidate.context,
                )
                continue

            entry = lag_entries.setdefault(
                lag_type,
                {
                    "lag_type": lag_type,
                    "label": display_label,
                    "opp_types": set(),
                    "severity_rank": 0,
                    "source_confidence": self._config.stage_1.conf_min,
                    "overview": {},
                    "_top_map": {},
                    "lag_scores": [],
                    "triggered_rules": set(),
                },
            )
            if display_label:
                entry.setdefault("label", display_label)
            entry["opp_types"].add(opp_type)
            entry["severity_rank"] = max(entry["severity_rank"], severity_rank)
            entry["source_confidence"] = max(entry["source_confidence"], confidence_value)
            if lag_score is not None:
                entry["lag_scores"].append(lag_score)
            if dimension.get("triggered_rules"):
                entry["triggered_rules"].update(dimension.get("triggered_rules"))

            for evidence in evidence_items:
                overview = evidence.get("overview") or {}
                if overview:
                    entry.setdefault("overview", {}).setdefault(opp_type, {}).update(overview)
                self._merge_top_opps(entry.setdefault("_top_map", {}), lag_type, evidence.get("top_opps"))
                self._merge_top_opps(overall_top_map, lag_type, evidence.get("top_opps"))

        lag_items_payload: list[dict[str, Any]] = []
        for lag_type, entry in lag_entries.items():
            overview = entry.get("overview") or {}
            top_map = entry.get("_top_map") or {}
            lag_item = {
                "lag_type": entry.get("label") or _LAG_TYPE_FILENAME_ALIASES.get(lag_type, lag_type),
                "opp_types": sorted(entry.get("opp_types") or ()),
                "severity": _severity_from_rank(entry.get("severity_rank", 0)),
                "source_confidence": round(entry.get("source_confidence", self._config.stage_1.conf_min), 4),
                "overview": {key: overview[key] for key in sorted(overview)},
                "top_opps": self._finalise_top_opps(top_map),
            }
            if entry.get("lag_scores"):
                lag_item["lag_score"] = max(entry["lag_scores"])
            if entry.get("triggered_rules"):
                lag_item["triggered_rules"] = sorted(entry["triggered_rules"])
            lag_items_payload.append(lag_item)

        lag_items_payload.sort(key=lambda item: (-_SEVERITY_ORDER.get(str(item.get("severity")).lower(), 1), item.get("lag_type")))
        top_list = self._finalise_top_opps(overall_top_map)
        top_csv = ""
        if top_list:
            top_csv = ",".join([row.get("opp_asin") for row in top_list[:8] if row.get("opp_asin")])

        base_context["opp_types"] = sorted(opp_types_seen)
        facts = {
            "context": base_context,
            "lag_items": lag_items_payload,
            "top_opp_asins_csv": top_csv,
            "allowed_action_codes": list(self._config.stage_2.allowed_action_codes),
            "allowed_root_cause_codes": list(self._config.stage_2.allowed_root_cause_codes),
            "output_language": "zh",
            "machine_json_schema": "competition_stage2_aggregate.schema.json",
        }
        if first_round_item:
            facts["first_round_item"] = first_round_item
        return facts

    def _load_cached_packet_evidence(
        self,
        context: Mapping[str, Any],
        lag_type: str,
        opp_type: str,
    ) -> Mapping[str, Any] | None:
        sql = """
          SELECT lag_type, opp_type, evidence_json
          FROM bi_amz_comp_llm_packet
          WHERE scene_tag=:scene_tag AND base_scene=:base_scene
            AND COALESCE(morphology,'') = COALESCE(:morphology,'')
            AND marketplace_id=:marketplace_id
            AND week=:week AND sunday=:sunday
            AND my_asin=:my_asin
        """
        params = {
            "scene_tag": context.get("scene_tag"),
            "base_scene": context.get("base_scene"),
            "morphology": context.get("morphology"),
            "marketplace_id": context.get("marketplace_id"),
            "week": context.get("week"),
            "sunday": context.get("sunday"),
            "my_asin": context.get("my_asin"),
        }
        try:
            rows = self._fetch_all(sql, params)
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage2_packet_lookup_failed asin=%s lag_type=%s error=%s",
                context.get("my_asin"),
                lag_type,
                exc,
            )
            return None
        if not rows:
            return None
        selected_row: Mapping[str, Any] | None = None
        normalized_target = _normalize_lag_type(lag_type)
        opp_type_lower = str(opp_type or "").lower()
        for row in rows:
            row_lag = _normalize_lag_type(row.get("lag_type"))
            row_opp = str(row.get("opp_type") or "").lower()
            if row_lag == normalized_target and (not opp_type_lower or row_opp == opp_type_lower):
                selected_row = row
                break
        if selected_row is None:
            selected_row = rows[0]

        payload_raw = selected_row.get("evidence_json")
        if not isinstance(payload_raw, str) or not payload_raw.strip():
            return None
        try:
            parsed = json.loads(payload_raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - malformed fixtures
            LOGGER.warning(
                "competition_llm.stage2_packet_parse_error asin=%s lag_type=%s error=%s",
                context.get("my_asin"),
                lag_type,
                exc,
            )
            return None
        if not isinstance(parsed, Mapping):
            return None

        overview = {}
        metrics = parsed.get("metrics")
        if isinstance(metrics, Mapping):
            overview = dict(metrics)

        top_opps: list[Mapping[str, Any]] = []
        competitors = parsed.get("top_competitors")
        if isinstance(competitors, Sequence):
            for item in competitors:
                if not isinstance(item, Mapping):
                    continue
                entry: dict[str, Any] = {
                    "opp_asin": item.get("asin"),
                    "opp_parent_asin": item.get("parent_asin") or item.get("parent"),
                    "opp_brand": item.get("brand"),
                }
                value = item.get("value")
                if value is not None:
                    coerced = _coerce_float(value)
                    entry["score"] = coerced if coerced is not None else value
                top_opps.append(entry)

        reasons: list[Mapping[str, Any]] = []
        drivers = parsed.get("drivers")
        if isinstance(drivers, Sequence):
            for driver in drivers:
                if not isinstance(driver, Mapping):
                    continue
                reason_entry: dict[str, Any] = {}
                metric_name = driver.get("name") or driver.get("metric")
                if metric_name:
                    reason_entry["metric"] = metric_name
                if driver.get("value") is not None:
                    reason_entry["value"] = driver.get("value")
                if driver.get("description"):
                    reason_entry["description"] = driver.get("description")
                if reason_entry:
                    reasons.append(reason_entry)

        top_csv = ",".join([row.get("opp_asin") for row in top_opps if row.get("opp_asin")])

        return {
            "channel": "cached",
            "lag_type": normalized_target,
            "opp_type": opp_type,
            "overview": overview,
            "top_opps": top_opps,
            "top_opp_asins_csv": top_csv,
            "top_diff_reasons": reasons,
            "prompt_hint": parsed.get("prompt_hint"),
        }

    def _write_stage2_output_aggregate(self, result: StageTwoAggregateResult) -> Sequence[Path]:
        context = result.context or {}
        week = str(context.get("week"))
        asin = context.get("my_asin", "unknown")
        stage2_dir = self._storage_root / week / "stage2"
        stage2_dir.mkdir(parents=True, exist_ok=True)

        base_name = result.storage_base_name or f"{asin}_ALL"
        main_path = stage2_dir / f"{base_name}.json"
        payload = {
            "machine_json": result.machine_json,
            "human_markdown": result.human_markdown,
        }
        main_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

        summary_paths: list[Path] = [main_path]
        summary_data = self._build_stage2_summary_aggregate(result)
        if summary_data:
            summary_json_path = stage2_dir / f"{base_name}_summary.json"
            summary_json_path.write_text(
                json.dumps(summary_data, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            summary_md_path = stage2_dir / f"{base_name}_summary.md"
            summary_md_path.write_text(
                self._render_stage2_summary_markdown(summary_data),
                encoding="utf-8",
            )
            summary_paths.extend([summary_json_path, summary_md_path])
        return tuple(summary_paths)

    def _build_stage2_summary_aggregate(self, result: StageTwoAggregateResult) -> Mapping[str, Any] | None:
        facts = result.facts or {}
        lag_items = facts.get("lag_items") or []
        if not lag_items:
            return None
        summary = {
            "context": facts.get("context"),
            "lag_items": lag_items,
            "top_opp_asins_csv": facts.get("top_opp_asins_csv"),
        }
        return summary

    def _render_stage2_summary_markdown(self, summary: Mapping[str, Any]) -> str:
        context = summary.get("context", {})
        lines: list[str] = []
        lines.append("# Stage-2 摘要（聚合）")
        lines.append("")
        lines.append(f"- 周次：{context.get('week', '未知')}")
        lines.append(f"- ASIN：{context.get('my_asin', '未知')}")
        opp_types = ", ".join(context.get("opp_types", ())) or "未知"
        lines.append(f"- 覆盖对手类型：{opp_types}")
        top_csv = summary.get("top_opp_asins_csv") or ""
        if top_csv:
            lines.append(f"- Top 对手：{top_csv}")
        lines.append("")

        lag_items = summary.get("lag_items") or []
        for item in lag_items:
            lag_type = item.get("lag_type", "unknown")
            lines.append(f"## 落后维度：{lag_type}")
            lines.append(f"- 对手类型：{', '.join(item.get('opp_types', ())) or '未知'}")
            lines.append(f"- 严重度：{item.get('severity', '未知')}")
            lines.append(f"- 置信度：{_format_metric_value(item.get('source_confidence'))}")
            if item.get("triggered_rules"):
                lines.append(f"- 触发规则：{', '.join(item['triggered_rules'])}")
            lines.append("")
            overview = item.get("overview") or {}
            for opp, metrics in overview.items():
                lines.append(f"### vs {opp}")
                if metrics:
                    for key, value in metrics.items():
                        lines.append(f"- {key}: {_format_metric_value(value)}")
                else:
                    lines.append("- 暂无概览数据")
                lines.append("")
            top_opps = item.get("top_opps") or []
            lines.append("### Top 对手差距")
            if not top_opps:
                lines.append("- 暂无数据")
            else:
                for row in top_opps:
                    label = _format_opp_label(row)
                    metrics = []
                    for key in _COMPARISON_METRIC_KEYS.get(lag_type, ()):
                        if row.get(key) is not None:
                            metrics.append(f"{key}={_format_metric_value(row.get(key))}")
                    text = f"- {label}"
                    if metrics:
                        text += f"：{', '.join(metrics)}"
                    lines.append(text)
            lines.append("")

        return "\n".join(lines) + "\n"

    def _resolve_week(self, week: str | None, marketplace_id: str | None) -> str:
        if week:
            return week
        filter_clause = "WHERE marketplace_id = :marketplace_id" if marketplace_id else ""
        sql = _STAGE1_LATEST_WEEK_SQL_BASE.format(marketplace_filter=filter_clause)
        params: dict[str, Any] = {}
        if marketplace_id:
            params["marketplace_id"] = marketplace_id
        row = self._fetch_one(sql, params)
        if not row:
            raise ValueError("Unable to determine latest week for Stage-1 inputs")
        resolved_week = str(row.get("week"))
        LOGGER.info(
            "competition_llm.latest_week_resolved week=%s sunday=%s marketplace_id=%s",
            resolved_week,
            row.get("sunday"),
            marketplace_id,
        )
        return resolved_week

    @staticmethod
    def _parse_confidence(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
        return None

    def _query_stage1_table(self, sql_base: str, week: str, marketplace_id: str | None) -> Sequence[Mapping[str, Any]]:
        sql = sql_base
        params: dict[str, Any] = {"week": week}
        if marketplace_id:
            sql += " AND marketplace_id = :marketplace_id"
            params["marketplace_id"] = marketplace_id
        return self._fetch_all(sql, params)

    def _load_rules_config(self) -> Mapping[str, Any]:
        path_value = self._config.stage_1.rules_config_path or str(_RULES_CONFIG_DEFAULT_PATH)
        path = Path(path_value)
        if not path.is_absolute():
            path = Path.cwd() / path

        data: Mapping[str, Any] | None = None
        try:
            text_data = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            LOGGER.warning("competition_llm.rules_config_missing path=%s", path)
        except OSError as exc:  # pragma: no cover - unexpected I/O errors
            LOGGER.warning("competition_llm.rules_config_read_error path=%s error=%s", path, exc)
        else:
            try:
                loaded = yaml.safe_load(text_data)
            except Exception as exc:  # pragma: no cover - YAML parser edge cases
                LOGGER.warning("competition_llm.rules_config_parse_error path=%s error=%s", path, exc)
            else:
                if isinstance(loaded, Mapping):
                    data = loaded
                elif loaded is not None:
                    LOGGER.warning(
                        "competition_llm.rules_config_invalid_type path=%s type=%s",
                        path,
                        type(loaded).__name__,
                    )

        if data is None:
            LOGGER.warning("competition_llm.rules_config_fallback_default path=%s", path)
            data = deepcopy(_DEFAULT_RULES_CONFIG)
        else:
            data = deepcopy(data)

        try:
            schema = load_schema(_RULES_CONFIG_SCHEMA_NAME)
            validate_schema(schema, data)
        except Exception as exc:  # pragma: no cover - schema warning path
            LOGGER.warning("competition_llm.rules_config_schema_warn path=%s error=%s", path, exc)

        return data

    def _load_active_rules(
        self, context: Mapping[str, Any], *, config_data: Mapping[str, Any]
    ) -> Sequence[LagRule]:
        defaults_raw = config_data.get("defaults")
        defaults = defaults_raw if isinstance(defaults_raw, Mapping) else {}
        rules_by_key: dict[tuple[str, str, str, str], LagRule] = {}

        def _emit(rule: Mapping[str, Any], *, opp_type_default: str = "any") -> None:
            if not isinstance(rule, Mapping):
                return
            if rule.get("is_active") is False:
                return
            lag_type = _normalize_lag_type(rule.get("lag_type"))
            metric = str(rule.get("metric") or "").strip()
            if not lag_type or not metric:
                return
            comparator_raw = rule.get("cmp") or rule.get("comparator") or defaults.get("comparator")
            comparator_alias = str(comparator_raw or "").strip().lower()
            comparator = _COMPARATOR_ALIASES.get(comparator_alias, comparator_raw)
            comparator = str(comparator or "").strip()
            if comparator not in _COMPARATOR_FUNCS:
                LOGGER.debug("competition_llm.stage1_skip_rule comparator=%s rule=%s", comparator_raw, rule)
                return
            threshold = _coerce_float(rule.get("threshold"))
            if threshold is None:
                LOGGER.debug("competition_llm.stage1_skip_rule threshold_missing rule=%s", rule)
                return
            weight = _coerce_float(rule.get("weight"))
            if weight is None:
                weight = _coerce_float(defaults.get("weight"))
            if weight is None:
                weight = 1.0
            opp_type_raw = rule.get("opp_type") or defaults.get("opp_type") or opp_type_default or "any"
            opp_type = str(opp_type_raw or "any").strip().lower()
            rule_name = str(rule.get("rule_name") or f"{metric}_{comparator}")
            rules_by_key[(lag_type, metric, opp_type, comparator)] = LagRule(
                rule_name=rule_name,
                lag_type=lag_type,
                metric=metric,
                opp_type=opp_type,
                comparator=comparator,
                threshold=float(threshold),
                weight=float(weight),
            )

        profiles = config_data.get("profiles") or ()
        for profile in profiles:
            if not isinstance(profile, Mapping):
                continue
            if not profile.get("is_active"):
                continue
            for rule in profile.get("rules") or ():
                if isinstance(rule, Mapping):
                    _emit(rule)

        ctx_source = context or {}

        def _norm(value: Any) -> str:
            return str(value or "").strip().lower()

        ctx = {
            "marketplace_id": _norm(ctx_source.get("marketplace_id") or self._current_marketplace_id),
            "scene_tag": _norm(ctx_source.get("scene_tag")),
            "base_scene": _norm(ctx_source.get("base_scene")),
            "morphology": _norm(ctx_source.get("morphology")),
            "opp_type": _norm(ctx_source.get("opp_type")),
        }

        def _match(conditions: Mapping[str, Any]) -> bool:
            for key, expected in conditions.items():
                if key not in ctx:
                    continue
                if _norm(expected) != ctx[key]:
                    return False
            return True

        overrides = config_data.get("overrides") or ()
        for override in overrides:
            if not isinstance(override, Mapping):
                continue
            if not override.get("is_active"):
                continue
            when = override.get("when")
            if not isinstance(when, Mapping):
                when = {}
            if when and not _match(when):
                continue
            opp_default = _norm(when.get("opp_type")) if when else ""
            if not opp_default:
                opp_default = "any"
            for rule in override.get("rules") or ():
                if isinstance(rule, Mapping):
                    if rule.get("is_active") is False:
                        lag_type = _normalize_lag_type(rule.get("lag_type"))
                        metric = str(rule.get("metric") or "").strip()
                        comparator_raw = rule.get("cmp") or rule.get("comparator") or defaults.get("comparator")
                        comparator_alias = str(comparator_raw or "").strip().lower()
                        comparator = _COMPARATOR_ALIASES.get(comparator_alias, comparator_raw)
                        comparator = str(comparator or "").strip()
                        opp_type_raw = rule.get("opp_type") or defaults.get("opp_type") or opp_default or "any"
                        opp_type = str(opp_type_raw or "any").strip().lower()
                        if lag_type and metric and comparator:
                            rules_by_key.pop((lag_type, metric, opp_type, comparator), None)
                        continue
                    _emit(rule, opp_type_default=opp_default)

        return tuple(rules_by_key.values())

    def _map_severity(self, score: float) -> str:
        thresholds = self._config.stage_1.severity_thresholds or {}
        high_threshold = float(thresholds.get("high", 2.0))
        mid_threshold = float(thresholds.get("mid", 1.0))
        if score >= high_threshold:
            return "high"
        if score >= mid_threshold:
            return "mid"
        return "low"

    def _fetch_all(self, sql: str, params: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        stmt = text(sql)
        with self._engine.connect() as conn:
            result = conn.execute(stmt, params)
            rows = result.fetchall()
            return [self._serialise_row(dict(row._mapping)) for row in rows]

    def _fetch_one(self, sql: str, params: Mapping[str, Any]) -> Mapping[str, Any] | None:
        stmt = text(sql)
        with self._engine.connect() as conn:
            result = conn.execute(stmt, params)
            row = result.fetchone()
            if not row:
                return None
            return self._serialise_row(dict(row._mapping))

    def _serialise_row(self, row: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        for key, value in list(row.items()):
            row[key] = _serialise_value(value)
        if "price_gap_leader" not in row and "price_gap" in row:
            row["price_gap_leader"] = row["price_gap"]
        if "traffic_gap" in row:
            if "ad_ratio_index_med" not in row:
                row["ad_ratio_index_med"] = row["traffic_gap"]
            if "ad_to_natural_gap" not in row:
                row["ad_to_natural_gap"] = row["traffic_gap"]
        return row

    def _build_stage1_key(self, row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row.get(field) for field in _STAGE1_KEY_FIELDS)

    def _build_context(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        context = {field: row.get(field) for field in _REQUIRED_CONTEXT_FIELDS}
        context["asin_priority"] = row.get("asin_priority", 0)
        return context

    def _invoke_with_retries(
        self,
        facts: Mapping[str, Any],
        *,
        schema: Mapping[str, Any],
        prompt: str,
        max_attempts: int,
    ) -> Mapping[str, Any]:
        attempts = 0
        last_error: Exception | None = None
        payload = facts
        while attempts < max_attempts:
            attempts += 1
            config = LLMRunConfig(
                prompt=prompt,
                facts=payload,
                schema=schema,
                model=self._config.llm.model,
                temperature=self._config.llm.temperature,
                response_format=self._config.llm.response_format,
                top_p=self._config.llm.top_p,
            )
            try:
                return self._llm.run(config, retry=False)
            except Exception as exc:  # pragma: no cover - aggregated error handling
                last_error = exc
                LOGGER.warning(
                    "competition_llm.llm_retry attempt=%s max_attempts=%s error=%s",
                    attempts,
                    max_attempts,
                    exc,
                )
                if attempts >= max_attempts:
                    break
        if last_error:
            raise last_error
        raise RuntimeError("LLM invocation failed without exception")

    def _materialize_evidence(
        self,
        payload: Mapping[str, Any],
        facts: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        data = dict(payload)
        root_causes = data.get("root_causes")
        if not isinstance(root_causes, Sequence):
            return data

        lag_index = self._build_lag_index(facts)
        if not lag_index:
            cleaned = []
            for cause in root_causes:
                if isinstance(cause, Mapping):
                    entry = dict(cause)
                    entry.pop("evidence_refs", None)
                    cleaned.append(entry)
                else:
                    cleaned.append(cause)
            data["root_causes"] = cleaned
            return data

        cleaned_causes: list[Mapping[str, Any]] = []
        for cause in root_causes:
            if not isinstance(cause, Mapping):
                cleaned_causes.append(cause)
                continue

            entry = dict(cause)
            raw_evidence: list[Any] = []
            if entry.get("evidence"):
                raw_evidence.extend(
                    self._normalise_evidence_sequence(entry.get("evidence"))
                )

            inferred_lag = self._infer_root_cause_dimension(entry, lag_index)
            lag_type = inferred_lag
            lag_data = lag_index.get(lag_type)

            hints = self._parse_evidence_refs(entry.get("evidence_refs"))
            if lag_data is None and len(lag_index) == 1:
                lag_type, lag_data = next(iter(lag_index.items()))
            if lag_data is None:
                for hint in hints:
                    hinted_type = hint.get("lag_type")
                    if hinted_type and hinted_type in lag_index:
                        lag_type = hinted_type
                        lag_data = lag_index.get(hinted_type)
                        break

            if lag_data is not None:
                for hint in hints:
                    raw_evidence.extend(
                        self._build_evidence_from_hint(lag_index, lag_type, hint)
                    )

                if not raw_evidence:
                    raw_evidence.extend(
                        self._extract_pairwise_evidence(lag_type, lag_data)
                    )

                if not raw_evidence:
                    raw_evidence.extend(
                        self._extract_overview_evidence(lag_type, lag_data)
                    )
            else:
                LOGGER.warning(
                    "competition_llm.stage2_evidence_lag_not_found lag_hint=%s available=%s",
                    inferred_lag,
                    sorted(lag_index),
                )

            normalised = self._deduplicate_evidence(
                self._normalise_evidence_sequence(raw_evidence)
            )

            if not normalised:
                LOGGER.warning(
                    "competition_llm.stage2_evidence_missing code=%s hints=%s",
                    entry.get("root_cause_code"),
                    hints,
                )

            entry["evidence"] = normalised[:3]
            entry.pop("evidence_refs", None)
            cleaned_causes.append(entry)

        data["root_causes"] = cleaned_causes
        return data

    def _build_evidence_from_hint(
        self,
        lag_index: Mapping[str, Mapping[str, Any]],
        default_lag: str,
        hint: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        lag_type = hint.get("lag_type") or default_lag
        lag_data = lag_index.get(lag_type)
        if lag_data is None:
            return []
        kind = str(hint.get("kind") or "").lower()
        metric = hint.get("metric")
        opp_type = hint.get("opp_type")
        opp_asin = hint.get("opp_asin")
        if kind == "overview":
            return self._extract_overview_evidence(
                lag_type,
                lag_data,
                metric=metric,
                opp_type=opp_type,
            )
        if kind == "pair":
            return self._extract_pairwise_evidence(
                lag_type,
                lag_data,
                metric=metric,
                opp_asin=opp_asin,
            )
        return []

    def _build_lag_index(self, facts: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
        lag_items = facts.get("lag_items")
        if not isinstance(lag_items, Sequence):
            return {}
        index: dict[str, Mapping[str, Any]] = {}
        for item in lag_items:
            if not isinstance(item, Mapping):
                continue
            lag_label = item.get("lag_type")
            normalized = _normalize_lag_type(lag_label)
            if not normalized:
                continue
            overview = item.get("overview")
            if not isinstance(overview, Mapping):
                overview = {}
            top_opps_raw = item.get("top_opps")
            top_opps: list[Mapping[str, Any]] = []
            if isinstance(top_opps_raw, Sequence):
                for row in top_opps_raw:
                    if isinstance(row, Mapping):
                        top_opps.append(row)
            index[normalized] = {
                "lag_type": normalized,
                "overview": overview,
                "top_opps": top_opps,
            }
        return index

    def _infer_root_cause_dimension(
        self, cause: Mapping[str, Any], lag_index: Mapping[str, Mapping[str, Any]]
    ) -> str:
        lag_dimension = _normalize_lag_type(
            cause.get("lag_dimension") or cause.get("lag_type")
        )
        if lag_dimension in lag_index:
            return lag_dimension
        code = str(cause.get("root_cause_code") or cause.get("code") or "").strip().lower()
        mapped = _ROOT_CAUSE_TO_LAG.get(code, lag_dimension)
        if mapped in lag_index:
            return mapped
        return mapped

    def _parse_evidence_refs(self, refs: Any) -> list[Mapping[str, Any]]:
        if not isinstance(refs, Sequence):
            return []
        hints: list[Mapping[str, Any]] = []
        for ref in refs:
            if isinstance(ref, Mapping):
                metric = str(ref.get("metric") or "").strip()
                if not metric:
                    continue
                hint: dict[str, Any] = {
                    "kind": "overview",
                    "metric": metric,
                }
                lag_type = _normalize_lag_type(ref.get("lag_type"))
                if lag_type:
                    hint["lag_type"] = lag_type
                opp_type = str(ref.get("opp_type") or "").strip().lower()
                if opp_type:
                    hint["opp_type"] = opp_type
                hints.append(hint)
            elif isinstance(ref, str):
                cleaned = ref.strip()
                if not cleaned:
                    continue
                parts = cleaned.split(".")
                if len(parts) >= 3 and parts[0] == "overview":
                    hint = {
                        "kind": "overview",
                        "opp_type": parts[1].lower(),
                        "metric": parts[2],
                    }
                    hints.append(hint)
                elif len(parts) >= 2 and parts[0] == "top_opps":
                    hint = {"kind": "pair"}
                    if len(parts) >= 3:
                        hint["metric"] = parts[-1]
                    if len(parts) >= 2 and parts[1]:
                        hint["opp_asin"] = parts[1]
                    hints.append(hint)
        return hints

    def _extract_overview_evidence(
        self,
        lag_type: str,
        lag_data: Mapping[str, Any],
        *,
        metric: Any = None,
        opp_type: Any = None,
    ) -> list[Mapping[str, Any]]:
        overview = lag_data.get("overview")
        if not isinstance(overview, Mapping):
            return []
        target_opp_types: Sequence[str]
        if opp_type:
            target_opp_types = (str(opp_type).lower(),)
        else:
            target_opp_types = tuple(overview.keys())
        source = "page.overview" if lag_type in _PAGE_LAG_TYPES else "traffic.overview"
        results: list[Mapping[str, Any]] = []
        for opp in target_opp_types:
            metrics_map = overview.get(opp)
            if not isinstance(metrics_map, Mapping):
                continue
            for name, value in metrics_map.items():
                if metric and name != metric:
                    continue
                entry = self._build_overview_evidence_entry(
                    lag_type,
                    str(opp),
                    name,
                    value,
                    source,
                )
                if entry:
                    results.append(entry)
                if metric:
                    break
            if results and metric:
                break
        return results

    def _build_overview_evidence_entry(
        self,
        lag_type: str,
        opp_type: str,
        metric: str,
        value: Any,
        source: str,
    ) -> Mapping[str, Any] | None:
        if value is None:
            return None
        against = opp_type.lower()
        if against not in {"leader", "median"}:
            return None
        metric_lower = metric.lower()
        unit = None
        if "ratio" in metric_lower or "index" in metric_lower:
            unit = "ratio"
            opp_value: Any = 1.0
        else:
            return None
        evidence = {
            "metric": metric,
            "against": against,
            "my_value": self._format_evidence_value(value),
            "opp_value": opp_value,
            "unit": unit,
            "source": source,
        }
        return evidence

    def _extract_pairwise_evidence(
        self,
        lag_type: str,
        lag_data: Mapping[str, Any],
        *,
        metric: Any = None,
        opp_asin: Any = None,
        limit: int = 3,
    ) -> list[Mapping[str, Any]]:
        rows = lag_data.get("top_opps")
        if not isinstance(rows, Sequence):
            return []
        source = "pairs_each" if lag_type in _PAGE_LAG_TYPES else "traffic.pairs"
        candidates: list[tuple[int, float, Mapping[str, Any]]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            row_opp_asin = row.get("opp_asin")
            if opp_asin and row_opp_asin != opp_asin:
                continue
            for entry in self._build_pairwise_entries_from_row(
                lag_type,
                row,
                source,
            ):
                if metric and entry.get("metric") != metric:
                    continue
                payload = dict(entry)
                if row_opp_asin and "opp_asin" not in payload:
                    payload["opp_asin"] = row_opp_asin
                priority = int(payload.pop("_priority", 50))
                impact = float(payload.pop("_impact", 0.0))
                candidates.append((priority, impact, payload))

        ordered = sorted(candidates, key=lambda item: (item[0], -abs(item[1])))
        results: list[Mapping[str, Any]] = []
        for _, _, entry in ordered:
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    def _build_pairwise_entries_from_row(
        self,
        lag_type: str,
        row: Mapping[str, Any],
        source: str,
    ) -> list[Mapping[str, Any]]:
        if not isinstance(row, Mapping):
            return []
        if lag_type == "rank":
            entries = self._build_rank_entries(row, source)
        elif lag_type == "content":
            entries = self._build_content_entries(row, source)
        elif lag_type == "social":
            entries = self._build_social_entries(row, source)
        elif lag_type == "keyword":
            entries = self._build_keyword_entries(row, source)
        else:
            entries = []
        if not entries:
            entries = self._build_default_pairwise_entries(row, source)
        return entries

    def _build_rank_entries(
        self, row: Mapping[str, Any], source: str
    ) -> list[Mapping[str, Any]]:
        entries: list[Mapping[str, Any]] = []
        entry = self._build_metric_entry(
            row,
            "rank_leaf",
            source,
            priority=1,
            unit="rank",
            note_builder=_format_rank_leaf_note,
        )
        if entry:
            entries.append(entry)
        entry = self._build_metric_entry(
            row,
            "rank_pos_pct",
            source,
            priority=2,
            unit="pct",
            note_builder=_format_rank_pct_note,
        )
        if entry:
            entries.append(entry)
        return entries

    def _build_content_entries(
        self, row: Mapping[str, Any], source: str
    ) -> list[Mapping[str, Any]]:
        entries: list[Mapping[str, Any]] = []
        entry = self._build_metric_entry(
            row,
            "image_cnt",
            source,
            priority=1,
            note_builder=_format_image_note,
        )
        if entry:
            entries.append(entry)
        entry = self._build_metric_entry(
            row,
            "video_cnt",
            source,
            priority=2,
            note_builder=_format_video_note,
        )
        if entry:
            entries.append(entry)
        entry = self._build_metric_entry(
            row,
            "content_score",
            source,
            priority=3,
            note_builder=_format_content_score_note,
        )
        if entry:
            entries.append(entry)
        return entries

    def _build_social_entries(
        self, row: Mapping[str, Any], source: str
    ) -> list[Mapping[str, Any]]:
        entries: list[Mapping[str, Any]] = []
        entry = self._build_metric_entry(
            row,
            "rating",
            source,
            priority=1,
            note_builder=_format_rating_note,
        )
        if entry:
            entries.append(entry)
        entry = self._build_metric_entry(
            row,
            "reviews",
            source,
            priority=2,
            note_builder=_format_reviews_note,
        )
        if entry:
            entries.append(entry)
        entry = self._build_metric_entry(
            row,
            "social_proof",
            source,
            priority=3,
            note_builder=_format_social_proof_note,
        )
        if entry:
            entries.append(entry)
        return entries

    def _build_keyword_entries(
        self, row: Mapping[str, Any], source: str
    ) -> list[Mapping[str, Any]]:
        opp_asin = row.get("opp_asin")
        if not opp_asin:
            return []
        pairs = row.get("keyword_pairs")
        if not isinstance(pairs, Sequence):
            return []
        entries: list[Mapping[str, Any]] = []
        for index, pair in enumerate(pairs):
            if not isinstance(pair, Mapping):
                continue
            keyword = pair.get("keyword")
            opp_rank = pair.get("opp_rank")
            if not keyword or opp_rank is None:
                continue
            my_rank = pair.get("my_rank")
            my_share = pair.get("my_share")
            opp_share = pair.get("opp_share")
            note = _format_keyword_note(
                keyword,
                my_rank,
                opp_rank,
                my_share,
                opp_share,
                pair.get("tag"),
            )
            entry: dict[str, Any] = {
                "metric": "keyword_rank",
                "against": "asin",
                "my_value": my_rank if my_rank is not None else "无",
                "opp_value": opp_rank,
                "opp_asin": opp_asin,
                "unit": "rank",
                "source": "keywords.7d",
                "note": note,
                "_priority": 1 + index,
                "_impact": float(pair.get("impact", 0.0) or 0.0),
            }
            entries.append(entry)
        return entries

    def _build_default_pairwise_entries(
        self, row: Mapping[str, Any], source: str
    ) -> list[Mapping[str, Any]]:
        entries: list[Mapping[str, Any]] = []
        for key in row.keys():
            if not key.startswith("my_"):
                continue
            suffix = key[3:]
            entry = self._build_metric_entry(
                row,
                suffix,
                source,
                priority=50,
            )
            if entry:
                entries.append(entry)
        return entries

    def _build_metric_entry(
        self,
        row: Mapping[str, Any],
        suffix: str,
        source: str,
        *,
        priority: int,
        unit: str | None = None,
        note_builder: Callable[[Any, Any, Mapping[str, Any]], str | None] | None = None,
    ) -> Mapping[str, Any] | None:
        opp_asin = row.get("opp_asin")
        if not opp_asin:
            return None
        my_key = f"my_{suffix}"
        opp_key = f"opp_{suffix}"
        if my_key not in row or opp_key not in row:
            return None
        my_value = row.get(my_key)
        opp_value = row.get(opp_key)
        if my_value in (None, "") or opp_value in (None, ""):
            return None
        entry: dict[str, Any] = {
            "metric": suffix,
            "against": "asin",
            "my_value": self._format_evidence_value(my_value),
            "opp_value": self._format_evidence_value(opp_value),
            "opp_asin": opp_asin,
            "source": source,
            "_priority": priority,
            "_impact": _compute_pair_gap(my_value, opp_value),
        }
        if unit:
            entry["unit"] = unit
        else:
            inferred = self._infer_metric_unit(suffix)
            if inferred:
                entry["unit"] = inferred
        if note_builder:
            note = note_builder(my_value, opp_value, row)
            if note:
                entry["note"] = note
        return entry

    def _normalise_evidence_sequence(self, items: Any) -> list[Mapping[str, Any]]:
        if not isinstance(items, Sequence):
            return []
        result: list[Mapping[str, Any]] = []
        for item in items:
            normalised = self._normalise_evidence_entry(item)
            if normalised:
                result.append(normalised)
        return result

    def _normalise_evidence_entry(self, item: Any) -> Mapping[str, Any] | None:
        if not isinstance(item, Mapping):
            return None
        metric = str(item.get("metric") or "").strip()
        if not metric:
            return None
        against_raw = item.get("against") or item.get("opp_type")
        against = str(against_raw or "").strip().lower()
        if against not in {"leader", "median", "asin"}:
            return None
        my_value = item.get("my_value")
        opp_value = item.get("opp_value")
        if my_value is None or opp_value is None:
            return None
        opp_asin = item.get("opp_asin")
        if against == "asin" and not opp_asin:
            return None
        normalised: dict[str, Any] = {
            "metric": metric,
            "against": against,
            "my_value": self._format_evidence_value(my_value),
            "opp_value": self._format_evidence_value(opp_value),
        }
        if against == "asin":
            normalised["opp_asin"] = str(opp_asin)
        unit = item.get("unit")
        if unit is not None:
            normalised["unit"] = unit
        source = item.get("source")
        if source is not None:
            normalised["source"] = source
        note = item.get("note")
        if note is not None:
            normalised["note"] = note
        return normalised

    def _deduplicate_evidence(
        self, entries: Sequence[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        seen: set[tuple[Any, ...]] = set()
        result: list[Mapping[str, Any]] = []
        for entry in entries:
            key = (
                entry.get("metric"),
                entry.get("against"),
                entry.get("opp_asin"),
                self._format_evidence_value(entry.get("my_value")),
                self._format_evidence_value(entry.get("opp_value")),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(entry)
        return result

    def _format_evidence_value(self, value: Any) -> Any:
        numeric = _coerce_float(value)
        if numeric is None:
            return value
        return round(numeric, 4)

    def _infer_metric_unit(self, metric: str) -> str | None:
        lower = metric.lower()
        if lower.endswith("_pct") or "pct" in lower or lower.endswith("_share") or "share" in lower:
            return "pct"
        if "ratio" in lower or "index" in lower:
            return "ratio"
        if "price" in lower or lower.endswith("_usd"):
            return "USD"
        return None

    def _validate_stage2_machine_json(self, payload: Mapping[str, Any]) -> None:
        if isinstance(payload, MutableMapping):
            mutable_payload: MutableMapping[str, Any] = payload
        else:
            mutable_payload = dict(payload)

        if "actions" not in mutable_payload and "recommended_actions" in mutable_payload:
            mutable_payload["actions"] = mutable_payload.get("recommended_actions") or []

        raw_actions = mutable_payload.get("actions") or []
        if not isinstance(raw_actions, Sequence):
            LOGGER.warning(
                "competition_llm.stage2_drop_actions_invalid_container type=%s",
                type(raw_actions).__name__,
            )
            raw_actions = []

        allowed_actions = tuple(self._config.stage_2.allowed_action_codes or ())
        allowed_actions_set = {code for code in allowed_actions if code}
        cleaned_actions: list[dict[str, Any]] = []
        for index, action in enumerate(raw_actions):
            if not isinstance(action, Mapping):
                LOGGER.warning(
                    "competition_llm.stage2_drop_action_invalid_item index=%s type=%s",
                    index,
                    type(action).__name__,
                )
                continue
            code_raw = action.get("code")
            if code_raw is None:
                code_raw = action.get("action_code")
            code = str(code_raw).strip() if code_raw is not None else ""
            if not code:
                LOGGER.warning(
                    "competition_llm.stage2_drop_action_missing_code index=%s item=%s",
                    index,
                    action,
                )
                continue
            if allowed_actions_set and code not in allowed_actions_set:
                raise ValueError(f"Action code {code!r} is not permitted")
            normalized = dict(action)
            normalized["code"] = code
            cleaned_actions.append(normalized)

        mutable_payload["actions"] = cleaned_actions
        if "recommended_actions" in mutable_payload:
            mutable_payload["recommended_actions"] = cleaned_actions

        root_causes_raw = mutable_payload.get("root_causes") or []
        if not isinstance(root_causes_raw, Sequence):
            LOGGER.warning(
                "competition_llm.stage2_root_causes_invalid_container type=%s",
                type(root_causes_raw).__name__,
            )
            root_causes = []
        else:
            root_causes = list(root_causes_raw)
        mutable_payload["root_causes"] = root_causes

        allowed_root_causes = tuple(self._config.stage_2.allowed_root_cause_codes or ())
        allowed_root_causes_set = {code for code in allowed_root_causes if code}
        for index, cause in enumerate(root_causes):
            if not isinstance(cause, Mapping):
                LOGGER.warning(
                    "competition_llm.stage2_root_cause_invalid_item type=%s",
                    type(cause).__name__,
                )
                continue
            code_raw = cause.get("root_cause_code")
            if code_raw is None:
                code_raw = cause.get("code")
            code = str(code_raw).strip() if code_raw is not None else ""
            if not code:
                raise ValueError("Root cause code is required")
            if allowed_root_causes_set and code not in allowed_root_causes_set:
                raise ValueError(f"Root cause code {code!r} is not permitted")
            evidence_list = self._normalise_evidence_sequence(cause.get("evidence"))
            if not evidence_list:
                raise ValueError("Root cause evidence is required")
            evidence_list = self._deduplicate_evidence(evidence_list)[:3]
            if isinstance(cause, MutableMapping):
                cause.setdefault("root_cause_code", code)
                cause["evidence"] = evidence_list
                if "evidence_refs" in cause:
                    cause.pop("evidence_refs", None)
            else:
                updated = dict(cause)
                updated["root_cause_code"] = code
                updated["evidence"] = evidence_list
                updated.pop("evidence_refs", None)
                root_causes[index] = updated

        validate_schema(self._stage2_schema, mutable_payload)

    def _build_page_evidence(
        self, ctx: Mapping[str, Any], lag_type: str, opp_type: str
    ) -> Mapping[str, Any] | None:
        sql_overview = """
          SELECT price_index_med, price_gap_leader, price_z,
                 rank_pos_pct, content_gap, social_gap,
                 badge_diff, badge_delta_sum, pressure, intensity_band, confidence
          FROM bi_amz_comp_pairs
          WHERE scene_tag=:scene_tag AND base_scene=:base_scene
            AND COALESCE(morphology,'') = COALESCE(:morphology,'')
            AND marketplace_id=:marketplace_id
            AND week=:week AND sunday=:sunday
            AND my_asin=:my_asin AND opp_type=:opp_type
          LIMIT 1
        """
        try:
            overview = self._fetch_one(sql_overview, ctx)
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage2_page_overview_unavailable asin=%s lag_type=%s error=%s",
                ctx.get("my_asin"),
                lag_type,
                exc,
            )
            overview = None
        if not overview:
            return None

        metric_map = {
            "price": ("price_gap_each", "score_price"),
            "rank": ("rank_pos_delta", "score_rank"),
            "content": ("content_gap_each", "score_cont"),
            "social": ("social_gap_each", "score_soc"),
            "badge": ("badge_delta_sum", "score_badge"),
            "confidence": ("confidence", "confidence"),
        }
        metric_col, score_col = metric_map.get(lag_type, ("pressure", "pressure"))

        sql_each = f"""
          SELECT *,
                 {score_col} AS score
          FROM bi_amz_comp_pairs_each
          WHERE scene_tag=:scene_tag AND base_scene=:base_scene
            AND COALESCE(morphology,'') = COALESCE(:morphology,'')
            AND marketplace_id=:marketplace_id
            AND week=:week AND sunday=:sunday
            AND my_asin=:my_asin
          ORDER BY ABS({metric_col}) DESC NULLS LAST
          LIMIT 3
        """
        try:
            top_each = self._fetch_all(sql_each, ctx)
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage2_page_pairs_unavailable asin=%s lag_type=%s error=%s",
                ctx.get("my_asin"),
                lag_type,
                exc,
            )
            top_each = []
        if top_each:
            self._attach_page_entity_details(ctx, top_each)
        comparison_reasons = _build_comparison_reasons(lag_type, top_each)

        return {
            "channel": "page",
            "lag_type": lag_type,
            "opp_type": opp_type,
            "overview": overview,
            "top_opps": top_each,
            "top_opp_asins_csv": ",".join(
                [row.get("opp_asin") for row in top_each if row.get("opp_asin")][:8]
            ),
            "top_diff_reasons": comparison_reasons,
            "reason_code": f"{lag_type}_by_pairs",
            "prompt_hint": "优先解释 overview 与 top_opps 中的差异来源；引用已给字段名即可；不要推导新指标。",
        }

    def _attach_page_entity_details(
        self,
        ctx: Mapping[str, Any],
        rows: Sequence[MutableMapping[str, Any]] | Sequence[Mapping[str, Any]],
    ) -> None:
        if not rows:
            return

        my_detail = self._get_entity_detail(ctx, ctx.get("my_asin"))
        opponent_cache: dict[str, Mapping[str, Any] | None] = {}
        missing_logged: set[tuple[str, str]] = set()

        for row in rows:
            if not isinstance(row, MutableMapping):
                continue
            opp_asin = row.get("opp_asin")
            if not opp_asin or opp_asin in opponent_cache:
                continue
            opponent_cache[opp_asin] = self._get_entity_detail(ctx, opp_asin)

        for row in rows:
            if not isinstance(row, MutableMapping):
                continue
            if my_detail:
                self._inject_entity_fields(row, my_detail, prefix="my_")
            else:
                self._log_missing_entity_detail(
                    "my", ctx.get("my_asin"), ctx, missing_logged
                )
            opp_asin = row.get("opp_asin")
            if not opp_asin:
                continue
            opp_detail = opponent_cache.get(opp_asin)
            if opp_detail:
                self._inject_entity_fields(row, opp_detail, prefix="opp_")
            else:
                self._log_missing_entity_detail("opp", opp_asin, ctx, missing_logged)

    def _get_entity_detail(
        self, ctx: Mapping[str, Any], asin: Any
    ) -> Mapping[str, Any] | None:
        if not asin:
            return None
        sql = """
          SELECT price_current, price_list, coupon_pct, price_net,
                 rank_leaf, rank_root, rank_score,
                 image_cnt, video_cnt, bullet_cnt, title_len, aplus_flag, content_score,
                 rating, reviews, social_proof, badge_json
          FROM bi_amz_comp_entities_clean
          WHERE scene_tag=:scene_tag AND base_scene=:base_scene
            AND COALESCE(morphology,'') = COALESCE(:morphology,'')
            AND marketplace_id=:marketplace_id
            AND week=:week AND sunday=:sunday
            AND asin=:asin
          LIMIT 1
        """
        params = {
            "scene_tag": ctx.get("scene_tag"),
            "base_scene": ctx.get("base_scene"),
            "morphology": ctx.get("morphology"),
            "marketplace_id": ctx.get("marketplace_id"),
            "week": ctx.get("week"),
            "sunday": ctx.get("sunday"),
            "asin": asin,
        }
        try:
            detail = self._fetch_one(sql, params)
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage2_entity_lookup_failed asin=%s error=%s",
                asin,
                exc,
            )
            detail = None

        if detail:
            result = dict(detail)
        else:
            result = {}

        if not result.get("brand"):
            brand = self._lookup_brand(ctx, asin)
            if brand:
                result["brand"] = brand

        return result or None

    def _log_missing_entity_detail(
        self,
        side: str,
        asin: Any,
        ctx: Mapping[str, Any],
        logged: set[tuple[str, str]] | None = None,
    ) -> None:
        asin_key = str(asin) if asin else ""
        if not asin_key:
            return
        cache = logged if logged is not None else set()
        key = (side, asin_key)
        if key in cache:
            return
        cache.add(key)
        LOGGER.debug(
            "stage2_entity_detail_missing side=%s asin=%s week=%s marketplace_id=%s",
            side,
            asin_key,
            ctx.get("week"),
            ctx.get("marketplace_id"),
        )

    def _inject_entity_fields(
        self,
        target: MutableMapping[str, Any],
        source: Mapping[str, Any],
        *,
        prefix: str,
    ) -> None:
        for field in _ENTITY_DETAIL_FIELDS:
            key = f"{prefix}{field}"
            if key in target and target[key] not in (None, ""):
                continue
            value = source.get(field)
            if value in (None, ""):
                continue
            target[key] = value

    def _lookup_brand(self, ctx: Mapping[str, Any], asin: Any) -> str | None:
        if not asin:
            return None
        asin_key = str(asin)
        if asin_key in self._brand_cache:
            return self._brand_cache[asin_key]

        brand = self._fetch_brand_from_snapshot(ctx, asin_key)
        if not brand:
            LOGGER.debug(
                "competition_llm.stage2_brand_missing asin=%s week=%s marketplace_id=%s",
                asin_key,
                ctx.get("week"),
                ctx.get("marketplace_id"),
            )
        self._brand_cache[asin_key] = brand
        return brand

    def _fetch_brand_from_snapshot(
        self, ctx: Mapping[str, Any], asin: str
    ) -> str | None:
        marketplace_id = ctx.get("marketplace_id")
        week = ctx.get("week")
        params = {
            "marketplace_id": marketplace_id,
            "week": week,
            "asin": asin,
        }
        sql = """
          SELECT brand, payload
          FROM bi_amz_asin_product_snapshot
          WHERE marketplace_id=:marketplace_id
            AND week=:week
            AND asin=:asin
          LIMIT 1
        """
        try:
            row = self._fetch_one(sql, params)
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage2_brand_lookup_failed asin=%s error=%s",
                asin,
                exc,
            )
            row = None

        if not row:
            return None

        brand_value = row.get("brand")
        if isinstance(brand_value, str) and brand_value.strip():
            return brand_value.strip()

        payload = row.get("payload")
        if payload in (None, ""):
            return None
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8", errors="ignore")
        if isinstance(payload, str):
            try:
                payload_obj = json.loads(payload)
            except json.JSONDecodeError:
                LOGGER.debug(
                    "competition_llm.stage2_brand_payload_invalid asin=%s", asin
                )
                return None
        elif isinstance(payload, Mapping):
            payload_obj = payload
        else:
            return None

        tokens = self._resolve_brand_path_tokens()
        brand = _extract_json_path(payload_obj, tokens)
        if not brand:
            brand = _extract_json_path(payload_obj, ["brand"])
        if isinstance(brand, str):
            return brand.strip() or None
        return None

    def _resolve_brand_path_tokens(self) -> list[str]:
        if self._brand_path_tokens is not None:
            return self._brand_path_tokens

        sql = """
          SELECT json_path, path, field_path
          FROM bi_amz_comp_payload_path
          WHERE field_name = :field_name
          LIMIT 1
        """
        try:
            row = self._fetch_one(sql, {"field_name": "brand"})
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage2_brand_path_lookup_failed error=%s",
                exc,
            )
            row = None

        path = None
        if row:
            path = (
                row.get("json_path")
                or row.get("path")
                or row.get("field_path")
                or ""
            )
        tokens = _parse_json_path_tokens(str(path or "")) if path else []
        if not tokens:
            tokens = ["brand"]
        self._brand_path_tokens = tokens
        return tokens

    def _attach_keyword_details(
        self,
        ctx: Mapping[str, Any],
        rows: Sequence[MutableMapping[str, Any]] | Sequence[Mapping[str, Any]],
    ) -> None:
        if not rows:
            return
        my_asin = ctx.get("my_asin")
        asins: set[str] = {str(my_asin)} if my_asin else set()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            opp_asin = row.get("opp_asin")
            if opp_asin:
                asins.add(str(opp_asin))
        keyword_rankings = self._fetch_keyword_rankings(ctx, asins)
        if not keyword_rankings:
            return
        my_keywords = keyword_rankings.get(str(my_asin), []) if my_asin else []
        my_index = {item["keyword"]: item for item in my_keywords if item.get("keyword")}

        for row in rows:
            if not isinstance(row, MutableMapping):
                continue
            opp_asin = row.get("opp_asin")
            if not opp_asin:
                continue
            opp_keywords = keyword_rankings.get(str(opp_asin), [])
            row["keyword_pairs"] = self._compose_keyword_pairs(my_index, opp_keywords)

    def _fetch_keyword_rankings(
        self,
        ctx: Mapping[str, Any],
        asins: Sequence[str],
    ) -> Mapping[str, list[dict[str, Any]]]:
        if not asins:
            return {}
        marketplace_id = ctx.get("marketplace_id")
        sunday = ctx.get("sunday")
        if not marketplace_id or not sunday:
            return {}
        try:
            sunday_date = datetime.fromisoformat(str(sunday)).date()
        except ValueError:
            return {}
        start_date = sunday_date - timedelta(days=_KEYWORD_LOOKBACK_DAYS - 1)
        asin_list = sorted({str(asin) for asin in asins if asin})
        if not asin_list:
            return {}
        asin_params = {
            f"asin_{index}": asin for index, asin in enumerate(asin_list)
        }
        asin_placeholders = ", ".join(
            f":asin_{index}" for index in range(len(asin_list))
        )
        sql = f"""
          SELECT asin, keyword, AVG(ratio_score) AS ratio_score
          FROM vw_sif_keyword_daily_std
          WHERE marketplace_id=:marketplace_id
            AND asin IN ({asin_placeholders})
            AND snapshot_date BETWEEN :start_date AND :end_date
          GROUP BY asin, keyword
        """
        params: dict[str, Any] = {
            "marketplace_id": marketplace_id,
            "start_date": start_date.isoformat(),
            "end_date": sunday_date.isoformat(),
        }
        params.update(asin_params)
        try:
            rows = self._fetch_all(sql, params)
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage2_keyword_lookup_failed error=%s",
                exc,
            )
            return {}

        rankings: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            asin_value = str(row.get("asin") or "").strip()
            keyword = str(row.get("keyword") or "").strip()
            if not asin_value or not keyword:
                continue
            share = _coerce_float(row.get("ratio_score"))
            rankings[asin_value].append(
                {"keyword": keyword, "share": share if share is not None else 0.0}
            )

        if not rankings:
            return {}

        tag_lookup = self._fetch_keyword_tags(
            {item["keyword"] for items in rankings.values() for item in items}
        )

        for asin_value, items in rankings.items():
            items.sort(key=lambda item: item.get("share") or 0.0, reverse=True)
            limited = items[:10]
            for index, item in enumerate(limited, start=1):
                item["rank"] = index
                if item.get("keyword") in tag_lookup:
                    item["tag"] = tag_lookup[item["keyword"]]
            rankings[asin_value] = limited

        return rankings

    def _fetch_keyword_tags(self, keywords: set[str]) -> Mapping[str, str]:
        if not keywords:
            return {}
        keyword_list = sorted({kw for kw in keywords if kw})
        if not keyword_list:
            return {}
        params: dict[str, Any] = {
            f"kw_{index}": keyword_list[index]
            for index in range(len(keyword_list))
        }
        placeholders = ", ".join(f":kw_{index}" for index in range(len(keyword_list)))
        sql = f"""
          SELECT keyword, tag
          FROM bi_amz_comp_kw_tag
          WHERE keyword IN ({placeholders})
        """
        try:
            rows = self._fetch_all(sql, params)
        except SQLAlchemyError:
            return {}
        return {
            str(row.get("keyword") or ""): str(row.get("tag") or "")
            for row in rows
            if row.get("keyword")
        }

    def _compose_keyword_pairs(
        self,
        my_index: Mapping[str, Mapping[str, Any]],
        opponent_keywords: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        used: set[str] = set()
        for item in opponent_keywords:
            keyword = item.get("keyword")
            if not keyword:
                continue
            my_item = my_index.get(keyword)
            if not my_item:
                continue
            pairs.append(self._build_keyword_pair(keyword, my_item, item))
            used.add(keyword)
            if len(pairs) >= _KEYWORD_PER_PAIR_LIMIT:
                return pairs
        for item in opponent_keywords:
            if len(pairs) >= _KEYWORD_PER_PAIR_LIMIT:
                break
            keyword = item.get("keyword")
            if not keyword or keyword in used:
                continue
            my_item = my_index.get(keyword)
            pairs.append(self._build_keyword_pair(keyword, my_item, item))
            used.add(keyword)
        return pairs

    def _build_keyword_pair(
        self,
        keyword: str,
        my_item: Mapping[str, Any] | None,
        opp_item: Mapping[str, Any],
    ) -> dict[str, Any]:
        my_rank = _coerce_float(my_item.get("rank")) if my_item else None
        opp_rank = _coerce_float(opp_item.get("rank"))
        my_share = _coerce_float(my_item.get("share")) if my_item else None
        opp_share = _coerce_float(opp_item.get("share"))
        tag = (
            my_item.get("tag")
            if my_item and my_item.get("tag")
            else opp_item.get("tag")
        )
        impact = abs((opp_share or 0.0) - (my_share or 0.0))
        if my_rank is None:
            impact = (opp_share or 0.0) + 1.0
        pair = {
            "keyword": keyword,
            "my_rank": int(my_rank) if my_rank is not None else None,
            "opp_rank": int(opp_rank) if opp_rank is not None else None,
            "my_share": my_share,
            "opp_share": opp_share,
            "tag": tag,
            "impact": impact,
        }
        return pair

    def _build_traffic_evidence(
        self, ctx: Mapping[str, Any], lag_type: str, opp_type: str
    ) -> Mapping[str, Any] | None:
        sql_overview = """
          SELECT ad_ratio_gap_leader, ad_ratio_index_med, ad_to_natural_gap,
                 sp_share_in_ad_gap, sbv_share_in_ad_gap, sb_share_in_ad_gap,
                 kw_entropy_gap, kw_hhi_gap, kw_top3_share_gap,
                 kw_brand_share_gap, kw_competitor_share_gap,
                 t_pressure, t_intensity_band, t_confidence
          FROM bi_amz_comp_traffic_pairs
          WHERE scene_tag=:scene_tag AND base_scene=:base_scene
            AND COALESCE(morphology,'') = COALESCE(:morphology,'')
            AND marketplace_id=:marketplace_id
            AND week=:week AND sunday=:sunday
            AND my_asin=:my_asin AND opp_type=:opp_type
          LIMIT 1
        """
        try:
            overview = self._fetch_one(sql_overview, ctx)
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage2_traffic_overview_unavailable asin=%s lag_type=%s error=%s",
                ctx.get("my_asin"),
                lag_type,
                exc,
            )
            overview = None
        if not overview:
            return None

        if lag_type == "traffic_mix":
            order_metric = "ABS(ad_to_natural_gap_each)"
        else:
            order_metric = "ABS(my_kw_top3_share_7d_avg - opp_kw_top3_share_7d_avg)"

        sql_each = f"""
          SELECT opp_asin, opp_parent_asin,
                 my_ad_ratio, opp_ad_ratio, my_nf_ratio, opp_nf_ratio,
                 my_recommend_ratio, opp_recommend_ratio,
                 ad_ratio_gap_each, ad_to_natural_gap_each,
                 my_kw_top3_share_7d_avg, opp_kw_top3_share_7d_avg,
                 my_kw_brand_share_7d_avg, opp_kw_brand_share_7d_avg,
                 my_kw_competitor_share_7d_avg, opp_kw_competitor_share_7d_avg,
                 t_score_mix, t_score_kw, t_pressure
          FROM bi_amz_comp_traffic_pairs_each
          WHERE scene_tag=:scene_tag AND base_scene=:base_scene
            AND COALESCE(morphology,'') = COALESCE(:morphology,'')
            AND marketplace_id=:marketplace_id
            AND week=:week AND sunday=:sunday
            AND my_asin=:my_asin
          ORDER BY {order_metric} DESC NULLS LAST
          LIMIT 3
        """
        try:
            top_each = self._fetch_all(sql_each, ctx)
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage2_traffic_pairs_unavailable asin=%s lag_type=%s error=%s",
                ctx.get("my_asin"),
                lag_type,
                exc,
            )
            top_each = []
        if top_each:
            self._attach_page_entity_details(ctx, top_each)
            if lag_type == "keyword":
                self._attach_keyword_details(ctx, top_each)
        comparison_reasons = _build_comparison_reasons(lag_type, top_each)

        return {
            "channel": "traffic",
            "lag_type": lag_type,
            "opp_type": opp_type,
            "overview": overview,
            "top_opps": top_each,
            "top_opp_asins_csv": ",".join(
                [row.get("opp_asin") for row in top_each if row.get("opp_asin")][:8]
            ),
            "top_diff_reasons": comparison_reasons,
            "reason_code": f"{lag_type}_by_traffic_pairs",
            "prompt_hint": "从流量结构或关键词结构解释差异；不要重算，只引用 evidence_json 字段。",
        }

    def _merge_top_opps(
        self, bucket: MutableMapping[str, dict[str, Any]], lag_type: str, rows: Sequence[Mapping[str, Any]] | None
    ) -> None:
        if not rows:
            return
        for row in rows:
            opp_asin = row.get("opp_asin")
            if not opp_asin:
                continue
            impact = _compute_top_impact(lag_type, row)
            current = bucket.get(opp_asin)
            if not current or impact > current.get("impact", float("-inf")):
                bucket[opp_asin] = {"impact": impact, "row": dict(row)}

    def _finalise_top_opps(self, bucket: Mapping[str, dict[str, Any]]) -> list[Mapping[str, Any]]:
        if not bucket:
            return []
        ordered = sorted(bucket.values(), key=lambda item: item.get("impact", 0.0), reverse=True)
        return [item.get("row", {}) for item in ordered]


_OVERVIEW_METRIC_CONFIG: Mapping[str, Mapping[str, Sequence[tuple[str, str]]]] = {
    "price": {
        "leader": (("price_gap_leader", "与领先者价差"),),
        "median": (("price_index_med", "相对中位数价格指数"), ("price_z", "价格Z分")),
    },
    "rank": {
        "leader": (("rank_pos_pct", "领先者排名百分位"),),
        "median": (("pressure", "竞争压力"), ("intensity_band", "竞争强度档位")),
    },
    "content": {
        "leader": (("content_gap", "内容差距"),),
        "median": (("pressure", "竞争压力"), ("intensity_band", "竞争强度档位")),
    },
    "social": {
        "leader": (("social_gap", "社交口碑差距"),),
        "median": (("pressure", "竞争压力"), ("intensity_band", "竞争强度档位")),
    },
    "badge": {
        "leader": (("badge_diff", "权益标识差值"),),
        "median": (("badge_delta_sum", "权益差异累计"),),
    },
    "confidence": {
        "leader": (("confidence", "置信度"),),
        "median": (("pressure", "竞争压力"),),
    },
    "traffic_mix": {
        "leader": (
            ("ad_ratio_gap_leader", "广告占比差距"),
            ("ad_to_natural_gap", "广告对自然流量差距"),
        ),
        "median": (
            ("ad_ratio_index_med", "广告占比指数"),
            ("t_pressure", "流量竞争压力"),
            ("t_intensity_band", "流量竞争强度档位"),
        ),
    },
    "keyword": {
        "leader": (
            ("kw_top3_share_gap", "Top3 关键词份额差距"),
            ("kw_competitor_share_gap", "竞品关键词份额差距"),
        ),
        "median": (
            ("kw_entropy_gap", "关键词覆盖度差距"),
            ("kw_hhi_gap", "关键词集中度差距"),
            ("t_pressure", "流量竞争压力"),
        ),
    },
}


_COMPARISON_METRIC_KEYS: Mapping[str, Sequence[str]] = {
    "price": ("price_gap_each", "price_ratio_each", "score"),
    "rank": ("rank_pos_delta", "score"),
    "content": ("content_gap_each", "score"),
    "social": ("social_gap_each", "score"),
    "badge": ("badge_delta_sum", "score"),
    "confidence": ("confidence", "score"),
    "traffic_mix": (
        "ad_ratio_gap_each",
        "ad_to_natural_gap_each",
        "my_ad_ratio",
        "opp_ad_ratio",
        "my_nf_ratio",
        "opp_nf_ratio",
        "t_score_mix",
        "t_pressure",
    ),
    "keyword": (
        "my_kw_top3_share_7d_avg",
        "opp_kw_top3_share_7d_avg",
        "my_kw_brand_share_7d_avg",
        "opp_kw_brand_share_7d_avg",
        "my_kw_competitor_share_7d_avg",
        "opp_kw_competitor_share_7d_avg",
        "t_score_kw",
        "t_pressure",
    ),
}


def _severity_from_rank(rank: int) -> str:
    if rank >= _SEVERITY_ORDER["high"]:
        return "high"
    if rank >= _SEVERITY_ORDER["mid"]:
        return "mid"
    return "low"


def _stage2_cache_key(context: Mapping[str, Any], lag_type: str, opp_type: str, channel: str) -> tuple[Any, ...]:
    return (
        channel,
        lag_type,
        opp_type,
        context.get("scene_tag"),
        context.get("base_scene"),
        context.get("morphology"),
        context.get("marketplace_id"),
        context.get("week"),
        context.get("sunday"),
        context.get("my_asin"),
    )


def _compare(value: float, threshold: float, comparator: str) -> bool:
    func = _COMPARATOR_FUNCS.get(comparator)
    if func is None:
        return False
    return func(value, threshold)


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
    return None


def _compute_top_impact(lag_type: str, row: Mapping[str, Any]) -> float:
    score = _coerce_float(row.get("score"))
    if score is not None:
        return score
    metric_key = None
    if lag_type == "price":
        metric_key = "price_gap_each"
    elif lag_type == "rank":
        metric_key = "rank_pos_delta"
    elif lag_type == "content":
        metric_key = "content_gap_each"
    elif lag_type == "social":
        metric_key = "social_gap_each"
    elif lag_type == "badge":
        metric_key = "badge_delta_sum"
    elif lag_type == "confidence":
        metric_key = "confidence"
    elif lag_type == "traffic_mix":
        metric_key = "ad_to_natural_gap_each"
    elif lag_type == "keyword":
        my_share = _coerce_float(row.get("my_kw_top3_share_7d_avg"))
        opp_share = _coerce_float(row.get("opp_kw_top3_share_7d_avg"))
        if my_share is not None and opp_share is not None:
            return abs(my_share - opp_share)
    if metric_key:
        metric_value = _coerce_float(row.get(metric_key))
        if metric_value is not None:
            return abs(metric_value)
    return 0.0


def _parse_json_path_tokens(path: str) -> list[str]:
    cleaned = str(path or "").strip()
    if not cleaned:
        return []
    if cleaned.startswith("$"):
        cleaned = cleaned[1:]
    tokens: list[str] = []
    buffer = cleaned
    while buffer:
        if buffer.startswith("."):
            buffer = buffer[1:]
            continue
        bracket_index = buffer.find("[")
        dot_index = buffer.find(".")
        if bracket_index == -1 and dot_index == -1:
            tokens.append(buffer)
            break
        if bracket_index != -1 and (dot_index == -1 or bracket_index < dot_index):
            if bracket_index > 0:
                tokens.append(buffer[:bracket_index])
            buffer = buffer[bracket_index:]
            match = re.match(r"\[(\d+)\](.*)", buffer)
            if match:
                tokens.append(match.group(1))
                buffer = match.group(2)
            else:
                break
        else:
            tokens.append(buffer[:dot_index])
            buffer = buffer[dot_index:]
    return [token for token in tokens if token]


def _extract_json_path(obj: Any, tokens: Sequence[str]) -> Any:
    current: Any = obj
    for token in tokens:
        if isinstance(current, Mapping):
            current = current.get(token)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            try:
                index = int(token)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _compute_pair_gap(my_value: Any, opp_value: Any) -> float:
    my_num = _coerce_float(my_value)
    opp_num = _coerce_float(opp_value)
    if my_num is None or opp_num is None:
        return 0.0
    return abs(my_num - opp_num)


def _format_integer_text(value: Any) -> str:
    number = _coerce_float(value)
    if number is None:
        return "缺失"
    return f"{int(round(number)):,}"


def _format_decimal_text(value: Any, decimals: int = 2) -> str:
    number = _coerce_float(value)
    if number is None:
        return "缺失"
    format_spec = f"{{:.{decimals}f}}"
    return format_spec.format(number)


def _format_percentage_text(value: Any) -> str:
    number = _coerce_float(value)
    if number is None:
        return "缺失"
    return f"{number * 100:.2f}%"


def _format_rank_leaf_note(my_value: Any, opp_value: Any, _: Mapping[str, Any]) -> str:
    my_text = _format_integer_text(my_value)
    opp_text = _format_integer_text(opp_value)
    my_num = _coerce_float(my_value)
    opp_num = _coerce_float(opp_value)
    gap_text = ""
    if my_num is not None and opp_num is not None:
        gap = my_num - opp_num
        if abs(gap) >= 1:
            direction = "落后" if gap > 0 else "领先"
            gap_text = f"，{direction} {abs(int(round(gap)))} 位"
    return f"类目排名：我方 {my_text}，对手 {opp_text}{gap_text}"


def _format_rank_pct_note(my_value: Any, opp_value: Any, _: Mapping[str, Any]) -> str:
    my_text = _format_percentage_text(my_value)
    opp_text = _format_percentage_text(opp_value)
    my_num = _coerce_float(my_value)
    opp_num = _coerce_float(opp_value)
    gap_text = ""
    if my_num is not None and opp_num is not None:
        gap = my_num - opp_num
        if abs(gap) >= 0.005:
            direction = "落后" if gap > 0 else "领先"
            gap_text = f"，{direction} {abs(gap) * 100:.2f}%"
    return f"排名百分位：我方 {my_text}，对手 {opp_text}{gap_text}"


def _format_image_note(my_value: Any, opp_value: Any, _: Mapping[str, Any]) -> str:
    return _format_count_note("图片数", my_value, opp_value, "张")


def _format_video_note(my_value: Any, opp_value: Any, _: Mapping[str, Any]) -> str:
    return _format_count_note("视频数", my_value, opp_value, "个")


def _format_count_note(label: str, my_value: Any, opp_value: Any, unit: str) -> str:
    my_text = _format_integer_text(my_value)
    opp_text = _format_integer_text(opp_value)
    my_num = _coerce_float(my_value)
    opp_num = _coerce_float(opp_value)
    gap_text = ""
    if my_num is not None and opp_num is not None:
        gap = opp_num - my_num
        if abs(gap) >= 1:
            if gap > 0:
                gap_text = f"，缺少 {int(round(abs(gap)))}{unit}"
            else:
                gap_text = f"，多出 {int(round(abs(gap)))}{unit}"
    return f"{label}：我方 {my_text}，对手 {opp_text}{gap_text}"


def _format_content_score_note(
    my_value: Any, opp_value: Any, _: Mapping[str, Any]
) -> str:
    my_text = _format_decimal_text(my_value)
    opp_text = _format_decimal_text(opp_value)
    gap = _compute_pair_gap(my_value, opp_value)
    gap_text = f"，差距 {gap:.2f}" if gap >= 0.01 else ""
    return f"内容得分：我方 {my_text}，对手 {opp_text}{gap_text}"


def _format_rating_note(my_value: Any, opp_value: Any, _: Mapping[str, Any]) -> str:
    my_text = _format_decimal_text(my_value)
    opp_text = _format_decimal_text(opp_value)
    gap = _compute_pair_gap(my_value, opp_value)
    gap_text = f"，相差 {gap:.2f}" if gap >= 0.01 else ""
    return f"评分：我方 {my_text}，对手 {opp_text}{gap_text}"


def _format_reviews_note(my_value: Any, opp_value: Any, _: Mapping[str, Any]) -> str:
    note = _format_count_note("评论数", my_value, opp_value, "条")
    return note


def _format_social_proof_note(
    my_value: Any, opp_value: Any, _: Mapping[str, Any]
) -> str:
    my_text = _format_decimal_text(my_value)
    opp_text = _format_decimal_text(opp_value)
    gap = _compute_pair_gap(my_value, opp_value)
    gap_text = f"，差距 {gap:.2f}" if gap >= 0.01 else ""
    return f"社交证明：我方 {my_text}，对手 {opp_text}{gap_text}"


def _format_keyword_note(
    keyword: str,
    my_rank: Any,
    opp_rank: Any,
    my_share: Any,
    opp_share: Any,
    tag: Any,
) -> str:
    my_rank_text = f"TOP{int(my_rank)}" if my_rank is not None else "无"
    opp_rank_text = f"TOP{int(opp_rank)}" if opp_rank is not None else "缺失"
    my_share_text = _format_percentage_text(my_share)
    opp_share_text = _format_percentage_text(opp_share)
    keyword_label = str(keyword)
    if tag:
        keyword_label = f"{keyword_label}（{tag}）"
    return (
        f"关键词 {keyword_label}：我方{my_rank_text}，对手{opp_rank_text}；"
        f"7天份额 我方{my_share_text} 对手{opp_share_text}"
    )


def _compose_overview_summary(
    overview: Mapping[str, Any], metrics: Sequence[tuple[str, str]]
) -> Mapping[str, Any]:
    metrics_payload: dict[str, Any] = {}
    parts: list[str] = []
    for column, label in metrics:
        if column in overview:
            value = overview.get(column)
            metrics_payload[column] = value
            if value is not None and value != "":
                parts.append(f"{label}{_format_metric_value(value)}")
    summary_text = "、".join(parts) if parts else "暂无数据"
    return {"summary": summary_text, "metrics": metrics_payload}


def _build_comparison_reasons(
    lag_type: str, top_rows: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    reasons: list[Mapping[str, Any]] = []
    if not top_rows:
        return reasons
    keys = _COMPARISON_METRIC_KEYS.get(lag_type, ())
    for row in top_rows:
        metrics = {key: row.get(key) for key in keys if key in row}
        if "score" in row and "score" not in metrics:
            metrics["score"] = row.get("score")
        reasons.append(
            {
                "opp_asin": row.get("opp_asin"),
                "opp_parent_asin": row.get("opp_parent_asin"),
                "summary": _compose_reason_sentence(lag_type, row),
                "metrics": metrics,
            }
        )
    return reasons


def _compose_reason_sentence(lag_type: str, row: Mapping[str, Any]) -> str:
    label = _format_opp_label(row)
    if lag_type == "price":
        gap = _format_metric_value(row.get("price_gap_each"))
        ratio = _format_metric_value(row.get("price_ratio_each"))
        return f"竞品 {label} 的价格差为 {gap}，我方/竞品价格比 {ratio}，价格优势不足。"
    if lag_type == "rank":
        delta = _format_metric_value(row.get("rank_pos_delta"))
        return f"竞品 {label} 的排名领先差值 {delta}，我方自然位次需提升。"
    if lag_type == "content":
        gap = _format_metric_value(row.get("content_gap_each"))
        return f"竞品 {label} 的内容表现领先 {gap}，需要优化前台素材。"
    if lag_type == "social":
        gap = _format_metric_value(row.get("social_gap_each"))
        return f"竞品 {label} 的社交口碑差距为 {gap}，需加强评价与评分建设。"
    if lag_type == "badge":
        diff = _format_metric_value(row.get("badge_delta_sum"))
        return f"竞品 {label} 拥有更多权益标识差值 {diff}，需补齐权益标签。"
    if lag_type == "confidence":
        confidence = _format_metric_value(row.get("confidence"))
        return f"竞品 {label} 的信号置信度 {confidence} 更高，需补充可靠证据。"
    if lag_type == "traffic_mix":
        ad_gap = _format_metric_value(row.get("ad_ratio_gap_each"))
        mix_gap = _format_metric_value(row.get("ad_to_natural_gap_each"))
        return (
            f"竞品 {label} 的广告占比差距 {ad_gap}，广告/自然流量组合差距 {mix_gap}，流量结构劣势明显。"
        )
    if lag_type == "keyword":
        my_share = _format_metric_value(row.get("my_kw_top3_share_7d_avg"))
        opp_share = _format_metric_value(row.get("opp_kw_top3_share_7d_avg"))
        return f"竞品 {label} 的Top3关键词份额为 {opp_share}，我方仅 {my_share}，关键词覆盖不足。"
    score = _format_metric_value(row.get("score"))
    return f"竞品 {label} 的差距评分为 {score}，需结合明细进一步分析。"


def _format_metric_value(value: Any) -> str:
    if value is None or value == "":
        return "缺失"
    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            return "是" if value else "否"
        abs_value = abs(float(value))
        if abs_value >= 100 or abs_value == 0:
            return f"{value:.0f}" if abs_value >= 100 else f"{value:.2f}"
        if abs_value >= 1:
            return f"{value:.2f}"
        return f"{value:.2%}"
    return str(value)


def _format_opp_label(row: Mapping[str, Any]) -> str:
    brand = row.get("opp_brand")
    opp_parent = row.get("opp_parent_asin") or ""
    opp_asin = row.get("opp_asin") or "未知ASIN"
    parts: list[str] = []
    if brand:
        parts.append(str(brand))
    if opp_parent and opp_parent != opp_asin:
        parts.append(str(opp_parent))
    parts.append(str(opp_asin))
    return " ".join(part for part in parts if part)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)


def _serialise_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


__all__ = [
    "CompetitionLLMOrchestrator",
    "CompetitionRunResult",
    "StageOneLLMResult",
    "StageOneResult",
    "StageTwoAggregateResult",
]
