"""Stage-1 and Stage-2 orchestration for the competition LLM workflow."""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml
from sqlalchemy import text
from sqlalchemy.engine import Engine

from scpc.llm.orchestrator import LLMOrchestrator, LLMRunConfig, validate_schema
from scpc.prompts import load_prompt
from scpc.schemas import load_schema

from .competition_config import CompetitionLLMConfig

LOGGER = logging.getLogger(__name__)


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
        self._stage2_schema = load_schema("competition_stage2_aggregate.schema.json")
        self._stage2_prompt = load_prompt("competition_stage2_aggregate.md")
        self._current_marketplace_id: str | None = None

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
        if run_stage1:
            stage1_inputs = self._collect_stage1_inputs(target_week, marketplace_id)
            stage1_outputs = self._execute_stage1_code(stage1_inputs)
            for item in stage1_outputs:
                storage_paths.append(self._write_stage1_output(item))
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
            if not isinstance(machine_json.get("context"), Mapping):
                machine_json = dict(machine_json)
                machine_json["context"] = facts.get("context", {})
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

        for candidate in group.items:
            dimension = candidate.dimension
            lag_type = _normalize_lag_type(dimension.get("lag_type"))
            opp_type = str(candidate.context.get("opp_type") or dimension.get("source_opp_type") or "").lower()
            if not lag_type or not opp_type:
                LOGGER.debug("competition_llm.stage2_skip_invalid_dimension dimension=%s", dimension)
                continue
            severity_label = str(dimension.get("severity") or "low").lower()
            severity_rank = _SEVERITY_ORDER.get(severity_label, 1)
            opp_types_seen.add(opp_type)
            confidence_value = self._parse_confidence(dimension.get("source_confidence")) or self._config.stage_1.conf_min
            lag_score = _coerce_float(dimension.get("lag_score"))

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
                    "opp_types": set(),
                    "severity_rank": 0,
                    "source_confidence": self._config.stage_1.conf_min,
                    "overview": {},
                    "_top_map": {},
                    "lag_scores": [],
                    "triggered_rules": set(),
                },
            )
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
                "lag_type": lag_type,
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
        return facts

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

    def _validate_stage2_machine_json(self, payload: Mapping[str, Any]) -> None:
        validate_schema(self._stage2_schema, payload)
        actions = payload.get("recommended_actions", [])
        for action in actions:
            code = action.get("action_code")
            if self._config.stage_2.allowed_action_codes and code not in self._config.stage_2.allowed_action_codes:
                raise ValueError(f"Action code {code!r} is not permitted")
        root_causes = payload.get("root_causes", [])
        for cause in root_causes:
            code = cause.get("root_cause_code")
            if self._config.stage_2.allowed_root_cause_codes and code not in self._config.stage_2.allowed_root_cause_codes:
                raise ValueError(f"Root cause code {code!r} is not permitted")

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
        overview = self._fetch_one(sql_overview, ctx)
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

        select_columns = [
            "opp_asin",
            "opp_parent_asin",
            "price_gap_each",
            "price_ratio_each",
            "rank_pos_delta",
            "content_gap_each",
            "social_gap_each",
            "badge_delta_sum",
        ]
        if lag_type == "confidence":
            select_columns.append("confidence")
        select_columns.append(f"{score_col} AS score")
        select_clause = ",\n                 ".join(select_columns)
        sql_each = f"""
          SELECT {select_clause}
          FROM bi_amz_comp_pairs_each
          WHERE scene_tag=:scene_tag AND base_scene=:base_scene
            AND COALESCE(morphology,'') = COALESCE(:morphology,'')
            AND marketplace_id=:marketplace_id
            AND week=:week AND sunday=:sunday
            AND my_asin=:my_asin
          ORDER BY ABS({metric_col}) DESC NULLS LAST
          LIMIT 3
        """
        top_each = self._fetch_all(sql_each, ctx)
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
        overview = self._fetch_one(sql_overview, ctx)
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
        top_each = self._fetch_all(sql_each, ctx)
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
    opp_parent = row.get("opp_parent_asin") or ""
    opp_asin = row.get("opp_asin") or "未知ASIN"
    if opp_parent and opp_parent != opp_asin:
        return f"{opp_parent} {opp_asin}"
    return str(opp_asin)


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
    "StageOneResult",
    "StageTwoAggregateResult",
]
