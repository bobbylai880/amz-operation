"""Stage-1 and Stage-2 orchestration for the competition LLM workflow."""
from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, MutableMapping, Sequence

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

_STAGE3_CURRENT_WEEK_SUNDAY_SQL_BASE = """
SELECT sunday
FROM bi_amz_comp_pairs
WHERE week = :week
{marketplace_filter}
ORDER BY sunday DESC
LIMIT 1
"""

_STAGE3_PREVIOUS_WEEK_SQL_BASE = """
SELECT week, sunday
FROM bi_amz_comp_pairs
WHERE sunday < :sunday
{marketplace_filter}
ORDER BY sunday DESC
LIMIT 1
"""

_STAGE3_DELTA_SQL_BASE = """
SELECT
  scene_tag,
  base_scene,
  morphology,
  marketplace_id,
  window_id,
  my_asin,
  opp_type,
  week_w0,
  week_w1,
  my_parent_asin,
  d_price_net,
  d_rank_score,
  d_social_proof,
  d_content_score,
  badge_change,
  d_price_gap_leader,
  d_price_index_med,
  d_rank_pos_pct,
  d_content_gap,
  d_social_gap,
  delta_pressure
FROM bi_amz_comp_delta
WHERE week_w0 = :week_w0
  AND week_w1 = :week_w1
{marketplace_filter}
"""

_STAGE3_PAIRS_SQL_BASE = """
SELECT
  scene_tag,
  base_scene,
  morphology,
  marketplace_id,
  week,
  sunday,
  my_asin,
  opp_type,
  my_parent_asin,
  opp_asin,
  price_gap_leader,
  price_index_med,
  rank_pos_pct,
  content_gap,
  social_gap,
  badge_delta_sum,
  pressure
FROM bi_amz_comp_pairs
WHERE week IN (:week_w0, :week_w1)
{marketplace_filter}
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
_PAIRWISE_ONLY_LAG_TYPES = {"rank", "content", "social", "keyword"}

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

_DEFAULT_DIRECTION_TOLERANCE = 1e-6
_DIRECTION_KEYWORDS = {
    "lower_better": "越低越好",
    "higher_better": "越高越好",
}
_METRIC_DIRECTION_MAP: Mapping[str, str] = {
    "rank": "lower_better",
    "rank_leaf": "lower_better",
    "rank_root": "lower_better",
    "rank_pos_pct": "lower_better",
    "rank_pos_delta": "lower_better",
    "rank_delta": "lower_better",
    "rank_score": "higher_better",
    "content_score": "higher_better",
    "price_net": "lower_better",
}

_ABSOLUTE_RANK_METRICS: frozenset[str] = frozenset({"rank_leaf", "rank_root"})

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
    direction: str | None = None


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


Channel = Literal["page", "traffic"]


@dataclass(slots=True)
class StageThreeChangeRecord:
    scene_context: Mapping[str, Any]
    week_w0: str
    week_w1: str
    window_id: str
    channel: Channel
    my_asin: str
    my_parent_asin: str | None
    opp_type: str
    leader_asin_w0: str | None
    leader_asin_w1: str | None
    leader_changed: bool | None
    lag_type: str
    metric: str
    change_type: str
    direction: str | None
    value_w0: float | None
    value_w1: float | None
    delta_value: float | None
    change_summary: str
    change_payload: Mapping[str, Any]


@dataclass(slots=True)
class StageThreeDimensionChange:
    scene_context: Mapping[str, Any]
    week_w0: str
    week_w1: str
    channel: Channel
    lag_type: str
    total_changes: int
    total_changes_leader: int
    total_changes_median: int
    records: Sequence[StageThreeChangeRecord]


@dataclass(slots=True)
class StageThreeResult:
    context: Mapping[str, Any]
    page_dimensions: Sequence[StageThreeDimensionChange]
    traffic_dimensions: Sequence[StageThreeDimensionChange]


_PAGE_METRIC_TO_LAG_TYPE: Mapping[str, str] = {
    "d_price_net": "price",
    "d_price_gap_leader": "price",
    "d_price_index_med": "price",
    "d_rank_score": "rank",
    "d_rank_pos_pct": "rank",
    "d_content_score": "content",
    "d_content_gap": "content",
    "d_social_proof": "social",
    "d_social_gap": "social",
    "badge_change": "badge",
    "delta_pressure": "confidence",
}

_PAGE_METRIC_TO_CHANGE_TYPE: Mapping[str, str] = {
    "d_price_net": "self_price_change",
    "d_price_gap_leader": "gap_price_vs_leader_change",
    "d_price_index_med": "price_index_med_change",
    "d_rank_score": "self_rank_score_change",
    "d_rank_pos_pct": "rank_pos_pct_change",
    "d_content_score": "self_content_score_change",
    "d_content_gap": "content_gap_change",
    "d_social_proof": "self_social_proof_change",
    "d_social_gap": "social_gap_change",
    "badge_change": "badge_count_change",
    "delta_pressure": "pressure_change",
}

_PAGE_METRIC_TO_PAIR_FIELD: Mapping[str, str] = {
    "d_price_gap_leader": "price_gap_leader",
    "d_price_index_med": "price_index_med",
    "d_rank_pos_pct": "rank_pos_pct",
    "d_content_gap": "content_gap",
    "d_social_gap": "social_gap",
    "badge_change": "badge_delta_sum",
    "delta_pressure": "pressure",
}

_STAGE3_PAGE_CHANNEL: Channel = "page"


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
        env_flag = os.getenv("COMP_RANK_FIX_ENABLED")
        if env_flag is None:
            self._rank_fix_enabled = bool(
                getattr(self._config.stage_2, "rank_direction_fix_enabled", False)
            )
        else:
            self._rank_fix_enabled = env_flag.strip().lower() in {"1", "true", "yes", "on"}
        tolerance_config = getattr(
            self._config.stage_2, "direction_tolerance", _DEFAULT_DIRECTION_TOLERANCE
        )
        try:
            tolerance_value = float(tolerance_config)
        except (TypeError, ValueError):
            tolerance_value = _DEFAULT_DIRECTION_TOLERANCE
        if tolerance_value < 0:
            tolerance_value = 0.0
        self._direction_tolerance = tolerance_value if tolerance_value is not None else _DEFAULT_DIRECTION_TOLERANCE
        if self._direction_tolerance < 0:
            self._direction_tolerance = 0.0
        self._rank_fix_counters: dict[str, int] = {
            "total": 0,
            "corrected": 0,
            "dropped": 0,
            "evidence_dropped": 0,
        }

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

    def run_stage3(
        self,
        week: str | None,
        *,
        marketplace_id: str | None = None,
    ) -> Sequence[StageThreeResult]:
        """Compute Stage-3 structured change facts without invoking the LLM."""

        if not getattr(self._config, "stage_3", None) or not self._config.stage_3.enabled:
            LOGGER.info("competition_llm.stage3_disabled")
            return ()

        target_week = self._resolve_week(week, marketplace_id)
        prev_week = self._resolve_previous_week(target_week, marketplace_id)
        if not prev_week:
            LOGGER.warning(
                "competition_llm.stage3_prev_week_missing week=%s marketplace_id=%s",
                target_week,
                marketplace_id,
            )
            return ()

        delta_rows = self._fetch_stage3_delta_rows(target_week, prev_week, marketplace_id)
        if not delta_rows:
            LOGGER.debug(
                "competition_llm.stage3_no_delta week_w0=%s week_w1=%s marketplace_id=%s",
                target_week,
                prev_week,
                marketplace_id,
            )
            return ()

        pairs_w0, pairs_w1 = self._fetch_stage3_pairs(target_week, prev_week, marketplace_id)
        config = self._config.stage_3
        all_records: list[StageThreeChangeRecord] = []
        for row in delta_rows:
            records = self._build_stage3_change_records(row, pairs_w0, pairs_w1, config)
            if records:
                all_records.extend(records)

        if not all_records:
            LOGGER.debug(
                "competition_llm.stage3_no_records week_w0=%s week_w1=%s marketplace_id=%s",
                target_week,
                prev_week,
                marketplace_id,
            )
            return ()

        results = self._assemble_stage3_results(all_records, config.max_records_per_dimension)

        storage_paths: list[Path] = []
        for result in results:
            try:
                storage_paths.append(self._write_stage3_output(result))
            except OSError as exc:  # pragma: no cover - unexpected filesystem issue
                LOGGER.warning(
                    "competition_llm.stage3_write_failed scene=%s error=%s",
                    result.context,
                    exc,
                )

        LOGGER.info(
            "competition_llm.stage3_completed week_w0=%s week_w1=%s scenes=%s records=%s",
            target_week,
            prev_week,
            len(results),
            len(all_records),
        )

        return tuple(results)

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
            direction_registry: defaultdict[str, dict[str, str]] = defaultdict(dict)

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
                            direction_value = rule.direction or _resolve_direction(rule.metric, rule.lag_type)
                            if direction_value:
                                direction_registry[rule.lag_type][rule.metric] = direction_value
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
                metric_dir = direction_registry.get(lag_type)
                if metric_dir:
                    payload["metric_directions"] = {
                        metric: metric_dir[metric]
                        for metric in sorted(metric_dir)
                    }
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
            human_markdown_raw = llm_response.get("human_markdown")
            if not isinstance(machine_json, Mapping):
                raise ValueError("Stage-2 machine_json missing or invalid")
            machine_json = dict(machine_json)
            if not isinstance(machine_json.get("context"), Mapping):
                machine_json["context"] = facts.get("context", {})
            machine_json = self._materialize_evidence(machine_json, facts)
            if self._rank_fix_enabled:
                self._fix_directional_metrics(machine_json, facts=facts)
            self._validate_stage2_machine_json(machine_json)
            llm_markdown = human_markdown_raw if isinstance(human_markdown_raw, str) else None
            if not isinstance(human_markdown_raw, str):
                LOGGER.debug(
                    "competition_llm.stage2_markdown_non_string type=%s",
                    type(human_markdown_raw).__name__,
                )
            human_markdown = self._render_stage2_diff_markdown(
                machine_json, facts, llm_markdown=llm_markdown
            )
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
            metric_dirs_raw = dimension.get("metric_directions")
            metric_dirs = _normalise_direction_map(metric_dirs_raw) if metric_dirs_raw else {}
            if metric_dirs:
                entry.setdefault("metric_directions", {}).update(metric_dirs)

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
            if entry.get("metric_directions"):
                lag_item["metric_directions"] = {
                    metric: entry["metric_directions"][metric]
                    for metric in sorted(entry["metric_directions"])
                }
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

        return (main_path,)

    def _render_stage2_diff_markdown(
        self,
        machine_json: Mapping[str, Any],
        facts: Mapping[str, Any] | None,
        *,
        llm_markdown: str | None = None,
    ) -> str:
        root_causes = machine_json.get("root_causes")
        if not isinstance(root_causes, Sequence) or not root_causes:
            return "# 差异点与对比对象\n\n- 本周未发现对我不利的客观差异\n"

        row_index = self._build_evidence_row_index(facts or {})
        lines: list[str] = ["# 差异点与对比对象", ""]

        lines.append("## 落后维度概览")
        overview_lines: list[str] = []
        for cause in root_causes:
            if not isinstance(cause, Mapping):
                continue
            title = str(
                cause.get("summary")
                or cause.get("root_cause_code")
                or cause.get("lag_dimension")
                or "差异"
            ).strip()
            if title:
                overview_lines.append(f"- {title}")
        if overview_lines:
            lines.extend(overview_lines)
        else:
            lines.append("- 暂无维度")
        lines.append("")

        lines.append("## Top 对手差距")
        top_rows = self._collect_top_opponents(facts or {}, limit=3)
        if top_rows:
            for row in top_rows:
                lines.append(f"- {_format_opp_label(row)}")
        else:
            lines.append("- 暂无对手差距")
        lines.append("")

        lines.append("## 客观证据明细")
        lines.append("")

        for cause in root_causes:
            if not isinstance(cause, Mapping):
                continue
            title = str(
                cause.get("summary")
                or cause.get("root_cause_code")
                or cause.get("lag_dimension")
                or "差异"
            ).strip() or "差异"
            lines.append(f"### {title}")
            evidence_items = cause.get("evidence")
            formatted: list[str] = []
            if isinstance(evidence_items, Sequence):
                for item in evidence_items:
                    if not isinstance(item, Mapping):
                        continue
                    text = self._format_evidence_markdown_line(item, row_index)
                    if text:
                        formatted.append(text)
            if not formatted:
                lines.append("- 暂无证据")
            else:
                for text in formatted:
                    lines.append(f"- {text}")
            lines.append("")

        diagnostics = machine_json.get("diagnostics")
        actions_section: Sequence[Mapping[str, Any]] | None = None
        if isinstance(diagnostics, Mapping):
            validated = diagnostics.get("validated_actions")
            if isinstance(validated, Sequence):
                actions_section = [
                    action
                    for action in validated
                    if isinstance(action, Mapping)
                ]

        action_lines: list[str] = []
        if actions_section:
            for action in actions_section:
                text = (
                    str(action.get("how") or action.get("why") or action.get("expected_impact") or action.get("code") or "")
                ).strip()
                if not text:
                    continue
                if not text.startswith("-"):
                    text = f"- {text}"
                action_lines.append(text)
        if llm_markdown and llm_markdown.strip():
            for raw_line in llm_markdown.strip().splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                if not stripped.startswith("-"):
                    stripped = f"- {stripped}"
                action_lines.append(stripped)
        if action_lines:
            lines.append("## 建议动作")
            lines.extend(action_lines)
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _build_evidence_row_index(
        self, facts: Mapping[str, Any]
    ) -> Mapping[str, Mapping[str, Any]]:
        if not isinstance(facts, Mapping):
            return {}
        lag_items = facts.get("lag_items")
        if not isinstance(lag_items, Sequence):
            return {}
        index: dict[str, Mapping[str, Any]] = {}
        for item in lag_items:
            if not isinstance(item, Mapping):
                continue
            top_opps = item.get("top_opps")
            if not isinstance(top_opps, Sequence):
                continue
            for row in top_opps:
                if not isinstance(row, Mapping):
                    continue
                opp_asin = str(row.get("opp_asin") or "").strip()
                if not opp_asin:
                    continue
                index.setdefault(opp_asin, row)
        return index

    def _collect_top_opponents(
        self, facts: Mapping[str, Any], *, limit: int = 3
    ) -> list[Mapping[str, Any]]:
        if not isinstance(facts, Mapping):
            return []
        lag_items = facts.get("lag_items")
        if not isinstance(lag_items, Sequence):
            return []
        results: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for item in lag_items:
            if not isinstance(item, Mapping):
                continue
            top_opps = item.get("top_opps")
            if not isinstance(top_opps, Sequence):
                continue
            for row in top_opps:
                if not isinstance(row, Mapping):
                    continue
                opp_asin = str(row.get("opp_asin") or "").strip()
                if not opp_asin or opp_asin in seen:
                    continue
                seen.add(opp_asin)
                results.append(row)
                if len(results) >= limit:
                    return results
        return results

    def _format_evidence_markdown_line(
        self,
        evidence: Mapping[str, Any],
        row_index: Mapping[str, Mapping[str, Any]],
    ) -> str | None:
        metric = str(evidence.get("metric") or "").strip()
        against = str(evidence.get("against") or "").strip().lower()
        my_value = evidence.get("my_value")
        opp_value = evidence.get("opp_value")
        if not metric or my_value is None or opp_value is None:
            return None

        direction = _normalize_direction(evidence.get("direction"))
        delta_value = evidence.get("delta")
        worse_flag = evidence.get("worse")

        if against != "asin":
            if against == "leader":
                label = "领先者"
            elif against == "median":
                label = "中位数"
            else:
                label = "对手"
            if direction:
                direction_text = _DIRECTION_KEYWORDS.get(direction)
                metric_label = f"{metric}（{direction_text}）" if direction_text else metric
                delta_formatted = _format_signed_value(delta_value if delta_value is not None else _compute_delta(my_value, opp_value))
                conclusion = "更差" if worse_flag is True else ("更好" if worse_flag is False else "待确认")
                my_text = _format_metric_value(my_value)
                opp_text = _format_metric_value(opp_value)
                return f"{metric_label}：我方 {my_text} / {label} {opp_text}，差值 {delta_formatted} → {conclusion}"
            my_text = _format_metric_value(my_value)
            opp_text = _format_metric_value(opp_value)
            return f"{metric}：我方 {my_text} / {label} {opp_text}"

        opp_asin = str(evidence.get("opp_asin") or "").strip()
        base_row: Mapping[str, Any]
        row = row_index.get(opp_asin)
        if isinstance(row, Mapping):
            temp_row = dict(row)
        else:
            temp_row = {}
        if opp_asin and "opp_asin" not in temp_row:
            temp_row["opp_asin"] = opp_asin
        base_row = temp_row

        note = evidence.get("note")
        if isinstance(note, str) and note.strip() and not direction:
            return note.strip()

        if direction:
            direction_text = _DIRECTION_KEYWORDS.get(direction)
            metric_label = f"{metric}（{direction_text}）" if direction_text else metric
            my_text = _format_metric_value(my_value)
            opp_text = _format_metric_value(opp_value)
            delta_metric = delta_value if delta_value is not None else _compute_delta(my_value, opp_value)
            delta_formatted = _format_signed_value(delta_metric)
            conclusion = "更差" if worse_flag is True else ("更好" if worse_flag is False else "待确认")
            label = _format_opponent_display_label(base_row)
            return f"{metric_label}：我方 {my_text} / 对手({label}) {opp_text}，差值 {delta_formatted} → {conclusion}"

        formatter_map: Mapping[str, Callable[[Any, Any, Mapping[str, Any]], str]] = {
            "rank_leaf": _format_rank_leaf_note,
            "rank_pos_pct": _format_rank_pct_note,
            "image_cnt": _format_image_note,
            "video_cnt": _format_video_note,
            "content_score": _format_content_score_note,
            "rating": _format_rating_note,
            "reviews": _format_reviews_note,
            "social_proof": _format_social_proof_note,
        }

        formatter = formatter_map.get(metric)
        if formatter:
            return formatter(my_value, opp_value, base_row)

        label = _format_opponent_display_label(base_row)
        my_text = _format_metric_value(my_value)
        opp_text = _format_metric_value(opp_value)
        return f"{metric}：我方 {my_text} / 对手({label}) {opp_text}"

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

    def _resolve_previous_week(self, week: str, marketplace_id: str | None) -> str | None:
        marketplace_filter = " AND marketplace_id = :marketplace_id" if marketplace_id else ""
        current_sql = _STAGE3_CURRENT_WEEK_SUNDAY_SQL_BASE.format(
            marketplace_filter=marketplace_filter
        )
        params: dict[str, Any] = {"week": week}
        if marketplace_id:
            params["marketplace_id"] = marketplace_id
        current_row = self._fetch_one(current_sql, params)
        sunday = current_row.get("sunday") if current_row else None
        if not sunday:
            LOGGER.warning(
                "competition_llm.stage3_current_week_missing week=%s marketplace_id=%s",
                week,
                marketplace_id,
            )
            return None

        prev_sql = _STAGE3_PREVIOUS_WEEK_SQL_BASE.format(marketplace_filter=marketplace_filter)
        prev_params: dict[str, Any] = {"sunday": sunday}
        if marketplace_id:
            prev_params["marketplace_id"] = marketplace_id
        prev_row = self._fetch_one(prev_sql, prev_params)
        if not prev_row or not prev_row.get("week"):
            return None
        prev_week = str(prev_row.get("week"))
        LOGGER.info(
            "competition_llm.stage3_prev_week_resolved week=%s prev_week=%s marketplace_id=%s",
            week,
            prev_week,
            marketplace_id,
        )
        return prev_week

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

    def _fetch_stage3_delta_rows(
        self, week_w0: str, week_w1: str, marketplace_id: str | None
    ) -> Sequence[Mapping[str, Any]]:
        marketplace_filter = " AND marketplace_id = :marketplace_id" if marketplace_id else ""
        sql = _STAGE3_DELTA_SQL_BASE.format(marketplace_filter=marketplace_filter)
        params: dict[str, Any] = {"week_w0": week_w0, "week_w1": week_w1}
        if marketplace_id:
            params["marketplace_id"] = marketplace_id
        return self._fetch_all(sql, params)

    def _fetch_stage3_pairs(
        self, week_w0: str, week_w1: str, marketplace_id: str | None
    ) -> tuple[Mapping[tuple[Any, ...], Mapping[str, Any]], Mapping[tuple[Any, ...], Mapping[str, Any]]]:
        marketplace_filter = " AND marketplace_id = :marketplace_id" if marketplace_id else ""
        sql = _STAGE3_PAIRS_SQL_BASE.format(marketplace_filter=marketplace_filter)
        params: dict[str, Any] = {"week_w0": week_w0, "week_w1": week_w1}
        if marketplace_id:
            params["marketplace_id"] = marketplace_id
        rows = self._fetch_all(sql, params)
        pairs_w0: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        pairs_w1: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for row in rows:
            scene_key = (
                str(row.get("scene_tag") or ""),
                str(row.get("base_scene") or ""),
                str(row.get("morphology") or ""),
                str(row.get("marketplace_id") or ""),
            )
            key = (scene_key, str(row.get("my_asin") or ""), str(row.get("opp_type") or ""))
            week_value = str(row.get("week") or "")
            if week_value == week_w0:
                pairs_w0[key] = row
            elif week_value == week_w1:
                pairs_w1[key] = row
        return pairs_w0, pairs_w1

    def _build_stage3_change_records(
        self,
        row_delta: Mapping[str, Any],
        pairs_w0: Mapping[tuple[Any, ...], Mapping[str, Any]],
        pairs_w1: Mapping[tuple[Any, ...], Mapping[str, Any]],
        config: "StageThreeConfig",
    ) -> Sequence[StageThreeChangeRecord]:
        scene_key = (
            str(row_delta.get("scene_tag") or ""),
            str(row_delta.get("base_scene") or ""),
            str(row_delta.get("morphology") or ""),
            str(row_delta.get("marketplace_id") or ""),
        )
        week_w0 = str(row_delta.get("week_w0") or "")
        week_w1 = str(row_delta.get("week_w1") or "")
        window_id = str(row_delta.get("window_id") or "")
        my_asin = str(row_delta.get("my_asin") or "")
        opp_type = str(row_delta.get("opp_type") or "")
        my_parent_asin = row_delta.get("my_parent_asin")

        key_pairs = (scene_key, str(row_delta.get("my_asin") or ""), opp_type)
        row_w0 = pairs_w0.get(key_pairs)
        row_w1 = pairs_w1.get(key_pairs)

        leader_asin_w0: str | None = None
        leader_asin_w1: str | None = None
        leader_changed: bool | None = None
        if opp_type.lower() == "leader":
            leader_asin_w0 = str(row_w0.get("opp_asin")) if row_w0 and row_w0.get("opp_asin") else None
            leader_asin_w1 = str(row_w1.get("opp_asin")) if row_w1 and row_w1.get("opp_asin") else None
            if leader_asin_w0 and leader_asin_w1:
                leader_changed = leader_asin_w0 != leader_asin_w1

        scene_context = {
            "scene_tag": scene_key[0],
            "base_scene": scene_key[1],
            "morphology": scene_key[2],
            "marketplace_id": scene_key[3],
        }

        records: list[StageThreeChangeRecord] = []
        for metric, lag_type in _PAGE_METRIC_TO_LAG_TYPE.items():
            delta_raw = row_delta.get(metric)
            delta_numeric = _coerce_float(delta_raw)
            if delta_numeric is None:
                continue
            if abs(delta_numeric) < config.delta_tolerance:
                continue

            change_type = _PAGE_METRIC_TO_CHANGE_TYPE.get(metric)
            if not change_type:
                continue

            direction: str | None
            if delta_numeric > 0:
                direction = "up"
            elif delta_numeric < 0:
                direction = "down"
            else:
                direction = "neutral"

            value_field = _PAGE_METRIC_TO_PAIR_FIELD.get(metric)
            value_w0 = _coerce_float(row_w0.get(value_field)) if row_w0 and value_field else None
            value_w1 = _coerce_float(row_w1.get(value_field)) if row_w1 and value_field else None

            change_summary = _build_stage3_change_summary(lag_type, metric, direction, delta_numeric)

            change_payload = {
                "metric": metric,
                "lag_type": lag_type,
                "delta": float(delta_numeric),
                "week_w0": week_w0,
                "week_w1": week_w1,
                "value_w0": value_w0,
                "value_w1": value_w1,
                "opp_type": opp_type,
                "leader_asin_w0": leader_asin_w0,
                "leader_asin_w1": leader_asin_w1,
                "window_id": window_id,
                "my_asin": my_asin,
            }
            if value_field:
                change_payload["value_field"] = value_field

            records.append(
                StageThreeChangeRecord(
                    scene_context=scene_context,
                    week_w0=week_w0,
                    week_w1=week_w1,
                    window_id=window_id,
                    channel=_STAGE3_PAGE_CHANNEL,
                    my_asin=my_asin,
                    my_parent_asin=my_parent_asin,
                    opp_type=opp_type,
                    leader_asin_w0=leader_asin_w0,
                    leader_asin_w1=leader_asin_w1,
                    leader_changed=leader_changed,
                    lag_type=lag_type,
                    metric=metric,
                    change_type=change_type,
                    direction=direction,
                    value_w0=value_w0,
                    value_w1=value_w1,
                    delta_value=float(delta_numeric),
                    change_summary=change_summary,
                    change_payload=change_payload,
                )
            )

        return tuple(records)

    def _assemble_stage3_results(
        self, records: Sequence[StageThreeChangeRecord], max_records_per_dimension: int
    ) -> Sequence[StageThreeResult]:
        grouped: dict[tuple[Any, ...], list[StageThreeChangeRecord]] = {}
        for record in records:
            key = (
                record.scene_context.get("scene_tag"),
                record.scene_context.get("base_scene"),
                record.scene_context.get("morphology"),
                record.scene_context.get("marketplace_id"),
                record.week_w0,
                record.week_w1,
            )
            grouped.setdefault(key, []).append(record)

        results: list[StageThreeResult] = []
        for key, scene_records in grouped.items():
            scene_context = {
                "scene_tag": key[0],
                "base_scene": key[1],
                "morphology": key[2],
                "marketplace_id": key[3],
                "week_w0": key[4],
                "week_w1": key[5],
            }
            dimension_map: dict[tuple[Channel, str], list[StageThreeChangeRecord]] = {}
            for record in scene_records:
                dim_key = (record.channel, record.lag_type)
                dimension_map.setdefault(dim_key, []).append(record)

            page_dimensions: list[StageThreeDimensionChange] = []
            traffic_dimensions: list[StageThreeDimensionChange] = []
            for (channel, lag_type), dim_records in sorted(
                dimension_map.items(), key=lambda item: (item[0][0], item[0][1])
            ):
                sorted_records = sorted(
                    dim_records,
                    key=lambda r: (
                        0 if str(r.opp_type).lower() == "leader" else 1,
                        -abs(r.delta_value or 0.0),
                        r.my_asin,
                    ),
                )
                limited_records = tuple(sorted_records[: max_records_per_dimension or len(sorted_records)])
                dimension = StageThreeDimensionChange(
                    scene_context=scene_context,
                    week_w0=scene_context["week_w0"],
                    week_w1=scene_context["week_w1"],
                    channel=channel,
                    lag_type=lag_type,
                    total_changes=len(dim_records),
                    total_changes_leader=sum(
                        1 for item in dim_records if str(item.opp_type).lower() == "leader"
                    ),
                    total_changes_median=sum(
                        1 for item in dim_records if str(item.opp_type).lower() == "median"
                    ),
                    records=limited_records,
                )
                if channel == _STAGE3_PAGE_CHANNEL:
                    page_dimensions.append(dimension)
                else:
                    traffic_dimensions.append(dimension)

            results.append(
                StageThreeResult(
                    context=scene_context,
                    page_dimensions=tuple(page_dimensions),
                    traffic_dimensions=tuple(traffic_dimensions),
                )
            )

        return tuple(results)

    def _write_stage3_output(self, result: StageThreeResult) -> Path:
        context = result.context
        week_w0 = str(context.get("week_w0") or "unknown")
        storage_name = self._build_stage3_storage_name(context)
        path = self._storage_root / week_w0 / "stage3" / f"{storage_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(result)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return path

    def _build_stage3_storage_name(self, context: Mapping[str, Any]) -> str:
        parts = [
            _sanitize_storage_fragment(context.get("scene_tag")),
            _sanitize_storage_fragment(context.get("base_scene")),
            _sanitize_storage_fragment(context.get("morphology")),
        ]
        name = "_".join(part for part in parts if part)
        if not name:
            name = "scene"
        marketplace = _sanitize_storage_fragment(context.get("marketplace_id"))
        if marketplace:
            name = f"{name}_{marketplace}"
        return name

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
            direction_raw = rule.get("direction") or defaults.get("direction")
            direction = _normalize_direction(direction_raw) or _resolve_direction(metric, lag_type)
            rules_by_key[(lag_type, metric, opp_type, comparator)] = LagRule(
                rule_name=rule_name,
                lag_type=lag_type,
                metric=metric,
                opp_type=opp_type,
                comparator=comparator,
                threshold=float(threshold),
                weight=float(weight),
                direction=direction,
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
                raw_evidence.extend(entry.get("evidence"))

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

            pairwise_only = lag_type in _PAIRWISE_ONLY_LAG_TYPES
            if lag_data is not None:
                for hint in hints:
                    if pairwise_only and str(hint.get("kind") or "").lower() == "overview":
                        continue
                    raw_evidence.extend(
                        self._build_evidence_from_hint(lag_index, lag_type, hint)
                    )

                pairwise_entries = self._extract_pairwise_evidence(lag_type, lag_data)
                if pairwise_only:
                    if pairwise_entries:
                        raw_evidence = list(pairwise_entries) + list(raw_evidence)
                elif not raw_evidence:
                    raw_evidence.extend(pairwise_entries)

                if not raw_evidence and not pairwise_only:
                    raw_evidence.extend(
                        self._extract_overview_evidence(lag_type, lag_data)
                    )
            else:
                LOGGER.warning(
                    "competition_llm.stage2_evidence_lag_not_found lag_hint=%s available=%s",
                    inferred_lag,
                    sorted(lag_index),
                )

            direction_hints = _normalise_direction_map(
                lag_data.get("metric_directions") if isinstance(lag_data, Mapping) else {}
            )
            normalised = self._deduplicate_evidence(
                self._normalise_evidence_sequence(
                    raw_evidence,
                    lag_type=lag_type,
                    direction_hints=direction_hints,
                )
            )

            filtered_evidence = self._filter_normalised_evidence(
                lag_type, normalised, lag_data, direction_hints=direction_hints
            )

            if not filtered_evidence:
                if self._config.stage_2.require_unfavorable_evidence:
                    LOGGER.debug(
                        "competition_llm.stage2_drop_root_cause_no_evidence code=%s lag_type=%s",
                        entry.get("root_cause_code"),
                        lag_type,
                    )
                    continue
                LOGGER.warning(
                    "competition_llm.stage2_evidence_missing code=%s hints=%s",
                    entry.get("root_cause_code"),
                    hints,
                )

            limit = 4 if lag_type == "rank" else 3
            evidence_slice = filtered_evidence[:limit]
            if not evidence_slice and self._config.stage_2.require_unfavorable_evidence:
                LOGGER.debug(
                    "competition_llm.stage2_drop_root_cause_empty_slice code=%s lag_type=%s",
                    entry.get("root_cause_code"),
                    lag_type,
                )
                continue
            entry["evidence"] = evidence_slice
            entry.pop("evidence_refs", None)
            cleaned_causes.append(entry)

        if not cleaned_causes:
            data["root_causes"] = []
            data["lag_type"] = "none"
        else:
            data["root_causes"] = cleaned_causes
            data["lag_type"] = self._determine_machine_lag_type(
                cleaned_causes,
                lag_index,
                original=data.get("lag_type"),
            )
        return data

    def _fix_directional_metrics(
        self,
        machine_json: MutableMapping[str, Any],
        *,
        facts: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(machine_json, MutableMapping):
            return
        root_causes = machine_json.get("root_causes")
        if not isinstance(root_causes, Sequence):
            return

        lag_index = self._build_lag_index(facts or {}) if facts else {}
        direction_hints_map = {
            lag: _normalise_direction_map(data.get("metric_directions"))
            for lag, data in lag_index.items()
        }

        total = 0
        corrected = 0
        dropped = 0
        dropped_evidence_total = 0
        updated_causes: list[Mapping[str, Any]] = []

        for cause in root_causes:
            if not isinstance(cause, Mapping):
                updated_causes.append(cause)
                continue

            code = str(cause.get("root_cause_code") or cause.get("code") or "").lower()
            if code != "rank_gap":
                updated_causes.append(cause)
                continue

            total += 1
            lag_type = _normalize_lag_type(cause.get("lag_dimension") or "rank") or "rank"
            direction_hints = direction_hints_map.get(lag_type, {})
            original_evidence_raw = (
                cause.get("evidence") if isinstance(cause.get("evidence"), Sequence) else []
            )
            need_metadata_fix = self._has_missing_directional_fields(original_evidence_raw)
            evidence_items = self._normalise_evidence_sequence(
                original_evidence_raw,
                lag_type=lag_type,
                direction_hints=direction_hints,
            )
            local_dropped_evidence = 0
            evidence_changed = False
            cleaned_evidence: list[Mapping[str, Any]] = []
            for item in evidence_items:
                if not isinstance(item, Mapping):
                    continue
                metric_name = str(item.get("metric") or "").strip().lower()
                against = str(item.get("against") or "").strip().lower()
                my_value = item.get("my_value")
                opp_value = item.get("opp_value")
                if against == "asin" and _is_rank_metric(metric_name):
                    if _is_invalid_rank_value(my_value) or _is_invalid_rank_value(opp_value):
                        local_dropped_evidence += 1
                        evidence_changed = True
                        LOGGER.debug(
                            "competition_llm.stage2_drop_rank_evidence_missing metric=%s my_value=%s opp_value=%s",
                            metric_name,
                            my_value,
                            opp_value,
                        )
                        continue

                updated_item = dict(item)
                direction_value = _normalize_direction(updated_item.get("direction"))
                resolved_direction = direction_value or _resolve_direction(
                    metric_name, lag_type, direction_hints
                )
                if resolved_direction != updated_item.get("direction"):
                    updated_item["direction"] = resolved_direction
                    evidence_changed = True

                delta_value = _compute_delta(my_value, opp_value)
                new_delta = round(delta_value, 6) if delta_value is not None else None
                if new_delta != updated_item.get("delta"):
                    updated_item["delta"] = new_delta
                    evidence_changed = True

                if resolved_direction:
                    new_worse = _compute_worse(
                        resolved_direction, my_value, opp_value, self._direction_tolerance
                    )
                else:
                    new_worse = None
                if new_worse != updated_item.get("worse"):
                    updated_item["worse"] = new_worse
                    evidence_changed = True

                cleaned_evidence.append(updated_item)

            dropped_evidence_total += local_dropped_evidence

            if not cleaned_evidence:
                LOGGER.info(
                    "competition_llm.stage2_rankgap_dropped_no_evidence summary=%s",
                    cause.get("summary"),
                )
                dropped += 1
                continue

            any_worse = any(item.get("worse") is True for item in cleaned_evidence)
            if not any_worse:
                LOGGER.info(
                    "competition_llm.stage2_rankgap_dropped_no_worse summary=%s opp_asins=%s",
                    cause.get("summary"),
                    [item.get("opp_asin") for item in cleaned_evidence if item.get("opp_asin")],
                )
                dropped += 1
                continue

            summary_before = str(cause.get("summary") or "").strip()
            summary_after = self._rewrite_rank_summary(summary_before, lag_type)
            did_correct = (
                need_metadata_fix
                or summary_after != summary_before
                or evidence_changed
                or local_dropped_evidence > 0
            )

            updated_cause = dict(cause)
            updated_cause["evidence"] = cleaned_evidence
            if summary_after != summary_before:
                updated_cause["summary"] = summary_after
            updated_causes.append(updated_cause)

            if did_correct:
                corrected += 1
                LOGGER.info(
                    "competition_llm.stage2_direction_fixed summary_before=%s summary_after=%s metadata_fix=%s evidence_changed=%s",
                    summary_before,
                    summary_after,
                    need_metadata_fix,
                    evidence_changed or local_dropped_evidence > 0,
                )

        if total:
            diagnostics = machine_json.setdefault("diagnostics", {})
            stage2_metrics = diagnostics.setdefault("stage2_rank_fix", {})
            stage2_metrics.update(
                {
                    "total_causes": total,
                    "corrected_count": corrected,
                    "dropped_count": dropped,
                    "dropped_evidence_count": dropped_evidence_total,
                }
            )
            self._rank_fix_counters["total"] += total
            self._rank_fix_counters["corrected"] += corrected
            self._rank_fix_counters["dropped"] += dropped
            self._rank_fix_counters["evidence_dropped"] += dropped_evidence_total
            LOGGER.info(
                "stage2_rank_fix.metrics total_causes=%s corrected_count=%s dropped_count=%s dropped_evidence_count=%s",
                total,
                corrected,
                dropped,
                dropped_evidence_total,
            )

        if updated_causes:
            machine_json["root_causes"] = updated_causes
            machine_json["lag_type"] = self._determine_machine_lag_type(
                updated_causes, lag_index, original=machine_json.get("lag_type")
            )
        else:
            machine_json["root_causes"] = []
            machine_json["lag_type"] = "none"

    def _has_missing_directional_fields(self, evidence: Any) -> bool:
        if not isinstance(evidence, Sequence):
            return True
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            direction = _normalize_direction(item.get("direction"))
            delta = item.get("delta")
            worse = item.get("worse")
            if direction is None or delta is None or worse is None:
                return True
        return False

    def _rewrite_rank_summary(self, summary: str, lag_type: str) -> str:
        if not summary:
            return summary
        if _normalize_lag_type(lag_type) != "rank":
            return summary
        updated = summary
        replacements = {
            "百分位低于": "百分位高于",
            "百分位更低": "百分位更高",
            "排名低于": "排名高于",
            "排名更低": "排名更高",
        }
        for old, new in replacements.items():
            if old in updated:
                updated = updated.replace(old, new)
        direction_phrase = _DIRECTION_KEYWORDS.get("lower_better")
        if direction_phrase and direction_phrase not in updated:
            stripped = updated.rstrip("。")
            updated = f"{stripped}（{direction_phrase}）"
            if summary.endswith("。"):
                updated += "。"
        return updated

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
            if lag_type in _PAIRWISE_ONLY_LAG_TYPES:
                return []
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
                "metric_directions": _normalise_direction_map(item.get("metric_directions")),
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

    def _determine_machine_lag_type(
        self,
        causes: Sequence[Any],
        lag_index: Mapping[str, Mapping[str, Any]],
        *,
        original: Any = None,
    ) -> str:
        if not causes:
            return "none"
        inferred: list[str] = []
        for cause in causes:
            if not isinstance(cause, Mapping):
                continue
            inferred_lag = self._infer_root_cause_dimension(cause, lag_index)
            if inferred_lag:
                inferred.append(inferred_lag)
        unique_normalized = {
            _normalize_lag_type(value) for value in inferred if value
        }
        unique_normalized.discard("")
        unique_normalized.discard(None)  # type: ignore[arg-type]
        if len(unique_normalized) == 1:
            normalized = next(iter(unique_normalized))
            return _LAG_TYPE_FILENAME_ALIASES.get(normalized, normalized)
        if not unique_normalized and isinstance(original, str):
            cleaned = original.strip()
            if cleaned:
                normalized_original = _normalize_lag_type(cleaned)
                if normalized_original:
                    return _LAG_TYPE_FILENAME_ALIASES.get(
                        normalized_original, cleaned
                    )
                return cleaned
        return "mixed"

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
        limit: int | None = None,
    ) -> list[Mapping[str, Any]]:
        rows = lag_data.get("top_opps")
        if not isinstance(rows, Sequence):
            return []
        source = "pairs_each" if lag_type in _PAGE_LAG_TYPES else "traffic.pairs"
        direction_hints = _normalise_direction_map(lag_data.get("metric_directions"))
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
                direction_hints=direction_hints,
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
        effective_limit = limit if limit is not None else (4 if lag_type == "rank" else 3)
        results: list[Mapping[str, Any]] = []
        for _, _, entry in ordered:
            results.append(entry)
            if len(results) >= effective_limit:
                break
        return results

    def _build_pairwise_entries_from_row(
        self,
        lag_type: str,
        row: Mapping[str, Any],
        source: str,
        *,
        direction_hints: Mapping[str, str] | None = None,
    ) -> list[Mapping[str, Any]]:
        if not isinstance(row, Mapping):
            return []
        if lag_type == "rank":
            entries = self._build_rank_entries(row, source, direction_hints=direction_hints)
        elif lag_type == "content":
            entries = self._build_content_entries(row, source, direction_hints=direction_hints)
        elif lag_type == "social":
            entries = self._build_social_entries(row, source, direction_hints=direction_hints)
        elif lag_type == "keyword":
            entries = self._build_keyword_entries(row, source, direction_hints=direction_hints)
        else:
            entries = []
        if not entries:
            entries = self._build_default_pairwise_entries(
                row,
                source,
                lag_type,
                direction_hints=direction_hints,
            )
        return entries

    def _build_rank_entries(
        self,
        row: Mapping[str, Any],
        source: str,
        *,
        direction_hints: Mapping[str, str] | None = None,
    ) -> list[Mapping[str, Any]]:
        entries: list[Mapping[str, Any]] = []
        entry = self._build_metric_entry(
            row,
            "rank_leaf",
            source,
            priority=1,
            lag_type="rank",
            unit="rank",
            note_builder=_format_rank_leaf_note,
            direction_hints=direction_hints,
        )
        if entry:
            entries.append(entry)
        entry = self._build_metric_entry(
            row,
            "rank_pos_pct",
            source,
            priority=2,
            lag_type="rank",
            unit="pct",
            note_builder=_format_rank_pct_note,
            direction_hints=direction_hints,
        )
        if entry:
            entries.append(entry)
        return entries

    def _build_content_entries(
        self,
        row: Mapping[str, Any],
        source: str,
        *,
        direction_hints: Mapping[str, str] | None = None,
    ) -> list[Mapping[str, Any]]:
        entries: list[Mapping[str, Any]] = []
        entry = self._build_metric_entry(
            row,
            "image_cnt",
            source,
            priority=1,
            lag_type="content",
            note_builder=_format_image_note,
            direction_hints=direction_hints,
        )
        if entry:
            entries.append(entry)
        entry = self._build_metric_entry(
            row,
            "video_cnt",
            source,
            priority=2,
            lag_type="content",
            note_builder=_format_video_note,
            direction_hints=direction_hints,
        )
        if entry:
            entries.append(entry)
        entry = self._build_metric_entry(
            row,
            "content_score",
            source,
            priority=3,
            lag_type="content",
            note_builder=_format_content_score_note,
            direction_hints=direction_hints,
        )
        if entry:
            entries.append(entry)
        return entries

    def _build_social_entries(
        self,
        row: Mapping[str, Any],
        source: str,
        *,
        direction_hints: Mapping[str, str] | None = None,
    ) -> list[Mapping[str, Any]]:
        entries: list[Mapping[str, Any]] = []
        entry = self._build_metric_entry(
            row,
            "rating",
            source,
            priority=1,
            lag_type="social",
            note_builder=_format_rating_note,
            direction_hints=direction_hints,
        )
        if entry:
            entries.append(entry)
        entry = self._build_metric_entry(
            row,
            "reviews",
            source,
            priority=2,
            lag_type="social",
            note_builder=_format_reviews_note,
            direction_hints=direction_hints,
        )
        if entry:
            entries.append(entry)
        entry = self._build_metric_entry(
            row,
            "social_proof",
            source,
            priority=3,
            lag_type="social",
            note_builder=_format_social_proof_note,
            direction_hints=direction_hints,
        )
        if entry:
            entries.append(entry)
        return entries

    def _build_keyword_entries(
        self,
        row: Mapping[str, Any],
        source: str,
        *,
        direction_hints: Mapping[str, str] | None = None,
    ) -> list[Mapping[str, Any]]:
        opp_asin = row.get("opp_asin")
        if not opp_asin:
            return []
        pairs = row.get("keyword_pairs")
        if not isinstance(pairs, Sequence):
            return []
        entries: list[Mapping[str, Any]] = []
        limit = max(1, int(self._config.stage_2.keyword_max_pairs_per_opp or 1))
        for pair in pairs:
            if not isinstance(pair, Mapping):
                continue
            if len(entries) >= limit:
                break
            keyword = pair.get("keyword")
            opp_rank = pair.get("opp_rank")
            if not keyword or opp_rank is None:
                continue
            my_rank = pair.get("my_rank")
            my_share = pair.get("my_share")
            opp_share = pair.get("opp_share")
            if not self._is_unfavorable_keyword_pair(my_rank, opp_rank, my_share, opp_share):
                continue
            note = _format_keyword_note(
                row,
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
                "_priority": len(entries) + 1,
                "_impact": float(pair.get("impact", 0.0) or 0.0),
            }
            opp_brand = row.get("opp_brand")
            if opp_brand:
                entry["opp_brand"] = opp_brand
            direction = _resolve_direction("keyword_rank", "keyword", direction_hints) or "lower_better"
            entry["direction"] = direction
            delta_value = _compute_delta(my_rank, opp_rank)
            entry["delta"] = round(delta_value, 6) if delta_value is not None else None
            entry["worse"] = _compute_worse(
                direction,
                my_rank,
                opp_rank,
                self._direction_tolerance,
            )
            entries.append(entry)
        return entries

    def _build_default_pairwise_entries(
        self,
        row: Mapping[str, Any],
        source: str,
        lag_type: str,
        *,
        direction_hints: Mapping[str, str] | None = None,
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
                lag_type=lag_type,
                direction_hints=direction_hints,
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
        lag_type: str | None = None,
        unit: str | None = None,
        note_builder: Callable[[Any, Any, Mapping[str, Any]], str | None] | None = None,
        direction_hints: Mapping[str, str] | None = None,
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
        if _normalize_lag_type(lag_type) == "rank" and _is_absolute_rank_metric(suffix, lag_type):
            if _is_invalid_rank_value(my_value) or _is_invalid_rank_value(opp_value):
                return None
        always_include = self._metric_always_included(lag_type, suffix)
        if lag_type and not always_include and not self._is_unfavorable_metric(
            lag_type, suffix, my_value, opp_value, row=row
        ):
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
        opp_brand = row.get("opp_brand")
        if opp_brand:
            entry["opp_brand"] = opp_brand
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
        direction = _resolve_direction(suffix, lag_type, direction_hints)
        entry["direction"] = direction
        delta_value = _compute_delta(my_value, opp_value)
        entry["delta"] = round(delta_value, 6) if delta_value is not None else None
        if direction:
            entry["worse"] = _compute_worse(
                direction, my_value, opp_value, self._direction_tolerance
            )
        else:
            entry["worse"] = None
        return entry

    def _metric_always_included(self, lag_type: str | None, metric: str) -> bool:
        if not lag_type or not metric:
            return False
        mapping = getattr(self._config.stage_2, "always_include_metrics", {}) or {}
        metrics = mapping.get(lag_type.lower())
        if not metrics:
            return False
        return metric.lower() in metrics

    def _is_unfavorable_metric(
        self,
        lag_type: str,
        metric: str,
        my_value: Any,
        opp_value: Any,
        *,
        row: Mapping[str, Any] | None = None,
        direction_hints: Mapping[str, str] | None = None,
    ) -> bool:
        if not self._config.stage_2.require_unfavorable_evidence:
            return True
        metric_key = str(metric or "").lower()
        my_num = _coerce_float(my_value)
        opp_num = _coerce_float(opp_value)
        direction = _resolve_direction(metric_key, lag_type, direction_hints)
        if lag_type == "rank":
            if direction:
                worse_flag = _compute_worse(direction, my_num, opp_num, self._direction_tolerance)
                return bool(worse_flag)
        if lag_type == "content" and metric_key in {"image_cnt", "video_cnt", "content_score"}:
            if my_num is None or opp_num is None:
                return False
            return my_num < opp_num
        if lag_type == "social" and metric_key in {"rating", "reviews", "social_proof"}:
            priority = 0
            if row is not None:
                priority = self._evaluate_social_priority(row)
                if priority < 0:
                    return False
                if priority > 0:
                    return True
            if direction:
                worse_flag = _compute_worse(direction, my_num, opp_num, self._direction_tolerance)
                if worse_flag is not None:
                    return worse_flag
            if my_num is None or opp_num is None:
                return False
            return my_num < opp_num
        if direction:
            worse_flag = _compute_worse(direction, my_num, opp_num, self._direction_tolerance)
            if worse_flag is None:
                return False
            return worse_flag
        if my_num is None or opp_num is None:
            return False
        return True

    def _filter_unfavorable_rows(
        self, lag_type: str, rows: Sequence[Mapping[str, Any]] | None
    ) -> list[MutableMapping[str, Any]]:
        if not rows:
            return []
        if not self._config.stage_2.require_unfavorable_evidence:
            return [row if isinstance(row, MutableMapping) else dict(row) for row in rows if isinstance(row, Mapping)]
        filtered: list[MutableMapping[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if not self._row_has_unfavorable_gap(lag_type, row):
                continue
            if isinstance(row, MutableMapping):
                filtered.append(row)
            else:
                filtered.append(dict(row))
        return filtered

    def _row_has_unfavorable_gap(
        self, lag_type: str, row: Mapping[str, Any]
    ) -> bool:
        if not self._config.stage_2.require_unfavorable_evidence:
            return True

        def _greater(my: Any, opp: Any) -> bool:
            my_num = _coerce_float(my)
            opp_num = _coerce_float(opp)
            return my_num is not None and opp_num is not None and my_num > opp_num

        def _less(my: Any, opp: Any) -> bool:
            my_num = _coerce_float(my)
            opp_num = _coerce_float(opp)
            return my_num is not None and opp_num is not None and my_num < opp_num

        if lag_type == "rank":
            worse_leaf = _compute_worse(
                "lower_better",
                row.get("my_rank_leaf"),
                row.get("opp_rank_leaf"),
                self._direction_tolerance,
            )
            if worse_leaf:
                return True
            worse_pct = _compute_worse(
                "lower_better",
                row.get("my_rank_pos_pct"),
                row.get("opp_rank_pos_pct"),
                self._direction_tolerance,
            )
            return bool(worse_pct)
        if lag_type == "content":
            return (
                _less(row.get("my_image_cnt"), row.get("opp_image_cnt"))
                or _less(row.get("my_video_cnt"), row.get("opp_video_cnt"))
                or _less(row.get("my_content_score"), row.get("opp_content_score"))
            )
        if lag_type == "social":
            priority = self._evaluate_social_priority(row)
            if priority < 0:
                return False
            if priority > 0:
                return True
            return (
                _less(row.get("my_rating"), row.get("opp_rating"))
                or _less(row.get("my_reviews"), row.get("opp_reviews"))
                or _less(row.get("my_social_proof"), row.get("opp_social_proof"))
            )
        if lag_type == "keyword":
            pairs = row.get("keyword_pairs")
            if isinstance(pairs, Sequence):
                for pair in pairs:
                    if not isinstance(pair, Mapping):
                        continue
                    if self._is_unfavorable_keyword_pair(
                        pair.get("my_rank"),
                        pair.get("opp_rank"),
                        pair.get("my_share"),
                        pair.get("opp_share"),
                    ):
                        return True
            my_share = _coerce_float(row.get("my_kw_top3_share_7d_avg"))
            opp_share = _coerce_float(row.get("opp_kw_top3_share_7d_avg"))
            if opp_share is None:
                return False
            if my_share is None:
                return True
            return my_share < opp_share
        return True

    def _evaluate_social_priority(self, row: Mapping[str, Any]) -> int:
        min_reviews = getattr(self._config.stage_2, "min_reviews_for_rating_priority", 0)
        try:
            min_reviews_value = int(min_reviews)
        except (TypeError, ValueError):
            min_reviews_value = 0
        if min_reviews_value < 0:
            min_reviews_value = 0

        rating_margin = getattr(self._config.stage_2, "rating_margin", 0.0)
        try:
            rating_margin_value = float(rating_margin)
        except (TypeError, ValueError):
            rating_margin_value = 0.0
        if rating_margin_value < 0:
            rating_margin_value = 0.0

        my_rating = _coerce_float(row.get("my_rating"))
        opp_rating = _coerce_float(row.get("opp_rating"))
        if my_rating is None or opp_rating is None:
            return 0
        rating_diff = my_rating - opp_rating

        my_reviews = _coerce_float(row.get("my_reviews"))
        opp_reviews = _coerce_float(row.get("opp_reviews"))

        if (
            my_reviews is not None
            and my_reviews >= min_reviews_value
            and rating_diff >= rating_margin_value
        ):
            return -1
        if (
            opp_reviews is not None
            and opp_reviews >= min_reviews_value
            and (-rating_diff) >= rating_margin_value
        ):
            return 1
        return 0

    def _is_unfavorable_keyword_pair(
        self, my_rank: Any, opp_rank: Any, my_share: Any, opp_share: Any
    ) -> bool:
        if not self._config.stage_2.require_unfavorable_evidence:
            return True
        opp_rank_num = _coerce_float(opp_rank)
        if opp_rank_num is None:
            return False
        my_rank_num = _coerce_float(my_rank)
        if my_rank_num is None:
            return True
        if my_rank_num > opp_rank_num:
            return True
        opp_share_num = _coerce_float(opp_share)
        if opp_share_num is None:
            return False
        my_share_num = _coerce_float(my_share)
        if my_share_num is None:
            return True
        return my_share_num < opp_share_num

    def _normalise_evidence_sequence(
        self,
        items: Any,
        *,
        lag_type: str | None = None,
        direction_hints: Mapping[str, str] | None = None,
    ) -> list[Mapping[str, Any]]:
        if not isinstance(items, Sequence):
            return []
        result: list[Mapping[str, Any]] = []
        for item in items:
            normalised = self._normalise_evidence_entry(
                item,
                lag_type=lag_type,
                direction_hints=direction_hints,
            )
            if normalised:
                result.append(normalised)
        return result

    def _normalise_evidence_entry(
        self,
        item: Any,
        *,
        lag_type: str | None = None,
        direction_hints: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any] | None:
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
        if _is_absolute_rank_metric(metric, lag_type):
            if _is_invalid_rank_value(my_value) or _is_invalid_rank_value(opp_value):
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
            opp_brand = item.get("opp_brand")
            if opp_brand not in (None, ""):
                normalised["opp_brand"] = str(opp_brand)
        unit = item.get("unit")
        if unit is not None:
            normalised["unit"] = unit
        source = item.get("source")
        if source is not None:
            normalised["source"] = source
        note = item.get("note")
        if note is not None:
            normalised["note"] = note
        direction = _normalize_direction(item.get("direction"))
        resolved_direction = direction or _resolve_direction(metric, lag_type, direction_hints)
        normalised["direction"] = resolved_direction
        delta_value = _compute_delta(my_value, opp_value)
        normalised["delta"] = round(delta_value, 6) if delta_value is not None else None
        if resolved_direction:
            normalised["worse"] = _compute_worse(
                resolved_direction, my_value, opp_value, self._direction_tolerance
            )
        else:
            normalised["worse"] = None
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

    def _filter_normalised_evidence(
        self,
        lag_type: str | None,
        entries: Sequence[Mapping[str, Any]],
        lag_data: Mapping[str, Any] | None,
        *,
        direction_hints: Mapping[str, str] | None = None,
    ) -> list[Mapping[str, Any]]:
        if not entries:
            return []
        if not lag_type:
            return list(entries)

        lag_type_lower = str(lag_type).lower()
        row_lookup: Mapping[str, Mapping[str, Any]] = {}
        if isinstance(lag_data, Mapping):
            row_lookup = self._build_lag_row_lookup(lag_data.get("top_opps"))
        hints = direction_hints or {}

        filtered: list[Mapping[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            metric = entry.get("metric")
            if not metric:
                continue
            source = str(entry.get("source") or "").lower()
            against = str(entry.get("against") or "").lower()
            # 1. 对排名/社交滞后，只关心成对比较，跳过页面概览来源
            if lag_type_lower in {"rank", "social"} and source == "page.overview":
                continue

            # 2. 仅对成对比较执行“不利”检查
            if against == "asin":
                row_context: Mapping[str, Any] | None = None
                opp_asin = entry.get("opp_asin")
                if opp_asin is not None:
                    row_context = row_lookup.get(str(opp_asin))

                if not self._is_unfavorable_metric(
                    lag_type_lower,
                    metric,
                    entry.get("my_value"),
                    entry.get("opp_value"),
                    row=row_context,
                    direction_hints=hints,
                ):
                    continue
            filtered.append(entry)
        return filtered

    def _build_lag_row_lookup(
        self, rows: Any
    ) -> Mapping[str, Mapping[str, Any]]:
        if not isinstance(rows, Sequence):
            return {}
        lookup: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            opp_asin = row.get("opp_asin")
            if opp_asin in (None, ""):
                continue
            lookup[str(opp_asin)] = row
        return lookup

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

        allowed_actions_seq = self._config.stage_2.allowed_action_codes
        enforce_actions = allowed_actions_seq is not None
        allowed_actions = tuple(allowed_actions_seq or ())
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
            if enforce_actions:
                if not allowed_actions_set:
                    LOGGER.debug(
                        "competition_llm.stage2_drop_action_disallowed code=%s", code
                    )
                    raise ValueError(f"Action code {code!r} is not permitted")
                if code not in allowed_actions_set:
                    LOGGER.debug(
                        "competition_llm.stage2_drop_action_disallowed code=%s", code
                    )
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

        if cleaned_actions:
            diagnostics = mutable_payload.setdefault("diagnostics", {})
            diagnostics["validated_actions"] = cleaned_actions
        mutable_payload["actions"] = []
        if "recommended_actions" in mutable_payload:
            mutable_payload["recommended_actions"] = []

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
            top_each_raw = self._fetch_all(sql_each, ctx)
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage2_page_pairs_unavailable asin=%s lag_type=%s error=%s",
                ctx.get("my_asin"),
                lag_type,
                exc,
            )
            top_each_raw = []
        if top_each_raw:
            self._attach_page_entity_details(ctx, top_each_raw)
        top_each = self._filter_unfavorable_rows(lag_type, top_each_raw)
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
        sql_full = """
          SELECT brand, payload
          FROM bi_amz_asin_product_snapshot
          WHERE marketplace_id=:marketplace_id
            AND week=:week
            AND asin=:asin
          LIMIT 1
        """
        row: Mapping[str, Any] | None = None
        payload: Any | None = None
        need_payload_query = False
        try:
            row = self._fetch_one(sql_full, params)
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage2_brand_lookup_failed asin=%s error=%s",
                asin,
                exc,
            )
            need_payload_query = True

        if row:
            brand_value = row.get("brand")
            if isinstance(brand_value, str) and brand_value.strip():
                return brand_value.strip()
            payload = row.get("payload")
            if payload in (None, ""):
                payload = None
                need_payload_query = True
        else:
            need_payload_query = True

        if need_payload_query:
            sql_payload_only = """
              SELECT payload
              FROM bi_amz_asin_product_snapshot
              WHERE marketplace_id=:marketplace_id
                AND week=:week
                AND asin=:asin
              LIMIT 1
            """
            try:
                payload_row = self._fetch_one(sql_payload_only, params)
            except SQLAlchemyError as exc:  # pragma: no cover - optional table
                LOGGER.debug(
                    "competition_llm.stage2_brand_payload_lookup_failed asin=%s error=%s",
                    asin,
                    exc,
                )
                payload_row = None
            if payload_row:
                payload = payload_row.get("payload")

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
        limit = max(1, int(self._config.stage_2.keyword_max_pairs_per_opp or 1))
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
            if len(pairs) >= limit:
                return pairs
        for item in opponent_keywords:
            if len(pairs) >= limit:
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
            top_each_raw = self._fetch_all(sql_each, ctx)
        except SQLAlchemyError as exc:  # pragma: no cover - optional table
            LOGGER.debug(
                "competition_llm.stage2_traffic_pairs_unavailable asin=%s lag_type=%s error=%s",
                ctx.get("my_asin"),
                lag_type,
                exc,
            )
            top_each_raw = []
        if top_each_raw:
            self._attach_page_entity_details(ctx, top_each_raw)
            if lag_type == "keyword":
                self._attach_keyword_details(ctx, top_each_raw)
        top_each = self._filter_unfavorable_rows(lag_type, top_each_raw)
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


def _normalize_direction(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if cleaned in {"lower", "lower-better", "lower_better"}:
        return "lower_better"
    if cleaned in {"higher", "higher-better", "higher_better"}:
        return "higher_better"
    return None


def _resolve_direction(
    metric: str | None,
    lag_type: str | None = None,
    hints: Mapping[str, str] | None = None,
) -> str | None:
    if not metric:
        return None
    metric_key = str(metric).strip().lower()
    if hints:
        hint_value = hints.get(metric_key)
        normalized_hint = _normalize_direction(hint_value)
        if normalized_hint:
            return normalized_hint
    if metric_key in _METRIC_DIRECTION_MAP:
        return _METRIC_DIRECTION_MAP[metric_key]
    if lag_type and _normalize_lag_type(lag_type) == "rank":
        if metric_key.startswith("rank_") or metric_key.endswith("_rank"):
            return "lower_better"
    return None


def _is_absolute_rank_metric(metric: str | None, lag_type: str | None = None) -> bool:
    if not metric:
        return False
    metric_key = str(metric).strip().lower()
    if metric_key in _ABSOLUTE_RANK_METRICS:
        return True
    return False


def _is_rank_metric(metric: str | None) -> bool:
    if not metric:
        return False
    metric_key = str(metric).strip().lower()
    if metric_key == "rank":
        return True
    if metric_key.startswith("rank_"):
        return True
    if metric_key.endswith("_rank"):
        return True
    return False


def _is_invalid_rank_value(value: Any) -> bool:
    if value in (None, ""):
        return True
    if isinstance(value, (int, float, Decimal)):
        if isinstance(value, Decimal):
            if value.is_nan():
                return True
            numeric = float(value)
        else:
            numeric = float(value)
        if math.isnan(numeric):
            return True
        return numeric <= 0
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        try:
            numeric = float(stripped)
        except ValueError:
            return False
        if math.isnan(numeric):
            return True
        return numeric <= 0
    return False


def _is_missing_rank_value(value: Any) -> bool:
    return _is_invalid_rank_value(value)


def _normalise_direction_map(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        normalised = _normalize_direction(value)
        if normalised:
            result[key.strip().lower()] = normalised
    return result


def _compute_delta(my_value: Any, opp_value: Any) -> float | None:
    my_num = _coerce_float(my_value)
    opp_num = _coerce_float(opp_value)
    if my_num is None or opp_num is None:
        return None
    return my_num - opp_num


def _compute_worse(
    direction: str, my_value: Any, opp_value: Any, tolerance: float
) -> bool | None:
    my_num = _coerce_float(my_value)
    opp_num = _coerce_float(opp_value)
    if my_num is None or opp_num is None:
        return None
    if direction == "lower_better":
        return my_num > opp_num + tolerance
    if direction == "higher_better":
        return my_num < opp_num - tolerance
    return None


def _build_stage3_change_summary(
    lag_type: str, metric: str, direction: str | None, delta_value: float
) -> str:
    normalized_direction = direction or "neutral"
    amount_text = _format_stage3_amount(delta_value)
    if metric == "d_price_net":
        if normalized_direction == "up":
            return f"我方净价较上周上升 {amount_text}"
        if normalized_direction == "down":
            return f"我方净价较上周下降 {amount_text}"
        return "我方净价与上周基本持平"
    if metric == "d_price_gap_leader":
        if normalized_direction == "up":
            return f"与 leader 的净价差拉大 {amount_text}"
        if normalized_direction == "down":
            return f"与 leader 的净价差缩小 {amount_text}"
        return "与 leader 的净价差基本持平"
    if metric == "d_price_index_med":
        if normalized_direction == "up":
            return f"相对中位数的价格指数上升 {amount_text}"
        if normalized_direction == "down":
            return f"相对中位数的价格指数下降 {amount_text}"
        return "相对中位数的价格指数保持稳定"
    if metric == "d_rank_score":
        if normalized_direction == "up":
            return f"排名得分提升 {amount_text}"
        if normalized_direction == "down":
            return f"排名得分下降 {amount_text}"
        return "排名得分保持稳定"
    if metric == "d_rank_pos_pct":
        if normalized_direction == "up":
            return f"排名百分位上升 {amount_text}"
        if normalized_direction == "down":
            return f"排名百分位下降 {amount_text}"
        return "排名百分位基本持平"
    if metric == "d_content_score":
        if normalized_direction == "up":
            return f"内容得分提升 {amount_text}"
        if normalized_direction == "down":
            return f"内容得分下降 {amount_text}"
        return "内容得分保持稳定"
    if metric == "d_content_gap":
        if normalized_direction == "up":
            return f"与竞品的内容差距扩大 {amount_text}"
        if normalized_direction == "down":
            return f"与竞品的内容差距缩小 {amount_text}"
        return "与竞品的内容差距基本持平"
    if metric == "d_social_proof":
        if normalized_direction == "up":
            return f"我方社交证据增加 {amount_text}"
        if normalized_direction == "down":
            return f"我方社交证据下降 {amount_text}"
        return "我方社交证据变化不大"
    if metric == "d_social_gap":
        if normalized_direction == "up":
            return f"与竞品的社交差距扩大 {amount_text}"
        if normalized_direction == "down":
            return f"与竞品的社交差距缩小 {amount_text}"
        return "与竞品的社交差距保持稳定"
    if metric == "badge_change":
        count = _format_stage3_badge_delta(delta_value)
        if normalized_direction == "up":
            return f"我方新增徽章 {count} 个"
        if normalized_direction == "down":
            return f"我方减少徽章 {count} 个"
        return "我方徽章数保持不变"
    if metric == "delta_pressure":
        if normalized_direction == "up":
            return f"相对压力上升 {amount_text}"
        if normalized_direction == "down":
            return f"相对压力下降 {amount_text}"
        return "相对压力保持稳定"

    direction_phrase = _stage3_direction_phrase(normalized_direction)
    if direction_phrase:
        return f"{lag_type} 指标{direction_phrase}{amount_text}"
    return f"{lag_type} 指标变化 {amount_text}"


def _format_stage3_amount(delta_value: float) -> str:
    magnitude = abs(float(delta_value))
    if not math.isfinite(magnitude):
        return "0.00"
    if magnitude >= 100:
        return f"{magnitude:.0f}"
    return f"{magnitude:.2f}"


def _format_stage3_badge_delta(delta_value: float) -> int:
    magnitude = abs(float(delta_value))
    if not math.isfinite(magnitude):
        return 0
    return int(round(magnitude))


def _stage3_direction_phrase(direction: str) -> str:
    if direction == "up":
        return "上升 "
    if direction == "down":
        return "下降 "
    return ""


def _sanitize_storage_fragment(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def _format_signed_value(value: Any) -> str:
    number = _coerce_float(value)
    if number is None:
        return "缺失"
    if abs(number) < 1:
        return f"{number:+.2%}"
    if abs(number) >= 100 or number.is_integer():
        return f"{number:+.0f}"
    return f"{number:+.2f}"


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


def _format_opponent_display_label(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        row = {}
    return _format_opp_label(row)


def _format_rank_leaf_note(my_value: Any, opp_value: Any, row: Mapping[str, Any]) -> str:
    label = _format_opponent_display_label(row)
    my_text = _format_integer_text(my_value)
    opp_text = _format_integer_text(opp_value)
    my_display = my_text if my_text == "缺失" else f"{my_text} 位"
    opp_display = opp_text if opp_text == "缺失" else f"{opp_text} 位"
    my_num = _coerce_float(my_value)
    opp_num = _coerce_float(opp_value)
    gap_text = ""
    if my_num is not None and opp_num is not None:
        gap = my_num - opp_num
        if abs(gap) >= 1:
            direction = "落后" if gap > 0 else "领先"
            gap_text = f"（{direction} {abs(int(round(gap)))} 位）"
    return f"排名：我方 {my_display} / 对手({label}) {opp_display}{gap_text}"


def _format_rank_pct_note(my_value: Any, opp_value: Any, row: Mapping[str, Any]) -> str:
    label = _format_opponent_display_label(row)
    my_text = _format_percentage_text(my_value)
    opp_text = _format_percentage_text(opp_value)
    my_num = _coerce_float(my_value)
    opp_num = _coerce_float(opp_value)
    gap_text = ""
    if my_num is not None and opp_num is not None:
        gap = my_num - opp_num
        if abs(gap) >= 0.005:
            direction = "落后" if gap < 0 else "领先"
            gap_text = f"（{direction} {abs(gap) * 100:.2f}%）"
    return f"排名百分位：我方 {my_text} / 对手({label}) {opp_text}{gap_text}"


def _format_image_note(my_value: Any, opp_value: Any, row: Mapping[str, Any]) -> str:
    return _format_count_note("图片", my_value, opp_value, "张", row)


def _format_video_note(my_value: Any, opp_value: Any, row: Mapping[str, Any]) -> str:
    return _format_count_note("视频", my_value, opp_value, "个", row)


def _format_count_note(
    label: str, my_value: Any, opp_value: Any, unit: str, row: Mapping[str, Any]
) -> str:
    label_text = _format_opponent_display_label(row)
    my_text = _format_integer_text(my_value)
    opp_text = _format_integer_text(opp_value)
    my_display = my_text if my_text == "缺失" else f"{my_text} {unit}"
    opp_display = opp_text if opp_text == "缺失" else f"{opp_text} {unit}"
    my_num = _coerce_float(my_value)
    opp_num = _coerce_float(opp_value)
    gap_text = ""
    if my_num is not None and opp_num is not None:
        gap = opp_num - my_num
        if abs(gap) >= 1:
            if gap > 0:
                gap_text = f"（缺少 {int(round(abs(gap)))}{unit}）"
            else:
                gap_text = f"（多出 {int(round(abs(gap)))}{unit}）"
    return f"{label}：我方 {my_display} / 对手({label_text}) {opp_display}{gap_text}"


def _format_content_score_note(
    my_value: Any, opp_value: Any, row: Mapping[str, Any]
) -> str:
    label = _format_opponent_display_label(row)
    my_text = _format_decimal_text(my_value)
    opp_text = _format_decimal_text(opp_value)
    gap = _compute_pair_gap(my_value, opp_value)
    gap_text = f"（差距 {gap:.2f}）" if gap >= 0.01 else ""
    return f"内容得分：我方 {my_text} / 对手({label}) {opp_text}{gap_text}"


def _format_rating_note(my_value: Any, opp_value: Any, row: Mapping[str, Any]) -> str:
    label = _format_opponent_display_label(row)
    my_text = _format_decimal_text(my_value)
    opp_text = _format_decimal_text(opp_value)
    gap = _compute_pair_gap(my_value, opp_value)
    gap_text = f"（相差 {gap:.2f}）" if gap >= 0.01 else ""
    return f"评分：我方 {my_text} / 对手({label}) {opp_text}{gap_text}"


def _format_reviews_note(my_value: Any, opp_value: Any, row: Mapping[str, Any]) -> str:
    return _format_count_note("评论", my_value, opp_value, "条", row)


def _format_social_proof_note(
    my_value: Any, opp_value: Any, row: Mapping[str, Any]
) -> str:
    label = _format_opponent_display_label(row)
    my_text = _format_decimal_text(my_value)
    opp_text = _format_decimal_text(opp_value)
    gap = _compute_pair_gap(my_value, opp_value)
    gap_text = f"（差距 {gap:.2f}）" if gap >= 0.01 else ""
    return f"社交证明：我方 {my_text} / 对手({label}) {opp_text}{gap_text}"


def _format_keyword_note(
    row: Mapping[str, Any],
    keyword: str,
    my_rank: Any,
    opp_rank: Any,
    my_share: Any,
    opp_share: Any,
    tag: Any,
) -> str:
    label = _format_opponent_display_label(row)
    my_rank_text = f"TOP{int(my_rank)}" if my_rank is not None else "无排名"
    opp_rank_text = f"TOP{int(opp_rank)}" if opp_rank is not None else "缺失"
    my_share_text = _format_percentage_text(my_share)
    opp_share_text = _format_percentage_text(opp_share)
    keyword_label = str(keyword)
    if tag:
        keyword_label = f"{keyword_label}（{tag}）"
    return (
        f"关键词「{keyword_label}」：我方 {my_rank_text} ({my_share_text}) / "
        f"对手({label}) {opp_rank_text} ({opp_share_text})"
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
        return f"竞品 {label} 价格差 {gap}，我方/竞品价格比 {ratio}。"
    if lag_type == "rank":
        delta = _format_metric_value(row.get("rank_pos_delta"))
        return f"竞品 {label} 排名领先差值 {delta}，我方自然位次落后。"
    if lag_type == "content":
        gap = _format_metric_value(row.get("content_gap_each"))
        return f"竞品 {label} 内容表现领先 {gap}。"
    if lag_type == "social":
        gap = _format_metric_value(row.get("social_gap_each"))
        return f"竞品 {label} 社交口碑差距 {gap}。"
    if lag_type == "badge":
        diff = _format_metric_value(row.get("badge_delta_sum"))
        return f"竞品 {label} 权益标识差值 {diff}。"
    if lag_type == "confidence":
        confidence = _format_metric_value(row.get("confidence"))
        return f"竞品 {label} 信号置信度 {confidence} 更高。"
    if lag_type == "traffic_mix":
        ad_gap = _format_metric_value(row.get("ad_ratio_gap_each"))
        mix_gap = _format_metric_value(row.get("ad_to_natural_gap_each"))
        return f"竞品 {label} 广告占比差距 {ad_gap}，广告/自然流量差距 {mix_gap}。"
    if lag_type == "keyword":
        my_share = _format_metric_value(row.get("my_kw_top3_share_7d_avg"))
        opp_share = _format_metric_value(row.get("opp_kw_top3_share_7d_avg"))
        return f"竞品 {label} Top3关键词份额 {opp_share}，我方 {my_share}。"
    score = _format_metric_value(row.get("score"))
    return f"竞品 {label} 差距评分 {score}。"


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
    "StageThreeChangeRecord",
    "StageThreeDimensionChange",
    "StageThreeResult",
    "StageOneLLMResult",
    "StageOneResult",
    "StageTwoAggregateResult",
]
