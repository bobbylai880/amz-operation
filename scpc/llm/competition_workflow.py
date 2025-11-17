"""Stage-1 orchestration for the competition LLM workflow (Stage-2 removed)."""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, MutableMapping, Sequence

from scpc.utils.dependencies import ensure_packages

ensure_packages([("yaml", "PyYAML")])

import yaml
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from scpc.llm.orchestrator import LLMOrchestrator, LLMRunConfig, validate_schema
from scpc.prompts import load_prompt
from scpc.schemas import load_schema

from .competition_config import CompetitionLLMConfig, StageThreeConfig

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

_STAGE3_OVERVIEW_WEEK_SQL_BASE = """
SELECT week, sunday
FROM vw_amz_comp_llm_overview
WHERE week = :week
{marketplace_filter}
ORDER BY sunday DESC
LIMIT 1
"""

_STAGE3_OVERVIEW_PREVIOUS_WEEK_SQL_BASE = """
SELECT week, sunday
FROM vw_amz_comp_llm_overview
WHERE sunday < :sunday
{marketplace_filter}
ORDER BY sunday DESC
LIMIT 1
"""

_STAGE3_PACKET_SQL_BASE = """
SELECT
  scene_tag,
  base_scene,
  morphology,
  marketplace_id,
  week,
  sunday,
  my_asin,
  opp_type,
  lag_type,
  evidence_json
FROM bi_amz_comp_llm_packet
WHERE week = :week
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
    "price_index_med": "lower_better",
    "price_gap_leader": "lower_better",
    "content_gap": "higher_better",
    "social_gap": "higher_better",
    "badge_delta_sum": "higher_better",
    "traffic_gap": "higher_better",
    "ad_ratio_index_med": "higher_better",
    "ad_to_natural_gap": "higher_better",
    "sp_share_in_ad_gap": "higher_better",
    "kw_top3_share_gap": "higher_better",
    "kw_brand_share_gap": "higher_better",
    "kw_competitor_share_gap": "higher_better",
    "pressure": "lower_better",
    "t_confidence": "higher_better",
    "confidence": "higher_better",
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
class CompetitionRunResult:
    week: str
    stage1_processed: int
    storage_paths: Sequence[Path]


EntityRole = Literal["self", "leader", "competitor"]
Channel = Literal["page", "traffic"]


@dataclass(slots=True)
class StageThreeEntityDelta:
    scene_context: Mapping[str, Any]
    week: str
    prev_week: str
    my_asin: str
    entity_asin: str | None
    entity_role: EntityRole
    opp_type: str | None
    channel: Channel
    metric_deltas: Mapping[str, Mapping[str, float | str | None]]
    leader_changed: bool | None = None


@dataclass(slots=True)
class StageThreeGapDelta:
    scene_context: Mapping[str, Any]
    week: str
    prev_week: str
    my_asin: str
    channel: Channel
    gap_deltas: Mapping[str, Mapping[str, float | str | None]]


@dataclass(slots=True)
class StageThreeDimensionChange:
    scene_context: Mapping[str, Any]
    week: str
    prev_week: str
    lag_type: str
    channel: Channel
    aggregates: Mapping[str, int | float]
    top_changes: Sequence[Mapping[str, Any]]


@dataclass(slots=True)
class StageThreeResult:
    context: Mapping[str, Any]
    self_entities: Sequence[StageThreeEntityDelta]
    leader_entities: Sequence[StageThreeEntityDelta]
    gap_deltas: Sequence[StageThreeGapDelta]
    dimensions: Sequence[StageThreeDimensionChange]


@dataclass(slots=True)
class StageThreeRunSummary:
    """Lightweight summary for the most recent Stage-3 execution referencing the current week and the comparison week."""

    week_w0: str | None
    week_w1: str | None
    scene_count: int
    record_count: int
    reason: str | None = None


_STAGE3_PAGE_CHANNEL: Channel = "page"

_PAGE_ENTITY_METRICS: tuple[str, ...] = (
    "price_net",
    "rank_score",
    "rank_pos_pct",
    "content_score",
    "social_proof",
    "confidence",
)

_PAGE_GAP_METRICS: tuple[str, ...] = (
    "price_gap_leader",
    "price_index_med",
    "content_gap",
    "social_gap",
    "badge_delta_sum",
    "pressure",
)

_TRAFFIC_ENTITY_METRICS: tuple[str, ...] = ("t_confidence",)

_TRAFFIC_GAP_METRICS: tuple[str, ...] = (
    "traffic_gap",
    "ad_ratio_index_med",
    "ad_to_natural_gap",
    "sp_share_in_ad_gap",
    "kw_top3_share_gap",
    "kw_brand_share_gap",
    "kw_competitor_share_gap",
)

_STAGE3_METRIC_LAG_MAP: Mapping[str, str] = {
    "price_net": "price",
    "rank_score": "rank",
    "rank_pos_pct": "rank",
    "content_score": "content",
    "social_proof": "social",
    "confidence": "confidence",
    "price_gap_leader": "price",
    "price_index_med": "price",
    "content_gap": "content",
    "social_gap": "social",
    "badge_delta_sum": "badge",
    "pressure": "confidence",
    "traffic_gap": "traffic_mix",
    "ad_ratio_index_med": "traffic_mix",
    "ad_to_natural_gap": "traffic_mix",
    "sp_share_in_ad_gap": "traffic_mix",
    "kw_top3_share_gap": "keyword",
    "kw_brand_share_gap": "keyword",
    "kw_competitor_share_gap": "keyword",
    "t_confidence": "confidence",
}


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
    """Coordinate the Stage-1 competition workflow using a rule engine plus a single LLM round."""

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
        self._current_marketplace_id: str | None = None
        self._brand_cache: dict[str, str | None] = {}
        self._brand_path_tokens: list[str] | None = None
        self._stage3_last_summary: StageThreeRunSummary | None = None

    @property
    def stage3_last_summary(self) -> StageThreeRunSummary | None:
        """Return the cached summary for the latest Stage-3 invocation."""

        return self._stage3_last_summary

    def run(
        self,
        week: str | None,
        *,
        marketplace_id: str | None = None,
        stages: Sequence[str] | None = None,
    ) -> CompetitionRunResult:
        """Execute Stage-1 for the provided week."""

        self._current_marketplace_id = marketplace_id
        target_week = self._resolve_week(week, marketplace_id)
        config = self._config.stage_3
        raw_stage_request = {str(stage).lower() for stage in (stages or ("stage1",)) if stage}
        if not raw_stage_request:
            raw_stage_request = {"stage1"}
        requested_stages = set(raw_stage_request)
        run_stage1 = "stage1" in requested_stages

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

        LOGGER.info(
            "competition_llm.end week=%s stage1=%s",
            target_week,
            len(stage1_outputs),
        )
        self._current_marketplace_id = None
        return CompetitionRunResult(
            week=target_week,
            stage1_processed=len(stage1_outputs),
            storage_paths=tuple(storage_paths),
        )

    def run_stage3(
        self,
        week: str | None,
        *,
        marketplace_id: str | None = None,
        previous_week: str | None = None,
        compare_tables: Mapping[str, Any] | None = None,
    ) -> Sequence[StageThreeResult]:
        """Compute Stage-3 structured change facts without invoking the LLM."""

        stage3_config = getattr(self._config, "stage_3", None)
        if not stage3_config or not stage3_config.enabled:
            self._stage3_last_summary = StageThreeRunSummary(
                week_w0=None,
                week_w1=None,
                scene_count=0,
                record_count=0,
                reason="disabled",
            )
            LOGGER.info("competition_llm.stage3_disabled")
            return ()

        config = stage3_config

        target_week = self._resolve_week(week, marketplace_id)
        compare_delta_rows: Sequence[Mapping[str, Any]] | None = None
        if compare_tables:
            compare_delta_rows = self._normalise_stage3_records(compare_tables.get("delta"))

        prev_week = self._infer_previous_week_from_delta(compare_delta_rows, target_week)
        if not prev_week and previous_week:
            prev_week = str(previous_week)
        if not prev_week:
            prev_week = self._resolve_previous_week(target_week, marketplace_id)

        def set_summary(
            reason: str | None,
            *,
            scene_count: int = 0,
            record_count: int = 0,
        ) -> None:
            self._stage3_last_summary = StageThreeRunSummary(
                week_w0=target_week,
                week_w1=prev_week,
                scene_count=scene_count,
                record_count=record_count,
                reason=reason,
            )

        if not prev_week:
            set_summary("previous_week_missing")
            LOGGER.warning(
                "competition_llm.stage3_prev_week_missing week=%s marketplace_id=%s",
                target_week,
                marketplace_id,
            )
            return ()

        overview_rows_curr = self._fetch_stage3_overview_rows(target_week, marketplace_id)
        overview_rows_prev = self._fetch_stage3_overview_rows(prev_week, marketplace_id)
        traffic_rows_curr = self._fetch_stage3_traffic_rows(target_week, marketplace_id)
        traffic_rows_prev = self._fetch_stage3_traffic_rows(prev_week, marketplace_id)
        packet_rows_curr = self._fetch_stage3_packets(target_week, marketplace_id)
        packet_rows_prev = self._fetch_stage3_packets(prev_week, marketplace_id)
        aligned_source_rows = self._align_stage3_source_rows(
            current_overview=overview_rows_curr,
            previous_overview=overview_rows_prev,
            current_traffic=traffic_rows_curr,
            previous_traffic=traffic_rows_prev,
            current_packets=packet_rows_curr,
            previous_packets=packet_rows_prev,
        )
        LOGGER.debug(
            "competition_llm.stage3_source_alignment week_w0=%s week_w1=%s marketplace_id=%s key_count=%s",
            target_week,
            prev_week,
            marketplace_id,
            len(aligned_source_rows),
        )
        leader_index_w0 = self._build_stage3_leader_index(packet_rows_curr)
        leader_index_w1 = self._build_stage3_leader_index(packet_rows_prev)
        self_entity_deltas, leader_entity_deltas = self._build_stage3_entity_deltas(
            overview_rows_curr,
            overview_rows_prev,
            traffic_rows_curr,
            traffic_rows_prev,
            leader_index_w0,
            leader_index_w1,
            config,
        )

        gap_deltas = self._build_stage3_gap_deltas(
            overview_rows_curr,
            overview_rows_prev,
            traffic_rows_curr,
            traffic_rows_prev,
            config,
        )
        dimension_changes = self._build_stage3_dimension_changes(
            self_entity_deltas,
            leader_entity_deltas,
            gap_deltas,
            config,
        )

        if not (
            self_entity_deltas
            or leader_entity_deltas
            or gap_deltas
            or dimension_changes
        ):
            reason = "no_stage3_data"
            set_summary(reason)
            LOGGER.info(
                "competition_llm.stage3_skipped week_w0=%s week_w1=%s marketplace_id=%s reason=%s",
                target_week,
                prev_week,
                marketplace_id,
                reason,
            )
            return ()

        results = self._assemble_stage3_results(
            self_entities=self_entity_deltas,
            leader_entities=leader_entity_deltas,
            gap_deltas=gap_deltas,
            dimension_changes=dimension_changes,
        )

        if not results:
            reason = "no_stage3_results"
            set_summary(reason)
            LOGGER.info(
                "competition_llm.stage3_skipped week_w0=%s week_w1=%s marketplace_id=%s reason=%s",
                target_week,
                prev_week,
                marketplace_id,
                reason,
            )
            return ()

        for result in results:
            try:
                self._write_stage3_output(result)
            except OSError as exc:  # pragma: no cover - unexpected filesystem issue
                LOGGER.warning(
                    "competition_llm.stage3_write_failed scene=%s error=%s",
                    result.context,
                    exc,
                )

        total_changes = sum(
            len(dimension.top_changes)
            for result in results
            for dimension in result.dimensions
        )

        set_summary(None, scene_count=len(results), record_count=total_changes)
        LOGGER.info(
            "competition_llm.stage3_completed week_w0=%s week_w1=%s scenes=%s records=%s",
            target_week,
            prev_week,
            len(results),
            total_changes,
        )

        return tuple(results)

    @staticmethod
    def _normalise_stage3_records(data: Any) -> Sequence[Mapping[str, Any]]:
        if data is None:
            return ()

        if hasattr(data, "to_dict"):
            to_dict = getattr(data, "to_dict")
            if callable(to_dict):
                try:
                    data = to_dict(orient="records")  # type: ignore[call-arg]
                except TypeError:
                    try:
                        data = to_dict("records")  # type: ignore[call-arg]
                    except TypeError:
                        data = to_dict()

        def sanitize(mapping: Mapping[str, Any]) -> dict[str, Any]:
            cleaned: dict[str, Any] = {}
            for key, value in mapping.items():
                if isinstance(value, float) and math.isnan(value):
                    cleaned[key] = None
                else:
                    cleaned[key] = value
            return cleaned

        def as_mapping(item: Any) -> Mapping[str, Any] | None:
            if isinstance(item, Mapping):
                return sanitize(dict(item))
            if hasattr(item, "_asdict") and callable(getattr(item, "_asdict")):
                return sanitize(dict(item._asdict()))
            return None

        if isinstance(data, Mapping):
            keys = list(data.keys())
            if not keys:
                return ()
            lengths = []
            rows: list[dict[str, Any]] = []
            for key in keys:
                values = data[key]
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    lengths.append(len(values))
                else:
                    return ()
            row_count = min(lengths) if lengths else 0
            for index in range(row_count):
                row = {key: data[key][index] for key in keys}
                rows.append(sanitize(row))
            return tuple(rows)

        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            rows = []
            for item in data:
                mapping = as_mapping(item)
                if mapping is not None:
                    rows.append(mapping)
            return tuple(rows)

        mapping = as_mapping(data)
        return (mapping,) if mapping is not None else ()

    @staticmethod
    def _infer_previous_week_from_delta(
        delta_rows: Sequence[Mapping[str, Any]] | None, target_week: str
    ) -> str | None:
        if not delta_rows:
            return None

        candidates: set[str] = set()
        for row in delta_rows:
            if not isinstance(row, Mapping):
                continue
            week_w0 = str(row.get("week_w0") or "")
            week_w1 = str(row.get("week_w1") or "")
            if week_w0 and week_w0 != target_week:
                continue
            if week_w1:
                candidates.add(week_w1)

        if len(candidates) == 1:
            return next(iter(candidates))
        if candidates:
            return sorted(candidates)[0]
        return None

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
    "StageThreeEntityDelta",
    "StageThreeGapDelta",
    "StageThreeDimensionChange",
    "StageThreeResult",
    "StageThreeRunSummary",
    "StageOneLLMResult",
    "StageOneResult",
]
