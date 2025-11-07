"""Stage-1 and Stage-2 orchestration for the competition LLM workflow."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

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

_STAGE2_PACKET_SQL = """
SELECT *
FROM bi_amz_comp_llm_packet
WHERE scene_tag = :scene_tag
  AND base_scene = :base_scene
  AND COALESCE(morphology, '') = COALESCE(:morphology, '')
  AND marketplace_id = :marketplace_id
  AND week = :week
  AND sunday = :sunday
  AND my_asin = :my_asin
  AND lag_type = :lag_type
  AND opp_type = :opp_type
"""

_STAGE2_LAG_INSIGHT_SQL = """
SELECT reason_code, severity, reason_detail, top_opp_asins_csv
FROM bi_amz_comp_lag_insights
WHERE scene_tag = :scene_tag
  AND base_scene = :base_scene
  AND COALESCE(morphology, '') = COALESCE(:morphology, '')
  AND marketplace_id = :marketplace_id
  AND week = :week
  AND sunday = :sunday
  AND my_asin = :my_asin
  AND lag_type = :lag_type
  AND opp_type = :opp_type
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


def _normalize_lag_type(value: object) -> str:
    v = str(value or "").strip().lower()
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
    return mapping.get(v, v)


@dataclass(slots=True)
class StageOneLLMResult:
    context: Mapping[str, Any]
    summary: str
    dimensions: Sequence[Mapping[str, Any]]


@dataclass(slots=True)
class StageTwoLLMResult:
    context: Mapping[str, Any]
    lag_type: str
    machine_json: Mapping[str, Any]
    human_markdown: str
    evidence: Mapping[str, Any]


@dataclass(slots=True)
class CompetitionRunResult:
    week: str
    stage1_processed: int
    stage2_candidates: int
    stage2_processed: int
    storage_paths: Sequence[Path]


class CompetitionLLMOrchestrator:
    """Coordinate the Stage-1/Stage-2 competition LLM workflow."""

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
        self._stage1_schema = load_schema("competition_stage1.schema.json")
        self._stage2_schema = load_schema("competition_stage2.schema.json")
        self._stage1_prompt = load_prompt("competition_stage1.md")
        self._stage2_prompt = load_prompt("competition_stage2.md")

    def run(self, week: str | None, *, marketplace_id: str | None = None) -> CompetitionRunResult:
        """Execute Stage-1 and Stage-2 for the provided week."""

        target_week = self._resolve_week(week, marketplace_id)
        LOGGER.info(
            "competition_llm.start week=%s marketplace_id=%s",
            target_week,
            marketplace_id,
        )

        stage1_inputs = self._collect_stage1_inputs(target_week, marketplace_id)
        stage1_outputs = self._execute_stage1(stage1_inputs)
        storage_paths: list[Path] = []
        for item in stage1_outputs:
            storage_paths.append(self._write_stage1_output(item))

        stage2_candidates = self._prepare_stage2_candidates(stage1_outputs)
        stage2_outputs: list[StageTwoLLMResult] = []
        if self._config.stage_2.enabled and stage2_candidates:
            stage2_outputs = self._execute_stage2(stage2_candidates)
            for item in stage2_outputs:
                storage_paths.extend(self._write_stage2_output(item))
        else:
            LOGGER.info(
                "competition_llm.stage2_skipped enabled=%s candidate_count=%s",
                self._config.stage_2.enabled,
                len(stage2_candidates),
            )

        LOGGER.info(
            "competition_llm.end week=%s stage1=%s stage2_candidates=%s stage2=%s",
            target_week,
            len(stage1_outputs),
            len(stage2_candidates),
            len(stage2_outputs),
        )
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
        return inputs

    def _execute_stage1(
        self,
        inputs: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]],
    ) -> Sequence[StageOneLLMResult]:
        results: list[StageOneLLMResult] = []
        for context, overview_rows, traffic_rows in inputs:
            payload = {
                "context": context,
                "overview_rows": overview_rows,
                "traffic_rows": traffic_rows,
                "thresholds": dict(self._config.stage_1.thresholds),
                "conf_min": self._config.stage_1.conf_min,
                "band_weight": self._config.stage_1.band_weight,
                "opp_weight": self._config.stage_1.opp_weight,
                "output_language": "zh",
                "response_schema": self._stage1_schema,
            }
            llm_result = self._invoke_with_retries(
                payload,
                schema=self._stage1_schema,
                prompt=self._stage1_prompt,
                max_attempts=self._config.stage_1.max_retries,
            )
            context_result = llm_result.get("context")
            if not context_result:
                raise ValueError("Stage-1 response missing context")
            summary = llm_result.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("Stage-1 response missing summary")
            dimensions = llm_result.get("dimensions")
            if not isinstance(dimensions, Sequence) or not dimensions:
                raise ValueError("Stage-1 response missing dimensions")
            results.append(
                StageOneLLMResult(
                    context=context_result,
                    summary=summary.strip(),
                    dimensions=tuple(dimensions),
                )
            )
        return tuple(results)

    def _write_stage1_output(self, result: StageOneLLMResult) -> Path:
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

    def _prepare_stage2_candidates(self, stage1_results: Sequence[StageOneLLMResult]) -> Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        candidates: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        threshold = self._config.stage_1.conf_min
        allowed_statuses = tuple(
            status.lower() for status in getattr(self._config.stage_2, "trigger_status", ("lag",))
        ) or ("lag",)
        allowed_set = set(allowed_statuses)
        for result in stage1_results:
            for dimension in result.dimensions:
                if isinstance(dimension, Mapping) and "lag_type" in dimension:
                    dimension = dict(dimension)
                    dimension["lag_type"] = _normalize_lag_type(dimension.get("lag_type"))
                raw_status = str(dimension.get("status", "")).lower()
                status_aliases = {raw_status}
                if raw_status == "parity":
                    status_aliases.add("neutral")
                if raw_status == "neutral":
                    status_aliases.add("parity")
                if allowed_set and status_aliases.isdisjoint(allowed_set):
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
                candidates.append((result.context, dimension))
        return tuple(candidates)

    def _execute_stage2(
        self, candidates: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ) -> Sequence[StageTwoLLMResult]:
        outputs: list[StageTwoLLMResult] = []
        for context, dimension in candidates:
            packet = self._fetch_stage2_packet(context, dimension)
            if packet is None:
                LOGGER.warning(
                    "competition_llm.missing_stage2_evidence context=%s dimension=%s",
                    context,
                    dimension,
                )
                continue
            lag_insight = self._fetch_optional_lag_insight(context, dimension)
            evidence_payload = packet.get("evidence_json") or {}
            facts = {
                "first_round_item": {
                    "context": context,
                    "lag_type": dimension.get("lag_type"),
                    "status": dimension.get("status"),
                    "severity": dimension.get("severity"),
                    "source_opp_type": dimension.get("source_opp_type"),
                    "source_confidence": dimension.get("source_confidence"),
                },
                "llm_packet": packet,
                "lag_insight": lag_insight or {},
                "machine_json_schema": self._stage2_schema,
                "allowed_action_codes": list(self._config.stage_2.allowed_action_codes),
                "allowed_root_cause_codes": list(self._config.stage_2.allowed_root_cause_codes),
                "output_language": "zh",
            }
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
            self._validate_stage2_machine_json(machine_json)
            if not isinstance(human_markdown, str):
                raise ValueError("Stage-2 human_markdown must be a string")
            outputs.append(
                StageTwoLLMResult(
                    context=machine_json.get("context", context),
                    lag_type=str(machine_json.get("lag_type", dimension.get("lag_type", ""))),
                    machine_json=machine_json,
                    human_markdown=human_markdown,
                    evidence=evidence_payload,
                )
            )
        return tuple(outputs)

    def _write_stage2_output(self, result: StageTwoLLMResult) -> Sequence[Path]:
        context = result.context
        week = str(context.get("week"))
        asin = context.get("my_asin", "unknown")
        lag_type = result.lag_type or "unknown"
        opp_type = context.get("opp_type", "na")
        stage2_dir = self._storage_root / week / "stage2"
        stage2_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{asin}_{lag_type}_{opp_type}"
        main_path = stage2_dir / f"{base_name}.json"
        payload = {
            "machine_json": result.machine_json,
            "human_markdown": result.human_markdown,
        }
        main_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

        summary_data = self._build_stage2_summary(result)
        summary_paths: list[Path] = [main_path]
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

    def _build_stage2_summary(self, result: StageTwoLLMResult) -> Mapping[str, Any] | None:
        evidence = result.evidence or {}
        if not evidence:
            return None
        overview = dict(evidence.get("overview") or {})
        metrics_cfg = _OVERVIEW_METRIC_CONFIG.get(result.lag_type, {})
        leader_summary = _compose_overview_summary(overview, metrics_cfg.get("leader", ()))
        median_summary = _compose_overview_summary(overview, metrics_cfg.get("median", ()))
        context_snapshot = {field: result.context.get(field) for field in _REQUIRED_CONTEXT_FIELDS}
        summary: dict[str, Any] = {
            "context": context_snapshot,
            "lag_type": result.lag_type,
            "channel": evidence.get("channel"),
            "vs_leader": leader_summary,
            "vs_median": median_summary,
            "top_diff_reasons": list(evidence.get("top_diff_reasons") or ()),
        }
        if overview:
            summary["overview"] = overview
        top_opps = evidence.get("top_opps")
        if top_opps:
            summary["top_opps"] = top_opps
        top_csv = evidence.get("top_opp_asins_csv")
        if top_csv:
            summary["top_opp_asins_csv"] = top_csv
        return summary

    def _render_stage2_summary_markdown(self, summary: Mapping[str, Any]) -> str:
        context = summary.get("context", {})
        lines: list[str] = []
        title = f"# Stage-2 摘要（{summary.get('lag_type', 'unknown')}）"
        lines.append(title)
        lines.append("")
        lines.append(f"- 周次：{context.get('week', '未知')}")
        lines.append(f"- ASIN：{context.get('my_asin', '未知')}")
        lines.append(f"- 对手类型：{context.get('opp_type', '未知')}")
        lines.append("")

        leader = summary.get("vs_leader", {})
        lines.append("## 本品 vs 领先者")
        leader_summary = leader.get("summary") or "暂无数据"
        lines.append(f"- {leader_summary}")
        leader_metrics = leader.get("metrics") or {}
        if leader_metrics:
            lines.append("")
            lines.append("**关键指标：**")
            for key, value in leader_metrics.items():
                lines.append(f"- {key}: {_format_metric_value(value)}")
        lines.append("")

        median = summary.get("vs_median", {})
        lines.append("## 本品 vs 中位数")
        median_summary = median.get("summary") or "暂无数据"
        lines.append(f"- {median_summary}")
        median_metrics = median.get("metrics") or {}
        if median_metrics:
            lines.append("")
            lines.append("**关键指标：**")
            for key, value in median_metrics.items():
                lines.append(f"- {key}: {_format_metric_value(value)}")
        lines.append("")

        reasons = summary.get("top_diff_reasons") or []
        lines.append("## Top3 差距原因")
        if not reasons:
            lines.append("- 暂无数据")
        else:
            for item in reasons:
                label = _format_opp_label(item)
                summary_text = item.get("summary") or ""
                lines.append(f"- {label}：{summary_text}")
                metrics = item.get("metrics") or {}
                if metrics:
                    lines.append("  - 指标：")
                    for key, value in metrics.items():
                        lines.append(f"    - {key}: {_format_metric_value(value)}")

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
        """Normalize raw confidence values emitted by Stage-1."""

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

    def _fetch_stage2_packet(self, context: Mapping[str, Any], dimension: Mapping[str, Any]) -> Mapping[str, Any] | None:
        params = self._build_stage2_params(context, dimension)
        lag_type = params.get("lag_type")
        opp_type = str(params.get("opp_type"))
        common = {
            **params,
            "my_parent_asin": context.get("my_parent_asin"),
        }

        if lag_type in ("price", "rank", "content", "social", "badge", "confidence"):
            evidence = self._build_page_evidence(common, lag_type, opp_type)
        elif lag_type in ("traffic_mix", "keyword"):
            evidence = self._build_traffic_evidence(common, lag_type, opp_type)
        else:
            LOGGER.warning(
                "competition_llm.unsupported_lag_type lag_type=%s context=%s",
                lag_type,
                context,
            )
            return None

        if not evidence:
            return None

        return {
            **common,
            "reason_code": evidence.get("reason_code", f"{lag_type}_auto"),
            "severity": evidence.get("severity"),
            "evidence_json": evidence,
            "prompt_hint": evidence.get(
                "prompt_hint", "仅使用 evidence_json 中已给出的指标，不要自行重算。"
            ),
        }

    def _fetch_optional_lag_insight(
        self, context: Mapping[str, Any], dimension: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        params = self._build_stage2_params(context, dimension)
        row = self._fetch_one(_STAGE2_LAG_INSIGHT_SQL, params)
        return dict(row) if row else None

    def _build_stage2_params(self, context: Mapping[str, Any], dimension: Mapping[str, Any]) -> dict[str, Any]:
        params = {
            "scene_tag": context.get("scene_tag"),
            "base_scene": context.get("base_scene"),
            "morphology": context.get("morphology"),
            "marketplace_id": context.get("marketplace_id"),
            "week": context.get("week"),
            "sunday": context.get("sunday"),
            "my_asin": context.get("my_asin"),
            "lag_type": _normalize_lag_type(dimension.get("lag_type")),
            "opp_type": context.get("opp_type"),
        }
        missing = [key for key, value in params.items() if value is None]
        if missing:
            raise ValueError(f"Stage-2 parameter missing required fields: {missing}")
        return params

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
                payload = dict(facts)
                payload["validation_error"] = str(exc)
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
    keys = _COMPARISON_METRIC_KEYS.get(lag_type, ("score",))
    for row in top_rows[:3]:
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
    "StageOneLLMResult",
    "StageTwoLLMResult",
]
