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
                storage_paths.append(self._write_stage2_output(item))
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
        for result in stage1_results:
            for dimension in result.dimensions:
                status = str(dimension.get("status", "")).lower()
                severity = str(dimension.get("severity", "")).lower()
                if status != "lag" or severity not in {"mid", "high"}:
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
                    "competition_llm.missing_packet context=%s dimension=%s",
                    context,
                    dimension,
                )
                continue
            lag_insight = self._fetch_optional_lag_insight(context, dimension)
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
                )
            )
        return tuple(outputs)

    def _write_stage2_output(self, result: StageTwoLLMResult) -> Path:
        context = result.context
        week = str(context.get("week"))
        asin = context.get("my_asin", "unknown")
        lag_type = result.lag_type or "unknown"
        opp_type = context.get("opp_type", "na")
        path = self._storage_root / week / "stage2" / f"{asin}_{lag_type}_{opp_type}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "machine_json": result.machine_json,
            "human_markdown": result.human_markdown,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        return path

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
        row = self._fetch_one(_STAGE2_PACKET_SQL, params)
        if not row:
            return None
        packet = dict(row)
        evidence = packet.get("evidence_json")
        if isinstance(evidence, str):
            try:
                packet["evidence_json"] = json.loads(evidence)
            except json.JSONDecodeError:
                LOGGER.warning("competition_llm.bad_evidence_json context=%s", context)
                return None
        return packet

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
            "lag_type": dimension.get("lag_type"),
            "opp_type": context.get("opp_type"),
        }
        missing = [key for key, value in params.items() if value is None]
        if missing:
            raise ValueError(f"Stage-2 parameter missing required fields: {missing}")
        return params

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
