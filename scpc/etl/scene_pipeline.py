"""Orchestrator for the Scene: AI 分析大盘 pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from scpc.db.engine import create_doris_engine
from scpc.db.io import replace_into
from scpc.etl.scene_clean import CleanResult, clean_keyword_panel
from scpc.etl.scene_drivers import compute_scene_drivers
from scpc.etl.scene_features import compute_scene_features
from scpc.llm.summarize_scene import SceneSummarizationError, summarize_scene
from scpc.reports.builder import build_scene_markdown

LOGGER = logging.getLogger(__name__)


KEYWORD_SQL = """
SELECT scene, keyword_norm, marketplace_id, weight
FROM bi_amz_vw_scene_keyword
WHERE scene = :scene AND marketplace_id = :mk AND is_active = 1
ORDER BY keyword_norm
"""

FACT_SQL = text(
    """
    SELECT marketplace_id, keyword_norm, year, week_num, startDate, endDate,
           vol, rank, clickShare, conversionShare,
           asin1, asin1_clickShare, asin1_conversionShare,
           asin2, asin2_clickShare, asin2_conversionShare,
           asin3, asin3_clickShare, asin3_conversionShare,
           update_time
    FROM bi_amz_vw_kw_week
    WHERE marketplace_id = :mk
      AND keyword_norm IN :kw_list
      AND (year * 100 + week_num) >= :min_yrwk
    ORDER BY year, week_num
    """
).bindparams(bindparam("kw_list", expanding=True))

COVERAGE_SQL = """
SELECT scene, marketplace_id, year, week_num, start_date, coverage
FROM bi_amz_mv_scene_week
WHERE scene = :scene AND marketplace_id = :mk
  AND (year * 100 + week_num) >= :min_yrwk
ORDER BY year, week_num
"""

LATEST_YEARWEEK_SQL = text(
    """
    SELECT MAX(year * 100 + week_num) AS yearweek
    FROM bi_amz_scene_features
    WHERE scene = :scene AND marketplace_id = :mk
    """
)


def _compute_min_yrwk(weeks_back: int) -> int:
    today = date.today()
    target = today - timedelta(weeks=weeks_back)
    iso = target.isocalendar()
    return iso[0] * 100 + iso[1]


def _collect_week_index(facts: pd.DataFrame, coverage: pd.DataFrame) -> list[date]:
    weeks: set[date] = set()
    if not facts.empty:
        if "startDate" in facts.columns:
            weeks.update(pd.to_datetime(facts["startDate"]).dropna().dt.date.tolist())
        if "year" in facts.columns and "week_num" in facts.columns:
            derived = [
                (date.fromisocalendar(int(row.year), int(row.week_num), 1) - timedelta(days=1))
                for row in facts[["year", "week_num"]].itertuples(index=False)
            ]
            weeks.update(derived)
    if coverage is not None and not coverage.empty:
        weeks.update(pd.to_datetime(coverage["start_date"]).dropna().dt.date.tolist())
    ordered = sorted(weeks)
    return ordered


def _latest_yearweek(outputs: dict[str, pd.DataFrame]) -> int | None:
    for name in ("features", "drivers", "clean"):
        frame = outputs.get(name)
        if frame is None or frame.empty:
            continue
        if {"year", "week_num"}.issubset(frame.columns):
            values = frame["year"].astype(int) * 100 + frame["week_num"].astype(int)
            if not values.empty:
                latest = int(values.max())
                return latest
    return None


def _fetch_latest_yearweek(engine: Engine, scene: str, marketplace_id: str) -> int | None:
    with engine.connect() as conn:
        result = conn.execute(
            LATEST_YEARWEEK_SQL,
            {"scene": scene, "mk": marketplace_id},
        )
        value = result.scalar()
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        LOGGER.warning(
            "scene_pipeline_yearweek_invalid scene=%s mk=%s value=%s",
            scene,
            marketplace_id,
            value,
            extra={"scene": scene, "mk": marketplace_id, "value": value},
        )
        return None


def _ensure_output_directory(base: Path, scene: str, marketplace_id: str, yearweek: int) -> Path:
    target = base / scene / marketplace_id / str(yearweek)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _sanitise_component(value: str, fallback: str) -> str:
    cleaned = value.strip() if value else ""
    if not cleaned:
        cleaned = fallback
    cleaned = cleaned.replace(os.sep, "_").replace("/", "_")
    cleaned = re.sub(r"[\\?%*:|\"<>]", "_", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _write_json_output(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("scene_pipeline_write_json path=%s", path, extra={"path": str(path)})


def _write_text_output(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    LOGGER.info("scene_pipeline_write_text path=%s", path, extra={"path": str(path)})


def _prepare_clean_output(clean: CleanResult, scene: str, marketplace_id: str) -> pd.DataFrame:
    frame = clean.data.copy()
    if frame.empty:
        return frame
    frame["scene"] = scene
    frame["marketplace_id"] = marketplace_id
    frame = frame[
        [
            "scene",
            "marketplace_id",
            "keyword_norm",
            "year",
            "week_num",
            "start_date",
            "vol_s",
            "gap_flag",
            "winsor_low",
            "winsor_high",
            "z",
        ]
    ]
    frame["start_date"] = pd.to_datetime(frame["start_date"]).dt.date
    frame["gap_flag"] = frame["gap_flag"].astype(int)
    return frame


def _prepare_features_output(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features
    output = features.copy()
    output["VOL"] = output["VOL"].fillna(0).round().astype(int)
    numeric_cols = [
        "wow",
        "yoy",
        "season",
        "wow_sa",
        "slope8",
        "breadth_wow_pos",
        "breadth_yoy_pos",
        "HHI_kw",
        "volatility_8w",
        "coverage",
        "new_kw_share",
        "forecast_p10",
        "forecast_p50",
        "forecast_p90",
        "confidence",
    ]
    for col in numeric_cols:
        if col in output.columns:
            output[col] = output[col].astype(float)
    output["start_date"] = pd.to_datetime(output["start_date"]).dt.date
    return output


def _prepare_drivers_output(drivers: pd.DataFrame) -> pd.DataFrame:
    if drivers.empty:
        return drivers
    output = drivers.copy()
    output["vol_delta"] = output["vol_delta"].fillna(0).round().astype(int)
    float_cols = ["contrib", "rank_delta", "clickShare_delta", "conversionShare_delta"]
    for col in float_cols:
        if col in output.columns:
            output[col] = output[col].astype(float)
    output["is_new_kw"] = output["is_new_kw"].astype(int)
    output["start_date"] = pd.to_datetime(output["start_date"]).dt.date
    return output


@contextmanager
def _stage_timer(stage: str, scene: str, marketplace_id: str, *, call: str | None = None):
    start = perf_counter()
    call_desc = call or stage
    LOGGER.info(
        "scene_pipeline_stage_start scene=%s mk=%s stage=%s call=%s",
        scene,
        marketplace_id,
        stage,
        call_desc,
        extra={"scene": scene, "mk": marketplace_id, "stage": stage, "call": call_desc},
    )
    metrics: dict[str, object] = {}

    def _record(**kwargs: object) -> None:
        metrics.update(kwargs)

    try:
        yield _record
    except OperationalError:
        LOGGER.error(
            "scene_pipeline_stage_error scene=%s mk=%s stage=%s call=%s",
            scene,
            marketplace_id,
            stage,
            call_desc,
            extra={"scene": scene, "mk": marketplace_id, "stage": stage, "call": call_desc},
            exc_info=False,
        )
        raise
    except Exception:
        LOGGER.exception(
            "scene_pipeline_stage_error scene=%s mk=%s stage=%s call=%s",
            scene,
            marketplace_id,
            stage,
            call_desc,
            extra={"scene": scene, "mk": marketplace_id, "stage": stage, "call": call_desc},
        )
        raise
    else:
        duration_ms = round((perf_counter() - start) * 1000, 2)
        LOGGER.info(
            "scene_pipeline_stage_complete scene=%s mk=%s stage=%s call=%s duration_ms=%s %s",
            scene,
            marketplace_id,
            stage,
            call_desc,
            duration_ms,
            " ".join(f"{key}={value}" for key, value in metrics.items()),
            extra={
                "scene": scene,
                "mk": marketplace_id,
                "stage": stage,
                "call": call_desc,
                "duration_ms": duration_ms,
                **metrics,
            },
        )


def run_scene_pipeline(
    scene: str,
    marketplace_id: str,
    weeks_back: int,
    *,
    engine: Engine | None = None,
    write: bool,
    topn: int,
) -> dict[str, pd.DataFrame]:
    close_engine = False
    if engine is None:
        engine = create_doris_engine()
        close_engine = True
    try:
        min_yrwk = _compute_min_yrwk(weeks_back)
        total_start = perf_counter()
        LOGGER.info(
            "scene_pipeline_start scene=%s mk=%s min_yrwk=%s",
            scene,
            marketplace_id,
            min_yrwk,
            extra={"scene": scene, "mk": marketplace_id, "min_yrwk": min_yrwk},
        )

        with _stage_timer(
            "load_keywords",
            scene,
            marketplace_id,
            call="pd.read_sql_query(bi_amz_vw_scene_keyword)",
        ) as record:
            with engine.connect() as conn:
                keywords_df = pd.read_sql_query(text(KEYWORD_SQL), conn, params={"scene": scene, "mk": marketplace_id})
            record(keyword_count=len(keywords_df))
        if keywords_df.empty:
            LOGGER.warning(
                "no_keywords scene=%s mk=%s stage=load_keywords",
                scene,
                marketplace_id,
                extra={"scene": scene, "mk": marketplace_id, "stage": "load_keywords"},
            )
            return {"clean": pd.DataFrame(), "features": pd.DataFrame(), "drivers": pd.DataFrame()}

        kw_list = keywords_df["keyword_norm"].unique().tolist()
        if not kw_list:
            LOGGER.warning(
                "no_keywords_active scene=%s mk=%s stage=load_keywords",
                scene,
                marketplace_id,
                extra={"scene": scene, "mk": marketplace_id, "stage": "load_keywords"},
            )
            return {"clean": pd.DataFrame(), "features": pd.DataFrame(), "drivers": pd.DataFrame()}

        with _stage_timer(
            "load_facts",
            scene,
            marketplace_id,
            call="pd.read_sql_query(bi_amz_vw_kw_week)",
        ) as record:
            with engine.connect() as conn:
                facts_df = pd.read_sql_query(
                    FACT_SQL,
                    conn,
                    params={"mk": marketplace_id, "kw_list": kw_list, "min_yrwk": min_yrwk},
                )
            record(fact_rows=len(facts_df))

        coverage_df = pd.DataFrame()
        with _stage_timer(
            "load_coverage",
            scene,
            marketplace_id,
            call="pd.read_sql_query(bi_amz_mv_scene_week)",
        ) as record:
            with engine.connect() as conn:
                coverage_df = pd.read_sql_query(
                    text(COVERAGE_SQL),
                    conn,
                    params={"scene": scene, "mk": marketplace_id, "min_yrwk": min_yrwk},
                )
            record(coverage_rows=len(coverage_df))

        with _stage_timer(
            "prepare_clean_data",
            scene,
            marketplace_id,
            call="clean_keyword_panel",
        ) as record:
            week_index = _collect_week_index(facts_df, coverage_df)
            clean_result = clean_keyword_panel(facts_df, week_index=week_index)
            clean_df = _prepare_clean_output(clean_result, scene, marketplace_id)
            record(clean_weeks=len(week_index), clean_rows=len(clean_df))

        with _stage_timer(
            "compute_features",
            scene,
            marketplace_id,
            call="compute_scene_features",
        ) as record:
            features_result = compute_scene_features(
                clean_result.data,
                keywords_df,
                coverage_df,
                scene=scene,
                marketplace_id=marketplace_id,
            )
            features_df = _prepare_features_output(features_result.data)
            record(feature_rows=len(features_df))

        with _stage_timer(
            "compute_drivers",
            scene,
            marketplace_id,
            call="compute_scene_drivers",
        ) as record:
            drivers_result = compute_scene_drivers(
                clean_result.data,
                facts_df,
                features_df,
                keywords_df,
                scene=scene,
                marketplace_id=marketplace_id,
                topn=topn,
            )
            drivers_df = _prepare_drivers_output(drivers_result.data)
            record(driver_rows=len(drivers_df))

        if write:
            with _stage_timer(
                "persist_results",
                scene,
                marketplace_id,
                call="replace_into(bi_amz_scene_kw_week_clean|features|drivers)",
            ) as record:
                clean_rows = replace_into(engine, "bi_amz_scene_kw_week_clean", clean_df) if not clean_df.empty else 0
                feature_rows = replace_into(engine, "bi_amz_scene_features", features_df) if not features_df.empty else 0
                driver_rows = replace_into(engine, "bi_amz_scene_drivers", drivers_df) if not drivers_df.empty else 0
                record(clean_rows=clean_rows, feature_rows=feature_rows, driver_rows=driver_rows)

        LOGGER.info(
            "scene_pipeline_complete scene=%s mk=%s clean_weeks=%s feature_rows=%s driver_rows=%s duration_ms=%s",
            scene,
            marketplace_id,
            len(clean_result.week_index),
            len(features_df),
            len(drivers_df),
            round((perf_counter() - total_start) * 1000, 2),
            extra={
                "scene": scene,
                "mk": marketplace_id,
                "clean_weeks": len(clean_result.week_index),
                "feature_rows": len(features_df),
                "driver_rows": len(drivers_df),
                "duration_ms": round((perf_counter() - total_start) * 1000, 2),
            },
        )
        return {"clean": clean_df, "features": features_df, "drivers": drivers_df}
    finally:
        if close_engine:
            engine.dispose()


def _configure_logging(level: str, log_file: Path | None = None) -> None:
    root = logging.getLogger()
    desired_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(desired_level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(desired_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(desired_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def _normalise_argv(argv: Sequence[str]) -> list[str]:
    normalised: list[str] = []
    for arg in argv:
        cleaned = arg.strip()
        if cleaned:
            normalised.append(cleaned)
    return normalised


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scene AI pipeline runner")
    parser.add_argument("--scene", required=True, help="Scene name")
    parser.add_argument("--mk", required=True, help="Marketplace identifier")
    parser.add_argument("--weeks-back", type=int, default=60, help="Number of weeks to backfill")
    parser.add_argument("--write", action="store_true", help="Persist results to Doris")
    parser.add_argument(
        "--scene-topn",
        "--topn",
        dest="scene_topn",
        type=int,
        help="Driver TopN override",
    )
    parser.add_argument("--with-llm", action="store_true", help="Invoke LLM summarisation after ETL")
    parser.add_argument("--emit-json", action="store_true", help="Write scene summary JSON to disk")
    parser.add_argument("--emit-md", action="store_true", help="Write Markdown report to disk")
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Skip ETL and only invoke DeepSeek summarisation using existing tables",
    )
    parser.add_argument(
        "--outputs-dir",
        default="storage/outputs/scene",
        help="Base directory for generated outputs",
    )

    if argv is None:
        argv_list = sys.argv[1:]
    else:
        argv_list = list(argv)

    namespace = parser.parse_args(_normalise_argv(argv_list))
    if namespace.llm_only and namespace.write:
        parser.error("--llm-only cannot be combined with --write")
    if namespace.llm_only:
        namespace.with_llm = True
    return namespace


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_dir = Path(os.getenv("SCPC_LOG_DIR", "storage/logs"))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_scene = _sanitise_component(args.scene, "scene")
    log_mk = _sanitise_component(args.mk, "mk")
    log_file = log_dir / f"scene_pipeline_{log_scene}_{log_mk}_{timestamp}.log"
    _configure_logging(log_level, log_file=log_file)
    LOGGER.info("scene_pipeline_log_file path=%s", log_file, extra={"path": str(log_file)})
    env_topn = os.getenv("SCENE_TOPN", "10")
    try:
        default_topn = int(env_topn)
    except ValueError:
        LOGGER.warning("invalid_scene_topn_env value=%s", env_topn)
        default_topn = 10
    topn = args.scene_topn if args.scene_topn is not None else default_topn
    engine = create_doris_engine()
    try:
        outputs: dict[str, pd.DataFrame] = {}
        if not args.llm_only:
            try:
                outputs = run_scene_pipeline(
                    args.scene,
                    args.mk,
                    args.weeks_back,
                    engine=engine,
                    write=args.write,
                    topn=topn,
                )
            except OperationalError as exc:
                LOGGER.error(
                    "scene_pipeline_connection_failed scene=%s mk=%s call=%s",
                    args.scene,
                    args.mk,
                    "run_scene_pipeline",
                    extra={"scene": args.scene, "mk": args.mk, "call": "run_scene_pipeline"},
                    exc_info=False,
                )
                host = os.getenv("DORIS_HOST", "<unknown>")
                port = os.getenv("DORIS_PORT", "<unknown>")
                details = getattr(exc.orig, "args", ())
                if isinstance(details, tuple) and details:
                    hint = " ".join(str(item) for item in details)
                elif details:
                    hint = str(details)
                else:
                    hint = str(exc)
                message = (
                    f"Failed to connect to Doris at {host}:{port}. "
                    "Please verify network connectivity, VPN access, and credentials. "
                    f"Original error: {hint}"
                )
                raise SystemExit(message)

        if not args.llm_only and not args.write:
            summary = {name: len(df) for name, df in outputs.items()}
            print(json.dumps(summary, default=str))

        summary_payload: Mapping[str, Any] | None = None
        summary_error: SceneSummarizationError | None = None
        if args.with_llm:
            features_df = outputs.get("features", pd.DataFrame())
            if not args.llm_only and features_df.empty:
                LOGGER.warning(
                    "scene_pipeline_llm_skipped scene=%s mk=%s reason=no_features",
                    args.scene,
                    args.mk,
                    extra={"scene": args.scene, "mk": args.mk, "reason": "no_features"},
                )
            else:
                try:
                    LOGGER.info(
                        "scene_pipeline_llm_start scene=%s mk=%s call=%s",
                        args.scene,
                        args.mk,
                        "summarize_scene",
                        extra={"scene": args.scene, "mk": args.mk, "call": "summarize_scene"},
                    )
                    summary_payload = summarize_scene(
                        engine=engine,
                        scene=args.scene,
                        mk=args.mk,
                        topn=topn,
                    )
                    LOGGER.info(
                        "scene_pipeline_llm_complete scene=%s mk=%s call=%s",
                        args.scene,
                        args.mk,
                        "summarize_scene",
                        extra={"scene": args.scene, "mk": args.mk, "call": "summarize_scene"},
                    )
                except SceneSummarizationError as exc:
                    summary_error = exc
                    message = "scene_pipeline_llm_failed scene=%s mk=%s call=%s error=%s" % (
                        args.scene,
                        args.mk,
                        "summarize_scene",
                        exc,
                    )
                    LOGGER.error(
                        message,
                        extra={
                            "scene": args.scene,
                            "mk": args.mk,
                            "call": "summarize_scene",
                            "error": str(exc),
                        },
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    summary_error = SceneSummarizationError(str(exc))
                    message = "scene_pipeline_llm_failed scene=%s mk=%s call=%s unexpected" % (
                        args.scene,
                        args.mk,
                        "summarize_scene",
                    )
                    LOGGER.exception(
                        message,
                        extra={"scene": args.scene, "mk": args.mk, "call": "summarize_scene"},
                    )

        if args.with_llm and (args.emit_json or args.emit_md):
            yearweek = _latest_yearweek(outputs) if outputs else None
            if yearweek is None and args.llm_only:
                yearweek = _fetch_latest_yearweek(engine, args.scene, args.mk)
            if yearweek is None:
                LOGGER.warning(
                    "scene_pipeline_outputs_skipped scene=%s mk=%s reason=no_yearweek",
                    args.scene,
                    args.mk,
                    extra={"scene": args.scene, "mk": args.mk, "reason": "no_yearweek"},
                )
            else:
                outdir = _ensure_output_directory(Path(args.outputs_dir), args.scene, args.mk, yearweek)
                if args.emit_json:
                    if summary_payload is not None:
                        _write_json_output(outdir / "scene_summary.json", summary_payload)
                    elif summary_error is not None:
                        error_payload = {
                            "error": str(summary_error),
                            "details": getattr(summary_error, "details", []),
                        }
                        raw = getattr(summary_error, "raw", None)
                        if raw is not None:
                            error_payload["raw"] = raw
                        _write_json_output(outdir / "scene_summary.errors.json", error_payload)
                if args.emit_md and summary_payload is not None:
                    try:
                        markdown = build_scene_markdown(summary_payload)
                        _write_text_output(outdir / "scene_report.md", markdown)
                    except Exception:  # pragma: no cover - defensive guard
                        LOGGER.warning(
                            "scene_pipeline_markdown_failed scene=%s mk=%s call=%s",
                            args.scene,
                            args.mk,
                            "build_scene_markdown",
                            extra={"scene": args.scene, "mk": args.mk, "call": "build_scene_markdown"},
                            exc_info=True,
                        )
        elif args.emit_json or args.emit_md:
            LOGGER.info(
                "scene_pipeline_emit_skipped scene=%s mk=%s reason=llm_disabled",
                args.scene,
                args.mk,
                extra={"scene": args.scene, "mk": args.mk, "reason": "llm_disabled"},
            )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
