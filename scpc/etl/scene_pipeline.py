"""Orchestrator for the Scene: AI 分析大盘 pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import date, timedelta
from time import perf_counter
from typing import Sequence

import pandas as pd
from sqlalchemy import bindparam, text

from scpc.db.engine import create_doris_engine
from scpc.db.io import replace_into
from scpc.etl.scene_clean import CleanResult, clean_keyword_panel
from scpc.etl.scene_drivers import compute_scene_drivers
from scpc.etl.scene_features import compute_scene_features

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
def _stage_timer(stage: str, scene: str, marketplace_id: str):
    start = perf_counter()
    LOGGER.info(
        "scene_pipeline_stage_start scene=%s mk=%s stage=%s",
        scene,
        marketplace_id,
        stage,
        extra={"scene": scene, "mk": marketplace_id, "stage": stage},
    )
    metrics: dict[str, object] = {}

    def _record(**kwargs: object) -> None:
        metrics.update(kwargs)

    try:
        yield _record
    except Exception:
        LOGGER.exception(
            "scene_pipeline_stage_error scene=%s mk=%s stage=%s",
            scene,
            marketplace_id,
            stage,
            extra={"scene": scene, "mk": marketplace_id, "stage": stage},
        )
        raise
    else:
        duration_ms = round((perf_counter() - start) * 1000, 2)
        LOGGER.info(
            "scene_pipeline_stage_complete scene=%s mk=%s stage=%s duration_ms=%s %s",
            scene,
            marketplace_id,
            stage,
            duration_ms,
            " ".join(f"{key}={value}" for key, value in metrics.items()),
            extra={
                "scene": scene,
                "mk": marketplace_id,
                "stage": stage,
                "duration_ms": duration_ms,
                **metrics,
            },
        )


def run_scene_pipeline(scene: str, marketplace_id: str, weeks_back: int, *, write: bool, topn: int) -> dict[str, pd.DataFrame]:
    engine = create_doris_engine()
    min_yrwk = _compute_min_yrwk(weeks_back)
    total_start = perf_counter()
    LOGGER.info(
        "scene_pipeline_start scene=%s mk=%s min_yrwk=%s",
        scene,
        marketplace_id,
        min_yrwk,
        extra={"scene": scene, "mk": marketplace_id, "min_yrwk": min_yrwk},
    )

    with _stage_timer("load_keywords", scene, marketplace_id) as record:
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

    with _stage_timer("load_facts", scene, marketplace_id) as record:
        with engine.connect() as conn:
            facts_df = pd.read_sql_query(
                FACT_SQL,
                conn,
                params={"mk": marketplace_id, "kw_list": kw_list, "min_yrwk": min_yrwk},
            )
            coverage_df = pd.read_sql_query(
                text(COVERAGE_SQL),
                conn,
                params={"scene": scene, "mk": marketplace_id, "min_yrwk": min_yrwk},
            )
        record(fact_rows=len(facts_df), coverage_rows=len(coverage_df))

    with _stage_timer("prepare_clean_data", scene, marketplace_id) as record:
        week_index = _collect_week_index(facts_df, coverage_df)
        clean_result = clean_keyword_panel(facts_df, week_index=week_index)
        clean_df = _prepare_clean_output(clean_result, scene, marketplace_id)
        record(clean_weeks=len(week_index), clean_rows=len(clean_df))

    with _stage_timer("compute_features", scene, marketplace_id) as record:
        features_result = compute_scene_features(
            clean_result.data,
            keywords_df,
            coverage_df,
            scene=scene,
            marketplace_id=marketplace_id,
        )
        features_df = _prepare_features_output(features_result.data)
        record(feature_rows=len(features_df))

    with _stage_timer("compute_drivers", scene, marketplace_id) as record:
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
        with _stage_timer("persist_results", scene, marketplace_id) as record:
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


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


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
    parser.add_argument("--topn", type=int, default=int(os.getenv("SCENE_TOPN", "10")), help="Driver TopN override")

    if argv is None:
        argv_list = sys.argv[1:]
    else:
        argv_list = list(argv)

    return parser.parse_args(_normalise_argv(argv_list))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    log_level = os.getenv("LOG_LEVEL", "INFO")
    _configure_logging(log_level)
    outputs = run_scene_pipeline(
        args.scene,
        args.mk,
        args.weeks_back,
        write=args.write,
        topn=args.topn,
    )
    if not args.write:
        summary = {name: len(df) for name, df in outputs.items()}
        print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
