"""Command line runner for competition data cleaning and feature ingestion."""

from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scpc.utils.dependencies import ensure_packages

ensure_packages(["pandas", "numpy", "sqlalchemy"])

import numpy as np
import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from scpc.db.engine import create_doris_engine
from scpc.db.io import replace_into
from scpc.llm.competition_config import load_competition_llm_config
from scpc.llm.competition_workflow import CompetitionLLMOrchestrator
from scpc.llm.deepseek_client import create_client_from_env
from scpc.llm.orchestrator import LLMOrchestrator
from scpc.etl.competition_features import (
    build_competition_tables_from_entities,
    build_traffic_features,
    clean_competition_entities,
)

LOGGER = logging.getLogger(__name__)


SNAPSHOT_SQL = """
SELECT asin, marketplace_id, week, sunday, parent_asin,
       price_current, price_list, coupon_pct,
       rank_root, rank_leaf, rating, reviews,
       image_cnt, video_cnt, bullet_cnt, title_len,
       aplus_flag, badge_json
FROM bi_amz_asin_product_snapshot
WHERE marketplace_id = :mk AND week = :week
"""

LATEST_SNAPSHOT_WEEK_SQL = """
SELECT week, sunday
FROM bi_amz_asin_product_snapshot
WHERE marketplace_id = :mk
ORDER BY sunday DESC
LIMIT 1
"""

SCENE_TAG_SQL = """
SELECT scene_tag, base_scene, morphology, asin, marketplace_id, hyy_asin
FROM bi_amz_asin_scene_tag
WHERE marketplace_id = :mk
"""

# Doris 2.1 rewrites ``monday = ?`` predicates on the weekly flow table into
# invalid expressions when pushing them down to the daily source. Avoid the
# rewrite by removing the ``monday`` predicate from the SQL and performing the
# date filter client-side.
FLOW_SQL_BASE = """
SELECT asin,
       marketplace_id,
       monday,
       `广告流量占比` AS ad_ratio,
       `自然流量占比` AS nf_ratio,
       `推荐流量占比` AS recommend_ratio,
       `SP广告流量占比` AS sp_ratio,
       `视频广告流量占比` AS sbv_ratio,
       `品牌广告流量占比` AS sb_ratio
FROM hyy.bi_sif_asin_flow_overview_weekly
WHERE marketplace_id = :mk
"""

TRAFFIC_ONLY_COLUMNS: set[str] = {
    "ad_ratio",
    "nf_ratio",
    "recommend_ratio",
    "sp_ratio",
    "sbv_ratio",
    "sb_ratio",
    "sp_share_in_ad",
    "sbv_share_in_ad",
    "sb_share_in_ad",
    "ad_to_natural",
    "kw_entropy_7d_avg",
    "kw_hhi_7d_avg",
    "kw_top1_share_7d_avg",
    "kw_top3_share_7d_avg",
    "kw_top10_share_7d_avg",
    "kw_brand_share_7d_avg",
    "kw_competitor_share_7d_avg",
    "kw_generic_share_7d_avg",
    "kw_attribute_share_7d_avg",
    "kw_days_covered",
    "kw_coverage_ratio",
}

ENTITIES_TABLE = "bi_amz_comp_entities_clean"
TRAFFIC_TABLE = "bi_amz_comp_traffic_entities_weekly"
PAIRS_TABLE = "bi_amz_comp_pairs"
TRAFFIC_PAIRS_TABLE = "bi_amz_comp_traffic_pairs"
PAIRS_EACH_TABLE = "bi_amz_comp_pairs_each"
TRAFFIC_PAIRS_EACH_TABLE = "bi_amz_comp_traffic_pairs_each"
DELTA_TABLE = "bi_amz_comp_delta"
SCENE_WEEK_TABLE = "bi_amz_comp_scene_week_metrics"

ENTITIES_SELECT_SQL = f"SELECT * FROM {ENTITIES_TABLE} WHERE marketplace_id = :mk AND week = :week"
TRAFFIC_SELECT_SQL = f"SELECT * FROM {TRAFFIC_TABLE} WHERE marketplace_id = :mk AND week = :week"

PAIRS_SELECT_SQL = (
    f"SELECT week, sunday FROM {PAIRS_TABLE} "
    "WHERE marketplace_id = :mk ORDER BY sunday DESC LIMIT 1"
)

KEYWORD_SQL = """
SELECT asin, marketplace_id, keyword, snapshot_date, ratio_score
FROM vw_sif_keyword_daily_std
WHERE marketplace_id = :mk
  AND snapshot_date BETWEEN :start_date AND :end_date
"""

KEYWORD_TAG_SQL = """
SELECT keyword, tag
FROM bi_amz_comp_kw_tag
WHERE is_active = 1
"""


def _iso_week_to_dates(week: str) -> tuple[date, date]:
    """Return the Monday and Sunday for an ISO formatted ``YYYYWww`` label."""

    if not week:
        raise ValueError(f"Invalid ISO week label: {week}")

    normalized = week.strip()
    match = re.fullmatch(r"(\d{4})-?W(\d{1,2})", normalized)
    if not match:
        raise ValueError(f"Invalid ISO week label: {week}")

    year = int(match.group(1))
    week_num = int(match.group(2))
    if week_num < 1 or week_num > 53:
        raise ValueError(f"Invalid ISO week label: {week}")

    monday = date.fromisocalendar(year, week_num, 1)
    sunday = monday + timedelta(days=6)
    return monday, sunday


def _previous_week_label(week: str) -> str:
    """Return the ISO label for the week preceding ``week``."""

    monday, _ = _iso_week_to_dates(week)
    previous_monday = monday - timedelta(days=7)
    iso = previous_monday.isocalendar()
    return f"{iso.year}W{iso.week:02d}"


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _read_dataframe(engine: Engine, sql: str, params: Mapping[str, object]) -> pd.DataFrame:
    stmt = text(sql)
    with engine.connect() as conn:
        result = conn.execute(stmt, params)
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=result.keys())
        return pd.DataFrame(rows, columns=result.keys())


def _normalise_flow_dataframe(flow: pd.DataFrame) -> pd.DataFrame:
    """Add calendar fields to weekly flow rows without touching the SQL layer."""

    if flow.empty:
        columns = [
            "asin",
            "marketplace_id",
            "monday",
            "ad_ratio",
            "nf_ratio",
            "recommend_ratio",
            "sp_ratio",
            "sbv_ratio",
            "sb_ratio",
            "sunday",
            "week",
        ]
        return pd.DataFrame(columns=columns)

    df = flow.copy()
    monday_ts = pd.to_datetime(df.get("monday"), errors="coerce")
    df["monday"] = monday_ts.dt.date

    sunday_ts = monday_ts + pd.Timedelta(days=6)
    df["sunday"] = sunday_ts.dt.date

    iso = monday_ts.dt.isocalendar()
    df["week"] = iso["year"].astype(str) + "W" + iso["week"].astype(str).str.zfill(2)

    missing_mask = monday_ts.isna()
    if missing_mask.any():
        df.loc[missing_mask, "week"] = pd.NA
        df.loc[missing_mask, "sunday"] = pd.NA

    preferred_order = [
        "asin",
        "marketplace_id",
        "monday",
        "ad_ratio",
        "nf_ratio",
        "recommend_ratio",
        "sp_ratio",
        "sbv_ratio",
        "sb_ratio",
        "sunday",
        "week",
    ]
    columns = [col for col in preferred_order if col in df.columns]
    extra_columns = [col for col in df.columns if col not in columns]
    return df.loc[:, columns + extra_columns]


def _augment_in_clause(
    base_sql: str,
    column: str,
    values: Sequence[object],
    prefix: str,
) -> tuple[str, dict[str, object]]:
    """Append an ``IN`` clause with bound parameters when ``values`` is not empty."""

    if not values:
        return base_sql, {}

    tokens: list[str] = []
    params: dict[str, object] = {}
    for idx, value in enumerate(values):
        key = f"{prefix}_{idx}"
        tokens.append(f":{key}")
        params[key] = value

    clause = ", ".join(tokens)
    sql = base_sql + f" AND {column} IN ({clause})"
    return sql, params


def _augment_scene_filters(base_sql: str, filters: Sequence[str] | None) -> tuple[str, dict[str, object]]:
    if not filters:
        return base_sql, {}
    tokens: list[str] = []
    params: dict[str, object] = {}
    for idx, value in enumerate(filters):
        key = f"scene_tag_{idx}"
        tokens.append(f":{key}")
        params[key] = value
    clause = ", ".join(tokens)
    sql = base_sql + f" AND scene_tag IN ({clause})"
    return sql, params


def _prepare_traffic_entities(
    traffic: pd.DataFrame,
    scene_tags: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich traffic features with scene metadata for Doris ingestion."""

    columns = [
        "scene_tag",
        "base_scene",
        "morphology",
        "marketplace_id",
        "week",
        "sunday",
        "asin",
        "parent_asin",
        "hyy_asin",
        "ad_ratio",
        "nf_ratio",
        "recommend_ratio",
        "sp_ratio",
        "sbv_ratio",
        "sb_ratio",
        "sp_share_in_ad",
        "sbv_share_in_ad",
        "sb_share_in_ad",
        "ad_to_natural",
        "kw_entropy_7d_avg",
        "kw_hhi_7d_avg",
        "kw_top1_share_7d_avg",
        "kw_top3_share_7d_avg",
        "kw_top10_share_7d_avg",
        "kw_brand_share_7d_avg",
        "kw_competitor_share_7d_avg",
        "kw_generic_share_7d_avg",
        "kw_attribute_share_7d_avg",
        "kw_days_covered",
        "kw_coverage_ratio",
    ]

    if traffic.empty:
        return pd.DataFrame(columns=columns)

    traffic_df = traffic.copy()
    scene_cols = ["asin", "marketplace_id", "scene_tag", "base_scene", "morphology", "hyy_asin"]
    tags = scene_tags.loc[:, scene_cols].drop_duplicates()
    merged = traffic_df.merge(tags, on=["asin", "marketplace_id"], how="left")

    parent_map = (
        snapshots.loc[:, ["asin", "marketplace_id", "parent_asin"]]
        .drop_duplicates()
        .rename(columns={"parent_asin": "parent_asin_snapshot"})
    )
    merged = merged.merge(
        parent_map,
        on=["asin", "marketplace_id"],
        how="left",
    )
    if "parent_asin" not in merged:
        merged["parent_asin"] = pd.NA
    parent_series = merged["parent_asin"]
    snapshot_parent = merged.get("parent_asin_snapshot")
    if snapshot_parent is not None:
        merged["parent_asin"] = parent_series.where(parent_series.notna(), snapshot_parent)
    merged = merged.drop(columns=[col for col in merged.columns if col.endswith("_snapshot")])

    merged = merged.loc[merged["scene_tag"].notna()].copy()
    if merged.empty:
        return pd.DataFrame(columns=columns)

    merged["hyy_asin"] = merged.get("hyy_asin", 0).fillna(0).astype(int)
    merged["sunday"] = pd.to_datetime(merged["sunday"]).dt.date

    for share_col in (
        "ad_ratio",
        "nf_ratio",
        "recommend_ratio",
        "sp_ratio",
        "sbv_ratio",
        "sb_ratio",
        "sp_share_in_ad",
        "sbv_share_in_ad",
        "sb_share_in_ad",
        "ad_to_natural",
        "kw_entropy_7d_avg",
        "kw_hhi_7d_avg",
        "kw_top1_share_7d_avg",
        "kw_top3_share_7d_avg",
        "kw_top10_share_7d_avg",
        "kw_brand_share_7d_avg",
        "kw_competitor_share_7d_avg",
        "kw_generic_share_7d_avg",
        "kw_attribute_share_7d_avg",
        "kw_coverage_ratio",
    ):
        if share_col in merged:
            merged[share_col] = pd.to_numeric(merged[share_col], errors="coerce")

    merged["kw_days_covered"] = pd.to_numeric(
        merged.get("kw_days_covered"), errors="coerce"
    ).fillna(0).astype(int)

    merged = merged.drop_duplicates(
        subset=["scene_tag", "marketplace_id", "week", "asin"], keep="last"
    )
    return merged.reindex(columns=columns)


def _filter_scene_tags(scene_tags: pd.DataFrame, allowed: Iterable[str]) -> pd.DataFrame:
    allowed_set = set(allowed)
    if not allowed_set:
        return scene_tags
    return scene_tags.loc[scene_tags["scene_tag"].isin(allowed_set)].copy()


def _prune_traffic_columns(
    entities: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    """Remove traffic-only feature columns from entity records."""

    drop_cols = {column for column in TRAFFIC_ONLY_COLUMNS if column in entities.columns}
    if not drop_cols:
        return entities, set()

    pruned = entities.drop(columns=sorted(drop_cols))
    return pruned, drop_cols


def _load_table_columns(engine: Engine, table: str) -> list[str]:
    """Return the column names for ``table`` in declaration order."""

    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(table)
    except SQLAlchemyError as exc:  # pragma: no cover - database connectivity guard
        LOGGER.warning(
            "competition_pipeline_table_columns_failed table=%s error=%s",
            table,
            exc,
        )
        return []
    except Exception as exc:  # pragma: no cover - fallback for non-SQLAlchemy errors
        LOGGER.warning(
            "competition_pipeline_table_columns_failed table=%s error=%s",
            table,
            exc,
        )
        return []

    ordered = []
    for column in columns:
        name = column.get("name")
        if name:
            ordered.append(name)
    return ordered


def _prune_to_table(
    df: pd.DataFrame,
    table_columns: Sequence[str] | None,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    """Align ``df`` with ``table_columns`` by dropping extras and preserving order."""

    if df.empty or not table_columns:
        return df, set(), set()

    allowed = [column for column in table_columns if column]
    allowed_set = set(allowed)

    drop_cols = {column for column in df.columns if column not in allowed_set}
    pruned = df.drop(columns=sorted(drop_cols)) if drop_cols else df

    # Reindex columns to match the table declaration order.
    ordered_columns = [column for column in allowed if column in pruned.columns]
    if ordered_columns:
        pruned = pruned.loc[:, ordered_columns]

    missing = {column for column in allowed if column not in pruned.columns}
    return pruned, drop_cols, missing


def _merge_entities_with_traffic(
    entities: pd.DataFrame,
    traffic: pd.DataFrame,
) -> pd.DataFrame:
    """Combine page and traffic feature tables into a single entity frame."""

    if entities.empty:
        return entities.copy()

    if traffic.empty:
        merged = entities.copy()
        for column in TRAFFIC_ONLY_COLUMNS:
            if column not in merged.columns:
                merged[column] = np.nan
        return merged

    join_candidates = [
        "scene_tag",
        "base_scene",
        "morphology",
        "marketplace_id",
        "week",
        "sunday",
        "asin",
        "parent_asin",
        "hyy_asin",
    ]
    join_keys = [col for col in join_candidates if col in entities.columns and col in traffic.columns]
    if not join_keys:
        join_keys = [
            col
            for col in ("asin", "marketplace_id", "week")
            if col in entities.columns and col in traffic.columns
        ]

    traffic_dedup = traffic.drop_duplicates(subset=join_keys) if join_keys else traffic
    merged = entities.merge(
        traffic_dedup,
        how="left",
        on=join_keys,
        suffixes=("", "_traffic"),
    )

    for column in list(merged.columns):
        if column.endswith("_traffic"):
            base = column[:-8]
            if base in merged.columns:
                merged.drop(columns=[column], inplace=True)
            else:
                merged.rename(columns={column: base}, inplace=True)

    for column in TRAFFIC_ONLY_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan

    return merged


def _latest_week_with_pairs(engine: Engine, marketplace_id: str) -> str | None:
    """Return the most recent week present in competition pair tables."""

    with engine.connect() as conn:
        result = conn.execute(text(PAIRS_SELECT_SQL), {"mk": marketplace_id})
        row = result.fetchone()
    if not row:
        return None
    try:
        week_value = row["week"]
    except (TypeError, KeyError):  # pragma: no cover - compatibility fallback
        week_value = row[0] if row else None
    return str(week_value) if week_value else None


def _load_feature_slice(
    engine: Engine,
    *,
    marketplace_id: str,
    week: str,
    scene_filters: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fetch page and traffic features for ``week`` from Doris and merge them."""

    if not week:
        return pd.DataFrame()

    entity_sql, scene_params = _augment_scene_filters(ENTITIES_SELECT_SQL, scene_filters)
    traffic_sql, traffic_params = _augment_scene_filters(TRAFFIC_SELECT_SQL, scene_filters)

    params = {"mk": marketplace_id, "week": week}
    entities = _read_dataframe(engine, entity_sql, {**params, **scene_params})
    traffic = _read_dataframe(engine, traffic_sql, {**params, **traffic_params})

    if entities.empty and traffic.empty:
        return entities

    merged = _merge_entities_with_traffic(entities, traffic)
    if "week" in merged.columns:
        merged["week"] = merged["week"].astype(str)
    return merged


def _collect_compare_entities(
    engine: Engine,
    *,
    marketplace_id: str,
    weeks: Iterable[str],
    scene_filters: Sequence[str] | None = None,
    current_entities: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Gather entity rows for comparison across ``weeks``."""

    frames: list[pd.DataFrame] = []
    seen_weeks: set[str] = set()

    if current_entities is not None and not current_entities.empty:
        frames.append(current_entities.copy())
        seen_weeks.update(str(w) for w in current_entities.get("week", []) if pd.notna(w))

    for week in weeks:
        if not week:
            continue
        if week in seen_weeks:
            continue
        loaded = _load_feature_slice(
            engine,
            marketplace_id=marketplace_id,
            week=week,
            scene_filters=scene_filters,
        )
        if loaded.empty:
            LOGGER.warning(
                "competition_compare_missing_features week=%s mk=%s filters=%s",
                week,
                marketplace_id,
                scene_filters,
            )
            continue
        frames.append(loaded)
        seen_weeks.add(week)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        subset=[
            col
            for col in ("scene_tag", "marketplace_id", "week", "asin", "parent_asin", "hyy_asin")
            if col in combined.columns
        ],
        keep="last",
    )
    return combined


def _run_compare_checks(
    engine: Engine,
    marketplace_id: str,
    week: str,
    previous_week: str | None = None,
) -> None:
    """Verify that compare tables contain rows for the requested weeks."""

    weeks = {week}
    if previous_week:
        weeks.add(previous_week)

    commands = [
        (
            PAIRS_TABLE,
            text(
                f"SELECT COUNT(*) AS row_count FROM {PAIRS_TABLE} "
                "WHERE marketplace_id = :mk AND week = :week"
            ),
        ),
        (
            TRAFFIC_PAIRS_TABLE,
            text(
                f"SELECT COUNT(*) AS row_count FROM {TRAFFIC_PAIRS_TABLE} "
                "WHERE marketplace_id = :mk AND week = :week"
            ),
        ),
        (
            PAIRS_EACH_TABLE,
            text(
                f"SELECT COUNT(*) AS row_count FROM {PAIRS_EACH_TABLE} "
                "WHERE marketplace_id = :mk AND week = :week"
            ),
        ),
        (
            TRAFFIC_PAIRS_EACH_TABLE,
            text(
                f"SELECT COUNT(*) AS row_count FROM {TRAFFIC_PAIRS_EACH_TABLE} "
                "WHERE marketplace_id = :mk AND week = :week"
            ),
        ),
        (
            DELTA_TABLE,
            text(
                f"SELECT COUNT(*) AS row_count FROM {DELTA_TABLE} "
                "WHERE marketplace_id = :mk AND week_w0 = :week"
            ),
        ),
        (
            SCENE_WEEK_TABLE,
            text(
                f"SELECT COUNT(*) AS row_count FROM {SCENE_WEEK_TABLE} "
                "WHERE marketplace_id = :mk AND week = :week"
            ),
        ),
    ]

    with engine.connect() as conn:
        for target_week in weeks:
            for table, stmt in commands:
                preview = str(stmt)
                LOGGER.info(
                    "competition_compare_check table=%s week=%s command=%s",
                    table,
                    target_week,
                    preview,
                )
                result = conn.execute(stmt, {"mk": marketplace_id, "week": target_week})
                row_count = result.scalar() or 0
                status = "ok" if row_count > 0 else "empty"
                log_fn = LOGGER.info if row_count > 0 else LOGGER.warning
                log_fn(
                    "competition_compare_check_result table=%s week=%s rows=%d status=%s",
                    table,
                    target_week,
                    row_count,
                    status,
                )
def _run_post_write_checks(engine: Engine, marketplace_id: str, week: str) -> None:
    """Execute verification queries to confirm rows exist for the target slice."""

    commands = [
        (
            ENTITIES_TABLE,
            text(
                f"SELECT COUNT(*) AS row_count FROM {ENTITIES_TABLE} "
                "WHERE marketplace_id = :mk AND week = :week"
            ),
        ),
        (
            TRAFFIC_TABLE,
            text(
                f"SELECT COUNT(*) AS row_count FROM {TRAFFIC_TABLE} "
                "WHERE marketplace_id = :mk AND week = :week"
            ),
        ),
    ]

    with engine.connect() as conn:
        for idx, (table, stmt) in enumerate(commands, start=1):
            command_preview = (
                f"SELECT COUNT(*) AS row_count FROM {table} "
                "WHERE marketplace_id = :mk AND week = :week"
            )
            LOGGER.info(
                "competition_pipeline_check_step step=%d table=%s command=%s",
                idx,
                table,
                command_preview,
            )
            result = conn.execute(stmt, {"mk": marketplace_id, "week": week})
            row_count = result.scalar() or 0
            status = "ok" if row_count > 0 else "empty"
            log_fn = LOGGER.info if row_count > 0 else LOGGER.warning
            log_fn(
                "competition_pipeline_check_result step=%d table=%s row_count=%d status=%s",
                idx,
                table,
                row_count,
                status,
            )


def run_competition_pipeline(
    week: str | None,
    marketplace_id: str,
    *,
    scene_filters: Sequence[str] | None = None,
    engine: Engine | None = None,
    write: bool = False,
    chunk_size: int = 500,
) -> dict[str, pd.DataFrame]:
    """Execute the competition ETL up to feature ingestion."""

    engine = engine or create_doris_engine()

    resolved_week = week
    if not resolved_week:
        resolved_week = _latest_week_with_data(engine, marketplace_id)
        LOGGER.info(
            "competition_pipeline_week_auto_selected marketplace=%s week=%s",
            marketplace_id,
            resolved_week,
        )

    monday, sunday = _iso_week_to_dates(resolved_week)
    LOGGER.info(
        "competition_pipeline_fetching week=%s marketplace=%s monday=%s sunday=%s",
        resolved_week,
        marketplace_id,
        monday,
        sunday,
    )

    snapshot_df = _read_dataframe(
        engine, SNAPSHOT_SQL, {"mk": marketplace_id, "week": resolved_week}
    )
    if "discount_rate" not in snapshot_df.columns:
        snapshot_df["discount_rate"] = pd.NA
    LOGGER.info(
        "competition_pipeline_snapshots_fetched week=%s rows=%d",
        resolved_week,
        len(snapshot_df),
    )
    if snapshot_df.empty:
        message = f"No product snapshots for marketplace={marketplace_id} week={resolved_week}"
        LOGGER.error("competition_pipeline_no_snapshots %s", message)
        raise RuntimeError(message)

    snapshot_df["sunday"] = pd.to_datetime(snapshot_df["sunday"]).dt.date

    scene_sql, scene_params = _augment_scene_filters(SCENE_TAG_SQL, scene_filters)
    scene_df = _read_dataframe(
        engine,
        scene_sql,
        {"mk": marketplace_id, **scene_params},
    )
    LOGGER.info(
        "competition_pipeline_scene_tags_fetched mk=%s filters=%s rows=%d",
        marketplace_id,
        scene_filters,
        len(scene_df),
    )
    if scene_df.empty:
        message = (
            f"No scene_tag mapping for marketplace={marketplace_id} filters={scene_filters}"
        )
        LOGGER.error("competition_pipeline_no_scene_tags %s", message)
        raise RuntimeError(message)

    relevant_asins = snapshot_df["asin"].dropna().unique().tolist()
    scene_df = scene_df.loc[scene_df["asin"].isin(relevant_asins)].copy()
    if scene_filters:
        scene_df = _filter_scene_tags(scene_df, scene_filters)
    if scene_df.empty:
        LOGGER.error(
            "competition_pipeline_scene_filter_no_match week=%s mk=%s filters=%s",
            resolved_week,
            marketplace_id,
            scene_filters,
        )
        raise RuntimeError(
            "No scene_tag rows match provided filters after aligning with snapshots"
        )

    flow_sql, flow_params_extra = _augment_in_clause(
        FLOW_SQL_BASE,
        "asin",
        relevant_asins,
        "asin",
    )
    flow_params = {"mk": marketplace_id, **flow_params_extra}
    flow_df = _read_dataframe(engine, flow_sql, flow_params)
    flow_df = _normalise_flow_dataframe(flow_df)
    LOGGER.info(
        "competition_pipeline_flow_raw rows=%d",
        len(flow_df),
    )
    if not flow_df.empty:
        flow_df = flow_df.loc[flow_df["monday"] == monday].copy()
    LOGGER.info(
        "competition_pipeline_flow_filtered rows=%d monday=%s",
        len(flow_df),
        monday,
    )

    keyword_df = _read_dataframe(engine, KEYWORD_SQL, {
        "mk": marketplace_id,
        "start_date": monday,
        "end_date": sunday,
    })
    LOGGER.info(
        "competition_pipeline_keywords_fetched marketplace=%s rows=%d",
        marketplace_id,
        len(keyword_df),
    )
    if not keyword_df.empty:
        keyword_df = keyword_df.loc[keyword_df["asin"].isin(relevant_asins)].copy()

    keyword_tag_df = _read_dataframe(engine, KEYWORD_TAG_SQL, {})
    LOGGER.info(
        "competition_pipeline_keyword_tags_fetched rows=%d",
        len(keyword_tag_df),
    )

    traffic_features = build_traffic_features(flow_df, keyword_df, keyword_tag_df)
    LOGGER.info(
        "competition_pipeline_traffic_features_built rows=%d",
        len(traffic_features),
    )

    my_asins = scene_df.loc[scene_df["hyy_asin"] == 1, "asin"].dropna().unique().tolist()
    if not my_asins:
        LOGGER.warning(
            "competition_pipeline_no_my_asins week=%s mk=%s",
            resolved_week,
            marketplace_id,
        )

    entities_full = clean_competition_entities(
        snapshot_df,
        my_asins=my_asins,
        scene_tags=scene_df,
        traffic=traffic_features,
    )

    entities = entities_full.copy()
    entities, dropped_columns = _prune_traffic_columns(entities)
    if dropped_columns:
        LOGGER.info(
            "competition_pipeline_entities_pruned columns=%s",
            sorted(dropped_columns),
        )

    traffic_entities = _prepare_traffic_entities(traffic_features, scene_df, snapshot_df)

    if write:
        entity_table_columns = _load_table_columns(engine, ENTITIES_TABLE)
        traffic_table_columns = _load_table_columns(engine, TRAFFIC_TABLE)

        entities, dropped_entity_table, missing_entity_table = _prune_to_table(
            entities,
            entity_table_columns,
        )
        if dropped_entity_table:
            LOGGER.info(
                "competition_pipeline_prune_to_table table=%s dropped_cols=%s",
                ENTITIES_TABLE,
                sorted(dropped_entity_table),
            )
        if missing_entity_table:
            LOGGER.warning(
                "competition_pipeline_table_missing_columns table=%s missing=%s",
                ENTITIES_TABLE,
                sorted(missing_entity_table),
            )

        traffic_entities, dropped_traffic_table, missing_traffic_table = _prune_to_table(
            traffic_entities,
            traffic_table_columns,
        )
        if dropped_traffic_table:
            LOGGER.info(
                "competition_pipeline_prune_to_table table=%s dropped_cols=%s",
                TRAFFIC_TABLE,
                sorted(dropped_traffic_table),
            )
        if missing_traffic_table:
            LOGGER.warning(
                "competition_pipeline_table_missing_columns table=%s missing=%s",
                TRAFFIC_TABLE,
                sorted(missing_traffic_table),
            )

    results = {
        "snapshots": snapshot_df,
        "scene_tags": scene_df,
        "traffic_raw": traffic_features,
        "entities": entities,
        "entities_full": entities_full,
        "traffic": traffic_entities,
        "resolved_week": resolved_week,
    }

    LOGGER.info(
        "competition_pipeline_entities rows=%d traffic=%d",
        len(entities),
        len(traffic_entities),
    )

    if write:
        ent_written = replace_into(engine, ENTITIES_TABLE, entities, chunk_size=chunk_size)
        LOGGER.info("competition_pipeline_entities_written rows=%d", ent_written)
        traffic_written = replace_into(
            engine,
            TRAFFIC_TABLE,
            traffic_entities,
            chunk_size=chunk_size,
        )
        LOGGER.info("competition_pipeline_traffic_written rows=%d", traffic_written)
        _run_post_write_checks(engine, marketplace_id, resolved_week)

    return results


def run_competition_compare_pipeline(
    marketplace_id: str,
    *,
    week: str,
    previous_week: str | None = None,
    scene_filters: Sequence[str] | None = None,
    engine: Engine | None = None,
    current_entities: pd.DataFrame | None = None,
    scoring_rules: pd.DataFrame | dict[str, Any] | None = None,
    rule_name: str = "default",
    traffic_scoring: Mapping[str, Any] | None = None,
    traffic_rule_name: str = "default_traffic",
    write: bool = False,
    chunk_size: int = 500,
) -> dict[str, pd.DataFrame]:
    """Build competition compare tables from precomputed features and optionally persist them."""

    engine = engine or create_doris_engine()

    target_weeks: set[str] = {week}
    if previous_week:
        target_weeks.add(previous_week)

    combined_entities = _collect_compare_entities(
        engine,
        marketplace_id=marketplace_id,
        weeks=target_weeks,
        scene_filters=scene_filters,
        current_entities=current_entities,
    )

    if combined_entities.empty:
        LOGGER.warning(
            "competition_compare_no_entities week=%s previous=%s mk=%s filters=%s",
            week,
            previous_week,
            marketplace_id,
            scene_filters,
        )

    tables = build_competition_tables_from_entities(
        combined_entities,
        week=week,
        previous_week=previous_week,
        scoring_rules=scoring_rules,
        rule_name=rule_name,
        traffic_scoring=traffic_scoring,
        traffic_rule_name=traffic_rule_name,
    )

    LOGGER.info(
        "competition_compare_rows pairs=%d traffic_pairs=%d pairs_each=%d traffic_pairs_each=%d delta=%d summary=%d",
        len(tables.pairs),
        len(tables.traffic_pairs),
        len(tables.pairs_each),
        len(tables.traffic_pairs_each),
        len(tables.delta),
        len(tables.summary),
    )

    if write:
        table_map = {
            PAIRS_TABLE: tables.pairs,
            TRAFFIC_PAIRS_TABLE: tables.traffic_pairs,
            PAIRS_EACH_TABLE: tables.pairs_each,
            TRAFFIC_PAIRS_EACH_TABLE: tables.traffic_pairs_each,
            DELTA_TABLE: tables.delta,
            SCENE_WEEK_TABLE: tables.summary,
        }

        for table_name, df in table_map.items():
            table_columns = _load_table_columns(engine, table_name)
            aligned, dropped, missing = _prune_to_table(df, table_columns)
            if dropped:
                LOGGER.info(
                    "competition_compare_prune table=%s dropped_cols=%s",
                    table_name,
                    sorted(dropped),
                )
            if missing:
                LOGGER.warning(
                    "competition_compare_missing_columns table=%s missing=%s",
                    table_name,
                    sorted(missing),
                )
            written = replace_into(engine, table_name, aligned, chunk_size=chunk_size)
            LOGGER.info(
                "competition_compare_written table=%s rows=%d",
                table_name,
                written,
            )

        _run_compare_checks(engine, marketplace_id, week, previous_week=previous_week)

    return {
        "entities": tables.entities,
        "traffic_entities": tables.traffic_entities,
        "pairs": tables.pairs,
        "traffic_pairs": tables.traffic_pairs,
        "pairs_each": tables.pairs_each,
        "traffic_pairs_each": tables.traffic_pairs_each,
        "delta": tables.delta,
        "summary": tables.summary,
    }


def _latest_week_with_data(engine: Engine, marketplace_id: str) -> str:
    with engine.connect() as conn:
        result = conn.execute(text(LATEST_SNAPSHOT_WEEK_SQL), {"mk": marketplace_id})
        row = result.fetchone()
    if not row:
        message = f"Cannot auto-detect week: no snapshots for marketplace={marketplace_id}"
        LOGGER.error("competition_pipeline_week_auto_failed %s", message)
        raise RuntimeError(message)
    try:
        week_value = row["week"]
    except (TypeError, KeyError):  # pragma: no cover - compatibility guard
        week_value = row[0] if row else None
    if not week_value:
        message = f"Latest snapshot query returned empty week for marketplace={marketplace_id}"
        LOGGER.error("competition_pipeline_week_auto_empty %s", message)
        raise RuntimeError(message)
    return str(week_value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Competition ETL runner (features + compare)"
    )
    parser.add_argument(
        "--week",
        required=False,
        help="ISO week label, e.g. 2025W10. If omitted the latest week with data is used.",
    )
    parser.add_argument(
        "--previous-week",
        dest="previous_week",
        required=False,
        help="Baseline ISO week for WoW delta. Defaults to the week preceding --week.",
    )
    parser.add_argument("--mk", required=True, help="Marketplace identifier (US/JP/DE...)")
    parser.add_argument(
        "--scene-tag",
        action="append",
        dest="scene_tags",
        help="Restrict to specific scene_tag (repeatable)",
    )
    parser.add_argument("--write", action="store_true", help="Persist features to Doris")
    parser.add_argument(
        "--with-compare",
        action="store_true",
        help="Compute compare tables in addition to feature ingestion",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip feature ingestion and only compute compare tables",
    )
    parser.add_argument(
        "--write-compare",
        action="store_true",
        help="Persist compare tables (pairs/deltas/scene summary) to Doris",
    )
    parser.add_argument(
        "--rule-name",
        default="default",
        help="Scoring rule name for page-side comparisons (default: default)",
    )
    parser.add_argument(
        "--traffic-rule-name",
        default="default_traffic",
        help="Traffic scoring rule name (default: default_traffic)",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Execute the competition LLM workflow after compare outputs",
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Skip ETL/compare and only run the competition LLM workflow",
    )
    parser.add_argument(
        "--llm-stage",
        choices=("both", "stage1", "stage2"),
        default="both",
        help="Select which LLM stages to execute when running the competition workflow (default: both)",
    )
    parser.add_argument(
        "--llm-config",
        default="configs/competition_llm.yaml",
        help="Path to the competition LLM configuration YAML",
    )
    parser.add_argument(
        "--llm-storage-root",
        default=None,
        help="Directory to store Stage-1/Stage-2 LLM artefacts (defaults to storage/competition_llm)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Batch size for Doris upsert",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_logging()
    log_dir = Path(os.getenv("SCPC_LOG_DIR", "storage/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_week = args.week or "auto"
    logfile = log_dir / f"competition_pipeline_{log_week}_{args.mk}_{timestamp}.log"
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    LOGGER.info("competition_pipeline_log_file path=%s", logfile)

    try:
        engine = create_doris_engine()
        llm_only = args.llm_only
        with_llm = args.with_llm or llm_only
        compare_only = args.compare_only

        if llm_only and compare_only:
            LOGGER.info("competition_pipeline_llm_only_overrides_compare")
            compare_only = False

        with_compare = args.with_compare or compare_only
        if with_llm and not llm_only:
            with_compare = True

        resolved_week = args.week

        if not resolved_week:
            if compare_only:
                resolved_week = _latest_week_with_pairs(engine, args.mk)
                if resolved_week:
                    LOGGER.info(
                        "competition_pipeline_week_auto_pairs marketplace=%s week=%s",
                        args.mk,
                        resolved_week,
                    )
            if not resolved_week:
                resolved_week = _latest_week_with_data(engine, args.mk)
                LOGGER.info(
                    "competition_pipeline_week_auto_features marketplace=%s week=%s",
                    args.mk,
                    resolved_week,
                )

        previous_week = args.previous_week
        if with_compare and not previous_week:
            try:
                previous_week = _previous_week_label(resolved_week)
                LOGGER.info(
                    "competition_pipeline_previous_week_auto week=%s previous=%s",
                    resolved_week,
                    previous_week,
                )
            except ValueError:
                previous_week = None
                LOGGER.warning(
                    "competition_pipeline_previous_week_unavailable week=%s",
                    resolved_week,
                )

        run_features = not compare_only and not llm_only
        run_compare = with_compare and not llm_only

        feature_results: dict[str, pd.DataFrame] | None = None
        if run_features:
            feature_results = run_competition_pipeline(
                resolved_week,
                args.mk,
                scene_filters=args.scene_tags,
                engine=engine,
                write=args.write,
                chunk_size=args.chunk_size,
            )

        if run_compare:
            current_entities = None
            if feature_results is not None:
                current_entities = feature_results.get("entities_full")
            run_competition_compare_pipeline(
                args.mk,
                week=resolved_week,
                previous_week=previous_week,
                scene_filters=args.scene_tags,
                engine=engine,
                current_entities=current_entities,
                rule_name=args.rule_name,
                traffic_rule_name=args.traffic_rule_name,
                write=args.write_compare,
                chunk_size=args.chunk_size,
            )

        if with_llm:
            llm_config = load_competition_llm_config(args.llm_config)
            client = create_client_from_env()
            try:
                llm_orchestrator = LLMOrchestrator(client)
                competition_orchestrator = CompetitionLLMOrchestrator(
                    engine=engine,
                    llm_orchestrator=llm_orchestrator,
                    config=llm_config,
                    storage_root=args.llm_storage_root,
                )
                target_week = resolved_week or args.week
                if args.llm_stage == "both":
                    stage_selection: Sequence[str] | None = ("stage1", "stage2")
                else:
                    stage_selection = (args.llm_stage,)
                result = competition_orchestrator.run(
                    target_week,
                    marketplace_id=args.mk,
                    stages=stage_selection,
                )
                LOGGER.info(
                    "competition_pipeline_llm_completed week=%s stage1=%s stage2_candidates=%s stage2=%s storage=%s",
                    result.week,
                    result.stage1_processed,
                    result.stage2_candidates,
                    result.stage2_processed,
                    [str(path) for path in result.storage_paths],
                )
            finally:
                client.close()
    except Exception as exc:  # pragma: no cover - CLI level safeguard
        LOGGER.exception(
            "competition_pipeline_failed week=%s mk=%s error=%s",
            args.week,
            args.mk,
            exc,
        )
        raise SystemExit(1) from exc
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()


__all__ = [
    "main",
    "parse_args",
    "run_competition_pipeline",
    "run_competition_compare_pipeline",
    "_prepare_traffic_entities",
    "_normalise_flow_dataframe",
    "_iso_week_to_dates",
    "_previous_week_label",
    "_latest_week_with_data",
    "_latest_week_with_pairs",
    "_prune_traffic_columns",
    "_prune_to_table",
    "_merge_entities_with_traffic",
    "_load_feature_slice",
    "_collect_compare_entities",
    "_run_post_write_checks",
    "_run_compare_checks",
    "_load_table_columns",
    "TRAFFIC_ONLY_COLUMNS",
    "ENTITIES_TABLE",
    "TRAFFIC_TABLE",
    "PAIRS_TABLE",
    "TRAFFIC_PAIRS_TABLE",
    "PAIRS_EACH_TABLE",
    "TRAFFIC_PAIRS_EACH_TABLE",
    "DELTA_TABLE",
    "SCENE_WEEK_TABLE",
]
