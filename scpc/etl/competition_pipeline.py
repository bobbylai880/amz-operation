"""Command line runner for competition data cleaning and feature ingestion."""

from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from scpc.utils.dependencies import ensure_packages

ensure_packages(["pandas", "numpy", "sqlalchemy"])

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from scpc.db.engine import create_doris_engine
from scpc.db.io import replace_into
from scpc.etl.competition_features import build_traffic_features, clean_competition_entities

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

    entities = clean_competition_entities(
        snapshot_df,
        my_asins=my_asins,
        scene_tags=scene_df,
        traffic=traffic_features,
    )

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
        "traffic": traffic_entities,
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
    parser = argparse.ArgumentParser(description="Competition ETL runner (clean + features)")
    parser.add_argument(
        "--week",
        required=False,
        help="ISO week label, e.g. 2025W10. If omitted the latest week with data is used.",
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
        run_competition_pipeline(
            args.week,
            args.mk,
            scene_filters=args.scene_tags,
            engine=engine,
            write=args.write,
            chunk_size=args.chunk_size,
        )
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
    "_prepare_traffic_entities",
    "_normalise_flow_dataframe",
    "_iso_week_to_dates",
    "_latest_week_with_data",
    "_prune_traffic_columns",
    "_prune_to_table",
    "_run_post_write_checks",
    "_load_table_columns",
    "TRAFFIC_ONLY_COLUMNS",
    "ENTITIES_TABLE",
    "TRAFFIC_TABLE",
]
