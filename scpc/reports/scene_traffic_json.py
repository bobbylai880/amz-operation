"""Generate flow and keyword change JSON modules for scene traffic analysis."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from scpc.db.engine import create_doris_engine
from scpc.db.io import fetch_dataframe

LOGGER = logging.getLogger(__name__)
WEEK_PATTERN = re.compile(r"^(\d{4})-?W(\d{2})$")
FLOW_TOP_N = 20
KEYWORD_SCENE_TOP_K = 50
KEYWORD_ASIN_HEAD_TOP_K = 20
KEYWORD_ASIN_TOP_N = 20
KEYWORD_CONTRIBUTOR_TOP_N = 5

SCENE_TAG_SQL = """
SELECT scene_tag, base_scene, morphology, asin, marketplace_id, hyy_asin
FROM bi_amz_asin_scene_tag
WHERE scene_tag = :scene_tag AND marketplace_id = :marketplace_id
"""

FLOW_SQL = """
SELECT asin,
       marketplace_id,
       today,
       ad_ratio,
       nf_ratio,
       recommend_ratio,
       sp_ratio,
       sbv_ratio,
       sb_ratio
FROM HYY_DW_MYSQL.hyy.sif_asin_flow_overview
WHERE marketplace_id = :marketplace_id
  AND today BETWEEN :date_start AND :date_end
"""

# `bi_sif_keyword_daily` 真实列名为「有效曝光流量占比」，此处做别名兼容下游逻辑。
KEYWORD_SQL = """
SELECT asin,
       marketplace_id,
       keyword,
       snapshot_date,
       `有效曝光流量占比` AS effective_impr_share,
       weekly_search_volume,
       last_rank,
       ad_last_rank
FROM bi_sif_keyword_daily
WHERE marketplace_id = :marketplace_id
  AND snapshot_date IN (:sunday_this, :sunday_last)
"""

SNAPSHOT_BRAND_SQL = """
SELECT asin, marketplace_id, brand
FROM bi_amz_asin_product_snapshot_v2
WHERE week = :week AND marketplace_id = :marketplace_id
"""

SNAPSHOT_BRAND_FALLBACK_SQL = """
SELECT asin, marketplace_id, brand
FROM bi_amz_asin_product_snapshot
WHERE week = :week AND marketplace_id = :marketplace_id
"""

DEFAULT_RULES = {
    "ad_change_thresholds": {
        "big_increase": 0.15,
        "small_increase": 0.05,
        "stable_range": 0.05,
        "small_decrease": 0.05,
        "big_decrease": 0.15,
    },
    "traffic_mix_thresholds": {
        "ad_dominant_min": 0.6,
        "organic_dominant_min": 0.6,
        "reco_dominant_min": 0.4,
    },
    "keyword_profile_change": {
        "high_threshold": 0.6,
        "medium_threshold": 0.3,
    },
    "keyword_volume_opportunity": {
        "high_volume_min": 50000.0,
        "rising_rate_min": 0.3,
        "self_share_low_max": 0.05,
        "scene_share_min": 0.02,
    },
    "keyword_position": {
        "organic": {"strong_max": 16.0, "medium_max": 64.0},
        "ad": {"strong_max": 4.0, "medium_max": 10.0},
    },
    "keyword_opportunity_rules": {
        "high_volume_min": 10000.0,
        "self_share_low_max": 0.05,
        "rising_rate_min": 0.3,
    },
    "asin_rank_trend": {"stable_threshold": 2.0},
}

AD_CHANGE_BUCKETS = [
    "广告占比大幅提升",
    "广告占比小幅提升",
    "广告占比基本稳定",
    "广告占比小幅下降",
    "广告占比大幅下降",
]


@dataclass(slots=True)
class SceneTrafficJobParams:
    """CLI parameters consumed by :class:`SceneTrafficJsonGenerator`."""

    week: str
    scene_tag: str
    marketplace_id: str
    storage_dir: Path


class SceneTrafficJsonError(RuntimeError):
    """Raised when traffic JSON generation cannot proceed."""


class SceneTrafficJsonGenerator:
    """Generate flow_change.json and keyword_change.json for a scene."""

    def __init__(
        self,
        engine: Engine | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.engine = engine or create_doris_engine()
        self.config_path = config_path or Path("configs/scene_traffic_rules.yml")

    # ------------------------------------------------------------------
    def run(self, params: SceneTrafficJobParams) -> dict[str, Path]:
        """Execute the ETL pipeline and return written module paths."""

        LOGGER.info(
            "scene_traffic_json_start",
            extra={
                "week": params.week,
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
                "storage": str(params.storage_dir),
            },
        )

        monday_this, sunday_this, monday_last, sunday_last = _resolve_week_boundaries(
            params.week
        )

        rules = _load_rules_config(self.config_path)

        scene_df = fetch_dataframe(
            self.engine,
            SCENE_TAG_SQL,
            {"scene_tag": params.scene_tag, "marketplace_id": params.marketplace_id},
        )
        if scene_df.empty:
            LOGGER.error(
                "scene_traffic_json_no_scene_scope",
                extra={
                    "week": params.week,
                    "scene_tag": params.scene_tag,
                    "marketplace": params.marketplace_id,
                },
            )
            raise SceneTrafficJsonError("No scene_tag rows for provided filters")

        scene_scope = _aggregate_scene_tags(scene_df)
        LOGGER.info(
            "scene_traffic_json_scene_scope",
            extra={
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
                "asin_rows": len(scene_df),
                "asin_scope": len(scene_scope),
            },
        )

        brand_scope = _fetch_brand_scope(self.engine, params)
        scene_scope = scene_scope.merge(brand_scope, on=["asin", "marketplace_id"], how="left")
        scene_scope["brand"] = scene_scope["brand"].fillna("")

        base_scene = _first_non_null(scene_scope.get("base_scene")) or params.scene_tag
        morphology = _first_non_null(scene_scope.get("morphology"))

        flow_df = fetch_dataframe(
            self.engine,
            FLOW_SQL,
            {
                "marketplace_id": params.marketplace_id,
                "date_start": monday_last.strftime("%Y%m%d"),
                "date_end": sunday_this.strftime("%Y%m%d"),
            },
        )
        df_flow = _prepare_flow_dataframe(scene_scope, flow_df, monday_this, monday_last, rules)

        keyword_df = fetch_dataframe(
            self.engine,
            KEYWORD_SQL,
            {
                "marketplace_id": params.marketplace_id,
                "sunday_this": sunday_this,
                "sunday_last": sunday_last,
            },
        )
        df_keyword = _prepare_keyword_dataframe(scene_scope, keyword_df)

        outputs: dict[str, Any] = {}
        outputs["traffic/flow_change"] = _build_flow_payload(
            params,
            base_scene,
            morphology,
            monday_this,
            monday_last,
            df_flow,
            rules,
        )
        outputs["traffic/keyword_change"] = _build_keyword_payload(
            params,
            base_scene,
            morphology,
            sunday_this,
            sunday_last,
            df_keyword,
            rules,
        )

        scene_dir = _scene_storage_dir(params.scene_tag)
        base_dir = params.storage_dir / params.week / scene_dir / "traffic"
        base_dir.mkdir(parents=True, exist_ok=True)

        written: dict[str, Path] = {}
        for module, payload in outputs.items():
            filename = module.split("/")[-1]
            path = base_dir / f"{filename}.json"
            _write_json(path, payload)
            LOGGER.info(
                "scene_traffic_json_written",
                extra={
                    "scene_tag": params.scene_tag,
                    "marketplace": params.marketplace_id,
                    "week": params.week,
                    "module": module,
                    "path": str(path),
                },
            )
            written[module] = path

        LOGGER.info(
            "scene_traffic_json_done",
            extra={
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
                "week": params.week,
                "modules": list(written.keys()),
            },
        )
        return written


# ----------------------------------------------------------------------
# Data loading helpers
# ----------------------------------------------------------------------


def _resolve_week_boundaries(week: str) -> tuple[date, date, date, date]:
    match = WEEK_PATTERN.match(week.strip())
    if not match:
        raise SceneTrafficJsonError(f"Invalid ISO week format: {week}")
    year = int(match.group(1))
    week_num = int(match.group(2))
    if week_num < 1 or week_num > 53:
        raise SceneTrafficJsonError(f"Invalid ISO week: {week}")
    monday_this = date.fromisocalendar(year, week_num, 1)
    sunday_this = monday_this + timedelta(days=6)
    monday_last = monday_this - timedelta(days=7)
    sunday_last = sunday_this - timedelta(days=7)
    return monday_this, sunday_this, monday_last, sunday_last


def _load_rules_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        LOGGER.warning(
            "scene_traffic_json_config_missing", extra={"path": str(config_path)}
        )
        return DEFAULT_RULES.copy()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning(
            "scene_traffic_json_config_load_failed",
            extra={"path": str(config_path), "error": str(exc)},
        )
        return DEFAULT_RULES.copy()
    defaults = raw.get("defaults", {}) if isinstance(raw, Mapping) else {}
    merged = {}
    for key, fallback in DEFAULT_RULES.items():
        section = defaults.get(key) if isinstance(defaults, Mapping) else None
        merged[key] = _merge_rule_section(fallback, section, [key])
    return merged


def _merge_rule_section(
    fallback: Any, section: Any, path: list[str]
) -> Any:  # pragma: no cover - exercised via _load_rules_config
    section_label = ".".join(path)
    if isinstance(fallback, Mapping):
        merged: dict[str, Any] = {}
        if section is None:
            LOGGER.warning(
                "scene_traffic_json_config_section_missing",
                extra={"section": section_label},
            )
            section_map: Mapping[str, Any] = {}
        elif isinstance(section, Mapping):
            section_map = section
        else:
            LOGGER.warning(
                "scene_traffic_json_config_section_invalid",
                extra={"section": section_label},
            )
            section_map = {}
        for field, default_value in fallback.items():
            next_path = path + [field]
            merged[field] = _merge_rule_section(
                default_value, section_map.get(field), next_path
            )
        return merged

    if section is None:
        parent = ".".join(path[:-1]) or path[-1]
        LOGGER.warning(
            "scene_traffic_json_config_missing_key",
            extra={"section": parent, "missing": path[-1]},
        )
        return fallback

    if _is_number(section):
        return float(section)

    LOGGER.warning(
        "scene_traffic_json_config_invalid_value",
        extra={"section": ".".join(path[:-1]) or path[-1], "field": path[-1], "value": section},
    )
    return fallback


def _aggregate_scene_tags(scene_df: pd.DataFrame) -> pd.DataFrame:
    df = scene_df.copy()
    df["hyy_asin"] = df["hyy_asin"].fillna(0).astype(int)
    grouped = (
        df.groupby(["asin", "marketplace_id"], as_index=False)
        .agg(
            {
                "hyy_asin": "max",
                "scene_tag": _first_non_null,
                "base_scene": _first_non_null,
                "morphology": _first_non_null,
            }
        )
        .reset_index(drop=True)
    )
    return grouped


def _fetch_brand_scope(engine: Engine, params: SceneTrafficJobParams) -> pd.DataFrame:
    query_params = {"week": params.week, "marketplace_id": params.marketplace_id}
    try:
        brand_df = fetch_dataframe(engine, SNAPSHOT_BRAND_SQL, query_params)
    except OperationalError as exc:
        if _is_unknown_table_error(exc):
            LOGGER.warning(
                "scene_traffic_json_brand_table_missing",
                extra={
                    "week": params.week,
                    "scene_tag": params.scene_tag,
                    "marketplace": params.marketplace_id,
                    "table": "bi_amz_asin_product_snapshot_v2",
                },
            )
            try:
                brand_df = fetch_dataframe(engine, SNAPSHOT_BRAND_FALLBACK_SQL, query_params)
            except OperationalError as fallback_exc:
                if _is_unknown_table_error(fallback_exc):
                    LOGGER.warning(
                        "scene_traffic_json_brand_table_fallback_missing",
                        extra={
                            "week": params.week,
                            "scene_tag": params.scene_tag,
                            "marketplace": params.marketplace_id,
                            "table": "bi_amz_asin_product_snapshot",
                        },
                    )
                    brand_df = pd.DataFrame(columns=["asin", "marketplace_id", "brand"])
                else:
                    raise
            else:
                LOGGER.info(
                    "scene_traffic_json_brand_table_fallback",
                    extra={
                        "week": params.week,
                        "scene_tag": params.scene_tag,
                        "marketplace": params.marketplace_id,
                        "table": "bi_amz_asin_product_snapshot",
                    },
                )
        else:
            raise
    return _aggregate_brands(brand_df)


def _aggregate_brands(brand_df: pd.DataFrame) -> pd.DataFrame:
    if brand_df.empty:
        return pd.DataFrame(columns=["asin", "marketplace_id", "brand"])
    df = brand_df.copy()
    df["brand"] = df["brand"].fillna("")
    df = df.sort_values(["asin", "marketplace_id", "brand"], ascending=[True, True, False])
    deduped = df.drop_duplicates(subset=["asin", "marketplace_id"], keep="first")
    return deduped.loc[:, ["asin", "marketplace_id", "brand"]]


# ----------------------------------------------------------------------
# Flow payload
# ----------------------------------------------------------------------


def _prepare_flow_dataframe(
    scene_scope: pd.DataFrame,
    flow_df: pd.DataFrame,
    monday_this: date,
    monday_last: date,
    rules: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    share_cols = [
        "ad_flow_share",
        "organic_flow_share",
        "reco_flow_share",
        "sp_flow_share",
        "video_flow_share",
        "brand_flow_share",
    ]

    df = flow_df.copy()
    if df.empty:
        LOGGER.warning("scene_traffic_json_flow_missing", extra={"rows": 0})
        df = pd.DataFrame(columns=["asin", "marketplace_id", "monday", *share_cols])
    elif "today" in df.columns:
        # 日表：按周聚合成周一锚点的数据
        today_raw = df["today"].apply(lambda v: str(v).split(".")[0])
        df["date"] = pd.to_datetime(today_raw, format="%Y%m%d", errors="coerce").dt.date
        df = df[df["date"].notna()].copy()
        df["monday"] = df["date"].apply(lambda d: d - timedelta(days=d.weekday()))
        df = df[df["monday"].isin({monday_this, monday_last})].copy()

        ratio_map = {
            "ad_ratio": "ad_flow_share",
            "nf_ratio": "organic_flow_share",
            "recommend_ratio": "reco_flow_share",
            "sp_ratio": "sp_flow_share",
            "sbv_ratio": "video_flow_share",
            "sb_ratio": "brand_flow_share",
        }
        agg_cols = {src: "mean" for src in ratio_map if src in df.columns}
        if agg_cols:
            grouped = (
                df.groupby(["asin", "marketplace_id", "monday"], as_index=False)
                .agg(agg_cols)
                .rename(columns={src: ratio_map[src] for src in agg_cols})
            )
        else:
            grouped = df.loc[:, ["asin", "marketplace_id", "monday"]].drop_duplicates()
        df = grouped
    else:
        df["monday"] = pd.to_datetime(df.get("monday"), errors="coerce").dt.date

    if "monday" in df.columns:
        df = df[df["monday"].notna()]
        df = df[df["monday"].isin({monday_this, monday_last})]
    else:
        df["monday"] = pd.NaT

    for col in share_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = math.nan

    def _pivot(subset_date: date, suffix: str) -> pd.DataFrame:
        subset = df[df["monday"] == subset_date].copy()
        if subset.empty:
            return pd.DataFrame(columns=["asin", "marketplace_id"])
        renamed = subset.rename(
            columns={
                "ad_flow_share": f"ad_flow_share_{suffix}",
                "organic_flow_share": f"organic_flow_share_{suffix}",
                "reco_flow_share": f"reco_flow_share_{suffix}",
                "sp_flow_share": f"sp_flow_share_{suffix}",
                "video_flow_share": f"video_flow_share_{suffix}",
                "brand_flow_share": f"brand_flow_share_{suffix}",
            }
        )
        renamed = renamed.loc[
            :, [
                "asin",
                "marketplace_id",
                f"ad_flow_share_{suffix}",
                f"organic_flow_share_{suffix}",
                f"reco_flow_share_{suffix}",
                f"sp_flow_share_{suffix}",
                f"video_flow_share_{suffix}",
                f"brand_flow_share_{suffix}",
            ]
        ]
        renamed = renamed.drop_duplicates(subset=["asin", "marketplace_id"], keep="last")
        return renamed

    flow_this = _pivot(monday_this, "this")
    flow_last = _pivot(monday_last, "last")

    merged = scene_scope.merge(flow_this, on=["asin", "marketplace_id"], how="left")
    merged = merged.merge(flow_last, on=["asin", "marketplace_id"], how="left")

    for metric in ["ad_flow_share", "organic_flow_share", "reco_flow_share"]:
        merged[f"{metric}_diff"] = (
            merged.get(f"{metric}_this") - merged.get(f"{metric}_last")
        )

    for metric in ["sp_flow_share", "video_flow_share", "brand_flow_share"]:
        if f"{metric}_last" not in merged.columns:
            merged[f"{metric}_last"] = math.nan

    thresholds = rules.get("ad_change_thresholds", {})
    mix_thresholds = rules.get("traffic_mix_thresholds", {})
    merged["ad_change_type"] = merged.apply(
        lambda row: _classify_ad_change(row.get("ad_flow_share_diff"), thresholds), axis=1
    )
    merged["traffic_mix_type"] = merged.apply(
        lambda row: _classify_traffic_mix(
            row.get("ad_flow_share_this"),
            row.get("organic_flow_share_this"),
            row.get("reco_flow_share_this"),
            mix_thresholds,
        ),
        axis=1,
    )
    return merged


def _classify_ad_change(delta: Any, thresholds: Mapping[str, float]) -> str | None:
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return None
    big_inc = thresholds.get("big_increase", DEFAULT_RULES["ad_change_thresholds"]["big_increase"])
    small_inc = thresholds.get(
        "small_increase", DEFAULT_RULES["ad_change_thresholds"]["small_increase"]
    )
    stable = thresholds.get("stable_range", DEFAULT_RULES["ad_change_thresholds"]["stable_range"])
    small_dec = thresholds.get(
        "small_decrease", DEFAULT_RULES["ad_change_thresholds"]["small_decrease"]
    )
    big_dec = thresholds.get("big_decrease", DEFAULT_RULES["ad_change_thresholds"]["big_decrease"])

    if delta >= big_inc:
        return "广告占比大幅提升"
    if delta >= small_inc:
        return "广告占比小幅提升"
    if -stable < delta < stable:
        return "广告占比基本稳定"
    if delta <= -big_dec:
        return "广告占比大幅下降"
    if delta <= -small_dec:
        return "广告占比小幅下降"
    return "广告占比小幅下降"


def _classify_traffic_mix(
    ad_share: Any,
    organic_share: Any,
    reco_share: Any,
    thresholds: Mapping[str, float],
) -> str:
    ad_min = thresholds.get(
        "ad_dominant_min", DEFAULT_RULES["traffic_mix_thresholds"]["ad_dominant_min"]
    )
    org_min = thresholds.get(
        "organic_dominant_min",
        DEFAULT_RULES["traffic_mix_thresholds"]["organic_dominant_min"],
    )
    reco_min = thresholds.get(
        "reco_dominant_min", DEFAULT_RULES["traffic_mix_thresholds"]["reco_dominant_min"]
    )

    ad_share = _safe_float(ad_share)
    organic_share = _safe_float(organic_share)
    reco_share = _safe_float(reco_share)

    if ad_share is not None and ad_share >= ad_min:
        return "广告主导型流量"
    if organic_share is not None and organic_share >= org_min:
        return "自然主导型流量"
    if reco_share is not None and reco_share >= reco_min:
        return "推荐主导型流量"
    return "混合型流量"


def _build_flow_payload(
    params: SceneTrafficJobParams,
    base_scene: str | None,
    morphology: str | None,
    monday_this: date,
    monday_last: date,
    df_flow: pd.DataFrame,
    rules: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    overview = {
        "self": _summarise_flow(df_flow, True),
        "competitor": _summarise_flow(df_flow, False),
    }
    top_lists = {
        "self": _build_flow_top_lists(df_flow, True),
        "competitor": _build_flow_top_lists(df_flow, False),
    }

    payload = {
        "week": params.week,
        "scene_tag": params.scene_tag,
        "base_scene": base_scene,
        "morphology": morphology,
        "marketplace_id": params.marketplace_id,
        "monday_this": monday_this.isoformat(),
        "monday_last": monday_last.isoformat(),
        "overview": overview,
        "top_lists": top_lists,
    }
    if df_flow.empty:
        LOGGER.warning(
            "scene_traffic_json_flow_join_empty",
            extra={
                "week": params.week,
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
            },
        )
    return payload


def _summarise_flow(df_flow: pd.DataFrame, is_self: bool) -> dict[str, Any]:
    if df_flow.empty:
        return {
            "asin_count": 0,
            "avg_ad_flow_share_this": None,
            "avg_ad_flow_share_last": None,
            "avg_ad_flow_share_diff": None,
            "avg_organic_flow_share_this": None,
            "avg_reco_flow_share_this": None,
            "avg_sp_flow_share_this": None,
            "avg_video_flow_share_this": None,
            "avg_brand_flow_share_this": None,
            "ad_change_distribution": {label: 0 for label in AD_CHANGE_BUCKETS},
        }
    mask = df_flow["hyy_asin"].fillna(0).astype(int) == (1 if is_self else 0)
    subset = df_flow[mask]
    base = {
        "asin_count": int(len(subset)),
        "avg_ad_flow_share_this": _mean_or_none(subset.get("ad_flow_share_this")),
        "avg_ad_flow_share_last": _mean_or_none(subset.get("ad_flow_share_last")),
        "avg_ad_flow_share_diff": _mean_or_none(subset.get("ad_flow_share_diff")),
        "avg_organic_flow_share_this": _mean_or_none(subset.get("organic_flow_share_this")),
        "avg_reco_flow_share_this": _mean_or_none(subset.get("reco_flow_share_this")),
        "avg_sp_flow_share_this": _mean_or_none(subset.get("sp_flow_share_this")),
        "avg_video_flow_share_this": _mean_or_none(subset.get("video_flow_share_this")),
        "avg_brand_flow_share_this": _mean_or_none(subset.get("brand_flow_share_this")),
    }
    distribution = {label: 0 for label in AD_CHANGE_BUCKETS}
    if not subset.empty:
        counts = subset["ad_change_type"].value_counts(dropna=True)
        for label, count in counts.items():
            if label in distribution:
                distribution[label] = int(count)
    base["ad_change_distribution"] = distribution
    return base


def _build_flow_top_lists(df_flow: pd.DataFrame, is_self: bool) -> dict[str, list[dict[str, Any]]]:
    if df_flow.empty:
        return {
            "ad_increase_top": [],
            "ad_decrease_top": [],
            "ad_heavy_this": [],
            "organic_heavy_this": [],
        }
    mask = df_flow["hyy_asin"].fillna(0).astype(int) == (1 if is_self else 0)
    subset = df_flow[mask]
    lists = {
        "ad_increase_top": _flow_rows_to_records(
            subset[subset["ad_flow_share_diff"].fillna(0) > 0]
            .sort_values("ad_flow_share_diff", ascending=False)
            .head(FLOW_TOP_N)
        ),
        "ad_decrease_top": _flow_rows_to_records(
            subset[subset["ad_flow_share_diff"].fillna(0) < 0]
            .sort_values("ad_flow_share_diff", ascending=True)
            .head(FLOW_TOP_N)
        ),
        "ad_heavy_this": _flow_rows_to_records(
            subset[subset["traffic_mix_type"] == "广告主导型流量"]
            .sort_values("ad_flow_share_this", ascending=False)
            .head(FLOW_TOP_N)
        ),
        "organic_heavy_this": _flow_rows_to_records(
            subset[subset["traffic_mix_type"] == "自然主导型流量"]
            .sort_values("organic_flow_share_this", ascending=False)
            .head(FLOW_TOP_N)
        ),
    }
    return lists


def _flow_rows_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record = {
            "asin": row.get("asin"),
            "brand": row.get("brand", ""),
            "hyy_asin": int(row.get("hyy_asin", 0) or 0),
            "marketplace_id": row.get("marketplace_id"),
            "scene_tag": row.get("scene_tag"),
            "ad_flow_share_this": _safe_float(row.get("ad_flow_share_this")),
            "ad_flow_share_last": _safe_float(row.get("ad_flow_share_last")),
            "ad_flow_share_diff": _safe_float(row.get("ad_flow_share_diff")),
            "organic_flow_share_this": _safe_float(row.get("organic_flow_share_this")),
            "reco_flow_share_this": _safe_float(row.get("reco_flow_share_this")),
            "sp_flow_share_this": _safe_float(row.get("sp_flow_share_this")),
            "video_flow_share_this": _safe_float(row.get("video_flow_share_this")),
            "brand_flow_share_this": _safe_float(row.get("brand_flow_share_this")),
            "ad_change_type": row.get("ad_change_type"),
            "traffic_mix_type": row.get("traffic_mix_type"),
        }
        records.append(record)
    return records


# ----------------------------------------------------------------------
# Keyword payload
# ----------------------------------------------------------------------


def _prepare_keyword_dataframe(
    scene_scope: pd.DataFrame,
    keyword_df: pd.DataFrame,
) -> pd.DataFrame:
    if keyword_df.empty:
        LOGGER.warning("scene_traffic_json_keyword_missing", extra={"rows": 0})
        df = pd.DataFrame(columns=["asin", "marketplace_id", "keyword", "snapshot_date"])
    else:
        df = keyword_df.copy()
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date
        df["effective_impr_share"] = pd.to_numeric(
            df["effective_impr_share"], errors="coerce"
        )
        if "weekly_search_volume" in df.columns:
            df["weekly_search_volume"] = pd.to_numeric(
                df["weekly_search_volume"], errors="coerce"
            )
        if "last_rank" in df.columns:
            df["last_rank"] = pd.to_numeric(df["last_rank"], errors="coerce")
        if "ad_last_rank" in df.columns:
            df["ad_last_rank"] = pd.to_numeric(df["ad_last_rank"], errors="coerce")
    joined = df.merge(scene_scope, on=["asin", "marketplace_id"], how="inner")
    joined["hyy_asin"] = joined["hyy_asin"].fillna(0).astype(int)
    joined["share_self"] = joined.apply(
        lambda row: row["effective_impr_share"] if row["hyy_asin"] == 1 else 0.0,
        axis=1,
    )
    joined["share_comp"] = joined.apply(
        lambda row: row["effective_impr_share"] if row["hyy_asin"] != 1 else 0.0,
        axis=1,
    )
    return joined


def _build_keyword_payload(
    params: SceneTrafficJobParams,
    base_scene: str | None,
    morphology: str | None,
    sunday_this: date,
    sunday_last: date,
    df_keyword: pd.DataFrame,
    rules: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    df_this = df_keyword[df_keyword["snapshot_date"] == sunday_this]
    df_last = df_keyword[df_keyword["snapshot_date"] == sunday_last]

    position_rules = rules.get("keyword_position", DEFAULT_RULES["keyword_position"])
    head_this = _build_scene_head_keywords(df_this, "this", position_rules)
    head_last = _build_scene_head_keywords(df_last, "last", position_rules)

    scene_head_keywords = {
        "this_week": head_this["list"],
        "last_week": head_last["list"],
        "diff": _build_scene_keyword_diff(head_this["dict"], head_last["dict"]),
    }

    asin_heads_this = _build_asin_head_keywords(df_this, "this")
    asin_heads_last = _build_asin_head_keywords(df_last, "last")
    asin_profile_change = _build_keyword_profile_change(
        df_keyword,
        asin_heads_this,
        asin_heads_last,
        rules.get("keyword_profile_change", {}),
    )

    asin_rank_rules = rules.get("asin_rank_trend", DEFAULT_RULES["asin_rank_trend"])
    contributors = _build_keyword_contributors(
        df_this, df_last, head_this["list"], asin_rank_rules
    )
    keyword_opportunity = _build_keyword_volume_opportunity(
        head_this["dict"],
        head_last["dict"],
        rules.get("keyword_volume_opportunity", {}),
        rules.get("keyword_opportunity_rules", {}),
    )

    payload = {
        "week": params.week,
        "scene_tag": params.scene_tag,
        "base_scene": base_scene,
        "morphology": morphology,
        "marketplace_id": params.marketplace_id,
        "sunday_this": sunday_this.isoformat(),
        "sunday_last": sunday_last.isoformat(),
        "scene_head_keywords": scene_head_keywords,
        "asin_keyword_profile_change": asin_profile_change,
        "keyword_asin_contributors": {"this_week": contributors},
        "keyword_opportunity_by_volume": keyword_opportunity,
    }
    if df_keyword.empty:
        LOGGER.warning(
            "scene_traffic_json_keyword_join_empty",
            extra={
                "week": params.week,
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
            },
        )
    return payload


def _build_scene_head_keywords(
    df: pd.DataFrame, prefix: str, position_rules: Mapping[str, Any]
) -> dict[str, Any]:
    if df.empty:
        return {"list": [], "dict": {}}
    agg_map: dict[str, str] = {
        "effective_impr_share": "sum",
        "share_self": "sum",
        "share_comp": "sum",
    }
    if "weekly_search_volume" in df.columns:
        agg_map["weekly_search_volume"] = "max"
    grouped = (
        df.groupby("keyword", as_index=False)
        .agg(agg_map)
        .rename(
            columns={
                "effective_impr_share": f"scene_kw_share_{prefix}",
                "share_self": f"scene_kw_self_share_{prefix}",
                "share_comp": f"scene_kw_comp_share_{prefix}",
                "weekly_search_volume": f"search_volume_{prefix}",
            }
        )
    )
    grouped = grouped.sort_values(f"scene_kw_share_{prefix}", ascending=False).head(
        KEYWORD_SCENE_TOP_K
    )
    grouped["rank"] = range(1, len(grouped) + 1)
    volume_col = f"search_volume_{prefix}"
    if volume_col not in grouped.columns:
        grouped[volume_col] = math.nan

    rank_series_map = {
        f"self_best_organic_rank_{prefix}": _best_rank_series(df, "last_rank", True),
        f"comp_best_organic_rank_{prefix}": _best_rank_series(df, "last_rank", False),
        f"self_best_ad_rank_{prefix}": _best_rank_series(df, "ad_last_rank", True),
        f"comp_best_ad_rank_{prefix}": _best_rank_series(df, "ad_last_rank", False),
    }
    rank_frames = [
        series.rename(column)
        for column, series in rank_series_map.items()
        if not series.empty
    ]
    if rank_frames:
        metrics = pd.concat(rank_frames, axis=1).reset_index()
        grouped = grouped.merge(metrics, on="keyword", how="left")
    for column in rank_series_map:
        if column not in grouped.columns:
            grouped[column] = math.nan

    organic_thresholds = _resolve_position_thresholds(position_rules, "organic")
    ad_thresholds = _resolve_position_thresholds(position_rules, "ad")
    grouped[f"self_organic_status_{prefix}"] = grouped[
        f"self_best_organic_rank_{prefix}"
    ].apply(lambda value: _rank_status(value, organic_thresholds))
    grouped[f"comp_organic_status_{prefix}"] = grouped[
        f"comp_best_organic_rank_{prefix}"
    ].apply(lambda value: _rank_status(value, organic_thresholds))
    grouped[f"self_ad_status_{prefix}"] = grouped[f"self_best_ad_rank_{prefix}"].apply(
        lambda value: _rank_status(value, ad_thresholds)
    )
    grouped[f"comp_ad_status_{prefix}"] = grouped[f"comp_best_ad_rank_{prefix}"].apply(
        lambda value: _rank_status(value, ad_thresholds)
    )

    payload_list = []
    keyword_dict = {}
    for _, row in grouped.iterrows():
        entry = {
            "keyword": row["keyword"],
            f"scene_kw_share_{prefix}": _safe_float(row.get(f"scene_kw_share_{prefix}")),
            f"scene_kw_self_share_{prefix}": _safe_float(row.get(f"scene_kw_self_share_{prefix}")),
            f"scene_kw_comp_share_{prefix}": _safe_float(row.get(f"scene_kw_comp_share_{prefix}")),
            f"rank_{prefix}": int(row["rank"]),
            volume_col: _safe_float(row.get(volume_col)),
            f"self_best_organic_rank_{prefix}": _safe_float(
                row.get(f"self_best_organic_rank_{prefix}")
            ),
            f"comp_best_organic_rank_{prefix}": _safe_float(
                row.get(f"comp_best_organic_rank_{prefix}")
            ),
            f"self_best_ad_rank_{prefix}": _safe_float(
                row.get(f"self_best_ad_rank_{prefix}")
            ),
            f"comp_best_ad_rank_{prefix}": _safe_float(
                row.get(f"comp_best_ad_rank_{prefix}")
            ),
            f"self_organic_status_{prefix}": row.get(
                f"self_organic_status_{prefix}"
            ),
            f"comp_organic_status_{prefix}": row.get(
                f"comp_organic_status_{prefix}"
            ),
            f"self_ad_status_{prefix}": row.get(f"self_ad_status_{prefix}"),
            f"comp_ad_status_{prefix}": row.get(f"comp_ad_status_{prefix}"),
        }
        payload_list.append(entry)
        keyword_dict[row["keyword"]] = entry
    return {"list": payload_list, "dict": keyword_dict}


def _build_scene_keyword_diff(
    head_this: Mapping[str, Mapping[str, Any]],
    head_last: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    keywords_this = set(head_this.keys())
    keywords_last = set(head_last.keys())
    added = []
    for keyword in sorted(keywords_this - keywords_last):
        entry = head_this[keyword]
        added.append(
            {
                "keyword": keyword,
                "scene_kw_share_this": entry.get("scene_kw_share_this"),
                "scene_kw_self_share_this": entry.get("scene_kw_self_share_this"),
                "scene_kw_comp_share_this": entry.get("scene_kw_comp_share_this"),
                "search_volume_this": entry.get("search_volume_this"),
                "self_best_organic_rank_this": entry.get("self_best_organic_rank_this"),
                "comp_best_organic_rank_this": entry.get("comp_best_organic_rank_this"),
                "self_best_ad_rank_this": entry.get("self_best_ad_rank_this"),
                "comp_best_ad_rank_this": entry.get("comp_best_ad_rank_this"),
                "self_organic_status_this": entry.get("self_organic_status_this"),
                "comp_organic_status_this": entry.get("comp_organic_status_this"),
                "self_ad_status_this": entry.get("self_ad_status_this"),
                "comp_ad_status_this": entry.get("comp_ad_status_this"),
            }
        )
    removed = []
    for keyword in sorted(keywords_last - keywords_this):
        entry = head_last[keyword]
        removed.append(
            {
                "keyword": keyword,
                "scene_kw_share_last": entry.get("scene_kw_share_last"),
                "scene_kw_self_share_last": entry.get("scene_kw_self_share_last"),
                "scene_kw_comp_share_last": entry.get("scene_kw_comp_share_last"),
                "search_volume_last": entry.get("search_volume_last"),
                "self_best_organic_rank_last": entry.get("self_best_organic_rank_last"),
                "comp_best_organic_rank_last": entry.get("comp_best_organic_rank_last"),
                "self_best_ad_rank_last": entry.get("self_best_ad_rank_last"),
                "comp_best_ad_rank_last": entry.get("comp_best_ad_rank_last"),
                "self_organic_status_last": entry.get("self_organic_status_last"),
                "comp_organic_status_last": entry.get("comp_organic_status_last"),
                "self_ad_status_last": entry.get("self_ad_status_last"),
                "comp_ad_status_last": entry.get("comp_ad_status_last"),
            }
        )
    common = []
    for keyword in sorted(keywords_this & keywords_last):
        entry_this = head_this[keyword]
        entry_last = head_last[keyword]
        diff = None
        if entry_this.get("scene_kw_share_this") is not None and entry_last.get(
            "scene_kw_share_last"
        ) is not None:
            diff = entry_this["scene_kw_share_this"] - entry_last["scene_kw_share_last"]
        search_volume_this = _safe_float(entry_this.get("search_volume_this"))
        search_volume_last = _safe_float(entry_last.get("search_volume_last"))
        search_volume_diff = None
        search_volume_change_rate = None
        if search_volume_this is not None and search_volume_last is not None:
            search_volume_diff = search_volume_this - search_volume_last
            search_volume_change_rate = _compute_change_rate(
                search_volume_this, search_volume_last
            )
        common.append(
            {
                "keyword": keyword,
                "scene_kw_share_this": entry_this.get("scene_kw_share_this"),
                "scene_kw_share_last": entry_last.get("scene_kw_share_last"),
                "scene_kw_share_diff": _safe_float(diff),
                "scene_kw_self_share_this": entry_this.get("scene_kw_self_share_this"),
                "scene_kw_self_share_last": entry_last.get("scene_kw_self_share_last"),
                "scene_kw_comp_share_this": entry_this.get("scene_kw_comp_share_this"),
                "scene_kw_comp_share_last": entry_last.get("scene_kw_comp_share_last"),
                "search_volume_this": search_volume_this,
                "search_volume_last": search_volume_last,
                "search_volume_diff": search_volume_diff,
                "search_volume_change_rate": search_volume_change_rate,
                "self_best_organic_rank_this": entry_this.get(
                    "self_best_organic_rank_this"
                ),
                "self_best_organic_rank_last": entry_last.get(
                    "self_best_organic_rank_last"
                ),
                "comp_best_organic_rank_this": entry_this.get(
                    "comp_best_organic_rank_this"
                ),
                "comp_best_organic_rank_last": entry_last.get(
                    "comp_best_organic_rank_last"
                ),
                "self_best_ad_rank_this": entry_this.get("self_best_ad_rank_this"),
                "self_best_ad_rank_last": entry_last.get("self_best_ad_rank_last"),
                "comp_best_ad_rank_this": entry_this.get("comp_best_ad_rank_this"),
                "comp_best_ad_rank_last": entry_last.get("comp_best_ad_rank_last"),
                "self_organic_status_this": entry_this.get("self_organic_status_this"),
                "self_organic_status_last": entry_last.get("self_organic_status_last"),
                "comp_organic_status_this": entry_this.get("comp_organic_status_this"),
                "comp_organic_status_last": entry_last.get("comp_organic_status_last"),
                "self_ad_status_this": entry_this.get("self_ad_status_this"),
                "self_ad_status_last": entry_last.get("self_ad_status_last"),
                "comp_ad_status_this": entry_this.get("comp_ad_status_this"),
                "comp_ad_status_last": entry_last.get("comp_ad_status_last"),
            }
        )
    return {
        "keywords_added": added,
        "keywords_removed": removed,
        "keywords_common": common,
    }


def _build_asin_head_keywords(
    df: pd.DataFrame, prefix: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if df.empty:
        return {}
    df_sorted = df.sort_values(
        ["asin", "marketplace_id", "effective_impr_share"], ascending=[True, True, False]
    )
    df_sorted["rank"] = df_sorted.groupby(["asin", "marketplace_id"]).cumcount()
    head = df_sorted[df_sorted["rank"] < KEYWORD_ASIN_HEAD_TOP_K]
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (asin, marketplace), group in head.groupby(["asin", "marketplace_id"]):
        keyword_list: list[dict[str, Any]] = []
        for _, row in group.iterrows():
            entry = {
                "keyword": row["keyword"],
                "share": _safe_float(row["effective_impr_share"]),
            }
            volume = _safe_float(row.get("weekly_search_volume"))
            if volume is not None:
                entry[f"search_volume_{prefix}"] = volume
            keyword_list.append(entry)
        result[(asin, marketplace)] = keyword_list
    return result


def _build_keyword_profile_change(
    df_keyword: pd.DataFrame,
    asin_heads_this: Mapping[tuple[str, str], list[dict[str, Any]]],
    asin_heads_last: Mapping[tuple[str, str], list[dict[str, Any]]],
    thresholds: Mapping[str, float],
) -> dict[str, list[dict[str, Any]]]:
    if df_keyword.empty:
        return {"self": [], "competitor": []}
    records: list[dict[str, Any]] = []
    grouped = df_keyword.groupby(["asin", "marketplace_id"], as_index=False).agg(
        {
            "scene_tag": "first",
            "brand": "first",
            "hyy_asin": "max",
        }
    )
    high = thresholds.get(
        "high_threshold", DEFAULT_RULES["keyword_profile_change"]["high_threshold"]
    )
    medium = thresholds.get(
        "medium_threshold", DEFAULT_RULES["keyword_profile_change"]["medium_threshold"]
    )
    for _, row in grouped.iterrows():
        key = (row["asin"], row["marketplace_id"])
        head_this = asin_heads_this.get(key, [])
        head_last = asin_heads_last.get(key, [])
        keywords_this = {entry["keyword"] for entry in head_this}
        keywords_last = {entry["keyword"] for entry in head_last}
        union = keywords_this | keywords_last
        intersection = keywords_this & keywords_last
        if not union:
            change_score = 0.0
        else:
            change_score = 1.0 - (len(intersection) / len(union))
        if change_score >= high:
            change_type = "关键词画像变化显著"
        elif change_score >= medium:
            change_type = "关键词画像有一定变化"
        else:
            change_type = "关键词画像基本稳定"
        record = {
            "asin": row["asin"],
            "brand": row.get("brand", ""),
            "hyy_asin": int(row.get("hyy_asin", 0) or 0),
            "marketplace_id": row["marketplace_id"],
            "scene_tag": row.get("scene_tag"),
            "change_score": round(change_score, 4),
            "change_type": change_type,
            "head_keywords_this": head_this,
            "head_keywords_last": head_last,
            "keywords_added": sorted(list(keywords_this - keywords_last)),
            "keywords_removed": sorted(list(keywords_last - keywords_this)),
        }
        records.append(record)
    records_df = pd.DataFrame(records)
    self_records = (
        records_df[records_df["hyy_asin"] == 1]
        .sort_values("change_score", ascending=False)
        .head(KEYWORD_ASIN_TOP_N)
    )
    comp_records = (
        records_df[records_df["hyy_asin"] != 1]
        .sort_values("change_score", ascending=False)
        .head(KEYWORD_ASIN_TOP_N)
    )
    return {
        "self": self_records.to_dict(orient="records"),
        "competitor": comp_records.to_dict(orient="records"),
    }


def _build_keyword_contributors(
    df_this: pd.DataFrame,
    df_last: pd.DataFrame,
    head_keywords: list[Mapping[str, Any]],
    rank_trend_rules: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if df_this.empty or not head_keywords:
        return []
    contributors = []
    keyword_order = [entry["keyword"] for entry in head_keywords]
    stable_threshold = _safe_float(
        (rank_trend_rules or {}).get("stable_threshold")
    ) or DEFAULT_RULES["asin_rank_trend"]["stable_threshold"]
    last_lookup = _asin_rank_lookup(df_last)
    for keyword in keyword_order:
        subset = df_this[df_this["keyword"] == keyword]
        if subset.empty:
            continue
        top_asin = []
        ranked = subset.sort_values("effective_impr_share", ascending=False).head(
            KEYWORD_CONTRIBUTOR_TOP_N
        )
        for _, row in ranked.iterrows():
            organic_rank_this = _safe_float(row.get("last_rank"))
            ad_rank_this = _safe_float(row.get("ad_last_rank"))
            last_organic, last_ad = last_lookup.get(
                (keyword, row["asin"]), (None, None)
            )
            organic_diff, organic_trend = _rank_diff_and_trend(
                organic_rank_this, last_organic, stable_threshold
            )
            ad_diff, ad_trend = _rank_diff_and_trend(
                ad_rank_this, last_ad, stable_threshold
            )
            top_asin.append(
                {
                    "asin": row["asin"],
                    "brand": row.get("brand", ""),
                    "hyy_asin": int(row.get("hyy_asin", 0) or 0),
                    "marketplace_id": row["marketplace_id"],
                    "scene_tag": row.get("scene_tag"),
                    "effective_impr_share_this": _safe_float(
                        row.get("effective_impr_share")
                    ),
                    "organic_rank_this": organic_rank_this,
                    "organic_rank_last": last_organic,
                    "organic_rank_diff": organic_diff,
                    "organic_rank_trend": organic_trend,
                    "ad_rank_this": ad_rank_this,
                    "ad_rank_last": last_ad,
                    "ad_rank_diff": ad_diff,
                    "ad_rank_trend": ad_trend,
                }
            )
        contributors.append({"keyword": keyword, "top_asin": top_asin})
    return contributors


def _asin_rank_lookup(
    df: pd.DataFrame,
) -> dict[tuple[str, str], tuple[float | None, float | None]]:
    lookup: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    if df is None or df.empty:
        return lookup
    for _, row in df.iterrows():
        keyword = row.get("keyword")
        asin = row.get("asin")
        if keyword is None or asin is None:
            continue
        if pd.isna(keyword) or pd.isna(asin):
            continue
        key = (keyword, asin)
        if key in lookup:
            continue
        lookup[key] = (
            _safe_float(row.get("last_rank")),
            _safe_float(row.get("ad_last_rank")),
        )
    return lookup


def _rank_diff_and_trend(
    rank_this: float | None,
    rank_last: float | None,
    stable_threshold: float,
) -> tuple[float | None, str]:
    if rank_this is None and rank_last is None:
        return None, "missing"
    if rank_this is not None and rank_last is None:
        return None, "new"
    if rank_this is None and rank_last is not None:
        return None, "lost"
    diff = rank_last - rank_this
    if abs(diff) <= stable_threshold:
        trend = "stable"
    elif diff > 0:
        trend = "up"
    else:
        trend = "down"
    return diff, trend


def _build_keyword_volume_opportunity(
    head_this: Mapping[str, Mapping[str, Any]],
    head_last: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, float],
    opportunity_rules: Mapping[str, float],
) -> dict[str, list[dict[str, Any]]]:
    defaults = DEFAULT_RULES["keyword_volume_opportunity"]
    high_volume_min = thresholds.get("high_volume_min", defaults["high_volume_min"])
    rising_rate_min = thresholds.get("rising_rate_min", defaults["rising_rate_min"])
    self_share_low_max = thresholds.get(
        "self_share_low_max", defaults["self_share_low_max"]
    )
    scene_share_min = thresholds.get("scene_share_min", defaults["scene_share_min"])

    result = {"high_volume_low_self": [], "rising_demand_self_lagging": []}

    for keyword, entry in head_this.items():
        search_volume_this = _safe_float(entry.get("search_volume_this"))
        if search_volume_this is None or search_volume_this < high_volume_min:
            continue
        scene_share = _safe_float(entry.get("scene_kw_share_this"))
        if scene_share is None or scene_share < scene_share_min:
            continue
        self_share = _safe_float(entry.get("scene_kw_self_share_this"))
        if self_share is None or self_share > self_share_low_max:
            continue
        comp_share = _safe_float(entry.get("scene_kw_comp_share_this"))
        last_entry = head_last.get(keyword, {})
        search_volume_last = _safe_float(last_entry.get("search_volume_last"))
        record = {
            "keyword": keyword,
            "search_volume_this": search_volume_this,
            "search_volume_last": search_volume_last,
            "search_volume_change_rate": _compute_change_rate(
                search_volume_this, search_volume_last
            ),
            "scene_kw_share_this": scene_share,
            "scene_kw_self_share_this": self_share,
            "scene_kw_comp_share_this": comp_share,
        }
        _attach_rank_snapshot(record, entry, suffix="_this")
        record["opportunity_type"] = _classify_opportunity_types(
            entry,
            record["search_volume_change_rate"],
            opportunity_rules,
            bucket="high_volume_low_self",
        )
        result["high_volume_low_self"].append(record)

    for keyword in sorted(set(head_this.keys()) & set(head_last.keys())):
        entry_this = head_this[keyword]
        entry_last = head_last[keyword]
        search_volume_this = _safe_float(entry_this.get("search_volume_this"))
        search_volume_last = _safe_float(entry_last.get("search_volume_last"))
        change_rate = _compute_change_rate(search_volume_this, search_volume_last)
        if change_rate is None or change_rate < rising_rate_min:
            continue
        self_share_this = _safe_float(entry_this.get("scene_kw_self_share_this"))
        if self_share_this is None or self_share_this > self_share_low_max:
            continue
        scene_share_this = _safe_float(entry_this.get("scene_kw_share_this"))
        if scene_share_this is None or scene_share_this < scene_share_min:
            continue
        record = {
            "keyword": keyword,
            "search_volume_this": search_volume_this,
            "search_volume_last": search_volume_last,
            "search_volume_change_rate": change_rate,
            "scene_kw_share_this": scene_share_this,
            "scene_kw_self_share_this": self_share_this,
            "scene_kw_self_share_last": _safe_float(
                entry_last.get("scene_kw_self_share_last")
            ),
            "scene_kw_comp_share_this": _safe_float(
                entry_this.get("scene_kw_comp_share_this")
            ),
        }
        _attach_rank_snapshot(record, entry_this, suffix="_this")
        record["opportunity_type"] = _classify_opportunity_types(
            entry_this,
            change_rate,
            opportunity_rules,
            bucket="rising_demand_self_lagging",
        )
        result["rising_demand_self_lagging"].append(record)

    result["high_volume_low_self"].sort(
        key=lambda entry: entry.get("search_volume_this") or 0, reverse=True
    )
    result["rising_demand_self_lagging"].sort(
        key=lambda entry: entry.get("search_volume_change_rate") or 0,
        reverse=True,
    )
    return result


def _best_rank_series(df: pd.DataFrame, column: str, is_self: bool) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    subset = df[df[column].notna()].copy()
    if subset.empty:
        return pd.Series(dtype=float)
    mask = subset["hyy_asin"] == 1 if is_self else subset["hyy_asin"] != 1
    subset = subset[mask]
    if subset.empty:
        return pd.Series(dtype=float)
    grouped = subset.groupby("keyword")[column].min()
    return grouped.astype(float)


def _resolve_position_thresholds(
    position_rules: Mapping[str, Any], channel: str
) -> dict[str, float]:
    defaults = DEFAULT_RULES["keyword_position"].get(channel, {})
    section = position_rules.get(channel) if isinstance(position_rules, Mapping) else None
    if not isinstance(section, Mapping):
        section = {}
    strong = section.get("strong_max", defaults.get("strong_max"))
    medium = section.get("medium_max", defaults.get("medium_max"))
    strong_value = strong if _is_number(strong) else defaults.get("strong_max")
    medium_value = medium if _is_number(medium) else defaults.get("medium_max")
    return {
        "strong_max": float(strong_value) if strong_value is not None else math.inf,
        "medium_max": float(medium_value) if medium_value is not None else math.inf,
    }


def _rank_status(value: Any, thresholds: Mapping[str, float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "missing"
    try:
        rank_value = float(value)
    except (TypeError, ValueError):
        return "missing"
    strong_max = thresholds.get("strong_max", math.inf)
    medium_max = thresholds.get("medium_max", math.inf)
    if rank_value <= strong_max:
        return "strong"
    if rank_value <= medium_max:
        return "medium"
    return "weak"


def _attach_rank_snapshot(record: dict[str, Any], entry: Mapping[str, Any], suffix: str) -> None:
    fields = [
        "self_best_organic_rank",
        "comp_best_organic_rank",
        "self_best_ad_rank",
        "comp_best_ad_rank",
        "self_organic_status",
        "comp_organic_status",
        "self_ad_status",
        "comp_ad_status",
    ]
    for field in fields:
        key = f"{field}{suffix}"
        record[key] = entry.get(key)


def _classify_opportunity_types(
    entry: Mapping[str, Any],
    change_rate: float | None,
    rules: Mapping[str, float],
    *,
    bucket: str,
) -> list[str]:
    defaults = DEFAULT_RULES["keyword_opportunity_rules"]
    rising_rate_min = _resolve_rule_value(
        rules, "rising_rate_min", defaults["rising_rate_min"]
    )
    classifications: list[str] = []
    self_org_status = entry.get("self_organic_status_this")
    comp_org_status = entry.get("comp_organic_status_this")
    self_ad_status = entry.get("self_ad_status_this")
    comp_ad_status = entry.get("comp_ad_status_this")
    has_self_ad = entry.get("self_best_ad_rank_this") is not None

    if bucket == "high_volume_low_self":
        if comp_org_status == "strong" and self_org_status in {"weak", "missing"}:
            classifications.append("high_volume_organic_gap")
        if (
            self_org_status in {"weak", "missing"}
            and self_ad_status in {"medium", "weak"}
            and has_self_ad
        ):
            classifications.append("ad_heavy_but_not_dominant")
        if (
            self_org_status == "strong"
            and self_ad_status in {"missing", "weak"}
            and comp_ad_status == "strong"
        ):
            classifications.append("organic_good_ad_gap")
    elif bucket == "rising_demand_self_lagging":
        if (
            change_rate is not None
            and change_rate >= rising_rate_min
            and entry.get("self_organic_status_this") in {"weak", "missing"}
            and entry.get("self_ad_status_this") in {"weak", "missing"}
        ):
            classifications.append("rising_demand_full_gap")

    seen: set[str] = set()
    ordered: list[str] = []
    for value in classifications:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _resolve_rule_value(
    rules: Mapping[str, float], key: str, fallback: float
) -> float:
    if isinstance(rules, Mapping):
        value = rules.get(key)
        if _is_number(value):
            return float(value)
    return fallback


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def _scene_storage_dir(scene_tag: str) -> str:
    candidate = scene_tag.strip()
    if not candidate:
        raise SceneTrafficJsonError("scene_tag cannot be empty")
    if candidate in {".", ".."}:
        raise SceneTrafficJsonError("scene_tag cannot be '.' or '..'")
    separators = {os.sep, "/", "\\"}
    if os.altsep:
        separators.add(os.altsep)
    safe = [ch if ch not in separators else "_" for ch in candidate]
    return "".join(safe)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _first_non_null(series: pd.Series | None) -> Any:
    if series is None or series.empty:
        return None
    for value in series:
        if pd.notna(value) and value not in ("", None):
            return value
    return series.iloc[0]


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_change_rate(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    if previous == 0:
        return None
    return (current - previous) / previous


def _mean_or_none(series: pd.Series | None) -> float | None:
    if series is None or series.empty:
        return None
    mean_value = pd.to_numeric(series, errors="coerce").mean()
    if pd.isna(mean_value):
        return None
    return float(mean_value)


def _is_unknown_table_error(exc: OperationalError) -> bool:
    orig = getattr(exc, "orig", exc)
    args = getattr(orig, "args", None)
    if args:
        code = args[0]
        if isinstance(code, int) and code in {1051, 1146}:
            return True
    message = str(orig).lower()
    return "unknown table" in message or "doesn't exist" in message

