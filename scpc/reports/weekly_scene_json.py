"""Generate structured JSON modules for weekly scene-level ASIN insights."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from pandas.api.types import is_scalar

try:  # pragma: no cover - numpy is optional at runtime
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from scpc.db.engine import create_doris_engine
from scpc.db.io import fetch_dataframe

LOGGER = logging.getLogger(__name__)
BRAND_WARN_THRESHOLD = 0.3

SCENE_TAG_SQL = """
SELECT scene_tag, base_scene, morphology, asin, marketplace_id, hyy_asin
FROM bi_amz_asin_scene_tag
WHERE scene_tag = :scene_tag AND marketplace_id = :marketplace_id
"""

WEEK_DIFF_SQL = """
SELECT *
FROM bi_amz_asin_product_week_diff
WHERE week_this = :week AND marketplace_id = :marketplace_id
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

RANK_TREND_BUCKETS: Mapping[str, set[str]] = {
    "rank_up": {"排名明显上升"},
    "rank_down": {"排名明显下降"},
    "rank_stable": {"排名基本稳定"},
    "new_asin": {"新ASIN"},
}

PRICE_ACTION_BUCKETS: Mapping[str, set[str]] = {
    "big_drop": {"大幅降价"},
    "small_drop": {"小幅降价"},
    "stable": {"价格稳定"},
    "small_raise": {"小幅涨价"},
    "big_raise": {"大幅涨价"},
    "new_asin": {"新ASIN"},
}

PROMO_ACTION_BUCKETS: Mapping[str, set[str]] = {
    "new_promo": {"新增优惠"},
    "cancel_promo": {"取消优惠"},
    "promo_changed": {"优惠内容变化"},
    "promo_none": {"优惠无", "优惠无明显变化"},
}

TOP_N = 20

SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(slots=True)
class WeeklySceneJobParams:
    """Container for CLI parameters shared across helper functions."""

    week: str
    scene_tag: str
    marketplace_id: str
    storage_dir: Path


class WeeklySceneJsonError(RuntimeError):
    """Raised when the weekly scene JSON generation cannot proceed."""


class WeeklySceneJsonGenerator:
    """End-to-end orchestrator that emits scene-level JSON payloads."""

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or create_doris_engine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, params: WeeklySceneJobParams) -> dict[str, Path]:
        """Generate JSON files and return their paths keyed by module name."""

        LOGGER.info(
            "weekly_scene_json_start",
            extra={
                "week": params.week,
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
                "storage": str(params.storage_dir),
            },
        )

        scene_df = fetch_dataframe(
            self.engine,
            SCENE_TAG_SQL,
            {"scene_tag": params.scene_tag, "marketplace_id": params.marketplace_id},
        )
        if scene_df.empty:
            LOGGER.warning(
                "weekly_scene_json_no_scene_tags",
                extra={
                    "scene_tag": params.scene_tag,
                    "marketplace": params.marketplace_id,
                },
            )
            raise WeeklySceneJsonError("No scene_tag records found for provided filters")

        scene_agg = _aggregate_scene_tags(scene_df)
        LOGGER.info(
            "weekly_scene_json_scene_scope",
            extra={
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
                "asin_rows": len(scene_df),
                "asin_scope": len(scene_agg),
            },
        )

        diff_df = fetch_dataframe(
            self.engine,
            WEEK_DIFF_SQL,
            {"week": params.week, "marketplace_id": params.marketplace_id},
        )
        if diff_df.empty:
            LOGGER.warning(
                "weekly_scene_json_no_week_diff",
                extra={
                    "scene_tag": params.scene_tag,
                    "marketplace": params.marketplace_id,
                    "week": params.week,
                },
            )
            raise WeeklySceneJsonError("No weekly diff data for provided week/marketplace")

        diff_scene = diff_df.merge(scene_agg, on=["asin", "marketplace_id"], how="inner")
        if diff_scene.empty:
            LOGGER.warning(
                "weekly_scene_json_no_joined_diff",
                extra={
                    "scene_tag": params.scene_tag,
                    "marketplace": params.marketplace_id,
                    "week": params.week,
                },
            )
            raise WeeklySceneJsonError("Scene ASINs have no matching week diff rows")

        LOGGER.info(
            "weekly_scene_json_joined_rows",
            extra={
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
                "week": params.week,
                "rows": len(diff_scene),
            },
        )

        brand_scope = _fetch_brand_scope(self.engine, params)

        diff_tagged = diff_scene.merge(
            brand_scope, on=["asin", "marketplace_id"], how="left"
        )
        diff_tagged["brand"] = diff_tagged["brand"].fillna("").astype(str)
        brand_missing = (diff_tagged["brand"].str.len() == 0).sum()
        if brand_missing and brand_missing / len(diff_tagged) > BRAND_WARN_THRESHOLD:
            LOGGER.warning(
                "weekly_scene_json_brand_missing_high",
                extra={
                    "scene_tag": params.scene_tag,
                    "marketplace": params.marketplace_id,
                    "week": params.week,
                    "missing": int(brand_missing),
                    "total": len(diff_tagged),
                },
            )
        LOGGER.info(
            "weekly_scene_json_brand_join",
            extra={
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
                "week": params.week,
                "rows": len(diff_tagged),
                "brand_missing": int(brand_missing),
            },
        )

        tagged = _prepare_diff_dataframe(diff_tagged)
        meta = _build_meta(tagged, params)

        outputs: dict[str, dict[str, Any]] = {
            "overall_summary": build_overall_summary(tagged, meta),
            "self_analysis": build_scope_analysis(tagged, meta, scope="self"),
            "competitor_analysis": build_scope_analysis(tagged, meta, scope="competitor"),
            "self_risk_opportunity": build_self_risk_and_opportunity(tagged, meta),
            "competitor_actions": build_competitor_actions(tagged, meta),
        }

        base_dir = params.storage_dir / params.week / _safe_scene_tag(meta.scene_tag)
        base_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for module_name, payload in outputs.items():
            path = base_dir / f"{module_name}.json"
            _write_json(path, payload)
            LOGGER.info(
                "weekly_scene_json_written",
                extra={
                    "scene_tag": params.scene_tag,
                    "marketplace": params.marketplace_id,
                    "week": params.week,
                    "module": module_name,
                    "path": str(path),
                },
            )
            written[module_name] = path

        LOGGER.info(
            "weekly_scene_json_done",
            extra={
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
                "week": params.week,
                "modules": list(written.keys()),
            },
        )
        return written


# ----------------------------------------------------------------------
# Data preparation helpers
# ----------------------------------------------------------------------


def _fetch_brand_scope(engine: Engine, params: WeeklySceneJobParams) -> pd.DataFrame:
    """Fetch and aggregate brand info with graceful fallback."""

    query_params = {"week": params.week, "marketplace_id": params.marketplace_id}
    try:
        brand_df = fetch_dataframe(engine, SNAPSHOT_BRAND_SQL, query_params)
    except OperationalError as exc:
        if _is_unknown_table_error(exc):
            LOGGER.warning(
                "weekly_scene_json_brand_table_missing",
                extra={
                    "scene_tag": params.scene_tag,
                    "marketplace": params.marketplace_id,
                    "week": params.week,
                    "table": "bi_amz_asin_product_snapshot_v2",
                },
            )
            try:
                brand_df = fetch_dataframe(engine, SNAPSHOT_BRAND_FALLBACK_SQL, query_params)
            except OperationalError as fallback_exc:
                if _is_unknown_table_error(fallback_exc):
                    LOGGER.warning(
                        "weekly_scene_json_brand_table_fallback_missing",
                        extra={
                            "scene_tag": params.scene_tag,
                            "marketplace": params.marketplace_id,
                            "week": params.week,
                            "table": "bi_amz_asin_product_snapshot",
                        },
                    )
                    brand_df = pd.DataFrame(columns=["asin", "marketplace_id", "brand"])
                else:
                    raise
            else:
                LOGGER.info(
                    "weekly_scene_json_brand_table_fallback",
                    extra={
                        "scene_tag": params.scene_tag,
                        "marketplace": params.marketplace_id,
                        "week": params.week,
                        "table": "bi_amz_asin_product_snapshot",
                    },
                )
        else:
            raise
    return _aggregate_brands(brand_df)


def _aggregate_scene_tags(scene_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scene-tag rows to unique (asin, marketplace) pairs."""

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


def _aggregate_brands(brand_df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest non-empty brand per (asin, marketplace)."""

    if brand_df.empty:
        return pd.DataFrame(columns=["asin", "marketplace_id", "brand"])

    df = brand_df.copy()
    df["brand"] = df["brand"].fillna("")
    df = df.sort_values(["asin", "marketplace_id", "brand"], ascending=[True, True, False])
    deduped = df.drop_duplicates(subset=["asin", "marketplace_id"], keep="first")
    return deduped.loc[:, ["asin", "marketplace_id", "brand"]]


def _is_unknown_table_error(exc: OperationalError) -> bool:
    """Best-effort detection for MySQL/Doris unknown-table errors."""

    orig = getattr(exc, "orig", exc)
    args = getattr(orig, "args", None)
    if args:
        code = args[0]
        if isinstance(code, int) and code in {1051, 1146}:
            return True
    message = str(orig).lower()
    return "unknown table" in message or "doesn't exist" in message


def _prepare_diff_dataframe(diff_df: pd.DataFrame) -> pd.DataFrame:
    """Standardise dtype/JSON columns on the diff dataset."""

    df = diff_df.copy()
    json_cols = ["badge_added_json", "badge_removed_json"]
    for col in json_cols:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
        df[col] = df[col].apply(_ensure_json_list)
    df["hyy_asin"] = df["hyy_asin"].fillna(0).astype(int)
    df["brand"] = df["brand"].fillna("")
    df["badge_removed_cnt"] = df["badge_removed_json"].apply(lambda x: len(x))
    df["new_reviews"] = _normalise_new_reviews(df)
    return df


def _normalise_new_reviews(df: pd.DataFrame) -> pd.Series:
    """Ensure exposed new_reviews values are non-negative integers."""

    index = df.index
    if "new_reviews" in df.columns:
        series = pd.to_numeric(df["new_reviews"], errors="coerce").fillna(0)
    elif {"reviews_this", "reviews_last"}.issubset(df.columns):
        series = (
            pd.to_numeric(df["reviews_this"], errors="coerce").fillna(0)
            - pd.to_numeric(df["reviews_last"], errors="coerce").fillna(0)
        )
    else:
        return pd.Series(0, index=index, dtype=int)

    return series.clip(lower=0).round().astype(int)


def _first_non_null(series: pd.Series) -> Any:
    for value in series:
        if pd.notna(value) and value not in ("", None):
            return value
    return series.iloc[0] if len(series) else None


def _ensure_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [value]


def _safe_scene_tag(scene_tag: str) -> str:
    slug = SAFE_FILENAME_RE.sub("_", scene_tag.strip()) or "scene"
    return slug.strip("_") or "scene"


def _build_meta(df: pd.DataFrame, params: WeeklySceneJobParams) -> "SceneMeta":
    base_scene = _first_non_null(df.get("base_scene", pd.Series([None]))) or params.scene_tag
    return SceneMeta(
        week=params.week,
        scene_tag=params.scene_tag,
        base_scene=str(base_scene),
        marketplace_id=params.marketplace_id,
    )


@dataclass(slots=True)
class SceneMeta:
    week: str
    scene_tag: str
    base_scene: str
    marketplace_id: str


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if np is not None and isinstance(obj, np.generic):  # type: ignore[arg-type]
        return obj.item()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serialisable")


# ----------------------------------------------------------------------
# Module builders
# ----------------------------------------------------------------------


def build_overall_summary(df: pd.DataFrame, meta: SceneMeta) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "week": meta.week,
        "scene_tag": meta.scene_tag,
        "base_scene": meta.base_scene,
        "marketplace_id": meta.marketplace_id,
        "asin_counts": _count_asins(df),
        "rank_trend_distribution": _count_bucket(df, "rank_trend", RANK_TREND_BUCKETS),
        "price_action_distribution": _count_bucket(df, "price_action", PRICE_ACTION_BUCKETS),
        "promo_action_distribution": _count_bucket(df, "promo_action", PROMO_ACTION_BUCKETS),
        "badge_changes": {
            "badge_change_asin_cnt": int(df.get("has_badge_change", pd.Series(dtype=int)).fillna(0).astype(int).gt(0).sum()),
        },
    }
    return payload


def build_scope_analysis(df: pd.DataFrame, meta: SceneMeta, *, scope: str) -> dict[str, Any]:
    if scope not in {"self", "competitor"}:
        raise ValueError(f"Unsupported scope {scope}")

    scoped = df[df["hyy_asin"].eq(1) if scope == "self" else ~df["hyy_asin"].eq(1)].copy()
    overview = {
        "asin_counts": {"total": int(len(scoped))},
        "rank_trend_distribution": _count_bucket(scoped, "rank_trend", RANK_TREND_BUCKETS),
        "price_action_distribution": _count_bucket(scoped, "price_action", PRICE_ACTION_BUCKETS),
        "promo_action_distribution": _count_bucket(scoped, "promo_action", PROMO_ACTION_BUCKETS),
        "badge_changes": {
            "badge_change_asin_cnt": int(
                scoped.get("has_badge_change", pd.Series(dtype=int)).fillna(0).astype(int).gt(0).sum()
            ),
        },
    }
    lists = _build_top_lists(scoped)
    payload: dict[str, Any] = {
        "week": meta.week,
        "scene_tag": meta.scene_tag,
        "base_scene": meta.base_scene,
        "marketplace_id": meta.marketplace_id,
        "scope": scope,
        "overview": overview,
        "top_lists": lists,
    }
    return payload


def build_self_risk_and_opportunity(df: pd.DataFrame, meta: SceneMeta) -> dict[str, Any]:
    scoped = df[df["hyy_asin"].eq(1)].copy()
    rules = {
        "risk": {
            "rank_trend": "排名明显下降",
            "rating_diff_threshold": 0.0,
            "new_reviews_min": 10,
            "promo_cancel_and_rank_down": True,
            "badge_removed": True,
        },
        "opportunity": {
            "rank_trend": "排名明显上升",
            "badge_added": True,
            "rank_up_and_price_not_high": True,
        },
    }
    risk_rows = _select_risk_rows(scoped, rules["risk"])
    opportunity_rows = _select_opportunity_rows(scoped, rules["opportunity"])
    payload: dict[str, Any] = {
        "week": meta.week,
        "scene_tag": meta.scene_tag,
        "base_scene": meta.base_scene,
        "marketplace_id": meta.marketplace_id,
        "scope": "self",
        "rules": rules,
        "risk_asin": _records_from_dataframe(risk_rows),
        "opportunity_asin": _records_from_dataframe(opportunity_rows),
    }
    return payload


def build_competitor_actions(df: pd.DataFrame, meta: SceneMeta) -> dict[str, Any]:
    scoped = df[~df["hyy_asin"].eq(1)].copy()
    payload: dict[str, Any] = {
        "week": meta.week,
        "scene_tag": meta.scene_tag,
        "base_scene": meta.base_scene,
        "marketplace_id": meta.marketplace_id,
        "scope": "competitor",
        "price_and_rank_moves": _records_from_dataframe(
            _select_price_rank_moves(scoped)
        ),
        "promo_and_rank_moves": _records_from_dataframe(
            _select_promo_rank_moves(scoped)
        ),
        "content_and_badge_moves": _records_from_dataframe(
            _select_content_badge_moves(scoped)
        ),
    }
    return payload


# ----------------------------------------------------------------------
# Counting helpers
# ----------------------------------------------------------------------


def _count_asins(df: pd.DataFrame) -> dict[str, int]:
    total = len(df)
    self_count = int(df["hyy_asin"].eq(1).sum())
    return {
        "total": int(total),
        "self": self_count,
        "competitor": int(total - self_count),
    }


def _count_bucket(df: pd.DataFrame, column: str, mapping: Mapping[str, Iterable[str]]) -> dict[str, int]:
    counts = {key: 0 for key in mapping}
    if df.empty or column not in df.columns:
        return counts
    series = df[column]
    for value in series.dropna():
        text = str(value)
        matched = False
        for key, labels in mapping.items():
            if text in labels:
                counts[key] += 1
                matched = True
                break
        if not matched and "rank_stable" in counts:
            counts["rank_stable"] += 1
    return counts


# ----------------------------------------------------------------------
# Top list builders
# ----------------------------------------------------------------------


TOP_LIST_COLUMNS = [
    "asin",
    "brand",
    "marketplace_id",
    "scene_tag",
    "rank_leaf_this",
    "rank_leaf_last",
    "rank_leaf_diff",
    "rank_trend",
    "price_current_this",
    "price_current_last",
    "price_action",
    "promo_action",
    "rating_this",
    "new_reviews",
    "badge_added_json",
    "badge_removed_json",
]


def _columns_present(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return ordered, de-duplicated columns that exist on the dataframe."""

    seen: set[str] = set()
    ordered: list[str] = []
    for column in columns:
        if column in df.columns and column not in seen:
            ordered.append(column)
            seen.add(column)
    return ordered


def _build_top_lists(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    lists = {
        "rank_up_top": _records_from_dataframe(_top_rank_movers(df, direction="up")),
        "rank_down_top": _records_from_dataframe(
            _top_rank_movers(df, direction="down")
        ),
        "price_drop_top": _records_from_dataframe(_top_price_moves(df, direction="drop")),
        "price_raise_top": _records_from_dataframe(
            _top_price_moves(df, direction="raise")
        ),
    }
    return lists


def _top_rank_movers(df: pd.DataFrame, *, direction: str) -> pd.DataFrame:
    if df.empty:
        return df.loc[:, TOP_LIST_COLUMNS]
    if direction == "up":
        subset = df[df["rank_leaf_diff"].fillna(0) > 0]
        ordered = subset.sort_values("rank_leaf_diff", ascending=False)
    else:
        subset = df[df["rank_leaf_diff"].fillna(0) < 0]
        ordered = subset.sort_values("rank_leaf_diff", ascending=True)
    return ordered.loc[:, TOP_LIST_COLUMNS].head(TOP_N)


def _top_price_moves(df: pd.DataFrame, *, direction: str) -> pd.DataFrame:
    if df.empty:
        return df.loc[:, TOP_LIST_COLUMNS]
    if direction == "drop":
        subset = df[df["price_action"].isin({"大幅降价", "小幅降价"})]
        ordered = subset.sort_values("price_current_diff", ascending=True)
    else:
        subset = df[df["price_action"].isin({"大幅涨价", "小幅涨价"})]
        ordered = subset.sort_values("price_current_diff", ascending=False)
    return ordered.loc[:, TOP_LIST_COLUMNS].head(TOP_N)


# ----------------------------------------------------------------------
# Risk & opportunity selection
# ----------------------------------------------------------------------


def _select_risk_rows(df: pd.DataFrame, rules: Mapping[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    rank_down = df["rank_trend"].eq(rules["rank_trend"])
    rating_drop = df["rating_diff"].fillna(0).lt(rules["rating_diff_threshold"])
    review_active = df["new_reviews"].fillna(0).ge(rules["new_reviews_min"])
    promo_cancel = df["promo_action"].eq("取消优惠")
    rank_loss = df["rank_leaf_diff"].fillna(0).lt(0)
    badge_removed = (
        df.get("has_badge_change", pd.Series(0, index=df.index)).fillna(0).eq(1)
        & df.get("badge_removed_cnt", pd.Series(0, index=df.index)).fillna(0).gt(0)
    )
    mask = rank_down | (rating_drop & review_active) | (promo_cancel & rank_loss) | badge_removed
    selected = df.loc[mask].copy()
    columns = TOP_LIST_COLUMNS + [
        "price_action",
        "rating_last",
        "rating_diff",
        "new_reviews",
        "has_badge_change",
    ]
    existing = _columns_present(selected, columns)
    return selected.loc[:, existing].head(TOP_N)


def _select_opportunity_rows(df: pd.DataFrame, rules: Mapping[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    rank_up = df["rank_trend"].eq(rules["rank_trend"])
    badge_added = df["badge_added_json"].apply(lambda x: bool(x))
    friendly_price = df["price_action"].isin({"价格稳定", "小幅降价", "大幅降价"})
    mask = rank_up & (badge_added | friendly_price)
    selected = df.loc[mask].copy()
    columns = TOP_LIST_COLUMNS + ["price_action", "coupon_pct_this", "coupon_description_this"]
    existing = _columns_present(selected, columns)
    return selected.loc[:, existing].head(TOP_N)


# ----------------------------------------------------------------------
# Competitor action selectors
# ----------------------------------------------------------------------


def _select_price_rank_moves(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["rank_trend"].isin({"排名明显上升", "排名明显下降"}) & df["price_action"].isin(
        {"大幅降价", "小幅降价", "小幅涨价", "大幅涨价"}
    )
    columns = TOP_LIST_COLUMNS + ["price_current_diff", "coupon_pct_this", "coupon_description_this"]
    existing = _columns_present(df, columns)
    return df.loc[mask, existing].sort_values("rank_leaf_diff", ascending=False).head(TOP_N)


def _select_promo_rank_moves(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["rank_trend"].isin({"排名明显上升", "排名明显下降"}) & df["promo_action"].isin(
        {"新增优惠", "优惠内容变化"}
    )
    columns = TOP_LIST_COLUMNS + ["promo_action", "has_coupon_this", "coupon_description_this"]
    existing = _columns_present(df, columns)
    return df.loc[mask, existing].sort_values("rank_leaf_diff", ascending=False).head(TOP_N)


def _select_content_badge_moves(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    content_mask = (
        df.get("image_cnt_diff", pd.Series(0)).fillna(0).ne(0)
        | df.get("video_cnt_diff", pd.Series(0)).fillna(0).ne(0)
        | df.get("aplus_flag_this", pd.Series(0)).fillna(0).eq(1)
        | df.get("has_badge_change", pd.Series(0)).fillna(0).ne(0)
    )
    columns = TOP_LIST_COLUMNS + [
        "image_cnt_diff",
        "video_cnt_diff",
        "aplus_flag_this",
        "has_badge_change",
    ]
    existing = _columns_present(df, columns)
    ordered = df.loc[content_mask, existing].sort_values("rank_leaf_diff", ascending=False)
    return ordered.head(TOP_N)


# ----------------------------------------------------------------------
# Serialisation helpers
# ----------------------------------------------------------------------


def _records_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    records = []
    for row in df.to_dict(orient="records"):
        records.append({key: _normalise_value(value) for key, value in row.items()})
    return records


def _normalise_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat()
    if np is not None and isinstance(value, np.generic):  # type: ignore[arg-type]
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (list, tuple)):
        return [_normalise_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalise_value(v) for k, v in value.items()}
    if is_scalar(value) and pd.isna(value):  # type: ignore[arg-type]
        return None
    return value


__all__ = [
    "WeeklySceneJobParams",
    "WeeklySceneJsonError",
    "WeeklySceneJsonGenerator",
    "build_overall_summary",
    "build_scope_analysis",
    "build_self_risk_and_opportunity",
    "build_competitor_actions",
]

