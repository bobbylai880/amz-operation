"""ETL pipeline that computes weekly ASIN deltas from snapshot data."""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sqlalchemy.engine import Engine

from scpc.db.engine import create_doris_engine
from scpc.db.io import fetch_dataframe, replace_into
from scpc.utils.dependencies import ensure_packages

LOGGER = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "asin_week_diff_rules.yml"
SNAPSHOT_TABLE = "bi_amz_asin_product_snapshot"
TARGET_TABLE = "bi_amz_asin_product_week_diff"

SUNDAY_LOOKUP_SQL = """
    SELECT DISTINCT sunday
    FROM bi_amz_asin_product_snapshot
    WHERE week = :week
    """

JOIN_SQL = """
    SELECT
      cur.asin,
      cur.marketplace_id,
      cur.sunday AS sunday_this,
      cur.week AS week_this,
      prev.sunday AS sunday_last,
      prev.week AS week_last,
      cur.rank_leaf AS rank_leaf_this,
      prev.rank_leaf AS rank_leaf_last,
      cur.rank_root AS rank_root_this,
      prev.rank_root AS rank_root_last,
      cur.price_current AS price_current_this,
      prev.price_current AS price_current_last,
      cur.price_list AS price_list_this,
      prev.price_list AS price_list_last,
      cur.coupon_pct AS coupon_pct_this,
      prev.coupon_pct AS coupon_pct_last,
      cur.coupon_description AS coupon_description_this,
      prev.coupon_description AS coupon_description_last,
      cur.rating AS rating_this,
      prev.rating AS rating_last,
      cur.reviews AS reviews_this,
      prev.reviews AS reviews_last,
      cur.image_cnt AS image_cnt_this,
      prev.image_cnt AS image_cnt_last,
      cur.video_cnt AS video_cnt_this,
      prev.video_cnt AS video_cnt_last,
      cur.bullet_cnt AS bullet_cnt_this,
      prev.bullet_cnt AS bullet_cnt_last,
      cur.title_len AS title_len_this,
      prev.title_len AS title_len_last,
      cur.aplus_flag AS aplus_flag_this,
      prev.aplus_flag AS aplus_flag_last,
      cur.badge_json AS badge_json_this,
      prev.badge_json AS badge_json_last
    FROM bi_amz_asin_product_snapshot cur
    LEFT JOIN bi_amz_asin_product_snapshot prev
      ON cur.asin = prev.asin
     AND cur.marketplace_id = prev.marketplace_id
     AND prev.sunday = :sunday_last
    WHERE cur.sunday = :sunday_this
    """

WEEK_DIFF_COLUMNS = [
    "asin",
    "marketplace_id",
    "sunday_this",
    "week_this",
    "sunday_last",
    "week_last",
    "rank_leaf_this",
    "rank_leaf_last",
    "rank_leaf_diff",
    "rank_root_this",
    "rank_root_last",
    "rank_root_diff",
    "price_current_this",
    "price_current_last",
    "price_current_diff",
    "price_change_rate",
    "price_list_this",
    "price_list_last",
    "coupon_pct_this",
    "coupon_pct_last",
    "coupon_description_this",
    "coupon_description_last",
    "has_coupon_this",
    "has_coupon_last",
    "rating_this",
    "rating_last",
    "rating_diff",
    "reviews_this",
    "reviews_last",
    "new_reviews",
    "image_cnt_this",
    "image_cnt_last",
    "image_cnt_diff",
    "video_cnt_this",
    "video_cnt_last",
    "video_cnt_diff",
    "bullet_cnt_this",
    "bullet_cnt_last",
    "bullet_cnt_diff",
    "title_len_this",
    "title_len_last",
    "title_len_diff",
    "aplus_flag_this",
    "aplus_flag_last",
    "badge_json_this",
    "badge_json_last",
    "badge_added_json",
    "badge_removed_json",
    "badge_added_cnt",
    "badge_removed_cnt",
    "has_badge_change",
    "price_action",
    "rank_trend",
    "promo_action",
    "etl_time",
]

DEFAULT_RULES = {
    "price_action": {
        "new_asin_label": "新ASIN",
        "unknown_label": "价格未知",
        "thresholds": {
            "big_drop": -0.10,
            "small_drop": -0.03,
            "small_raise": 0.03,
            "big_raise": 0.10,
        },
        "labels": {
            "big_drop": "大幅降价",
            "small_drop": "小幅降价",
            "stable": "价格稳定",
            "small_raise": "小幅涨价",
            "big_raise": "大幅涨价",
        },
    },
    "rank_trend": {
        "new_asin_label": "新ASIN",
        "unknown_label": "未知",
        "thresholds": {
            "big_up": 50,
            "big_down": -50,
        },
        "labels": {
            "big_up": "排名明显上升",
            "big_down": "排名明显下降",
            "stable": "排名基本稳定",
        },
    },
    "promo_action": {
        "labels": {
            "new_promo": "新增优惠",
            "cancel_promo": "取消优惠",
            "none": "优惠无",
            "changed": "优惠内容变化",
            "unchanged": "优惠无明显变化",
        }
    },
}


class WeekDiffJobError(RuntimeError):
    """Raised when the weekly diff ETL cannot proceed."""


def _deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value  # type: ignore[index]
    return target


def _detect_missing_fields(source: Mapping[str, Any], template: Mapping[str, Any], prefix: str = "") -> list[str]:
    missing: list[str] = []
    for key, expected in template.items():
        dotted = f"{prefix}{key}" if prefix else key
        if key not in source:
            missing.append(dotted)
            continue
        candidate = source[key]
        if isinstance(expected, Mapping):
            if not isinstance(candidate, Mapping):
                missing.append(dotted)
            else:
                missing.extend(_detect_missing_fields(candidate, expected, prefix=f"{dotted}."))
    return missing


def load_week_diff_rules(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Return the rule configuration used for label generation."""

    ensure_packages([("yaml", "PyYAML")])
    import yaml  # type: ignore

    path = Path(config_path)
    merged = deepcopy(DEFAULT_RULES)
    if not path.exists():
        LOGGER.error("Week diff rule file missing", extra={"path": str(path)})
        LOGGER.warning("Falling back to built-in defaults for week diff rules")
        return merged

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - configuration error
        LOGGER.error("Failed to parse week diff rules", extra={"path": str(path)}, exc_info=True)
        return merged
    except OSError as exc:  # pragma: no cover - IO error
        LOGGER.error("Unable to read week diff rules", extra={"path": str(path)}, exc_info=True)
        return merged

    if not isinstance(data, Mapping):
        LOGGER.error("Week diff rules must be a mapping", extra={"path": str(path)})
        return merged

    _deep_update(merged, data)
    missing_fields = _detect_missing_fields(data, DEFAULT_RULES)
    if missing_fields:
        LOGGER.warning(
            "Week diff rules missing fields",
            extra={"path": str(path), "missing": ", ".join(missing_fields)},
        )
    return merged


def _normalise_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except TypeError:
        pass
    return value


def _coerce_date(value: Any) -> date | None:
    value = _normalise_scalar(value)
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().date()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime().date()
    return None


def _diff(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    return a - b


def _compute_price_change_rate(price_diff: Any, price_last: Any) -> float | None:
    if price_diff is None or price_last is None:
        return None
    try:
        denominator = float(price_last)
    except (TypeError, ValueError):
        return None
    if math.isclose(denominator, 0.0, abs_tol=1e-9):
        return None
    try:
        numerator = float(price_diff)
    except (TypeError, ValueError):
        return None
    return numerator / denominator


def _normalise_coupon_text(value: Any) -> str | None:
    value = _normalise_scalar(value)
    if value is None:
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    normalised = " ".join(text_value.split())
    return normalised or None


def _has_coupon(pct: Any, description: str | None) -> int:
    pct_value = 0.0
    if pct is not None:
        try:
            pct_value = float(pct)
        except (TypeError, ValueError):
            pct_value = 0.0
    has_desc = bool(description)
    return 1 if (pct_value != 0.0 or has_desc) else 0


def _parse_badges(raw: Any) -> list[str]:
    raw = _normalise_scalar(raw)
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return []
    else:
        loaded = raw
    if isinstance(loaded, list):
        return [str(item) for item in loaded if item is not None]
    return []


def _classify_price_action(
    price_last: Any,
    price_change_rate: float | None,
    rules: Mapping[str, Any],
) -> str:
    labels = rules.get("labels", {})
    thresholds = rules.get("thresholds", {})
    if price_last is None:
        return rules.get("new_asin_label", "新ASIN")
    if price_change_rate is None:
        return rules.get("unknown_label", "价格未知")
    big_drop = float(thresholds.get("big_drop", -0.10))
    small_drop = float(thresholds.get("small_drop", -0.03))
    small_raise = float(thresholds.get("small_raise", 0.03))
    big_raise = float(thresholds.get("big_raise", 0.10))
    if price_change_rate <= big_drop:
        return labels.get("big_drop", "大幅降价")
    if price_change_rate <= small_drop:
        return labels.get("small_drop", "小幅降价")
    if price_change_rate < small_raise:
        return labels.get("stable", "价格稳定")
    if price_change_rate < big_raise:
        return labels.get("small_raise", "小幅涨价")
    return labels.get("big_raise", "大幅涨价")


def _classify_rank_trend(rank_last: Any, rank_diff: Any, rules: Mapping[str, Any]) -> str:
    labels = rules.get("labels", {})
    thresholds = rules.get("thresholds", {})
    if rank_last is None:
        return rules.get("new_asin_label", "新ASIN")
    if rank_diff is None:
        return rules.get("unknown_label", "未知")
    big_up = float(thresholds.get("big_up", 50))
    big_down = float(thresholds.get("big_down", -50))
    if rank_diff >= big_up:
        return labels.get("big_up", "排名明显上升")
    if rank_diff <= big_down:
        return labels.get("big_down", "排名明显下降")
    return labels.get("stable", "排名基本稳定")


def _determine_promo_action(has_last: int, has_this: int, pct_last: Any, pct_this: Any, desc_last: Any, desc_this: Any, labels: Mapping[str, str]) -> str:
    if has_last == 0 and has_this == 0:
        return labels.get("none", "优惠无")
    if has_last == 0 and has_this == 1:
        return labels.get("new_promo", "新增优惠")
    if has_last == 1 and has_this == 0:
        return labels.get("cancel_promo", "取消优惠")
    changed = False
    if _normalise_scalar(pct_last) != _normalise_scalar(pct_this):
        changed = True
    else:
        changed = _normalise_coupon_text(desc_last) != _normalise_coupon_text(desc_this)
    if changed:
        return labels.get("changed", "优惠内容变化")
    return labels.get("unchanged", "优惠无明显变化")


def compute_week_diff(
    source: pd.DataFrame,
    rules: Mapping[str, Any],
    sunday_last_default: date | None = None,
) -> pd.DataFrame:
    """Transform joined snapshot rows into the week diff payload."""

    if source.empty:
        return pd.DataFrame(columns=WEEK_DIFF_COLUMNS)

    now = datetime.utcnow()
    rows: list[dict[str, Any]] = []
    price_rules = rules.get("price_action", {})
    rank_rules = rules.get("rank_trend", {})
    promo_labels = rules.get("promo_action", {}).get("labels", {})

    for record in source.to_dict(orient="records"):
        asin = record.get("asin")
        marketplace_id = record.get("marketplace_id")
        sunday_this = _coerce_date(record.get("sunday_this"))
        sunday_last = _coerce_date(record.get("sunday_last")) or sunday_last_default
        week_this = _normalise_scalar(record.get("week_this"))
        week_last = _normalise_scalar(record.get("week_last"))
        rank_leaf_this = _normalise_scalar(record.get("rank_leaf_this"))
        rank_leaf_last = _normalise_scalar(record.get("rank_leaf_last"))
        rank_root_this = _normalise_scalar(record.get("rank_root_this"))
        rank_root_last = _normalise_scalar(record.get("rank_root_last"))
        rank_leaf_diff = _diff(rank_leaf_last, rank_leaf_this)
        rank_root_diff = _diff(rank_root_last, rank_root_this)
        price_current_this = _normalise_scalar(record.get("price_current_this"))
        price_current_last = _normalise_scalar(record.get("price_current_last"))
        price_current_diff = _diff(price_current_this, price_current_last)
        price_change_rate = _compute_price_change_rate(price_current_diff, price_current_last)
        price_list_this = _normalise_scalar(record.get("price_list_this"))
        price_list_last = _normalise_scalar(record.get("price_list_last"))
        coupon_pct_this = _normalise_scalar(record.get("coupon_pct_this"))
        coupon_pct_last = _normalise_scalar(record.get("coupon_pct_last"))
        coupon_description_this = _normalise_coupon_text(record.get("coupon_description_this"))
        coupon_description_last = _normalise_coupon_text(record.get("coupon_description_last"))
        has_coupon_this = _has_coupon(coupon_pct_this, coupon_description_this)
        has_coupon_last = _has_coupon(coupon_pct_last, coupon_description_last)
        rating_this = _normalise_scalar(record.get("rating_this"))
        rating_last = _normalise_scalar(record.get("rating_last"))
        rating_diff = _diff(rating_this, rating_last)
        reviews_this = _normalise_scalar(record.get("reviews_this"))
        reviews_last = _normalise_scalar(record.get("reviews_last"))
        if reviews_this is None:
            new_reviews = None
        elif reviews_last is None:
            new_reviews = reviews_this
        else:
            new_reviews = reviews_this - reviews_last
        image_cnt_this = _normalise_scalar(record.get("image_cnt_this"))
        image_cnt_last = _normalise_scalar(record.get("image_cnt_last"))
        image_cnt_diff = _diff(image_cnt_this, image_cnt_last)
        video_cnt_this = _normalise_scalar(record.get("video_cnt_this"))
        video_cnt_last = _normalise_scalar(record.get("video_cnt_last"))
        video_cnt_diff = _diff(video_cnt_this, video_cnt_last)
        bullet_cnt_this = _normalise_scalar(record.get("bullet_cnt_this"))
        bullet_cnt_last = _normalise_scalar(record.get("bullet_cnt_last"))
        bullet_cnt_diff = _diff(bullet_cnt_this, bullet_cnt_last)
        title_len_this = _normalise_scalar(record.get("title_len_this"))
        title_len_last = _normalise_scalar(record.get("title_len_last"))
        title_len_diff = _diff(title_len_this, title_len_last)
        aplus_flag_this = _normalise_scalar(record.get("aplus_flag_this"))
        aplus_flag_last = _normalise_scalar(record.get("aplus_flag_last"))
        badge_json_this = _normalise_scalar(record.get("badge_json_this"))
        badge_json_last = _normalise_scalar(record.get("badge_json_last"))
        badges_this = set(_parse_badges(badge_json_this))
        badges_last = set(_parse_badges(badge_json_last))
        badge_added = sorted(badges_this - badges_last)
        badge_removed = sorted(badges_last - badges_this)
        badge_added_json = json.dumps(badge_added, ensure_ascii=False)
        badge_removed_json = json.dumps(badge_removed, ensure_ascii=False)
        badge_added_cnt = len(badge_added)
        badge_removed_cnt = len(badge_removed)
        has_badge_change = 1 if (badge_added_cnt or badge_removed_cnt) else 0
        price_action = _classify_price_action(price_current_last, price_change_rate, price_rules)
        rank_trend = _classify_rank_trend(rank_leaf_last, rank_leaf_diff, rank_rules)
        promo_action = _determine_promo_action(
            has_coupon_last,
            has_coupon_this,
            coupon_pct_last,
            coupon_pct_this,
            coupon_description_last,
            coupon_description_this,
            promo_labels,
        )

        rows.append(
            {
                "asin": asin,
                "marketplace_id": marketplace_id,
                "sunday_this": sunday_this,
                "week_this": week_this,
                "sunday_last": sunday_last,
                "week_last": week_last,
                "rank_leaf_this": rank_leaf_this,
                "rank_leaf_last": rank_leaf_last,
                "rank_leaf_diff": rank_leaf_diff,
                "rank_root_this": rank_root_this,
                "rank_root_last": rank_root_last,
                "rank_root_diff": rank_root_diff,
                "price_current_this": price_current_this,
                "price_current_last": price_current_last,
                "price_current_diff": price_current_diff,
                "price_change_rate": price_change_rate,
                "price_list_this": price_list_this,
                "price_list_last": price_list_last,
                "coupon_pct_this": coupon_pct_this,
                "coupon_pct_last": coupon_pct_last,
                "coupon_description_this": coupon_description_this,
                "coupon_description_last": coupon_description_last,
                "has_coupon_this": has_coupon_this,
                "has_coupon_last": has_coupon_last,
                "rating_this": rating_this,
                "rating_last": rating_last,
                "rating_diff": rating_diff,
                "reviews_this": reviews_this,
                "reviews_last": reviews_last,
                "new_reviews": new_reviews,
                "image_cnt_this": image_cnt_this,
                "image_cnt_last": image_cnt_last,
                "image_cnt_diff": image_cnt_diff,
                "video_cnt_this": video_cnt_this,
                "video_cnt_last": video_cnt_last,
                "video_cnt_diff": video_cnt_diff,
                "bullet_cnt_this": bullet_cnt_this,
                "bullet_cnt_last": bullet_cnt_last,
                "bullet_cnt_diff": bullet_cnt_diff,
                "title_len_this": title_len_this,
                "title_len_last": title_len_last,
                "title_len_diff": title_len_diff,
                "aplus_flag_this": aplus_flag_this,
                "aplus_flag_last": aplus_flag_last,
                "badge_json_this": badge_json_this,
                "badge_json_last": badge_json_last,
                "badge_added_json": badge_added_json,
                "badge_removed_json": badge_removed_json,
                "badge_added_cnt": badge_added_cnt,
                "badge_removed_cnt": badge_removed_cnt,
                "has_badge_change": has_badge_change,
                "price_action": price_action,
                "rank_trend": rank_trend,
                "promo_action": promo_action,
                "etl_time": now,
            }
        )

    return pd.DataFrame(rows, columns=WEEK_DIFF_COLUMNS)


def _resolve_sunday(engine: Engine, week: str) -> date:
    df = fetch_dataframe(engine, SUNDAY_LOOKUP_SQL, {"week": week})
    if df.empty:
        raise WeekDiffJobError(f"Week {week} not found in {SNAPSHOT_TABLE}")
    sundays = {value for value in (_coerce_date(item) for item in df["sunday"].tolist()) if value}
    if len(sundays) != 1:
        raise WeekDiffJobError(f"Week {week} maps to multiple sunday values: {sorted(sundays)}")
    return sundays.pop()


def _fetch_joined_snapshots(engine: Engine, sunday_this: date, sunday_last: date) -> pd.DataFrame:
    params = {"sunday_this": sunday_this, "sunday_last": sunday_last}
    return fetch_dataframe(engine, JOIN_SQL, params)


def run_week_diff_etl(
    week: str,
    *,
    engine: Engine | None = None,
    config_path: str | Path = CONFIG_PATH,
) -> int:
    """Entry point for computing and persisting ASIN weekly differences."""

    rules = load_week_diff_rules(config_path)
    doris = engine or create_doris_engine()

    LOGGER.info("asin_week_diff.start", extra={"week": week})
    sunday_this = _resolve_sunday(doris, week)
    sunday_last = sunday_this - timedelta(days=7)
    LOGGER.info(
        "asin_week_diff.sunday_resolved",
        extra={"week": week, "sunday_this": str(sunday_this), "sunday_last": str(sunday_last)},
    )

    joined = _fetch_joined_snapshots(doris, sunday_this, sunday_last)
    LOGGER.info(
        "asin_week_diff.join_rows",
        extra={"week": week, "rows": int(joined.shape[0])},
    )
    if joined.empty:
        LOGGER.info("asin_week_diff.no_rows", extra={"week": week})
        return 0

    payload = compute_week_diff(joined, rules, sunday_last_default=sunday_last)
    LOGGER.info(
        "asin_week_diff.computed_rows",
        extra={"week": week, "rows": int(payload.shape[0])},
    )
    affected = replace_into(doris, TARGET_TABLE, payload)
    LOGGER.info(
        "asin_week_diff.write_complete",
        extra={"week": week, "rows": affected, "table": TARGET_TABLE},
    )
    LOGGER.info("asin_week_diff.finish", extra={"week": week})
    return affected


__all__ = [
    "CONFIG_PATH",
    "DEFAULT_RULES",
    "WEEK_DIFF_COLUMNS",
    "WeekDiffJobError",
    "compute_week_diff",
    "load_week_diff_rules",
    "run_week_diff_etl",
]
