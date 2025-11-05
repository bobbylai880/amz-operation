"""Feature engineering utilities for the competition module."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


BadgeValue = list[str]


@dataclass(slots=True)
class CompetitionFeatureResult:
    """Container holding competition pair features and aggregates."""

    metadata: dict[str, Any]
    pairs: list[dict[str, Any]]
    summary: dict[str, Any]
    insufficient_data: bool

    def as_dict(self) -> dict[str, Any]:
        """Serialise the feature bundle into JSON-friendly primitives."""

        payload = {
            "metadata": self.metadata.copy(),
            "pairs": self.pairs,
            "summary": self.summary,
            "insufficient_data": self.insufficient_data,
        }
        sunday = payload["metadata"].get("sunday")
        if isinstance(sunday, (pd.Timestamp, date)):
            payload["metadata"]["sunday"] = sunday.isoformat()
        previous_sunday = payload["metadata"].get("previous_sunday")
        if isinstance(previous_sunday, (pd.Timestamp, date)):
            payload["metadata"]["previous_sunday"] = previous_sunday.isoformat()
        return payload


@dataclass(slots=True)
class CompetitionTables:
    """Structured artefacts mirroring Doris competition tables."""

    entities: pd.DataFrame
    pairs: pd.DataFrame
    delta: pd.DataFrame
    summary: pd.DataFrame


DEFAULT_SCORING_RULES: dict[str, dict[str, float]] = {
    "price": {"theta": 0.0, "k": 1.5, "weight": 0.30},
    "rank": {"theta": 0.0, "k": 6.0, "weight": 0.25},
    "content": {"theta": 0.0, "k": 5.0, "weight": 0.20},
    "social": {"theta": 0.0, "k": 4.0, "weight": 0.15},
    "badge": {"theta": 0.0, "k": 3.0, "weight": 0.10},
}

DEFAULT_BAND_CUTS: dict[str, float] = {"C1": 0.25, "C2": 0.50, "C3": 0.75, "C4": 1.00}


def _attach_scene_tags(
    snapshots: pd.DataFrame,
    scene_tags: pd.DataFrame,
) -> pd.DataFrame:
    """Merge scene tagging metadata onto snapshot rows."""

    required_cols = {"asin", "marketplace_id", "scene_tag", "base_scene", "morphology"}
    missing = required_cols - set(scene_tags.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"scene_tags is missing required columns: {missing_cols}")

    tags = scene_tags.loc[:, ["asin", "marketplace_id", "scene_tag", "base_scene", "morphology"]].copy()
    merged = snapshots.merge(tags, on=["asin", "marketplace_id"], how="left", suffixes=("", "_tag"))

    for col in ("scene_tag", "base_scene", "morphology"):
        tag_col = f"{col}_tag"
        if col not in merged.columns:
            merged[col] = pd.NA
        if tag_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[tag_col])
            merged = merged.drop(columns=tag_col)

    merged = merged.loc[merged["scene_tag"].notna()].copy()
    return merged.reset_index(drop=True)


def clean_competition_entities(
    snapshots: pd.DataFrame,
    *,
    my_asins: Iterable[str],
    scene_tags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Clean ASIN snapshots, optionally merging scene tags, and derive features."""

    ordered_cols = [
        "scene_tag",
        "base_scene",
        "morphology",
        "marketplace_id",
        "week",
        "sunday",
        "parent_asin",
        "asin",
        "hyy_asin",
        "price_current",
        "price_list",
        "coupon_pct",
        "discount_rate",
        "rank_root",
        "rank_leaf",
        "rating",
        "reviews",
        "image_cnt",
        "video_cnt",
        "bullet_cnt",
        "title_len",
        "aplus_flag",
        "badge_json",
        "price_net",
        "rank_score",
        "social_proof",
        "content_score",
    ]

    if snapshots.empty:
        return pd.DataFrame(columns=ordered_cols)

    df = snapshots.copy()
    if scene_tags is not None:
        df = _attach_scene_tags(df, scene_tags)
        if df.empty:
            return pd.DataFrame(columns=ordered_cols)
    my_set = set(my_asins)
    df["hyy_asin"] = df["asin"].isin(my_set).astype(int)

    numeric_cols = [
        "price_current",
        "price_list",
        "coupon_pct",
        "discount_rate",
        "rank_root",
        "rank_leaf",
        "rating",
        "reviews",
        "image_cnt",
        "video_cnt",
        "bullet_cnt",
        "title_len",
        "aplus_flag",
    ]
    for col in numeric_cols:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["coupon_pct"] = df["coupon_pct"].fillna(0.0).clip(lower=0.0, upper=0.9)
    df["price_current"] = df["price_current"].fillna(0.0)
    df["price_list"] = df["price_list"].replace(0, np.nan)

    df["discount_rate"] = df["discount_rate"].where(~df["discount_rate"].isna())
    valid_mask = df["discount_rate"].isna() & df["price_list"].notna()
    df.loc[valid_mask, "discount_rate"] = 1 - (
        df.loc[valid_mask, "price_current"] / df.loc[valid_mask, "price_list"]
    )
    df["discount_rate"] = df["discount_rate"].fillna(0.0).clip(lower=-1.0, upper=1.0)

    df["price_net"] = df["price_current"] * (1 - df["coupon_pct"])

    group_cols = ["scene_tag", "base_scene", "morphology", "marketplace_id", "week"]

    df["rank_score"] = _normalise_series(df, "rank_root", group_cols, invert=True)

    reviews_term = np.log1p(df["reviews"].fillna(0.0))
    rating_term = (df["rating"].fillna(0.0) / 5).clip(lower=0.0, upper=1.0)
    df["social_proof"] = _normalise_series(df.assign(_social=rating_term * reviews_term), "_social", group_cols)

    image_norm = _normalise_series(df.fillna({"image_cnt": 0.0}), "image_cnt", group_cols)
    video_norm = _normalise_series(df.fillna({"video_cnt": 0.0}), "video_cnt", group_cols)
    bullet_norm = _normalise_series(df.fillna({"bullet_cnt": 0.0}), "bullet_cnt", group_cols)
    title_norm = _normalise_series(df.fillna({"title_len": 0.0}), "title_len", group_cols)
    aplus_norm = df["aplus_flag"].fillna(0.0).clip(lower=0.0, upper=1.0)
    df["content_score"] = (
        0.35 * image_norm
        + 0.25 * video_norm
        + 0.20 * bullet_norm
        + 0.10 * title_norm
        + 0.10 * aplus_norm
    ).clip(lower=0.0, upper=1.0)

    df["badge_json"] = df.get("badge_json", []).apply(_normalise_badges)

    return df.reindex(columns=ordered_cols)


def build_competition_pairs(
    entities: pd.DataFrame,
    *,
    scoring_rules: pd.DataFrame | dict[str, Any] | None = None,
    rule_name: str = "default",
) -> pd.DataFrame:
    """Generate pairwise comparison rows for leader and median competitors."""

    if entities.empty:
        return pd.DataFrame(columns=_pair_columns())

    feature_rules, band_cuts = _prepare_scoring_rules(scoring_rules, rule_name)

    group_cols = ["scene_tag", "base_scene", "morphology", "marketplace_id", "week", "sunday"]
    records: list[dict[str, Any]] = []

    for _, group in entities.groupby(group_cols, dropna=False):
        mine = group.loc[group["hyy_asin"] == 1]
        competitors = group.loc[group["hyy_asin"] == 0]
        if mine.empty or competitors.empty:
            continue

        competitors = competitors.sort_values(
            by=["rank_score", "rank_root"], ascending=[False, True]
        )
        leader = competitors.iloc[0]
        median = competitors.iloc[len(competitors) // 2]

        comp_prices = pd.to_numeric(competitors["price_current"], errors="coerce")
        comp_price_mean = float(comp_prices.mean()) if comp_prices.notna().any() else None
        comp_price_std = (
            float(comp_prices.std(ddof=0))
            if comp_prices.notna().sum() > 1
            else None
        )
        median_price = _clean_float(median.get("price_current"))

        for _, my_row in mine.iterrows():
            for opp_type, opp_row in (("leader", leader), ("median", median)):
                record = _compute_pair_row(
                    my_row,
                    opp_row,
                    opp_type=opp_type,
                    median_price=median_price,
                    comp_price_mean=comp_price_mean,
                    comp_price_std=comp_price_std,
                    feature_rules=feature_rules,
                    band_cuts=band_cuts,
                )
                records.append(record)

    if not records:
        return pd.DataFrame(columns=_pair_columns())
    return pd.DataFrame.from_records(records, columns=_pair_columns())


def build_competition_delta(
    entities: pd.DataFrame,
    *,
    pairs_current: pd.DataFrame,
    pairs_previous: pd.DataFrame,
    week: str,
    previous_week: str | None,
) -> pd.DataFrame:
    """Construct week-over-week delta rows for the competition window."""

    if previous_week is None or pairs_current.empty:
        return pd.DataFrame(columns=_delta_columns())

    entities_current = _filter_week(entities, week)
    entities_previous = _filter_week(entities, previous_week)

    previous_lookup = {
        (row["my_asin"], row["opp_type"]): row
        for _, row in pairs_previous.iterrows()
    }
    my_prev_entities = {row["asin"]: row for _, row in entities_previous.iterrows()}
    my_curr_entities = {row["asin"]: row for _, row in entities_current.iterrows()}

    rows: list[dict[str, Any]] = []
    for _, current in pairs_current.iterrows():
        key = (current["my_asin"], current["opp_type"])
        previous = previous_lookup.get(key)
        my_curr = my_curr_entities.get(current["my_asin"])
        my_prev = my_prev_entities.get(current["my_asin"])

        row = {
            "scene_tag": current.get("scene_tag"),
            "base_scene": current.get("base_scene"),
            "morphology": current.get("morphology"),
            "marketplace_id": current.get("marketplace_id"),
            "window_id": f"{week}__{previous_week}",
            "week_w0": week,
            "sunday_w0": current.get("sunday"),
            "week_w1": previous_week,
            "sunday_w1": previous.get("sunday") if previous is not None else None,
            "my_parent_asin": current.get("my_parent_asin"),
            "my_asin": current.get("my_asin"),
            "opp_type": current.get("opp_type"),
            "d_price_net": _diff_float(
                my_curr.get("price_net") if my_curr is not None else None,
                my_prev.get("price_net") if my_prev is not None else None,
            ),
            "d_rank_score": _diff_float(
                my_curr.get("rank_score") if my_curr is not None else None,
                my_prev.get("rank_score") if my_prev is not None else None,
            ),
            "d_social_proof": _diff_float(
                my_curr.get("social_proof") if my_curr is not None else None,
                my_prev.get("social_proof") if my_prev is not None else None,
            ),
            "d_content_score": _diff_float(
                my_curr.get("content_score") if my_curr is not None else None,
                my_prev.get("content_score") if my_prev is not None else None,
            ),
            "badge_change": (
                _badge_count(my_curr.get("badge_json")) - _badge_count(my_prev.get("badge_json"))
            )
            if my_curr is not None and my_prev is not None
            else None,
            "d_price_gap_leader": _diff_float(
                current.get("price_gap_leader"),
                previous.get("price_gap_leader") if previous is not None else None,
            ),
            "d_price_index_med": _diff_float(
                current.get("price_index_med"),
                previous.get("price_index_med") if previous is not None else None,
            ),
            "d_rank_pos_pct": _diff_float(
                current.get("rank_pos_pct"),
                previous.get("rank_pos_pct") if previous is not None else None,
            ),
            "d_content_gap": _diff_float(
                current.get("content_gap"),
                previous.get("content_gap") if previous is not None else None,
            ),
            "d_social_gap": _diff_float(
                current.get("social_gap"),
                previous.get("social_gap") if previous is not None else None,
            ),
            "delta_pressure": _diff_float(
                current.get("pressure"),
                previous.get("pressure") if previous is not None else None,
            ),
        }
        rows.append(row)

    return pd.DataFrame.from_records(rows, columns=_delta_columns())


def summarise_competition_scene(
    *,
    pairs_current: pd.DataFrame,
    delta_window: pd.DataFrame,
    entities_current: pd.DataFrame,
    entities_previous: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate weekly competition metrics for reporting."""

    if pairs_current.empty and entities_current.empty:
        return pd.DataFrame(columns=_summary_columns())

    summary_stats = _build_summary(
        pairs_current=pairs_current,
        delta_window=delta_window,
        entities_current=entities_current,
        entities_previous=entities_previous,
    )

    source = pairs_current if not pairs_current.empty else entities_current
    base = source.iloc[0]
    row = {
        "scene_tag": base.get("scene_tag"),
        "base_scene": base.get("base_scene"),
        "morphology": base.get("morphology"),
        "marketplace_id": base.get("marketplace_id"),
        "week": base.get("week"),
        "sunday": base.get("sunday"),
        "my_asin_cnt": summary_stats["my_asin_cnt"],
        "comp_cnt": summary_stats["comp_cnt"],
        "pressure_p50": summary_stats["pressure_p50"],
        "pressure_p90": summary_stats["pressure_p90"],
        "worsen_ratio": summary_stats["worsen_ratio"],
        "moves_coupon_up": summary_stats["moves"]["moves_coupon_up"],
        "moves_price_down": summary_stats["moves"]["moves_price_down"],
        "moves_new_video": summary_stats["moves"]["moves_new_video"],
        "moves_badge_gain": summary_stats["moves"]["moves_badge_gain"],
        "avg_score_price": summary_stats["avg_scores"].get("score_price"),
        "avg_score_rank": summary_stats["avg_scores"].get("score_rank"),
        "avg_score_cont": summary_stats["avg_scores"].get("score_cont"),
        "avg_score_soc": summary_stats["avg_scores"].get("score_soc"),
        "avg_score_badge": summary_stats["avg_scores"].get("score_badge"),
    }
    return pd.DataFrame([row], columns=_summary_columns())


def build_competition_tables(
    snapshots: pd.DataFrame,
    *,
    week: str,
    previous_week: str | None,
    my_asins: Iterable[str],
    scene_tags: pd.DataFrame | None = None,
    scoring_rules: pd.DataFrame | dict[str, Any] | None = None,
    rule_name: str = "default",
) -> CompetitionTables:
    """Produce Doris-aligned tables from snapshots, including scene tagging when provided."""

    entities = clean_competition_entities(snapshots, my_asins=my_asins, scene_tags=scene_tags)
    pairs = build_competition_pairs(entities, scoring_rules=scoring_rules, rule_name=rule_name)

    target_weeks = {week}
    if previous_week:
        target_weeks.add(previous_week)

    pairs_subset = pairs.loc[pairs["week"].isin(target_weeks)].copy() if not pairs.empty else pairs
    entities_subset = entities.loc[entities["week"].isin(target_weeks)].copy() if not entities.empty else entities

    pairs_current = _filter_week(pairs_subset, week)
    pairs_previous = _filter_week(pairs_subset, previous_week) if previous_week else pairs_subset.iloc[0:0]

    deltas = build_competition_delta(
        entities_subset,
        pairs_current=pairs_current,
        pairs_previous=pairs_previous,
        week=week,
        previous_week=previous_week,
    )

    summary = summarise_competition_scene(
        pairs_current=pairs_current,
        delta_window=deltas,
        entities_current=_filter_week(entities_subset, week),
        entities_previous=_filter_week(entities_subset, previous_week) if previous_week else entities_subset.iloc[0:0],
    )

    return CompetitionTables(
        entities=entities_subset,
        pairs=pairs_subset,
        delta=deltas,
        summary=summary,
    )


def compute_competition_features(
    snapshots: pd.DataFrame | None = None,
    *,
    entities: pd.DataFrame | None = None,
    pairs: pd.DataFrame | None = None,
    deltas: pd.DataFrame | None = None,
    week: str,
    previous_week: str | None = None,
    my_asins: Iterable[str] | None = None,
    scene_tags: pd.DataFrame | None = None,
    scoring_rules: pd.DataFrame | dict[str, Any] | None = None,
    rule_name: str = "default",
) -> CompetitionFeatureResult:
    """Build structured competition facts for the LLM layer from tables or raw snapshots."""

    if snapshots is not None:
        if my_asins is None:
            raise ValueError("my_asins must be provided when snapshots are supplied")
        tables = build_competition_tables(
            snapshots,
            week=week,
            previous_week=previous_week,
            my_asins=my_asins,
            scene_tags=scene_tags,
            scoring_rules=scoring_rules,
            rule_name=rule_name,
        )
        entities = tables.entities
        pairs = tables.pairs
        deltas = tables.delta

    if entities is None or pairs is None or deltas is None:
        raise ValueError("entities, pairs, and deltas must be provided")

    pairs_current = _filter_week(pairs, week)
    metadata = _extract_metadata(pairs_current, week)

    if previous_week:
        metadata["previous_week"] = previous_week

    if pairs_current.empty:
        summary = {
            "my_asin_cnt": 0,
            "comp_cnt": 0,
            "pressure_p50": None,
            "pressure_p90": None,
            "worsen_ratio": None,
            "moves": {
                "moves_coupon_up": 0,
                "moves_price_down": 0,
                "moves_new_video": 0,
                "moves_badge_gain": 0,
            },
            "avg_scores": {
                "score_price": None,
                "score_rank": None,
                "score_cont": None,
                "score_soc": None,
                "score_badge": None,
            },
        }
        return CompetitionFeatureResult(
            metadata=metadata,
            pairs=[],
            summary=summary,
            insufficient_data=True,
        )

    entities_current = _filter_week(entities, week)
    entities_previous = _filter_week(entities, previous_week) if previous_week else entities.iloc[0:0]
    pairs_previous = _filter_week(pairs, previous_week) if previous_week else pairs.iloc[0:0]

    if previous_week:
        metadata["previous_sunday"] = _extract_first_date(pairs_previous, "sunday") or _extract_first_date(
            entities_previous, "sunday"
        )

    delta_window = _filter_delta(deltas, week, previous_week)

    pair_features: list[dict[str, Any]] = []
    for row in pairs_current.itertuples(index=False):
        prev_row = _match_previous_pair(pairs_previous, row)
        delta_row = _match_delta(delta_window, row)
        my_current = _lookup_entity(entities_current, row.my_asin)
        my_previous = _lookup_entity(entities_previous, row.my_asin)
        opp_current = _lookup_entity(entities_current, row.opp_asin)
        opp_previous = _lookup_entity(entities_previous, row.opp_asin)

        pair_features.append(
            _build_pair_feature(
                row=row,
                prev_row=prev_row,
                delta_row=delta_row,
                my_current=my_current,
                my_previous=my_previous,
                opp_current=opp_current,
                opp_previous=opp_previous,
            )
        )

    summary = _build_summary(
        pairs_current=pairs_current,
        delta_window=delta_window,
        entities_current=entities_current,
        entities_previous=entities_previous,
    )

    return CompetitionFeatureResult(
        metadata=metadata,
        pairs=pair_features,
        summary=summary,
        insufficient_data=False,
    )


def _normalise_series(
    df: pd.DataFrame, column: str, group_cols: list[str], *, invert: bool = False
) -> pd.Series:
    if column not in df:
        return pd.Series(0.0, index=df.index, dtype=float)
    series = pd.to_numeric(df[column], errors="coerce")
    if series.empty:
        return series.astype(float)
    grouped = df[group_cols + [column]].groupby(group_cols, dropna=False)[column]
    normalised = grouped.transform(
        lambda s: _normalise_array(s.to_numpy(dtype=float), invert=invert)
    )
    normalised = pd.to_numeric(normalised, errors="coerce").fillna(0.0)
    return normalised.astype(float)


def _normalise_array(values: np.ndarray, *, invert: bool) -> np.ndarray:
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    min_val = float(np.nanmin(finite))
    max_val = float(np.nanmax(finite))
    if math.isclose(max_val, min_val):
        return np.zeros_like(values, dtype=float)
    norm = (values - min_val) / (max_val - min_val)
    return 1 - norm if invert else norm


def _prepare_scoring_rules(
    scoring_rules: pd.DataFrame | dict[str, Any] | None,
    rule_name: str,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    if scoring_rules is None:
        return DEFAULT_SCORING_RULES.copy(), DEFAULT_BAND_CUTS.copy()

    if isinstance(scoring_rules, dict):
        feature_rules = DEFAULT_SCORING_RULES.copy()
        feature_rules.update(scoring_rules)
        return feature_rules, DEFAULT_BAND_CUTS.copy()

    subset = scoring_rules.loc[scoring_rules["rule_name"] == rule_name]
    if subset.empty:
        return DEFAULT_SCORING_RULES.copy(), DEFAULT_BAND_CUTS.copy()

    feature_rules = DEFAULT_SCORING_RULES.copy()
    band_cuts: dict[str, float] | None = None
    for _, row in subset.iterrows():
        feature = row.get("feature_name")
        if not feature:
            continue
        feature_rules[feature] = {
            "theta": float(row.get("theta", feature_rules.get(feature, {}).get("theta", 0.0))),
            "k": float(row.get("k", feature_rules.get(feature, {}).get("k", 1.0))),
            "weight": float(row.get("weight", feature_rules.get(feature, {}).get("weight", 0.0))),
        }
        cuts = row.get("band_cuts")
        if band_cuts is None and cuts:
            if isinstance(cuts, str):
                try:
                    cuts = json.loads(cuts)
                except json.JSONDecodeError:
                    cuts = None
            if isinstance(cuts, dict):
                band_cuts = {str(k): float(v) for k, v in cuts.items()}
    if band_cuts is None:
        band_cuts = DEFAULT_BAND_CUTS.copy()
    return feature_rules, band_cuts


def _compute_pair_row(
    my_row: Mapping[str, Any],
    opp_row: Mapping[str, Any],
    *,
    opp_type: str,
    median_price: float | None,
    comp_price_mean: float | None,
    comp_price_std: float | None,
    feature_rules: dict[str, dict[str, float]],
    band_cuts: dict[str, float],
) -> dict[str, Any]:
    my_badges = _normalise_badges(my_row.get("badge_json"))
    opp_badges = _normalise_badges(opp_row.get("badge_json"))
    price_gap = _diff_float(my_row.get("price_current"), opp_row.get("price_current"))
    price_index_med = _safe_ratio(my_row.get("price_current"), median_price)
    price_z = _z_score(my_row.get("price_current"), comp_price_mean, comp_price_std)
    rank_pos_pct = _rank_pos_pct(my_row, opp_row)
    content_gap = _diff_float(my_row.get("content_score"), opp_row.get("content_score"))
    social_gap = _diff_float(my_row.get("social_proof"), opp_row.get("social_proof"))
    badge_diff, badge_delta_sum = _badge_gap(my_badges, opp_badges)

    score_price = _score_feature(price_gap, feature_rules.get("price"))
    score_rank = _score_feature(rank_pos_pct, feature_rules.get("rank"))
    score_cont = _score_feature(-content_gap if content_gap is not None else None, feature_rules.get("content"))
    score_soc = _score_feature(-social_gap if social_gap is not None else None, feature_rules.get("social"))
    score_badge = _score_feature(badge_delta_sum, feature_rules.get("badge"))

    pressure = _weighted_sum(
        [score_price, score_rank, score_cont, score_soc, score_badge],
        [
            feature_rules.get("price", {}).get("weight", 0.0),
            feature_rules.get("rank", {}).get("weight", 0.0),
            feature_rules.get("content", {}).get("weight", 0.0),
            feature_rules.get("social", {}).get("weight", 0.0),
            feature_rules.get("badge", {}).get("weight", 0.0),
        ],
    )
    intensity_band = _assign_band(pressure, band_cuts)
    confidence = _confidence([price_gap, rank_pos_pct, content_gap, social_gap, badge_delta_sum])

    return {
        "scene_tag": my_row.get("scene_tag"),
        "base_scene": my_row.get("base_scene"),
        "morphology": my_row.get("morphology"),
        "marketplace_id": my_row.get("marketplace_id"),
        "week": my_row.get("week"),
        "sunday": my_row.get("sunday"),
        "my_parent_asin": my_row.get("parent_asin"),
        "my_asin": my_row.get("asin"),
        "opp_type": opp_type,
        "opp_parent_asin": opp_row.get("parent_asin"),
        "opp_asin": opp_row.get("asin"),
        "price_index_med": price_index_med,
        "price_gap_leader": price_gap,
        "price_z": price_z,
        "rank_pos_pct": rank_pos_pct,
        "content_gap": content_gap,
        "social_gap": social_gap,
        "badge_diff": badge_diff,
        "badge_delta_sum": badge_delta_sum,
        "score_price": score_price,
        "score_rank": score_rank,
        "score_cont": score_cont,
        "score_soc": score_soc,
        "score_badge": score_badge,
        "pressure": pressure,
        "intensity_band": intensity_band,
        "confidence": confidence,
    }


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    num = _clean_float(numerator)
    den = _clean_float(denominator)
    if num is None or den in (None, 0):
        return None
    return num / den


def _z_score(value: Any, mean: float | None, std: float | None) -> float | None:
    val = _clean_float(value)
    if val is None or mean is None or std in (None, 0):
        return None
    return (val - mean) / std


def _rank_pos_pct(my_row: Mapping[str, Any], opp_row: Mapping[str, Any]) -> float | None:
    my_rank = _clean_float(my_row.get("rank_root")) or _clean_float(my_row.get("rank_leaf"))
    opp_rank = _clean_float(opp_row.get("rank_root")) or _clean_float(opp_row.get("rank_leaf"))
    if my_rank is None or opp_rank is None:
        return None
    denom = max(my_rank, opp_rank)
    if denom == 0:
        return 0.0
    return (my_rank - opp_rank) / denom


def _badge_gap(my_badges: BadgeValue, opp_badges: BadgeValue) -> tuple[str, int | None]:
    my_set = set(my_badges)
    opp_set = set(opp_badges)
    diff = {
        "my_only": sorted(my_set - opp_set),
        "opp_only": sorted(opp_set - my_set),
    }
    badge_delta = len(diff["opp_only"]) - len(diff["my_only"])
    return json.dumps(diff, ensure_ascii=False), badge_delta


def _score_feature(value: float | None, config: dict[str, float] | None) -> float | None:
    if value is None or config is None:
        return None
    theta = config.get("theta", 0.0)
    k = config.get("k", 1.0)
    try:
        score = 1 / (1 + math.exp(-k * (value - theta)))
    except OverflowError:
        score = 1.0 if k * (value - theta) > 0 else 0.0
    return float(score)


def _weighted_sum(values: list[float | None], weights: list[float]) -> float:
    total = 0.0
    weight_sum = 0.0
    for value, weight in zip(values, weights):
        if value is None or weight == 0:
            continue
        total += value * weight
        weight_sum += weight
    if weight_sum == 0:
        return 0.0
    return total / weight_sum


def _assign_band(pressure: float, band_cuts: dict[str, float]) -> str | None:
    if pressure is None:
        return None
    sorted_cuts = sorted(band_cuts.items(), key=lambda item: item[1])
    for label, threshold in sorted_cuts:
        if pressure <= threshold:
            return label
    return sorted_cuts[-1][0] if sorted_cuts else None


def _confidence(values: list[float | int | None]) -> float:
    available = sum(value is not None for value in values)
    return available / len(values) if values else 0.0


def _pair_columns() -> list[str]:
    return [
        "scene_tag",
        "base_scene",
        "morphology",
        "marketplace_id",
        "week",
        "sunday",
        "my_parent_asin",
        "my_asin",
        "opp_type",
        "opp_parent_asin",
        "opp_asin",
        "price_index_med",
        "price_gap_leader",
        "price_z",
        "rank_pos_pct",
        "content_gap",
        "social_gap",
        "badge_diff",
        "badge_delta_sum",
        "score_price",
        "score_rank",
        "score_cont",
        "score_soc",
        "score_badge",
        "pressure",
        "intensity_band",
        "confidence",
    ]


def _delta_columns() -> list[str]:
    return [
        "scene_tag",
        "base_scene",
        "morphology",
        "marketplace_id",
        "window_id",
        "week_w0",
        "sunday_w0",
        "week_w1",
        "sunday_w1",
        "my_parent_asin",
        "my_asin",
        "opp_type",
        "d_price_net",
        "d_rank_score",
        "d_social_proof",
        "d_content_score",
        "badge_change",
        "d_price_gap_leader",
        "d_price_index_med",
        "d_rank_pos_pct",
        "d_content_gap",
        "d_social_gap",
        "delta_pressure",
    ]


def _summary_columns() -> list[str]:
    return [
        "scene_tag",
        "base_scene",
        "morphology",
        "marketplace_id",
        "week",
        "sunday",
        "my_asin_cnt",
        "comp_cnt",
        "pressure_p50",
        "pressure_p90",
        "worsen_ratio",
        "moves_coupon_up",
        "moves_price_down",
        "moves_new_video",
        "moves_badge_gain",
        "avg_score_price",
        "avg_score_rank",
        "avg_score_cont",
        "avg_score_soc",
        "avg_score_badge",
    ]


def _filter_week(frame: pd.DataFrame, week: str | None) -> pd.DataFrame:
    if week is None or frame.empty:
        return frame.iloc[0:0]
    return frame.loc[frame["week"] == week].copy()


def _filter_delta(deltas: pd.DataFrame, week: str, previous_week: str | None) -> pd.DataFrame:
    if deltas.empty:
        return deltas.iloc[0:0]
    mask = deltas["week_w0"] == week
    if previous_week is not None and "week_w1" in deltas.columns:
        mask &= deltas["week_w1"] == previous_week
    return deltas.loc[mask].copy()


def _extract_metadata(pairs_current: pd.DataFrame, week: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "week": week,
        "previous_week": None,
        "scene_tag": None,
        "base_scene": None,
        "morphology": None,
        "marketplace_id": None,
        "sunday": None,
        "previous_sunday": None,
    }
    if pairs_current.empty:
        return metadata
    for field in ("scene_tag", "base_scene", "morphology", "marketplace_id", "sunday"):
        if field in pairs_current.columns:
            values = pairs_current[field].dropna().unique()
            if values.size == 1:
                metadata[field] = values[0]
    return metadata


def _extract_first_date(frame: pd.DataFrame, column: str) -> date | pd.Timestamp | None:
    if frame.empty or column not in frame.columns:
        return None
    values = frame[column].dropna().unique()
    if values.size == 0:
        return None
    return values[0]


def _match_previous_pair(pairs_previous: pd.DataFrame, row: Any) -> dict[str, Any] | None:
    if pairs_previous.empty:
        return None
    mask = (pairs_previous["my_asin"] == row.my_asin) & (pairs_previous["opp_type"] == row.opp_type)
    if "opp_asin" in pairs_previous.columns:
        mask &= pairs_previous["opp_asin"] == row.opp_asin
    result = pairs_previous.loc[mask]
    if result.empty:
        return None
    prev = result.iloc[0].to_dict()
    if "sunday" in prev:
        metadata_previous = prev.get("sunday")
        if isinstance(metadata_previous, (pd.Timestamp, date)):
            prev["sunday"] = metadata_previous
    return prev


def _match_delta(delta_window: pd.DataFrame, row: Any) -> dict[str, Any] | None:
    if delta_window.empty:
        return None
    mask = (delta_window["my_asin"] == row.my_asin) & (delta_window["opp_type"] == row.opp_type)
    result = delta_window.loc[mask]
    if result.empty:
        return None
    return result.iloc[0].to_dict()


def _lookup_entity(entities: pd.DataFrame, asin: str | None) -> dict[str, Any] | None:
    if not asin or entities.empty:
        return None
    subset = entities.loc[entities["asin"] == asin]
    if subset.empty:
        return None
    record = subset.iloc[0].to_dict()
    badge = record.get("badge_json")
    record["badge_json"] = _normalise_badges(badge)
    return record


def _build_pair_feature(
    *,
    row: Any,
    prev_row: dict[str, Any] | None,
    delta_row: dict[str, Any] | None,
    my_current: dict[str, Any] | None,
    my_previous: dict[str, Any] | None,
    opp_current: dict[str, Any] | None,
    opp_previous: dict[str, Any] | None,
) -> dict[str, Any]:
    current_gap = _extract_gap(row._asdict())
    previous_gap = _extract_gap(prev_row) if prev_row else None
    delta_gap = _extract_delta_gap(delta_row, row._asdict(), prev_row)
    my_snapshot = _build_entity_snapshot(my_current)
    my_change = _build_my_change(delta_row, my_current, my_previous)
    opp_snapshot = _build_entity_snapshot(opp_current)
    opp_change = _build_entity_change(opp_current, opp_previous)
    score_components = _extract_scores(row._asdict())

    return {
        "my_asin": row.my_asin,
        "my_parent_asin": getattr(row, "my_parent_asin", None),
        "opp_type": row.opp_type,
        "opp_asin": getattr(row, "opp_asin", None),
        "opp_parent_asin": getattr(row, "opp_parent_asin", None),
        "pressure": _clean_float(getattr(row, "pressure", None)),
        "intensity_band": getattr(row, "intensity_band", None),
        "confidence": _clean_float(getattr(row, "confidence", None)),
        "score_components": score_components,
        "current_gap": current_gap,
        "previous_gap": previous_gap,
        "delta_gap": delta_gap,
        "my_snapshot": my_snapshot,
        "my_change": my_change,
        "opp_snapshot": opp_snapshot,
        "opp_change": opp_change,
        "badge_diff": getattr(row, "badge_diff", None),
        "badge_delta_sum": _clean_int(getattr(row, "badge_delta_sum", None)),
        "delta_pressure": _clean_float(delta_row.get("delta_pressure")) if delta_row else None,
    }


def _extract_gap(row: dict[str, Any] | None) -> dict[str, float | None] | None:
    if row is None:
        return None
    mapping = {
        "price_gap_leader": row.get("price_gap_leader"),
        "price_index_med": row.get("price_index_med"),
        "price_z": row.get("price_z"),
        "rank_pos_pct": row.get("rank_pos_pct"),
        "content_gap": row.get("content_gap"),
        "social_gap": row.get("social_gap"),
    }
    return {key: _clean_float(value) for key, value in mapping.items()}


def _extract_delta_gap(
    delta_row: dict[str, Any] | None,
    current_row: dict[str, Any],
    previous_row: dict[str, Any] | None,
) -> dict[str, float | None]:
    if delta_row:
        mapping = {
            "price_gap_leader": delta_row.get("d_price_gap_leader"),
            "price_index_med": delta_row.get("d_price_index_med"),
            "rank_pos_pct": delta_row.get("d_rank_pos_pct"),
            "content_gap": delta_row.get("d_content_gap"),
            "social_gap": delta_row.get("d_social_gap"),
        }
        return {key: _clean_float(value) for key, value in mapping.items()}

    if previous_row:
        delta_map: dict[str, float | None] = {}
        for field in ("price_gap_leader", "price_index_med", "rank_pos_pct", "content_gap", "social_gap"):
            delta_map[field] = _diff_float(current_row.get(field), previous_row.get(field))
        return delta_map
    return {key: None for key in ("price_gap_leader", "price_index_med", "rank_pos_pct", "content_gap", "social_gap")}


def _extract_scores(row: dict[str, Any]) -> dict[str, float | None]:
    mapping = {
        "score_price": row.get("score_price"),
        "score_rank": row.get("score_rank"),
        "score_cont": row.get("score_cont"),
        "score_soc": row.get("score_soc"),
        "score_badge": row.get("score_badge"),
    }
    return {key: _clean_float(value) for key, value in mapping.items()}


def _build_entity_snapshot(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    fields = (
        "price_current",
        "price_list",
        "coupon_pct",
        "discount_rate",
        "price_net",
        "rank_root",
        "rank_leaf",
        "rank_score",
        "social_proof",
        "content_score",
        "rating",
        "reviews",
        "image_cnt",
        "video_cnt",
        "bullet_cnt",
        "title_len",
        "aplus_flag",
    )
    snapshot: dict[str, Any] = {field: _clean_float(record.get(field)) for field in fields}
    int_fields = (
        "image_cnt",
        "video_cnt",
        "bullet_cnt",
        "title_len",
        "aplus_flag",
        "rank_root",
        "rank_leaf",
        "reviews",
    )
    for field in int_fields:
        snapshot[field] = _clean_int(record.get(field))
    snapshot["badge_json"] = record.get("badge_json", [])
    snapshot["asin"] = record.get("asin")
    snapshot["hyy_asin"] = record.get("hyy_asin")
    return snapshot


def _build_my_change(
    delta_row: dict[str, Any] | None,
    current: dict[str, Any] | None,
    previous: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if delta_row:
        result = {
            "price_net": _clean_float(delta_row.get("d_price_net")),
            "rank_score": _clean_float(delta_row.get("d_rank_score")),
            "social_proof": _clean_float(delta_row.get("d_social_proof")),
            "content_score": _clean_float(delta_row.get("d_content_score")),
            "badge_change": _clean_int(delta_row.get("badge_change")),
        }
        return result
    if current and previous:
        return {
            "price_net": _diff_float(current.get("price_net"), previous.get("price_net")),
            "rank_score": _diff_float(current.get("rank_score"), previous.get("rank_score")),
            "social_proof": _diff_float(current.get("social_proof"), previous.get("social_proof")),
            "content_score": _diff_float(current.get("content_score"), previous.get("content_score")),
            "badge_change": _badge_count(current.get("badge_json")) - _badge_count(previous.get("badge_json")),
        }
    return None


def _build_entity_change(current: dict[str, Any] | None, previous: dict[str, Any] | None) -> dict[str, Any] | None:
    if not current or not previous:
        return None
    return {
        "price_net": _diff_float(current.get("price_net"), previous.get("price_net")),
        "rank_score": _diff_float(current.get("rank_score"), previous.get("rank_score")),
        "social_proof": _diff_float(current.get("social_proof"), previous.get("social_proof")),
        "content_score": _diff_float(current.get("content_score"), previous.get("content_score")),
        "badge_change": _badge_count(current.get("badge_json")) - _badge_count(previous.get("badge_json")),
    }


def _build_summary(
    *,
    pairs_current: pd.DataFrame,
    delta_window: pd.DataFrame,
    entities_current: pd.DataFrame,
    entities_previous: pd.DataFrame,
) -> dict[str, Any]:
    my_asin_cnt = int(pairs_current["my_asin"].nunique()) if not pairs_current.empty else 0
    if pairs_current.empty:
        comp_cnt = 0
    elif "opp_asin" in pairs_current.columns:
        comp_cnt = int(pairs_current["opp_asin"].nunique())
    else:
        comp_cnt = int(pairs_current["opp_parent_asin"].nunique())

    pressure_series = pd.to_numeric(pairs_current.get("pressure"), errors="coerce") if not pairs_current.empty else pd.Series(dtype=float)
    pressure_p50 = _quantile(pressure_series, 0.5)
    pressure_p90 = _quantile(pressure_series, 0.9)

    delta_pressure_series = (
        pd.to_numeric(delta_window.get("delta_pressure"), errors="coerce")
        if not delta_window.empty
        else pd.Series(dtype=float)
    )
    worsen_ratio = _worsen_ratio(delta_pressure_series)

    moves = _detect_moves(entities_current, entities_previous)
    avg_scores = {
        field: _nanmean(pd.to_numeric(pairs_current.get(field), errors="coerce"))
        for field in ("score_price", "score_rank", "score_cont", "score_soc", "score_badge")
    }

    summary: dict[str, Any] = {
        "my_asin_cnt": my_asin_cnt,
        "comp_cnt": comp_cnt,
        "pressure_p50": pressure_p50,
        "pressure_p90": pressure_p90,
        "worsen_ratio": worsen_ratio,
        "moves": moves,
        "avg_scores": avg_scores,
    }
    return summary


def _detect_moves(current: pd.DataFrame, previous: pd.DataFrame) -> dict[str, int]:
    current_my = current.loc[current.get("hyy_asin") == 1].copy()
    previous_my = previous.loc[previous.get("hyy_asin") == 1].copy() if not previous.empty else previous
    if current_my.empty or previous_my.empty:
        return {
            "moves_coupon_up": 0,
            "moves_price_down": 0,
            "moves_new_video": 0,
            "moves_badge_gain": 0,
        }

    prev_cols = [
        "asin",
        "coupon_pct",
        "price_current",
        "video_cnt",
        "badge_json",
    ]
    previous_my = previous_my[prev_cols].copy()
    previous_my["badge_json"] = previous_my["badge_json"].apply(_normalise_badges)
    previous_my = previous_my.rename(columns={
        "coupon_pct": "coupon_pct_prev",
        "price_current": "price_current_prev",
        "video_cnt": "video_cnt_prev",
        "badge_json": "badge_json_prev",
    })

    merged = current_my.merge(previous_my, on="asin", how="left")
    coupon_prev = merged["coupon_pct_prev"].fillna(merged["coupon_pct"])
    price_prev = merged["price_current_prev"].fillna(merged["price_current"])
    video_prev = merged["video_cnt_prev"].fillna(merged["video_cnt"])
    badge_prev = merged["badge_json_prev"].apply(_badge_count)
    badge_curr = merged["badge_json"].apply(_normalise_badges).apply(_badge_count)

    moves_coupon_up = int(((merged["coupon_pct"] - coupon_prev) > 1e-6).sum())
    moves_price_down = int(((price_prev - merged["price_current"]) > 1e-6).sum())
    moves_new_video = int(((merged["video_cnt"] - video_prev) > 0).sum())
    moves_badge_gain = int(((badge_curr - badge_prev) > 0).sum())

    return {
        "moves_coupon_up": moves_coupon_up,
        "moves_price_down": moves_price_down,
        "moves_new_video": moves_new_video,
        "moves_badge_gain": moves_badge_gain,
    }


def _quantile(series: pd.Series, quantile: float) -> float | None:
    if series.empty:
        return None
    values = series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None
    return float(np.nanquantile(values, quantile))


def _worsen_ratio(series: pd.Series) -> float | None:
    if series.empty:
        return None
    values = series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None
    worsen = np.sum(values > 0)
    return float(worsen / values.size)


def _nanmean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    values = series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None
    return float(np.nanmean(values))


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if isinstance(value, (pd.Timestamp, date)):
        return None
    if isinstance(value, (np.generic,)):
        value = float(value)
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)
    return None


def _clean_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = int(float(value))
        except ValueError:
            return None
    if isinstance(value, (np.integer,)):
        value = int(value)
    if isinstance(value, (int,)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    return None


def _diff_float(current: Any, previous: Any) -> float | None:
    curr = _clean_float(current)
    prev = _clean_float(previous)
    if curr is None or prev is None:
        return None
    return curr - prev


def _normalise_badges(value: Any) -> BadgeValue:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return [str(key) for key in sorted(value.keys())]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return _normalise_badges(parsed)
    return []


def _badge_count(value: Any) -> int:
    badges = _normalise_badges(value)
    return len(badges)


__all__ = [
    "CompetitionFeatureResult",
    "CompetitionTables",
    "build_competition_tables",
    "build_competition_pairs",
    "build_competition_delta",
    "clean_competition_entities",
    "compute_competition_features",
    "summarise_competition_scene",
]
