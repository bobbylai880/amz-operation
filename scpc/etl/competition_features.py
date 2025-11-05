"""Feature engineering utilities for the competition module."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import log
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


BadgeValue = list[str]


@dataclass(slots=True)
class TrafficScoringContext:
    """Container describing how traffic gaps should be scored."""

    mix_rules: dict[str, dict[str, float]]
    keyword_rules: dict[str, dict[str, float]]
    band_cuts: dict[str, float]
    pressure_weights: dict[str, float]
    coverage_threshold: int = 5
    eps: float = 1e-6

_SCORING_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "competition_scoring.yaml"

_FALLBACK_SCORING_RULES: dict[str, dict[str, float]] = {
    "price": {"theta": 0.0, "k": 1.5, "weight": 0.30},
    "rank": {"theta": 0.0, "k": 6.0, "weight": 0.25},
    "content": {"theta": 0.0, "k": 5.0, "weight": 0.20},
    "social": {"theta": 0.0, "k": 4.0, "weight": 0.15},
    "badge": {"theta": 0.0, "k": 3.0, "weight": 0.10},
}

_FALLBACK_BAND_CUTS: dict[str, float] = {"C1": 0.25, "C2": 0.50, "C3": 0.75, "C4": 1.00}

_FALLBACK_TRAFFIC_RULES: dict[str, dict[str, dict[str, float]]] = {
    "mix": {
        "ad_ratio_index_med": {"theta": 1.0, "k": 8.0, "weight": 0.35, "invert": True},
        "ad_to_natural_gap": {"theta": -0.05, "k": 6.0, "weight": 0.25, "invert": True},
        "sp_share_in_ad_gap": {"theta": 0.0, "k": 10.0, "weight": 0.20, "invert": True},
        "sbv_share_in_ad_gap": {"theta": 0.0, "k": 8.0, "weight": 0.10, "invert": True},
        "sb_share_in_ad_gap": {"theta": 0.0, "k": 8.0, "weight": 0.10, "invert": True},
    },
    "keyword": {
        "kw_top3_share_gap": {"theta": 0.0, "k": 8.0, "weight": 0.30, "invert": True},
        "kw_brand_share_gap": {"theta": 0.0, "k": 8.0, "weight": 0.25, "invert": True},
        "kw_competitor_share_gap": {"theta": 0.02, "k": 6.0, "weight": 0.25, "invert": False},
        "kw_entropy_gap": {"theta": 0.0, "k": 5.0, "weight": 0.20, "invert": True},
    },
    "bands": {"traffic_intensity": {"C1": 0.25, "C2": 0.50, "C3": 0.75, "C4": 1.00}},
    "pressure_weights": {"mix": 0.5, "keyword": 0.5},
}


def _clone_feature_rules(source: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Return a deep-ish copy of the feature-rule mapping."""

    return {feature: params.copy() for feature, params in source.items()}


def _clone_traffic_section(source: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Clone a nested mapping of traffic feature parameters."""

    return {feature: params.copy() for feature, params in source.items()}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _load_scoring_config(path: Path = _SCORING_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML scoring configuration, returning an empty mapping on failure."""

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _extract_rule_from_config(rule_name: str, *, config: dict[str, Any] | None = None) -> tuple[
    dict[str, dict[str, float]], dict[str, float]
]:
    """Return feature weights and band cuts for the requested rule name."""

    feature_rules = _clone_feature_rules(_FALLBACK_SCORING_RULES)
    band_cuts = _FALLBACK_BAND_CUTS.copy()

    config = config if config is not None else _SCORING_CONFIG
    rule_sets = config.get("rule_sets") if isinstance(config, dict) else None
    if not isinstance(rule_sets, dict):
        return feature_rules, band_cuts

    selected = rule_sets.get(rule_name)
    if selected is None and rule_name != "default":
        selected = rule_sets.get("default")
    if not isinstance(selected, dict):
        return feature_rules, band_cuts

    features = selected.get("features")
    if isinstance(features, dict):
        for feature, params in features.items():
            if not isinstance(params, dict):
                continue
            base = feature_rules.get(feature, {"theta": 0.0, "k": 1.0, "weight": 0.0}).copy()
            for key in ("theta", "k", "weight"):
                if key not in params:
                    continue
                try:
                    base[key] = float(params[key])
                except (TypeError, ValueError):
                    continue
            feature_rules[feature] = base

    band_section = selected.get("band_cuts")
    candidate: dict[str, Any] | None = None
    if isinstance(band_section, dict):
        if isinstance(band_section.get("values"), dict):
            candidate = band_section["values"]
        else:
            candidate = band_section
    if isinstance(candidate, dict):
        parsed: dict[str, float] = {}
        for band, threshold in candidate.items():
            try:
                parsed[str(band)] = float(threshold)
            except (TypeError, ValueError):
                continue
        if parsed:
            band_cuts = parsed

    return feature_rules, band_cuts


def _extract_traffic_rule_from_config(
    rule_name: str,
    *,
    config: dict[str, Any] | None = None,
) -> TrafficScoringContext:
    """Return the traffic-scoring context for the requested rule name."""

    config = config if config is not None else _SCORING_CONFIG

    mix_rules = _clone_traffic_section(_FALLBACK_TRAFFIC_RULES["mix"])
    keyword_rules = _clone_traffic_section(_FALLBACK_TRAFFIC_RULES["keyword"])
    band_cuts = _FALLBACK_TRAFFIC_RULES["bands"]["traffic_intensity"].copy()
    pressure_weights = _FALLBACK_TRAFFIC_RULES["pressure_weights"].copy()

    rule_sets = config.get("rule_sets") if isinstance(config, dict) else None
    selected: Mapping[str, Any] | None = None
    if isinstance(rule_sets, Mapping):
        selected = rule_sets.get(rule_name)
        if selected is None and rule_name != "default_traffic":
            selected = rule_sets.get("default_traffic")

    if isinstance(selected, Mapping):
        mix_section = selected.get("mix")
        if isinstance(mix_section, Mapping):
            for feature, params in mix_section.items():
                if not isinstance(params, Mapping):
                    continue
                base = mix_rules.get(feature, {}).copy()
                for key in ("theta", "k", "weight"):
                    if key not in params:
                        continue
                    try:
                        base[key] = float(params[key])
                    except (TypeError, ValueError):
                        continue
                if "invert" in params:
                    base["invert"] = _coerce_bool(params["invert"])
                mix_rules[feature] = base

        keyword_section = selected.get("keyword")
        if isinstance(keyword_section, Mapping):
            for feature, params in keyword_section.items():
                if not isinstance(params, Mapping):
                    continue
                base = keyword_rules.get(feature, {}).copy()
                for key in ("theta", "k", "weight"):
                    if key not in params:
                        continue
                    try:
                        base[key] = float(params[key])
                    except (TypeError, ValueError):
                        continue
                if "invert" in params:
                    base["invert"] = _coerce_bool(params["invert"])
                keyword_rules[feature] = base

        bands_section = selected.get("bands")
        if isinstance(bands_section, Mapping):
            if isinstance(bands_section.get("traffic_intensity"), Mapping):
                candidate = bands_section["traffic_intensity"]
            else:
                candidate = bands_section
            parsed: dict[str, float] = {}
            if isinstance(candidate, Mapping):
                for band, threshold in candidate.items():
                    try:
                        parsed[str(band)] = float(threshold)
                    except (TypeError, ValueError):
                        continue
            if parsed:
                band_cuts = parsed

        weight_section = selected.get("pressure_weights")
        if isinstance(weight_section, Mapping):
            parsed_weights: dict[str, float] = {}
            for key, value in weight_section.items():
                try:
                    parsed_weights[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            if parsed_weights:
                pressure_weights.update(parsed_weights)

    return TrafficScoringContext(
        mix_rules=mix_rules,
        keyword_rules=keyword_rules,
        band_cuts=band_cuts,
        pressure_weights=pressure_weights,
    )


def _prepare_traffic_scoring(
    traffic_scoring: Mapping[str, Any] | None,
    rule_name: str,
) -> TrafficScoringContext:
    if traffic_scoring is None:
        return _extract_traffic_rule_from_config(rule_name)

    mix_rules = _clone_traffic_section(_FALLBACK_TRAFFIC_RULES["mix"])
    keyword_rules = _clone_traffic_section(_FALLBACK_TRAFFIC_RULES["keyword"])
    band_cuts = _FALLBACK_TRAFFIC_RULES["bands"]["traffic_intensity"].copy()
    pressure_weights = _FALLBACK_TRAFFIC_RULES["pressure_weights"].copy()
    coverage_threshold = 5
    eps = 1e-6

    if "mix" in traffic_scoring and isinstance(traffic_scoring["mix"], Mapping):
        for feature, params in traffic_scoring["mix"].items():
            if not isinstance(params, Mapping):
                continue
            base = mix_rules.get(feature, {}).copy()
            for key in ("theta", "k", "weight"):
                if key not in params:
                    continue
                try:
                    base[key] = float(params[key])
                except (TypeError, ValueError):
                    continue
            if "invert" in params:
                base["invert"] = _coerce_bool(params["invert"])
            mix_rules[feature] = base

    if "keyword" in traffic_scoring and isinstance(traffic_scoring["keyword"], Mapping):
        for feature, params in traffic_scoring["keyword"].items():
            if not isinstance(params, Mapping):
                continue
            base = keyword_rules.get(feature, {}).copy()
            for key in ("theta", "k", "weight"):
                if key not in params:
                    continue
                try:
                    base[key] = float(params[key])
                except (TypeError, ValueError):
                    continue
            if "invert" in params:
                base["invert"] = _coerce_bool(params["invert"])
            keyword_rules[feature] = base

    if "bands" in traffic_scoring and isinstance(traffic_scoring["bands"], Mapping):
        candidate = traffic_scoring["bands"].get("traffic_intensity")
        candidate = candidate if isinstance(candidate, Mapping) else traffic_scoring["bands"]
        parsed: dict[str, float] = {}
        for band, threshold in candidate.items():
            try:
                parsed[str(band)] = float(threshold)
            except (TypeError, ValueError):
                continue
        if parsed:
            band_cuts = parsed

    if "pressure_weights" in traffic_scoring and isinstance(traffic_scoring["pressure_weights"], Mapping):
        for key, value in traffic_scoring["pressure_weights"].items():
            try:
                pressure_weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    if "coverage_threshold" in traffic_scoring:
        try:
            coverage_threshold = int(traffic_scoring["coverage_threshold"])
        except (TypeError, ValueError):
            coverage_threshold = 5

    if "eps" in traffic_scoring:
        try:
            eps = float(traffic_scoring["eps"])
        except (TypeError, ValueError):
            eps = 1e-6

    return TrafficScoringContext(
        mix_rules=mix_rules,
        keyword_rules=keyword_rules,
        band_cuts=band_cuts,
        pressure_weights=pressure_weights,
        coverage_threshold=max(1, coverage_threshold),
        eps=eps if eps > 0 else 1e-6,
    )


_SCORING_CONFIG = _load_scoring_config()

DEFAULT_SCORING_RULES, DEFAULT_BAND_CUTS = _extract_rule_from_config(
    "default", config=_SCORING_CONFIG
)


@dataclass(slots=True)
class CompetitionFeatureResult:
    """Container holding competition pair features, evidence, and aggregates."""

    metadata: dict[str, Any]
    pairs: list[dict[str, Any]]
    summary: dict[str, Any]
    insufficient_data: bool
    top_opponents: list[dict[str, Any]] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialise the feature bundle into JSON-friendly primitives."""

        payload = {
            "metadata": self.metadata.copy(),
            "pairs": self.pairs,
            "summary": self.summary,
            "insufficient_data": self.insufficient_data,
            "top_opponents": self.top_opponents or [],
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
    pairs_each: pd.DataFrame
    delta: pd.DataFrame
    summary: pd.DataFrame


def _to_normalised_timestamp(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize(None) if ts.tzinfo else ts
    return ts.normalize()


def _to_sunday(ts: pd.Timestamp | None) -> pd.Timestamp | None:
    if ts is None or pd.isna(ts):
        return None
    offset = (6 - ts.weekday()) % 7
    return (ts + pd.Timedelta(days=int(offset))).normalize()


def _iso_week_label(ts: pd.Timestamp | None) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    iso = ts.isocalendar()
    return f"{iso.year}W{iso.week:02d}"


def _standardise_flow_weekly(flow: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "asin",
        "marketplace_id",
        "sunday",
        "week",
        "ad_ratio",
        "nf_ratio",
        "recommend_ratio",
        "sp_ratio",
        "sbv_ratio",
        "sb_ratio",
    ]
    if flow is None or flow.empty:
        return pd.DataFrame(columns=columns)

    df = flow.copy()
    monday = df.get("monday") if "monday" in df.columns else df.get("week_start")
    if monday is not None:
        monday_ts = pd.to_datetime(monday, errors="coerce")
        df["sunday"] = monday_ts + pd.Timedelta(days=6)
    else:
        existing_sunday = df.get("sunday")
        df["sunday"] = pd.to_datetime(existing_sunday, errors="coerce")

    df["sunday"] = df["sunday"].map(lambda value: _to_sunday(_to_normalised_timestamp(value)))
    if "week" not in df.columns:
        df["week"] = df["sunday"].map(_iso_week_label)

    df = df.rename(columns={
        "country": "marketplace_id",
    })

    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA

    ratio_cols = ["ad_ratio", "nf_ratio", "recommend_ratio", "sp_ratio", "sbv_ratio", "sb_ratio"]
    for col in ratio_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.loc[:, columns]


def _standardise_keyword_daily(keywords: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "asin",
        "marketplace_id",
        "snapshot_date",
        "sunday",
        "week",
        "keyword",
        "ratio_score",
    ]
    if keywords is None or keywords.empty:
        return pd.DataFrame(columns=columns)

    df = keywords.copy()
    df = df.rename(columns={"country": "marketplace_id"})
    snapshot = df.get("snapshot_date") if "snapshot_date" in df.columns else df.get("date")
    df["snapshot_date"] = [
        _to_normalised_timestamp(value) for value in snapshot
    ]
    df["sunday"] = [
        _to_sunday(ts) for ts in df["snapshot_date"]
    ]
    if "week" not in df.columns:
        df["week"] = [
            _iso_week_label(ts) for ts in df["sunday"]
        ]

    df["ratio_score"] = pd.to_numeric(df.get("ratio_score"), errors="coerce")
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA

    return df.loc[:, columns]


def _aggregate_keyword_weekly(
    keywords: pd.DataFrame,
    keyword_tags: pd.DataFrame | Mapping[str, str] | None,
    *,
    coverage_threshold: int,
    eps: float,
) -> pd.DataFrame:
    columns = [
        "asin",
        "marketplace_id",
        "week",
        "sunday",
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
    if keywords.empty:
        return pd.DataFrame(columns=columns)

    if keyword_tags is None:
        tag_lookup: dict[str, str] = {}
    elif isinstance(keyword_tags, Mapping):
        tag_lookup = {str(k): str(v).lower() for k, v in keyword_tags.items()}
    else:
        tag_lookup = {
            str(row["keyword"]): str(row.get("tag") or row.get("tag_type") or row.get("label")).lower()
            for _, row in keyword_tags.iterrows()
            if row.get("keyword") is not None
        }

    group_cols = ["asin", "marketplace_id", "week", "sunday", "snapshot_date"]
    records: list[dict[str, Any]] = []

    for keys, group in keywords.groupby(group_cols, dropna=False):
        asin, marketplace, week, sunday, snapshot_date = keys
        ratios = pd.to_numeric(group["ratio_score"], errors="coerce").fillna(0.0)
        total = float(ratios.sum())
        if not math.isfinite(total) or total <= eps:
            record = {
                "asin": asin,
                "marketplace_id": marketplace,
                "week": week,
                "sunday": sunday,
                "snapshot_date": snapshot_date,
                "entropy": np.nan,
                "hhi": np.nan,
                "top1_share": np.nan,
                "top3_share": np.nan,
                "top10_share": np.nan,
                "brand_share": np.nan,
                "competitor_share": np.nan,
                "generic_share": np.nan,
                "attribute_share": np.nan,
                "valid": 0,
            }
            records.append(record)
            continue

        shares = ratios / total
        shares = shares.clip(lower=0.0)
        shares_sum = float(shares.sum())
        if not math.isclose(shares_sum, 1.0, rel_tol=1e-6):
            shares = shares / shares_sum if shares_sum > 0 else shares

        entropy_terms = shares[shares > 0]
        entropy = float(-np.sum(entropy_terms * np.log(entropy_terms))) if not entropy_terms.empty else 0.0
        hhi = float(np.sum(np.square(shares)))
        sorted_shares = np.sort(shares.to_numpy())[::-1]
        top1 = float(sorted_shares[:1].sum())
        top3 = float(sorted_shares[:3].sum())
        top10 = float(sorted_shares[:10].sum())

        tags = group.get("keyword")
        brand = competitor = generic = attribute = 0.0
        for share, keyword in zip(shares, tags):
            tag = tag_lookup.get(str(keyword), "")
            if tag == "brand":
                brand += float(share)
            elif tag == "competitor":
                competitor += float(share)
            elif tag == "generic":
                generic += float(share)
            elif tag == "attribute":
                attribute += float(share)

        record = {
            "asin": asin,
            "marketplace_id": marketplace,
            "week": week,
            "sunday": sunday,
            "snapshot_date": snapshot_date,
            "entropy": entropy,
            "hhi": hhi,
            "top1_share": top1,
            "top3_share": top3,
            "top10_share": top10,
            "brand_share": brand,
            "competitor_share": competitor,
            "generic_share": generic,
            "attribute_share": attribute,
            "valid": 1,
        }
        records.append(record)

    daily = pd.DataFrame.from_records(records)
    if daily.empty:
        return pd.DataFrame(columns=columns)

    agg = daily.groupby(["asin", "marketplace_id", "week", "sunday"], dropna=False).agg(
        {
            "entropy": "mean",
            "hhi": "mean",
            "top1_share": "mean",
            "top3_share": "mean",
            "top10_share": "mean",
            "brand_share": "mean",
            "competitor_share": "mean",
            "generic_share": "mean",
            "attribute_share": "mean",
            "valid": "sum",
        }
    )

    agg = agg.reset_index()
    agg["kw_days_covered"] = agg["valid"].astype(int)
    agg["kw_coverage_ratio"] = (
        agg["kw_days_covered"].clip(lower=0).astype(float) / 7.0
    ).clip(lower=0.0, upper=1.0)
    agg = agg.drop(columns=["valid"])

    agg = agg.rename(
        columns={
            "entropy": "kw_entropy_7d_avg",
            "hhi": "kw_hhi_7d_avg",
            "top1_share": "kw_top1_share_7d_avg",
            "top3_share": "kw_top3_share_7d_avg",
            "top10_share": "kw_top10_share_7d_avg",
            "brand_share": "kw_brand_share_7d_avg",
            "competitor_share": "kw_competitor_share_7d_avg",
            "generic_share": "kw_generic_share_7d_avg",
            "attribute_share": "kw_attribute_share_7d_avg",
        }
    )

    for col in columns:
        if col not in agg.columns:
            agg[col] = pd.NA

    agg.loc[:, "kw_days_covered"] = agg["kw_days_covered"].astype(int)

    insufficient_mask = agg["kw_days_covered"] < max(1, coverage_threshold)
    agg.loc[insufficient_mask, [
        "kw_entropy_7d_avg",
        "kw_hhi_7d_avg",
        "kw_top1_share_7d_avg",
        "kw_top3_share_7d_avg",
        "kw_top10_share_7d_avg",
        "kw_brand_share_7d_avg",
        "kw_competitor_share_7d_avg",
        "kw_generic_share_7d_avg",
        "kw_attribute_share_7d_avg",
    ]] = np.nan

    return agg.loc[:, columns]


def build_traffic_features(
    flow_weekly: pd.DataFrame | None,
    keyword_daily: pd.DataFrame | None,
    keyword_tags: pd.DataFrame | Mapping[str, str] | None = None,
    *,
    coverage_threshold: int = 5,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Combine flow and keyword inputs into weekly traffic features per ASIN."""

    flow = _standardise_flow_weekly(flow_weekly)
    keywords = _standardise_keyword_daily(keyword_daily)
    keyword_weekly = _aggregate_keyword_weekly(
        keywords,
        keyword_tags,
        coverage_threshold=coverage_threshold,
        eps=eps,
    )

    if flow.empty and keyword_weekly.empty:
        columns = [
            "asin",
            "marketplace_id",
            "sunday",
            "week",
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
        return pd.DataFrame(columns=columns)

    merged = flow.merge(
        keyword_weekly,
        on=["asin", "marketplace_id", "week", "sunday"],
        how="outer",
    )

    ratio_cols = ["ad_ratio", "nf_ratio", "recommend_ratio", "sp_ratio", "sbv_ratio", "sb_ratio"]
    for col in ratio_cols:
        merged[col] = pd.to_numeric(merged.get(col), errors="coerce")

    ad_ratio = merged["ad_ratio"].astype(float)
    nf_ratio = merged["nf_ratio"].astype(float)
    sp_ratio = merged["sp_ratio"].astype(float)
    sbv_ratio = merged["sbv_ratio"].astype(float)
    sb_ratio = merged["sb_ratio"].astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        merged["sp_share_in_ad"] = np.where(ad_ratio > eps, sp_ratio / ad_ratio, np.nan)
        merged["sbv_share_in_ad"] = np.where(ad_ratio > eps, sbv_ratio / ad_ratio, np.nan)
        merged["sb_share_in_ad"] = np.where(ad_ratio > eps, sb_ratio / ad_ratio, np.nan)
        merged["ad_to_natural"] = np.where(nf_ratio > eps, ad_ratio / nf_ratio, np.nan)

    merged["kw_days_covered"] = merged.get("kw_days_covered", 0).fillna(0).astype(int)
    merged["kw_coverage_ratio"] = (
        merged.get("kw_coverage_ratio", 0.0)
        .astype(float)
        .clip(lower=0.0, upper=1.0)
    )

    columns = [
        "asin",
        "marketplace_id",
        "sunday",
        "week",
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

    for col in columns:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged["sunday"] = merged["sunday"].map(lambda value: _to_sunday(_to_normalised_timestamp(value)))
    merged["week"] = merged.apply(
        lambda row: row["week"] if pd.notna(row.get("week")) else _iso_week_label(row.get("sunday")),
        axis=1,
    )

    merged = merged.drop_duplicates(subset=["asin", "marketplace_id", "week"], keep="last")

    return merged.loc[:, columns]


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
    traffic: pd.DataFrame | None = None,
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

    if snapshots.empty:
        return pd.DataFrame(columns=ordered_cols)

    df = snapshots.copy()
    if scene_tags is not None:
        df = _attach_scene_tags(df, scene_tags)
        if df.empty:
            return pd.DataFrame(columns=ordered_cols)
    my_set = set(my_asins)
    df["hyy_asin"] = df["asin"].isin(my_set).astype(int)

    if traffic is not None and not traffic.empty:
        traffic_subset = traffic.copy()
        traffic_subset = traffic_subset.rename(columns={"marketplace": "marketplace_id"})
        traffic_subset = traffic_subset.drop_duplicates(
            subset=["asin", "marketplace_id", "week"], keep="last"
        )
        df = df.merge(
            traffic_subset,
            on=["asin", "marketplace_id", "week"],
            how="left",
            suffixes=("", "_traffic"),
        )
        if "sunday_traffic" in df.columns:
            df["sunday"] = df["sunday"].where(df["sunday"].notna(), df["sunday_traffic"])
            df = df.drop(columns=["sunday_traffic"])
        for column in (
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
        ):
            traffic_col = f"{column}_traffic"
            if traffic_col in df.columns:
                df[column] = df[column].where(df[column].notna(), df[traffic_col])
                df = df.drop(columns=[traffic_col])
    else:
        for column in (
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
        ):
            if column not in df.columns:
                df[column] = np.nan

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
    ]
    for col in numeric_cols:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["kw_days_covered"] = pd.to_numeric(df.get("kw_days_covered"), errors="coerce").fillna(0).astype(int)
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
        "kw_top1_share_7d_avg",
        "kw_top3_share_7d_avg",
        "kw_top10_share_7d_avg",
        "kw_brand_share_7d_avg",
        "kw_competitor_share_7d_avg",
        "kw_generic_share_7d_avg",
        "kw_attribute_share_7d_avg",
        "kw_coverage_ratio",
    ):
        if share_col in df:
            df[share_col] = df[share_col].astype(float).clip(lower=0.0, upper=1.0)

    return df.reindex(columns=ordered_cols)


def build_competition_pairs(
    entities: pd.DataFrame,
    *,
    scoring_rules: pd.DataFrame | dict[str, Any] | None = None,
    rule_name: str = "default",
    traffic_scoring: Mapping[str, Any] | None = None,
    traffic_rule_name: str = "default_traffic",
) -> pd.DataFrame:
    """Generate pairwise comparison rows for leader and median competitors."""

    if entities.empty:
        return pd.DataFrame(columns=_pair_columns())

    feature_rules, band_cuts = _prepare_scoring_rules(scoring_rules, rule_name)
    traffic_context = _prepare_traffic_scoring(traffic_scoring, traffic_rule_name)

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

        traffic_median = _traffic_median_metrics(competitors)

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
                    traffic_context=traffic_context,
                    traffic_median=traffic_median,
                )
                records.append(record)

    if not records:
        return pd.DataFrame(columns=_pair_columns())
    return pd.DataFrame.from_records(records, columns=_pair_columns())


def build_competition_pairs_each(
    entities: pd.DataFrame,
    *,
    scoring_rules: pd.DataFrame | dict[str, Any] | None = None,
    rule_name: str = "default",
    traffic_scoring: Mapping[str, Any] | None = None,
    traffic_rule_name: str = "default_traffic",
) -> pd.DataFrame:
    """Generate one-to-one competition records for every opposing ASIN."""

    if entities.empty:
        return pd.DataFrame(columns=_pair_each_columns())

    feature_rules, band_cuts = _prepare_scoring_rules(scoring_rules, rule_name)
    traffic_context = _prepare_traffic_scoring(traffic_scoring, traffic_rule_name)

    group_cols = ["scene_tag", "base_scene", "morphology", "marketplace_id", "week", "sunday"]
    records: list[dict[str, Any]] = []

    for _, group in entities.groupby(group_cols, dropna=False):
        mine = group.loc[group["hyy_asin"] == 1]
        competitors = group.loc[group["hyy_asin"] == 0]
        if mine.empty or competitors.empty:
            continue

        traffic_median = _traffic_median_metrics(competitors)

        for _, my_row in mine.iterrows():
            for _, opp_row in competitors.iterrows():
                record = _compute_pair_each_row(
                    my_row,
                    opp_row,
                    feature_rules=feature_rules,
                    band_cuts=band_cuts,
                    traffic_context=traffic_context,
                    traffic_median=traffic_median,
                )
                records.append(record)

    if not records:
        return pd.DataFrame(columns=_pair_each_columns())
    return pd.DataFrame.from_records(records, columns=_pair_each_columns())


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
            "d_ad_ratio_gap": _diff_float(
                current.get("ad_ratio_gap"),
                previous.get("ad_ratio_gap") if previous is not None else None,
            ),
            "d_ad_ratio_index_med": _diff_float(
                current.get("ad_ratio_index_med"),
                previous.get("ad_ratio_index_med") if previous is not None else None,
            ),
            "d_ad_to_natural_gap": _diff_float(
                current.get("ad_to_natural_gap"),
                previous.get("ad_to_natural_gap") if previous is not None else None,
            ),
            "d_sp_share_in_ad_gap": _diff_float(
                current.get("sp_share_in_ad_gap"),
                previous.get("sp_share_in_ad_gap") if previous is not None else None,
            ),
            "d_sbv_share_in_ad_gap": _diff_float(
                current.get("sbv_share_in_ad_gap"),
                previous.get("sbv_share_in_ad_gap") if previous is not None else None,
            ),
            "d_sb_share_in_ad_gap": _diff_float(
                current.get("sb_share_in_ad_gap"),
                previous.get("sb_share_in_ad_gap") if previous is not None else None,
            ),
            "d_kw_entropy_gap": _diff_float(
                current.get("kw_entropy_gap"),
                previous.get("kw_entropy_gap") if previous is not None else None,
            ),
            "d_kw_hhi_gap": _diff_float(
                current.get("kw_hhi_gap"),
                previous.get("kw_hhi_gap") if previous is not None else None,
            ),
            "d_kw_top1_share_gap": _diff_float(
                current.get("kw_top1_share_gap"),
                previous.get("kw_top1_share_gap") if previous is not None else None,
            ),
            "d_kw_top3_share_gap": _diff_float(
                current.get("kw_top3_share_gap"),
                previous.get("kw_top3_share_gap") if previous is not None else None,
            ),
            "d_kw_top10_share_gap": _diff_float(
                current.get("kw_top10_share_gap"),
                previous.get("kw_top10_share_gap") if previous is not None else None,
            ),
            "d_kw_brand_share_gap": _diff_float(
                current.get("kw_brand_share_gap"),
                previous.get("kw_brand_share_gap") if previous is not None else None,
            ),
            "d_kw_competitor_share_gap": _diff_float(
                current.get("kw_competitor_share_gap"),
                previous.get("kw_competitor_share_gap") if previous is not None else None,
            ),
            "d_kw_generic_share_gap": _diff_float(
                current.get("kw_generic_share_gap"),
                previous.get("kw_generic_share_gap") if previous is not None else None,
            ),
            "d_kw_attribute_share_gap": _diff_float(
                current.get("kw_attribute_share_gap"),
                previous.get("kw_attribute_share_gap") if previous is not None else None,
            ),
            "d_t_score_mix": _diff_float(
                current.get("t_score_mix"),
                previous.get("t_score_mix") if previous is not None else None,
            ),
            "d_t_score_kw": _diff_float(
                current.get("t_score_kw"),
                previous.get("t_score_kw") if previous is not None else None,
            ),
            "d_t_pressure": _diff_float(
                current.get("t_pressure"),
                previous.get("t_pressure") if previous is not None else None,
            ),
            "d_t_confidence": _diff_float(
                current.get("t_confidence"),
                previous.get("t_confidence") if previous is not None else None,
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
        "traffic": summary_stats.get("traffic"),
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
    traffic: pd.DataFrame | None = None,
    traffic_scoring: Mapping[str, Any] | None = None,
    traffic_rule_name: str = "default_traffic",
) -> CompetitionTables:
    """Produce Doris-aligned tables from snapshots, including scene tagging when provided."""

    entities = clean_competition_entities(
        snapshots,
        my_asins=my_asins,
        scene_tags=scene_tags,
        traffic=traffic,
    )
    pairs = build_competition_pairs(
        entities,
        scoring_rules=scoring_rules,
        rule_name=rule_name,
        traffic_scoring=traffic_scoring,
        traffic_rule_name=traffic_rule_name,
    )
    pairs_each = build_competition_pairs_each(
        entities,
        scoring_rules=scoring_rules,
        rule_name=rule_name,
        traffic_scoring=traffic_scoring,
        traffic_rule_name=traffic_rule_name,
    )

    target_weeks = {week}
    if previous_week:
        target_weeks.add(previous_week)

    pairs_subset = pairs.loc[pairs["week"].isin(target_weeks)].copy() if not pairs.empty else pairs
    pairs_each_subset = (
        pairs_each.loc[pairs_each["week"].isin(target_weeks)].copy() if not pairs_each.empty else pairs_each
    )
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
        pairs_each=pairs_each_subset,
        delta=deltas,
        summary=summary,
    )


def compute_competition_features(
    snapshots: pd.DataFrame | None = None,
    *,
    entities: pd.DataFrame | None = None,
    pairs: pd.DataFrame | None = None,
    pairs_each: pd.DataFrame | None = None,
    deltas: pd.DataFrame | None = None,
    week: str,
    previous_week: str | None = None,
    my_asins: Iterable[str] | None = None,
    scene_tags: pd.DataFrame | None = None,
    scoring_rules: pd.DataFrame | dict[str, Any] | None = None,
    rule_name: str = "default",
    traffic: pd.DataFrame | None = None,
    traffic_scoring: Mapping[str, Any] | None = None,
    traffic_rule_name: str = "default_traffic",
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
            traffic=traffic,
            traffic_scoring=traffic_scoring,
            traffic_rule_name=traffic_rule_name,
        )
        entities = tables.entities
        pairs = tables.pairs
        pairs_each = tables.pairs_each
        deltas = tables.delta

    if entities is None or pairs is None or deltas is None or pairs_each is None:
        raise ValueError("entities, pairs, pairs_each, and deltas must be provided")

    pairs_current = _filter_week(pairs, week)
    pairs_each_current = _filter_week(pairs_each, week)
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
            "traffic": {
                "pressure_p50": None,
                "pressure_p90": None,
                "confidence_p50": None,
                "coverage_p50": None,
                "lagging_pairs": 0,
                "avg_mix_gap": None,
                "avg_keyword_gap": None,
            },
        }
        return CompetitionFeatureResult(
            metadata=metadata,
            pairs=[],
            summary=summary,
            insufficient_data=True,
            top_opponents=[],
        )

    entities_current = _filter_week(entities, week)
    entities_previous = _filter_week(entities, previous_week) if previous_week else entities.iloc[0:0]
    pairs_previous = _filter_week(pairs, previous_week) if previous_week else pairs.iloc[0:0]
    pairs_each_previous = _filter_week(pairs_each, previous_week) if previous_week else pairs_each.iloc[0:0]

    if previous_week:
        metadata["previous_sunday"] = _extract_first_date(pairs_previous, "sunday") or _extract_first_date(
            entities_previous, "sunday"
        )

    delta_window = _filter_delta(deltas, week, previous_week)

    pair_features: list[dict[str, Any]] = []
    pair_each_lookup = {
        (row["my_asin"], row["opp_asin"]): row
        for _, row in pairs_each_current.iterrows()
    }
    for row in pairs_current.itertuples(index=False):
        prev_row = _match_previous_pair(pairs_previous, row)
        delta_row = _match_delta(delta_window, row)
        my_current = _lookup_entity(entities_current, row.my_asin)
        my_previous = _lookup_entity(entities_previous, row.my_asin)
        opp_current = _lookup_entity(entities_current, row.opp_asin)
        opp_previous = _lookup_entity(entities_previous, row.opp_asin)
        pair_each_row = pair_each_lookup.get((row.my_asin, getattr(row, "opp_asin", None)))

        pair_features.append(
            _build_pair_feature(
                row=row,
                prev_row=prev_row,
                delta_row=delta_row,
                my_current=my_current,
                my_previous=my_previous,
                opp_current=opp_current,
                opp_previous=opp_previous,
                pair_each_row=pair_each_row,
            )
        )

    summary = _build_summary(
        pairs_current=pairs_current,
        delta_window=delta_window,
        entities_current=entities_current,
        entities_previous=entities_previous,
    )

    top_opponents = _build_top_opponents(
        pairs_each_current=pairs_each_current,
        pairs_each_previous=pairs_each_previous,
    )

    return CompetitionFeatureResult(
        metadata=metadata,
        pairs=pair_features,
        summary=summary,
        insufficient_data=False,
        top_opponents=top_opponents,
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
        return _extract_rule_from_config(rule_name)

    if isinstance(scoring_rules, dict):
        feature_rules = _clone_feature_rules(DEFAULT_SCORING_RULES)
        feature_rules.update(scoring_rules)
        return feature_rules, DEFAULT_BAND_CUTS.copy()

    subset = scoring_rules.loc[scoring_rules["rule_name"] == rule_name]
    if subset.empty:
        return _extract_rule_from_config(rule_name)

    feature_rules = _clone_feature_rules(DEFAULT_SCORING_RULES)
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


def _traffic_median_metrics(competitors: pd.DataFrame) -> dict[str, float | None]:
    fields = [
        "ad_ratio",
        "ad_to_natural",
        "sp_share_in_ad",
        "sbv_share_in_ad",
        "sb_share_in_ad",
        "kw_entropy_7d_avg",
        "kw_hhi_7d_avg",
        "kw_top1_share_7d_avg",
        "kw_top3_share_7d_avg",
        "kw_top10_share_7d_avg",
        "kw_brand_share_7d_avg",
        "kw_competitor_share_7d_avg",
        "kw_generic_share_7d_avg",
        "kw_attribute_share_7d_avg",
    ]
    medians: dict[str, float | None] = {}
    for field in fields:
        if field not in competitors.columns:
            medians[field] = None
            continue
        series = pd.to_numeric(competitors[field], errors="coerce")
        series = series.dropna()
        medians[field] = float(series.median()) if not series.empty else None
    return medians


def _compute_traffic_metrics(
    my_row: Mapping[str, Any],
    opp_row: Mapping[str, Any],
    *,
    traffic_median: Mapping[str, float | None],
    context: TrafficScoringContext,
) -> dict[str, Any]:
    my_ad_ratio = _clean_float(my_row.get("ad_ratio"))
    opp_ad_ratio = _clean_float(opp_row.get("ad_ratio"))
    median_ad_ratio = traffic_median.get("ad_ratio")
    ad_ratio_gap = _diff_float(my_ad_ratio, opp_ad_ratio)
    ad_ratio_index = _safe_ratio(my_ad_ratio, median_ad_ratio)

    my_ad_to_natural = _clean_float(my_row.get("ad_to_natural"))
    opp_ad_to_natural = _clean_float(opp_row.get("ad_to_natural"))
    median_ad_to_natural = traffic_median.get("ad_to_natural")
    ad_to_natural_gap = _diff_float(my_ad_to_natural, opp_ad_to_natural)

    my_sp_share = _clean_float(my_row.get("sp_share_in_ad"))
    opp_sp_share = _clean_float(opp_row.get("sp_share_in_ad"))
    sp_share_gap = _diff_float(my_sp_share, opp_sp_share)

    my_sbv_share = _clean_float(my_row.get("sbv_share_in_ad"))
    opp_sbv_share = _clean_float(opp_row.get("sbv_share_in_ad"))
    sbv_share_gap = _diff_float(my_sbv_share, opp_sbv_share)

    my_sb_share = _clean_float(my_row.get("sb_share_in_ad"))
    opp_sb_share = _clean_float(opp_row.get("sb_share_in_ad"))
    sb_share_gap = _diff_float(my_sb_share, opp_sb_share)

    my_kw_entropy = _clean_float(my_row.get("kw_entropy_7d_avg"))
    opp_kw_entropy = _clean_float(opp_row.get("kw_entropy_7d_avg"))
    kw_entropy_gap = _diff_float(my_kw_entropy, opp_kw_entropy)

    my_kw_hhi = _clean_float(my_row.get("kw_hhi_7d_avg"))
    opp_kw_hhi = _clean_float(opp_row.get("kw_hhi_7d_avg"))
    kw_hhi_gap = _diff_float(my_kw_hhi, opp_kw_hhi)

    def _gap(field: str) -> float | None:
        return _diff_float(_clean_float(my_row.get(field)), _clean_float(opp_row.get(field)))

    kw_top1_gap = _gap("kw_top1_share_7d_avg")
    kw_top3_gap = _gap("kw_top3_share_7d_avg")
    kw_top10_gap = _gap("kw_top10_share_7d_avg")
    kw_brand_gap = _gap("kw_brand_share_7d_avg")
    kw_competitor_gap = _gap("kw_competitor_share_7d_avg")
    kw_generic_gap = _gap("kw_generic_share_7d_avg")
    kw_attribute_gap = _gap("kw_attribute_share_7d_avg")

    mix_values = {
        "ad_ratio_index_med": ad_ratio_index,
        "ad_to_natural_gap": ad_to_natural_gap,
        "sp_share_in_ad_gap": sp_share_gap,
        "sbv_share_in_ad_gap": sbv_share_gap,
        "sb_share_in_ad_gap": sb_share_gap,
    }
    keyword_values = {
        "kw_entropy_gap": kw_entropy_gap,
        "kw_top3_share_gap": kw_top3_gap,
        "kw_brand_share_gap": kw_brand_gap,
        "kw_competitor_share_gap": kw_competitor_gap,
        "kw_generic_share_gap": kw_generic_gap,
        "kw_attribute_share_gap": kw_attribute_gap,
        "kw_top1_share_gap": kw_top1_gap,
        "kw_top10_share_gap": kw_top10_gap,
        "kw_hhi_gap": kw_hhi_gap,
    }

    mix_scores: dict[str, float | None] = {}
    for name, value in mix_values.items():
        mix_scores[name] = _score_feature(value, context.mix_rules.get(name))
    keyword_scores: dict[str, float | None] = {}
    for name, value in keyword_values.items():
        keyword_scores[name] = _score_feature(value, context.keyword_rules.get(name))

    mix_available = [
        (score, context.mix_rules.get(name, {}).get("weight", 0.0))
        for name, score in mix_scores.items()
    ]
    mix_pressure = (
        _weighted_sum(
            [score for score, weight in mix_available],
            [weight for score, weight in mix_available],
        )
        if any(score is not None and weight > 0 for score, weight in mix_available)
        else None
    )

    keyword_available = [
        (score, context.keyword_rules.get(name, {}).get("weight", 0.0))
        for name, score in keyword_scores.items()
    ]
    keyword_pressure = (
        _weighted_sum(
            [score for score, weight in keyword_available],
            [weight for score, weight in keyword_available],
        )
        if any(score is not None and weight > 0 for score, weight in keyword_available)
        else None
    )

    mix_confidence = _confidence(list(mix_values.values()))
    keyword_confidence = _confidence([
        kw_entropy_gap,
        kw_top3_gap,
        kw_brand_gap,
        kw_competitor_gap,
        kw_generic_gap,
        kw_attribute_gap,
        kw_top1_gap,
        kw_top10_gap,
        kw_hhi_gap,
    ])

    coverage_values = []
    for row in (my_row, opp_row):
        coverage = _clean_float(row.get("kw_coverage_ratio"))
        if coverage is not None:
            coverage_values.append(float(np.clip(coverage, 0.0, 1.0)))
    coverage_ratio = min(coverage_values) if coverage_values else 0.0

    confidence_components = [mix_confidence, keyword_confidence, coverage_ratio]
    available_conf = [comp for comp in confidence_components if comp is not None]
    t_confidence = float(np.mean(available_conf)) if available_conf else 0.0

    pressure_weights = context.pressure_weights
    overall_values: list[float | None] = []
    overall_weights: list[float] = []
    if mix_pressure is not None:
        overall_values.append(mix_pressure)
        overall_weights.append(float(pressure_weights.get("mix", 0.5)))
    if keyword_pressure is not None:
        overall_values.append(keyword_pressure)
        overall_weights.append(float(pressure_weights.get("keyword", 0.5)))

    t_pressure = (
        _weighted_sum(overall_values, overall_weights)
        if overall_values and any(weight > 0 for weight in overall_weights)
        else None
    )
    t_band = _assign_band(t_pressure, context.band_cuts) if t_pressure is not None else None

    return {
        "ad_ratio_gap": ad_ratio_gap,
        "ad_ratio_index_med": ad_ratio_index,
        "ad_to_natural_gap": ad_to_natural_gap,
        "sp_share_in_ad_gap": sp_share_gap,
        "sbv_share_in_ad_gap": sbv_share_gap,
        "sb_share_in_ad_gap": sb_share_gap,
        "kw_entropy_gap": kw_entropy_gap,
        "kw_hhi_gap": kw_hhi_gap,
        "kw_top1_share_gap": kw_top1_gap,
        "kw_top3_share_gap": kw_top3_gap,
        "kw_top10_share_gap": kw_top10_gap,
        "kw_brand_share_gap": kw_brand_gap,
        "kw_competitor_share_gap": kw_competitor_gap,
        "kw_generic_share_gap": kw_generic_gap,
        "kw_attribute_share_gap": kw_attribute_gap,
        "t_score_mix": mix_pressure,
        "t_score_kw": keyword_pressure,
        "t_pressure": t_pressure,
        "t_intensity_band": t_band,
        "t_mix_confidence": mix_confidence,
        "t_keyword_confidence": keyword_confidence,
        "t_coverage_ratio": coverage_ratio,
        "t_confidence": t_confidence,
        "t_mix_scores": mix_scores,
        "t_keyword_scores": keyword_scores,
    }
def _compute_pair_each_row(
    my_row: Mapping[str, Any],
    opp_row: Mapping[str, Any],
    *,
    feature_rules: dict[str, dict[str, float]],
    band_cuts: dict[str, float],
    traffic_context: TrafficScoringContext,
    traffic_median: Mapping[str, float | None],
) -> dict[str, Any]:
    my_badges = _normalise_badges(my_row.get("badge_json"))
    opp_badges = _normalise_badges(opp_row.get("badge_json"))

    my_price_net = _clean_float(my_row.get("price_net"))
    opp_price_net = _clean_float(opp_row.get("price_net"))
    price_gap_each = _diff_float(my_price_net, opp_price_net)
    price_ratio_each = _safe_ratio(my_price_net, opp_price_net)

    my_rank_leaf = _clean_int(my_row.get("rank_leaf"))
    opp_rank_leaf = _clean_int(opp_row.get("rank_leaf"))
    my_rank_pct = _clean_float(my_row.get("rank_score"))
    opp_rank_pct = _clean_float(opp_row.get("rank_score"))
    rank_pos_delta = _diff_float(my_rank_pct, opp_rank_pct)

    my_content_score = _clean_float(my_row.get("content_score"))
    opp_content_score = _clean_float(opp_row.get("content_score"))
    content_gap_each = _diff_float(my_content_score, opp_content_score)

    my_social_proof = _clean_float(my_row.get("social_proof"))
    opp_social_proof = _clean_float(opp_row.get("social_proof"))
    social_gap_each = _diff_float(my_social_proof, opp_social_proof)

    badge_diff, badge_delta_sum = _badge_gap(my_badges, opp_badges)

    score_price = _score_feature(price_gap_each, feature_rules.get("price"))
    score_rank = _score_feature(rank_pos_delta, feature_rules.get("rank"))
    score_cont = _score_feature(-content_gap_each if content_gap_each is not None else None, feature_rules.get("content"))
    score_soc = _score_feature(-social_gap_each if social_gap_each is not None else None, feature_rules.get("social"))
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
    confidence = _confidence([price_gap_each, rank_pos_delta, content_gap_each, social_gap_each, badge_delta_sum])

    traffic_metrics = _compute_traffic_metrics(
        my_row,
        opp_row,
        traffic_median=traffic_median,
        context=traffic_context,
    )

    traffic_gap = _extract_traffic_gap(traffic_metrics)
    traffic_scores = _extract_traffic_scores(traffic_metrics)
    traffic_confidence = _extract_traffic_confidence(traffic_metrics)

    record = {
        "scene_tag": my_row.get("scene_tag"),
        "base_scene": my_row.get("base_scene"),
        "morphology": my_row.get("morphology"),
        "marketplace_id": my_row.get("marketplace_id"),
        "week": my_row.get("week"),
        "sunday": my_row.get("sunday"),
        "my_parent_asin": my_row.get("parent_asin"),
        "my_asin": my_row.get("asin"),
        "opp_parent_asin": opp_row.get("parent_asin"),
        "opp_asin": opp_row.get("asin"),
        "my_price_net": my_price_net,
        "opp_price_net": opp_price_net,
        "my_rank_leaf": my_rank_leaf,
        "opp_rank_leaf": opp_rank_leaf,
        "my_rank_pos_pct": my_rank_pct,
        "opp_rank_pos_pct": opp_rank_pct,
        "my_content_score": my_content_score,
        "opp_content_score": opp_content_score,
        "my_social_proof": my_social_proof,
        "opp_social_proof": opp_social_proof,
        "price_gap_each": price_gap_each,
        "price_ratio_each": price_ratio_each,
        "rank_pos_delta": rank_pos_delta,
        "content_gap_each": content_gap_each,
        "social_gap_each": social_gap_each,
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
        "traffic_gap": traffic_gap,
        "traffic_scores": traffic_scores,
        "traffic_confidence": traffic_confidence,
    }
    record.update(traffic_metrics)
    return record


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
    traffic_context: TrafficScoringContext,
    traffic_median: Mapping[str, float | None],
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

    traffic_metrics = _compute_traffic_metrics(
        my_row,
        opp_row,
        traffic_median=traffic_median,
        context=traffic_context,
    )

    traffic_gap = _extract_traffic_gap(traffic_metrics)
    traffic_scores = _extract_traffic_scores(traffic_metrics)
    traffic_confidence = _extract_traffic_confidence(traffic_metrics)

    record = {
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
        "traffic_gap": traffic_gap,
        "traffic_scores": traffic_scores,
        "traffic_confidence": traffic_confidence,
    }
    record.update(traffic_metrics)
    return record


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
    invert = _coerce_bool(config.get("invert")) if isinstance(config, dict) else False
    if invert:
        value = -value
        theta = -theta
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
        "traffic_gap",
        "traffic_scores",
        "traffic_confidence",
        "ad_ratio_gap",
        "ad_ratio_index_med",
        "ad_to_natural_gap",
        "sp_share_in_ad_gap",
        "sbv_share_in_ad_gap",
        "sb_share_in_ad_gap",
        "kw_entropy_gap",
        "kw_hhi_gap",
        "kw_top1_share_gap",
        "kw_top3_share_gap",
        "kw_top10_share_gap",
        "kw_brand_share_gap",
        "kw_competitor_share_gap",
        "kw_generic_share_gap",
        "kw_attribute_share_gap",
        "t_score_mix",
        "t_score_kw",
        "t_pressure",
        "t_intensity_band",
        "t_mix_confidence",
        "t_keyword_confidence",
        "t_coverage_ratio",
        "t_confidence",
        "t_mix_scores",
        "t_keyword_scores",
    ]


def _pair_each_columns() -> list[str]:
    return [
        "scene_tag",
        "base_scene",
        "morphology",
        "marketplace_id",
        "week",
        "sunday",
        "my_parent_asin",
        "my_asin",
        "opp_parent_asin",
        "opp_asin",
        "my_price_net",
        "opp_price_net",
        "my_rank_leaf",
        "opp_rank_leaf",
        "my_rank_pos_pct",
        "opp_rank_pos_pct",
        "my_content_score",
        "opp_content_score",
        "my_social_proof",
        "opp_social_proof",
        "price_gap_each",
        "price_ratio_each",
        "rank_pos_delta",
        "content_gap_each",
        "social_gap_each",
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
        "traffic_gap",
        "traffic_scores",
        "traffic_confidence",
        "ad_ratio_gap",
        "ad_ratio_index_med",
        "ad_to_natural_gap",
        "sp_share_in_ad_gap",
        "sbv_share_in_ad_gap",
        "sb_share_in_ad_gap",
        "kw_entropy_gap",
        "kw_hhi_gap",
        "kw_top1_share_gap",
        "kw_top3_share_gap",
        "kw_top10_share_gap",
        "kw_brand_share_gap",
        "kw_competitor_share_gap",
        "kw_generic_share_gap",
        "kw_attribute_share_gap",
        "t_score_mix",
        "t_score_kw",
        "t_pressure",
        "t_intensity_band",
        "t_mix_confidence",
        "t_keyword_confidence",
        "t_coverage_ratio",
        "t_confidence",
        "t_mix_scores",
        "t_keyword_scores",
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
        "d_ad_ratio_gap",
        "d_ad_ratio_index_med",
        "d_ad_to_natural_gap",
        "d_sp_share_in_ad_gap",
        "d_sbv_share_in_ad_gap",
        "d_sb_share_in_ad_gap",
        "d_kw_entropy_gap",
        "d_kw_hhi_gap",
        "d_kw_top1_share_gap",
        "d_kw_top3_share_gap",
        "d_kw_top10_share_gap",
        "d_kw_brand_share_gap",
        "d_kw_competitor_share_gap",
        "d_kw_generic_share_gap",
        "d_kw_attribute_share_gap",
        "d_t_score_mix",
        "d_t_score_kw",
        "d_t_pressure",
        "d_t_confidence",
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
        "traffic",
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
    pair_each_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    current_gap = _extract_gap(row._asdict())
    previous_gap = _extract_gap(prev_row) if prev_row else None
    delta_gap = _extract_delta_gap(delta_row, row._asdict(), prev_row)
    traffic_gap = _extract_traffic_gap(row._asdict())
    traffic_delta = _extract_traffic_delta(delta_row, row._asdict(), prev_row)
    traffic_scores = _extract_traffic_scores(row._asdict())
    traffic_confidence = _extract_traffic_confidence(row._asdict())
    my_snapshot = _build_entity_snapshot(my_current)
    my_change = _build_my_change(delta_row, my_current, my_previous)
    opp_snapshot = _build_entity_snapshot(opp_current)
    opp_change = _build_entity_change(opp_current, opp_previous)
    score_components = _extract_scores(row._asdict())
    primary_competitor = _extract_pair_each_metrics(pair_each_row)

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
        "primary_competitor": primary_competitor,
        "traffic": {
            "gap": traffic_gap,
            "delta": traffic_delta,
            "scores": traffic_scores,
            "confidence": traffic_confidence,
        },
    }


def _extract_pair_each_metrics(row: Mapping[str, Any] | pd.Series | None) -> dict[str, Any] | None:
    if row is None:
        return None

    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = row

    traffic_gap = _extract_traffic_gap(data)
    traffic_scores = _extract_traffic_scores(data)
    traffic_confidence = _extract_traffic_confidence(data)

    fields_float = (
        "my_price_net",
        "opp_price_net",
        "price_gap_each",
        "price_ratio_each",
        "rank_pos_delta",
        "content_gap_each",
        "social_gap_each",
        "pressure",
        "confidence",
    )
    fields_int = ("my_rank_leaf", "opp_rank_leaf", "badge_delta_sum")
    result: dict[str, Any] = {
        "opp_asin": data.get("opp_asin"),
        "opp_parent_asin": data.get("opp_parent_asin"),
        "intensity_band": data.get("intensity_band"),
        "badge_diff": data.get("badge_diff"),
        "my_rank_pos_pct": _clean_float(data.get("my_rank_pos_pct")),
        "opp_rank_pos_pct": _clean_float(data.get("opp_rank_pos_pct")),
        "my_content_score": _clean_float(data.get("my_content_score")),
        "opp_content_score": _clean_float(data.get("opp_content_score")),
        "my_social_proof": _clean_float(data.get("my_social_proof")),
        "opp_social_proof": _clean_float(data.get("opp_social_proof")),
        "traffic_gap": traffic_gap,
        "traffic_scores": traffic_scores,
        "traffic_confidence": traffic_confidence,
    }
    for field in fields_float:
        result[field] = _clean_float(data.get(field))
    for field in fields_int:
        result[field] = _clean_int(data.get(field))
    return result


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


def _normalise_score_map(raw: Any) -> dict[str, float | None]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, pd.Series):
        raw = raw.to_dict()
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, float | None] = {}
    for key, value in raw.items():
        result[str(key)] = _clean_float(value)
    return result


def _extract_traffic_gap(row: Mapping[str, Any] | None) -> dict[str, dict[str, float | None]]:
    if row is None:
        return {"mix": {}, "keyword": {}}
    mix_fields = {
        "ad_ratio_gap": row.get("ad_ratio_gap"),
        "ad_ratio_index_med": row.get("ad_ratio_index_med"),
        "ad_to_natural_gap": row.get("ad_to_natural_gap"),
        "sp_share_in_ad_gap": row.get("sp_share_in_ad_gap"),
        "sbv_share_in_ad_gap": row.get("sbv_share_in_ad_gap"),
        "sb_share_in_ad_gap": row.get("sb_share_in_ad_gap"),
    }
    keyword_fields = {
        "kw_entropy_gap": row.get("kw_entropy_gap"),
        "kw_hhi_gap": row.get("kw_hhi_gap"),
        "kw_top1_share_gap": row.get("kw_top1_share_gap"),
        "kw_top3_share_gap": row.get("kw_top3_share_gap"),
        "kw_top10_share_gap": row.get("kw_top10_share_gap"),
        "kw_brand_share_gap": row.get("kw_brand_share_gap"),
        "kw_competitor_share_gap": row.get("kw_competitor_share_gap"),
        "kw_generic_share_gap": row.get("kw_generic_share_gap"),
        "kw_attribute_share_gap": row.get("kw_attribute_share_gap"),
    }
    return {
        "mix": {key: _clean_float(value) for key, value in mix_fields.items()},
        "keyword": {key: _clean_float(value) for key, value in keyword_fields.items()},
    }


def _extract_traffic_scores(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "mix_score": None,
            "keyword_score": None,
            "pressure": None,
            "band": None,
            "mix_components": {},
            "keyword_components": {},
        }
    mix_components = _normalise_score_map(row.get("t_mix_scores"))
    keyword_components = _normalise_score_map(row.get("t_keyword_scores"))
    return {
        "mix_score": _clean_float(row.get("t_score_mix")),
        "keyword_score": _clean_float(row.get("t_score_kw")),
        "pressure": _clean_float(row.get("t_pressure")),
        "band": row.get("t_intensity_band"),
        "mix_components": mix_components,
        "keyword_components": keyword_components,
    }


def _extract_traffic_confidence(row: Mapping[str, Any] | None) -> dict[str, float | None]:
    if row is None:
        return {"overall": None, "mix": None, "keyword": None, "coverage": None}
    return {
        "overall": _clean_float(row.get("t_confidence")),
        "mix": _clean_float(row.get("t_mix_confidence")),
        "keyword": _clean_float(row.get("t_keyword_confidence")),
        "coverage": _clean_float(row.get("t_coverage_ratio")),
    }


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


def _extract_traffic_delta(
    delta_row: dict[str, Any] | None,
    current_row: dict[str, Any],
    previous_row: dict[str, Any] | None,
) -> dict[str, Any]:
    mix_fields = (
        "ad_ratio_gap",
        "ad_ratio_index_med",
        "ad_to_natural_gap",
        "sp_share_in_ad_gap",
        "sbv_share_in_ad_gap",
        "sb_share_in_ad_gap",
    )
    keyword_fields = (
        "kw_entropy_gap",
        "kw_hhi_gap",
        "kw_top1_share_gap",
        "kw_top3_share_gap",
        "kw_top10_share_gap",
        "kw_brand_share_gap",
        "kw_competitor_share_gap",
        "kw_generic_share_gap",
        "kw_attribute_share_gap",
    )
    if delta_row:
        mix = {
            field: _clean_float(delta_row.get(f"d_{field}")) for field in mix_fields
        }
        keyword = {
            field: _clean_float(delta_row.get(f"d_{field}")) for field in keyword_fields
        }
        scores = {
            "mix": _clean_float(delta_row.get("d_t_score_mix")),
            "keyword": _clean_float(delta_row.get("d_t_score_kw")),
            "pressure": _clean_float(delta_row.get("d_t_pressure")),
            "confidence": _clean_float(delta_row.get("d_t_confidence")),
        }
        return {"mix": mix, "keyword": keyword, "scores": scores}
    if previous_row:
        mix = {
            field: _diff_float(current_row.get(field), previous_row.get(field)) for field in mix_fields
        }
        keyword = {
            field: _diff_float(current_row.get(field), previous_row.get(field)) for field in keyword_fields
        }
        scores = {
            "mix": _diff_float(current_row.get("t_score_mix"), previous_row.get("t_score_mix")),
            "keyword": _diff_float(current_row.get("t_score_kw"), previous_row.get("t_score_kw")),
            "pressure": _diff_float(current_row.get("t_pressure"), previous_row.get("t_pressure")),
            "confidence": _diff_float(current_row.get("t_confidence"), previous_row.get("t_confidence")),
        }
        return {"mix": mix, "keyword": keyword, "scores": scores}
    return {
        "mix": {field: None for field in mix_fields},
        "keyword": {field: None for field in keyword_fields},
        "scores": {"mix": None, "keyword": None, "pressure": None, "confidence": None},
    }


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
        "kw_days_covered",
    )
    for field in int_fields:
        snapshot[field] = _clean_int(record.get(field))
    snapshot["badge_json"] = record.get("badge_json", [])
    snapshot["asin"] = record.get("asin")
    snapshot["hyy_asin"] = record.get("hyy_asin")
    snapshot["kw_days_covered"] = _clean_int(record.get("kw_days_covered"))
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

    traffic_pressure_series = (
        pd.to_numeric(pairs_current.get("t_pressure"), errors="coerce")
        if not pairs_current.empty
        else pd.Series(dtype=float)
    )
    traffic_pressure_p50 = _quantile(traffic_pressure_series, 0.5)
    traffic_pressure_p90 = _quantile(traffic_pressure_series, 0.9)
    traffic_confidence_series = (
        pd.to_numeric(pairs_current.get("t_confidence"), errors="coerce")
        if not pairs_current.empty
        else pd.Series(dtype=float)
    )
    traffic_coverage_series = (
        pd.to_numeric(pairs_current.get("t_coverage_ratio"), errors="coerce")
        if not pairs_current.empty
        else pd.Series(dtype=float)
    )
    mix_gap_series = (
        pd.to_numeric(pairs_current.get("ad_ratio_gap"), errors="coerce")
        if not pairs_current.empty
        else pd.Series(dtype=float)
    )
    keyword_gap_series = (
        pd.to_numeric(pairs_current.get("kw_top3_share_gap"), errors="coerce")
        if not pairs_current.empty
        else pd.Series(dtype=float)
    )
    lagging_pairs = 0
    if not traffic_pressure_series.empty and not traffic_confidence_series.empty:
        lag_mask = (traffic_pressure_series > 0.6) & (traffic_confidence_series >= 0.5)
        lagging_pairs = int(lag_mask.sum())

    summary: dict[str, Any] = {
        "my_asin_cnt": my_asin_cnt,
        "comp_cnt": comp_cnt,
        "pressure_p50": pressure_p50,
        "pressure_p90": pressure_p90,
        "worsen_ratio": worsen_ratio,
        "moves": moves,
        "avg_scores": avg_scores,
        "traffic": {
            "pressure_p50": traffic_pressure_p50,
            "pressure_p90": traffic_pressure_p90,
            "confidence_p50": _quantile(traffic_confidence_series, 0.5),
            "coverage_p50": _quantile(traffic_coverage_series, 0.5),
            "lagging_pairs": lagging_pairs,
            "avg_mix_gap": _nanmean(mix_gap_series),
            "avg_keyword_gap": _nanmean(keyword_gap_series),
        },
    }
    return summary


def _build_top_opponents(
    *,
    pairs_each_current: pd.DataFrame,
    pairs_each_previous: pd.DataFrame,
) -> list[dict[str, Any]]:
    if pairs_each_current.empty:
        return []

    previous_lookup = {
        (row["scene_tag"], row["marketplace_id"], row["my_asin"], row["opp_asin"]): row
        for _, row in pairs_each_previous.iterrows()
    }

    group_cols = ["scene_tag", "base_scene", "morphology", "marketplace_id", "week", "my_asin"]
    results: list[dict[str, Any]] = []

    for _, group in pairs_each_current.groupby(group_cols, dropna=False):
        base = group.iloc[0]
        key_prefix = (base.get("scene_tag"), base.get("marketplace_id"), base.get("my_asin"))
        opponents: list[dict[str, Any]] = []

        sorted_group = group.sort_values(by="pressure", ascending=False, na_position="last").head(3)
        for _, opp_row in sorted_group.iterrows():
            prev_key = (*key_prefix, opp_row.get("opp_asin"))
            prev_row = previous_lookup.get(prev_key)
            opponents.append(
                {
                    "opp_asin": opp_row.get("opp_asin"),
                    "opp_parent_asin": opp_row.get("opp_parent_asin"),
                    "pressure": _clean_float(opp_row.get("pressure")),
                    "delta_pressure": _diff_float(
                        opp_row.get("pressure"),
                        prev_row.get("pressure") if prev_row is not None else None,
                    ),
                    "intensity_band": opp_row.get("intensity_band"),
                    "confidence": _clean_float(opp_row.get("confidence")),
                    "price_gap_each": _clean_float(opp_row.get("price_gap_each")),
                    "rank_pos_delta": _clean_float(opp_row.get("rank_pos_delta")),
                    "content_gap_each": _clean_float(opp_row.get("content_gap_each")),
                    "social_gap_each": _clean_float(opp_row.get("social_gap_each")),
                    "badge_delta_sum": _clean_int(opp_row.get("badge_delta_sum")),
                }
            )

        results.append(
            {
                "scene_tag": base.get("scene_tag"),
                "base_scene": base.get("base_scene"),
                "morphology": base.get("morphology"),
                "marketplace_id": base.get("marketplace_id"),
                "week": base.get("week"),
                "my_parent_asin": base.get("my_parent_asin"),
                "my_asin": base.get("my_asin"),
                "top_competitors": opponents,
            }
        )

    return results


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
