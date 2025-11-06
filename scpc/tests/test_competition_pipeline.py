from datetime import date

from scpc.utils.dependencies import ensure_packages

ensure_packages(["sqlalchemy", "pandas", "numpy"])

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from scpc.etl.competition_features import (
    build_traffic_features,
    clean_competition_entities,
)

from scpc.etl.competition_pipeline import (
    _augment_in_clause,
    _iso_week_to_dates,
    _latest_week_with_data,
    _normalise_flow_dataframe,
    _prepare_traffic_entities,
    _prune_traffic_columns,
    TRAFFIC_ONLY_COLUMNS,
)
from scpc.tests.data.competition_samples import (
    build_competition_snapshot_sample,
    build_keyword_daily_sample,
    build_keyword_tag_sample,
    build_scene_tag_sample,
    build_traffic_flow_sample,
)


@pytest.mark.parametrize("label", ["2025W10", "2025-W10"])
def test_iso_week_to_dates_returns_monday_sunday(label: str) -> None:
    monday, sunday = _iso_week_to_dates(label)
    assert monday == date(2025, 3, 3)
    assert sunday == date(2025, 3, 9)


@pytest.mark.parametrize("label", ["", "2025-XX", "2025W60"])
def test_iso_week_to_dates_rejects_invalid_inputs(label: str) -> None:
    with pytest.raises(ValueError):
        _iso_week_to_dates(label)


def test_prepare_traffic_entities_merges_scene_and_parent() -> None:
    traffic = build_traffic_features(
        build_traffic_flow_sample(),
        build_keyword_daily_sample(),
        build_keyword_tag_sample(),
    )
    # Restrict to a single week to simulate pipeline input
    traffic = traffic.loc[traffic["week"] == "2025W10"].reset_index(drop=True)

    scene_tags = build_scene_tag_sample()
    snapshots = build_competition_snapshot_sample()
    snapshots = snapshots.loc[snapshots["week"] == "2025W10"].reset_index(drop=True)

    enriched = _prepare_traffic_entities(traffic, scene_tags, snapshots)
    expected_columns = {
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
    }
    assert set(enriched.columns) == expected_columns
    assert not enriched.empty

    my_row = enriched.loc[(enriched["asin"] == "MY-ASIN-1")].iloc[0]
    assert my_row["scene_tag"] == "SCN-USBAG-01"
    assert my_row["parent_asin"] == "PARENT-1"
    assert my_row["hyy_asin"] == 1
    assert 0 <= my_row["kw_coverage_ratio"] <= 1


def test_prune_traffic_columns_removes_flow_metrics() -> None:
    snapshots = build_competition_snapshot_sample()
    snapshots = snapshots.loc[snapshots["week"] == "2025W10"].reset_index(drop=True)
    scene_tags = build_scene_tag_sample()

    traffic_features = build_traffic_features(
        build_traffic_flow_sample(),
        build_keyword_daily_sample(),
        build_keyword_tag_sample(),
    )
    traffic_features = traffic_features.loc[
        traffic_features["week"] == "2025W10"
    ].reset_index(drop=True)

    entities = clean_competition_entities(
        snapshots,
        my_asins={"MY-ASIN-1"},
        scene_tags=scene_tags,
        traffic=traffic_features,
    )
    pruned, dropped = _prune_traffic_columns(entities)

    expected_dropped = {col for col in TRAFFIC_ONLY_COLUMNS if col in entities.columns}
    assert dropped == expected_dropped
    for column in expected_dropped:
        assert column not in pruned.columns

    assert {"scene_tag", "price_current", "asin"}.issubset(pruned.columns)


def test_normalise_flow_dataframe_adds_calendar_fields() -> None:
    raw = pd.DataFrame(
        [
            {
                "asin": "A1",
                "marketplace_id": "US",
                "monday": "20250303",
                "广告流量占比": "0.25",
                "自然流量占比": 0.50,
                "推荐流量占比": 0.25,
                "SP广告流量占比": 0.10,
                "视频广告流量占比": 0.05,
                "品牌广告流量占比": 0.02,
            }
        ]
    )

    normalised = _normalise_flow_dataframe(raw)

    for column in [
        "asin",
        "marketplace_id",
        "monday",
        "广告流量占比",
        "自然流量占比",
        "推荐流量占比",
        "SP广告流量占比",
        "视频广告流量占比",
        "品牌广告流量占比",
        "sunday",
        "week",
    ]:
        assert column in normalised.columns
    assert normalised.loc[0, "monday"] == date(2025, 3, 3)
    assert normalised.loc[0, "sunday"] == date(2025, 3, 9)
    assert normalised.loc[0, "week"] == "2025W10"


def test_augment_in_clause_appends_tokens() -> None:
    base_sql = "SELECT * FROM demo WHERE marketplace_id = :mk"
    sql, params = _augment_in_clause(base_sql, "asin", ["A1", "A2"], "asin")

    assert sql.endswith("AND asin IN (:asin_0, :asin_1)")
    assert params == {"asin_0": "A1", "asin_1": "A2"}


def test_augment_in_clause_no_values_returns_base_sql() -> None:
    base_sql = "SELECT * FROM demo WHERE marketplace_id = :mk"
    sql, params = _augment_in_clause(base_sql, "asin", [], "asin")

    assert sql == base_sql
    assert params == {}


def test_latest_week_with_data_returns_latest_label() -> None:
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE bi_amz_asin_product_snapshot (
                    marketplace_id TEXT,
                    week TEXT,
                    sunday TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                "INSERT INTO bi_amz_asin_product_snapshot VALUES (:mk, :week, :sunday)"
            ),
            {"mk": "US", "week": "2025W10", "sunday": "2025-03-09"},
        )
        conn.execute(
            text(
                "INSERT INTO bi_amz_asin_product_snapshot VALUES (:mk, :week, :sunday)"
            ),
            {"mk": "US", "week": "2025W11", "sunday": "2025-03-16"},
        )
    assert _latest_week_with_data(engine, "US") == "2025W11"


def test_latest_week_with_data_raises_when_missing_snapshot() -> None:
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE bi_amz_asin_product_snapshot (
                    marketplace_id TEXT,
                    week TEXT,
                    sunday TEXT
                )
                """
            )
        )

    with pytest.raises(RuntimeError):
        _latest_week_with_data(engine, "US")

