from datetime import date

from scpc.utils.dependencies import ensure_packages

ensure_packages(["sqlalchemy", "pandas", "numpy"])

import pytest
from sqlalchemy import create_engine, text

from scpc.etl.competition_features import build_traffic_features
from scpc.etl.competition_pipeline import (
    _iso_week_to_dates,
    _latest_week_with_data,
    _prepare_traffic_entities,
)
from scpc.tests.data.competition_samples import (
    build_competition_snapshot_sample,
    build_keyword_daily_sample,
    build_keyword_tag_sample,
    build_scene_tag_sample,
    build_traffic_flow_sample,
)


def test_iso_week_to_dates_returns_monday_sunday() -> None:
    monday, sunday = _iso_week_to_dates("2025W10")
    assert monday == date(2025, 3, 3)
    assert sunday == date(2025, 3, 9)


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

