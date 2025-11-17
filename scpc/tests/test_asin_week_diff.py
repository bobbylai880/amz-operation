from __future__ import annotations

from datetime import date

import pandas as pd

from scpc.etl.asin_week_diff import DEFAULT_RULES, compute_week_diff, load_week_diff_rules


def _build_base_row() -> dict[str, object]:
    return {
        "asin": "B001",
        "marketplace_id": "US",
        "sunday_this": date(2025, 11, 9),
        "week_this": "2025-W45",
        "sunday_last": date(2025, 11, 2),
        "week_last": "2025-W44",
        "rank_leaf_this": 120,
        "rank_leaf_last": 200,
        "rank_root_this": 500,
        "rank_root_last": 450,
        "price_current_this": 18.0,
        "price_current_last": 20.0,
        "price_list_this": 25.0,
        "price_list_last": 25.0,
        "coupon_pct_this": 5.0,
        "coupon_pct_last": 0.0,
        "coupon_description_this": "  save 5%  ",
        "coupon_description_last": "",
        "rating_this": 4.5,
        "rating_last": 4.6,
        "reviews_this": 1200,
        "reviews_last": 1100,
        "image_cnt_this": 6,
        "image_cnt_last": 5,
        "video_cnt_this": 2,
        "video_cnt_last": 2,
        "bullet_cnt_this": 6,
        "bullet_cnt_last": 6,
        "title_len_this": 120,
        "title_len_last": 118,
        "aplus_flag_this": 1,
        "aplus_flag_last": 1,
        "badge_json_this": '["Amazon\'s Choice", "Climate Pledge Friendly"]',
        "badge_json_last": '["Amazon\'s Choice"]',
    }


def test_compute_week_diff_generates_expected_deltas_and_labels() -> None:
    df = pd.DataFrame([_build_base_row()])
    result = compute_week_diff(df, DEFAULT_RULES)
    row = result.iloc[0]

    assert row["rank_leaf_diff"] == 80
    assert row["price_current_diff"] == -2
    assert round(row["price_change_rate"], 4) == -0.1
    assert row["price_action"] == "大幅降价"
    assert row["rank_trend"] == "排名明显上升"
    assert row["new_reviews"] == 100
    assert row["has_coupon_last"] == 0
    assert row["has_coupon_this"] == 1
    assert row["promo_action"] == "新增优惠"
    assert row["badge_added_cnt"] == 1
    assert row["badge_removed_cnt"] == 0
    assert row["has_badge_change"] == 1


def test_compute_week_diff_handles_missing_previous_week() -> None:
    base = _build_base_row()
    base.update(
        {
            "rank_leaf_last": None,
            "rank_root_last": None,
            "price_current_last": None,
            "coupon_pct_last": None,
            "coupon_description_last": None,
            "rating_last": None,
            "reviews_last": None,
            "image_cnt_last": None,
            "video_cnt_last": None,
            "bullet_cnt_last": None,
            "title_len_last": None,
            "aplus_flag_last": None,
            "badge_json_last": None,
        }
    )
    df = pd.DataFrame([base])
    result = compute_week_diff(df, DEFAULT_RULES)
    row = result.iloc[0]

    assert row["price_action"] == "新ASIN"
    assert row["rank_trend"] == "新ASIN"
    assert row["new_reviews"] == base["reviews_this"]
    assert row["promo_action"] == "新增优惠"
    assert row["badge_added_cnt"] == 2
    assert row["badge_removed_cnt"] == 0


def test_load_week_diff_rules_warns_for_missing_sections(tmp_path, caplog) -> None:
    config_path = tmp_path / "rules.yml"
    config_path.write_text(
        "price_action:\n  thresholds:\n    big_drop: -0.2\n",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        rules = load_week_diff_rules(config_path)
    assert any("Week diff rules missing fields" in record.message for record in caplog.records)
    assert "rank_trend" in rules
    assert "promo_action" in rules
