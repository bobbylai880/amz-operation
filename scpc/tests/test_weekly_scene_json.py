"""Unit tests for weekly scene JSON helpers."""

import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype

from scpc.reports import weekly_scene_json as wsj


def test_normalise_new_reviews_clips_negative_and_handles_missing():
    df = pd.DataFrame(
        {
            "new_reviews": [5, -3, None],
        }
    )
    result = wsj._normalise_new_reviews(df)
    assert result.tolist() == [5, 0, 0]
    assert is_integer_dtype(result)


def test_normalise_new_reviews_derives_from_reviews_columns():
    df = pd.DataFrame(
        {
            "reviews_this": [120, 10],
            "reviews_last": [100, 25],
        }
    )
    result = wsj._normalise_new_reviews(df)
    assert result.tolist() == [20, 0]


def test_price_action_distribution_includes_new_asin_bucket():
    df = pd.DataFrame(
        {
            "hyy_asin": [1, 0, 0],
            "rank_trend": ["排名明显上升", "排名明显上升", "新ASIN"],
            "price_action": ["新ASIN", "大幅降价", "价格稳定"],
            "promo_action": ["优惠无", "优惠无", "优惠无"],
            "has_badge_change": [0, 0, 0],
        }
    )
    meta = wsj.SceneMeta(
        week="2025-W45",
        scene_tag="test",
        base_scene="test",
        marketplace_id="US",
    )
    payload = wsj.build_overall_summary(df, meta)
    price_dist = payload["price_action_distribution"]
    assert price_dist["new_asin"] == 1
    assert sum(price_dist.values()) == payload["asin_counts"]["total"]


def test_select_risk_rows_follows_defined_rules():
    df = pd.DataFrame(
        {
            "asin": [f"ASIN{i}" for i in range(6)],
            "brand": ["B"] * 6,
            "marketplace_id": ["US"] * 6,
            "scene_tag": ["test"] * 6,
            "rank_trend": [
                "排名明显下降",  # rank down
                "排名基本稳定",
                "排名基本稳定",
                "排名基本稳定",
                "排名基本稳定",
                "排名基本稳定",
            ],
            "rating_diff": [0.1, -0.5, 0.0, 0.0, 0.0, 0.1],
            "promo_action": ["优惠无", "优惠无", "取消优惠", "优惠无", "优惠无", "优惠无"],
            "rank_leaf_diff": [1, 1, -10, -5, 5, 5],
            "badge_removed_json": [[], [], [], ["Best"], [], []],
            "badge_removed_cnt": [0, 0, 0, 1, 0, 0],
            "has_badge_change": [0, 0, 0, 1, 0, 0],
            "price_action": ["价格稳定"] * 6,
            "rating_last": [4.5] * 6,
            "new_reviews": [0, 20, 5, 15, 15, 15],
            "hyy_asin": [1] * 6,
        }
    )
    rules = {
        "rank_trend": "排名明显下降",
        "rating_diff_threshold": 0.0,
        "new_reviews_min": 10,
        "promo_cancel_and_rank_down": True,
        "badge_removed": True,
    }
    selected = wsj._select_risk_rows(df, rules)
    assert list(selected["asin"]) == ["ASIN0", "ASIN1", "ASIN2", "ASIN3"]


def test_scene_storage_dir_accepts_unicode_and_trims_spaces():
    assert wsj._scene_storage_dir(" 浴室袋 ") == "浴室袋"


@pytest.mark.parametrize("invalid", ["", "..", ".", "../etc", "foo/bar", "foo\\bar"])
def test_scene_storage_dir_rejects_invalid_inputs(invalid: str) -> None:
    with pytest.raises(wsj.WeeklySceneJsonError):
        wsj._scene_storage_dir(invalid)
