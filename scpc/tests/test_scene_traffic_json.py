from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from scpc.reports import scene_traffic_json as module


def _scene_scope_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asin": "SELF1",
                "marketplace_id": "US",
                "scene_tag": "浴室袋",
                "base_scene": "base",
                "morphology": "std",
                "hyy_asin": 1,
                "brand": "Attmu",
            },
            {
                "asin": "COMP1",
                "marketplace_id": "US",
                "scene_tag": "浴室袋",
                "base_scene": "base",
                "morphology": "std",
                "hyy_asin": 0,
                "brand": "Lux",
            },
        ]
    )


def _keyword_source_df(sunday_this: date, sunday_last: date) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asin": "SELF1",
                "marketplace_id": "US",
                "keyword": "camping shower caddy",
                "snapshot_date": sunday_this,
                "effective_impr_share": 0.02,
                "weekly_search_volume": 120000,
            },
            {
                "asin": "COMP1",
                "marketplace_id": "US",
                "keyword": "camping shower caddy",
                "snapshot_date": sunday_this,
                "effective_impr_share": 0.18,
                "weekly_search_volume": 120000,
            },
            {
                "asin": "SELF1",
                "marketplace_id": "US",
                "keyword": "travel shower caddy",
                "snapshot_date": sunday_this,
                "effective_impr_share": 0.01,
                "weekly_search_volume": 80000,
            },
            {
                "asin": "COMP1",
                "marketplace_id": "US",
                "keyword": "travel shower caddy",
                "snapshot_date": sunday_this,
                "effective_impr_share": 0.14,
                "weekly_search_volume": 80000,
            },
            {
                "asin": "SELF1",
                "marketplace_id": "US",
                "keyword": "travel shower caddy",
                "snapshot_date": sunday_last,
                "effective_impr_share": 0.01,
                "weekly_search_volume": 40000,
            },
            {
                "asin": "COMP1",
                "marketplace_id": "US",
                "keyword": "travel shower caddy",
                "snapshot_date": sunday_last,
                "effective_impr_share": 0.04,
                "weekly_search_volume": 40000,
            },
        ]
    )


def test_keyword_payload_includes_search_volume_and_opportunities() -> None:
    params = module.SceneTrafficJobParams(
        week="2025-W45",
        scene_tag="浴室袋",
        marketplace_id="US",
        storage_dir=Path("/tmp"),
    )
    sunday_this = date(2025, 11, 9)
    sunday_last = date(2025, 11, 2)

    df_keyword = module._prepare_keyword_dataframe(
        _scene_scope_df(), _keyword_source_df(sunday_this, sunday_last)
    )
    rules = {
        "keyword_profile_change": module.DEFAULT_RULES["keyword_profile_change"],
        "keyword_volume_opportunity": {
            "high_volume_min": 50000,
            "rising_rate_min": 0.3,
            "self_share_low_max": 0.05,
            "scene_share_min": 0.02,
        },
    }

    payload = module._build_keyword_payload(
        params,
        base_scene="base",
        morphology="std",
        sunday_this=sunday_this,
        sunday_last=sunday_last,
        df_keyword=df_keyword,
        rules=rules,
    )

    this_week_keywords = payload["scene_head_keywords"]["this_week"]
    assert this_week_keywords[0]["search_volume_this"] == 120000

    diff_common = payload["scene_head_keywords"]["diff"]["keywords_common"]
    assert diff_common[0]["search_volume_diff"] == 40000
    assert diff_common[0]["search_volume_change_rate"] == 1.0

    opportunities = payload["keyword_opportunity_by_volume"]
    assert opportunities["high_volume_low_self"][0]["keyword"] == "camping shower caddy"
    assert opportunities["rising_demand_self_lagging"][0]["keyword"] == "travel shower caddy"

    asin_profile = payload["asin_keyword_profile_change"]["self"][0]
    assert asin_profile["head_keywords_this"][0]["search_volume_this"] == 120000
