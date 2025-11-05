from __future__ import annotations

from datetime import date, datetime

import pandas as pd


MY_ASINS_SAMPLE = {"MY-ASIN-1"}


def build_competition_snapshot_sample() -> pd.DataFrame:
    """Return two-week ASIN snapshots including our ASIN and two competitors."""

    records = [
        {
            "scene_tag": "SCN-USBAG-01",
            "base_scene": "收纳袋",
            "morphology": "standard",
            "marketplace_id": "US",
            "week": "2025W10",
            "sunday": date(2025, 3, 9),
            "parent_asin": "PARENT-1",
            "asin": "MY-ASIN-1",
            "price_current": 19.99,
            "price_list": 24.99,
            "coupon_pct": 0.10,
            "discount_rate": 0.15,
            "rank_root": 2500,
            "rank_leaf": 45,
            "rating": 4.6,
            "reviews": 120,
            "image_cnt": 6,
            "video_cnt": 1,
            "bullet_cnt": 5,
            "title_len": 82,
            "aplus_flag": 1,
            "badge_json": "[\"Best Seller\"]",
        },
        {
            "scene_tag": "SCN-USBAG-01",
            "base_scene": "收纳袋",
            "morphology": "standard",
            "marketplace_id": "US",
            "week": "2025W10",
            "sunday": date(2025, 3, 9),
            "parent_asin": "COMP-1",
            "asin": "COMP-ASIN-1",
            "price_current": 18.49,
            "price_list": 22.99,
            "coupon_pct": 0.00,
            "discount_rate": 0.20,
            "rank_root": 2100,
            "rank_leaf": 32,
            "rating": 4.8,
            "reviews": 260,
            "image_cnt": 7,
            "video_cnt": 2,
            "bullet_cnt": 6,
            "title_len": 88,
            "aplus_flag": 1,
            "badge_json": "{\"AmazonChoice\": true}",
        },
        {
            "scene_tag": "SCN-USBAG-01",
            "base_scene": "收纳袋",
            "morphology": "standard",
            "marketplace_id": "US",
            "week": "2025W10",
            "sunday": date(2025, 3, 9),
            "parent_asin": "COMP-2",
            "asin": "COMP-ASIN-2",
            "price_current": 20.49,
            "price_list": 23.99,
            "coupon_pct": 0.05,
            "discount_rate": 0.15,
            "rank_root": 2400,
            "rank_leaf": 40,
            "rating": 4.4,
            "reviews": 180,
            "image_cnt": 6,
            "video_cnt": 1,
            "bullet_cnt": 5,
            "title_len": 85,
            "aplus_flag": 1,
            "badge_json": "[\"ClimatePledgeFriendly\"]",
        },
        {
            "scene_tag": "SCN-USBAG-01",
            "base_scene": "收纳袋",
            "morphology": "standard",
            "marketplace_id": "US",
            "week": "2025W09",
            "sunday": date(2025, 3, 2),
            "parent_asin": "PARENT-1",
            "asin": "MY-ASIN-1",
            "price_current": 21.00,
            "price_list": 24.99,
            "coupon_pct": 0.05,
            "discount_rate": 0.10,
            "rank_root": 2800,
            "rank_leaf": 52,
            "rating": 4.5,
            "reviews": 110,
            "image_cnt": 6,
            "video_cnt": 0,
            "bullet_cnt": 5,
            "title_len": 82,
            "aplus_flag": 1,
            "badge_json": "[]",
        },
        {
            "scene_tag": "SCN-USBAG-01",
            "base_scene": "收纳袋",
            "morphology": "standard",
            "marketplace_id": "US",
            "week": "2025W09",
            "sunday": date(2025, 3, 2),
            "parent_asin": "COMP-1",
            "asin": "COMP-ASIN-1",
            "price_current": 17.99,
            "price_list": 22.99,
            "coupon_pct": 0.00,
            "discount_rate": 0.18,
            "rank_root": 2150,
            "rank_leaf": 35,
            "rating": 4.7,
            "reviews": 240,
            "image_cnt": 7,
            "video_cnt": 2,
            "bullet_cnt": 6,
            "title_len": 88,
            "aplus_flag": 1,
            "badge_json": "[\"AmazonChoice\"]",
        },
        {
            "scene_tag": "SCN-USBAG-01",
            "base_scene": "收纳袋",
            "morphology": "standard",
            "marketplace_id": "US",
            "week": "2025W09",
            "sunday": date(2025, 3, 2),
            "parent_asin": "COMP-2",
            "asin": "COMP-ASIN-2",
            "price_current": 21.49,
            "price_list": 23.99,
            "coupon_pct": 0.00,
            "discount_rate": 0.12,
            "rank_root": 2550,
            "rank_leaf": 48,
            "rating": 4.3,
            "reviews": 170,
            "image_cnt": 6,
            "video_cnt": 1,
            "bullet_cnt": 5,
            "title_len": 85,
            "aplus_flag": 1,
            "badge_json": "[]",
        },
    ]
    return pd.DataFrame.from_records(records)


def build_scene_tag_sample() -> pd.DataFrame:
    """Return ASIN-to-scene tagging records mirroring Doris table 1."""

    now = datetime(2025, 3, 10, 0, 0, 0)
    records = [
        {
            "scene_tag": "SCN-USBAG-01",
            "base_scene": "收纳袋",
            "morphology": "standard",
            "asin": "MY-ASIN-1",
            "marketplace_id": "US",
            "hyy_asin": 1,
            "update_time": now,
        },
        {
            "scene_tag": "SCN-USBAG-01",
            "base_scene": "收纳袋",
            "morphology": "standard",
            "asin": "COMP-ASIN-1",
            "marketplace_id": "US",
            "hyy_asin": 0,
            "update_time": now,
        },
        {
            "scene_tag": "SCN-USBAG-01",
            "base_scene": "收纳袋",
            "morphology": "standard",
            "asin": "COMP-ASIN-2",
            "marketplace_id": "US",
            "hyy_asin": 0,
            "update_time": now,
        },
    ]
    return pd.DataFrame.from_records(records)


def build_scoring_rules_sample() -> pd.DataFrame:
    """Return a default scoring-rule configuration compatible with Doris table 6."""

    rows = [
        {
            "rule_name": "default",
            "feature_name": "price",
            "theta": 0.0,
            "k": 1.5,
            "weight": 0.30,
            "band_cuts": "{\"C1\":0.25,\"C2\":0.50,\"C3\":0.75,\"C4\":1.0}",
            "is_active": 1,
        },
        {
            "rule_name": "default",
            "feature_name": "rank",
            "theta": 0.0,
            "k": 6.0,
            "weight": 0.25,
            "band_cuts": None,
            "is_active": 1,
        },
        {
            "rule_name": "default",
            "feature_name": "content",
            "theta": 0.0,
            "k": 5.0,
            "weight": 0.20,
            "band_cuts": None,
            "is_active": 1,
        },
        {
            "rule_name": "default",
            "feature_name": "social",
            "theta": 0.0,
            "k": 4.0,
            "weight": 0.15,
            "band_cuts": None,
            "is_active": 1,
        },
        {
            "rule_name": "default",
            "feature_name": "badge",
            "theta": 0.0,
            "k": 3.0,
            "weight": 0.10,
            "band_cuts": None,
            "is_active": 1,
        },
    ]
    return pd.DataFrame.from_records(rows)


__all__ = [
    "MY_ASINS_SAMPLE",
    "build_competition_snapshot_sample",
    "build_scoring_rules_sample",
]
