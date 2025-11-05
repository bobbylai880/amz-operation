import pytest

from scpc.etl.competition_features import (
    build_competition_delta,
    build_competition_pairs,
    build_competition_pairs_each,
    build_competition_tables,
    build_traffic_features,
    clean_competition_entities,
    compute_competition_features,
)
from scpc.tests.data import (
    MY_ASINS_SAMPLE,
    build_competition_snapshot_sample,
    build_keyword_daily_sample,
    build_keyword_tag_sample,
    build_scene_tag_sample,
    build_scoring_rules_sample,
    build_traffic_flow_sample,
)


def _build_traffic_features() -> tuple:
    flow = build_traffic_flow_sample()
    keyword_daily = build_keyword_daily_sample()
    keyword_tags = build_keyword_tag_sample()
    traffic = build_traffic_features(flow, keyword_daily, keyword_tags)
    return traffic, flow, keyword_daily, keyword_tags


def test_build_traffic_features_generates_mix_and_keyword_metrics() -> None:
    traffic, *_ = _build_traffic_features()

    current = traffic[(traffic["week"] == "2025W10") & (traffic["asin"] == "MY-ASIN-1")].iloc[0]
    assert current["ad_ratio"] == pytest.approx(0.45, rel=1e-3)
    assert current["sp_share_in_ad"] == pytest.approx(0.30 / 0.45, rel=1e-3)
    assert current["ad_to_natural"] == pytest.approx(0.45 / 0.40, rel=1e-3)
    assert current["kw_top3_share_7d_avg"] == pytest.approx(0.35 + 0.25 + 0.25, rel=1e-3)
    assert current["kw_days_covered"] == 7


def test_clean_competition_entities_derives_expected_features() -> None:
    snapshots = build_competition_snapshot_sample().drop(columns=["scene_tag", "base_scene", "morphology"])
    scene_tags = build_scene_tag_sample()
    traffic, *_ = _build_traffic_features()

    entities = clean_competition_entities(
        snapshots,
        my_asins=MY_ASINS_SAMPLE,
        scene_tags=scene_tags,
        traffic=traffic,
    )

    current = entities[(entities["week"] == "2025W10") & (entities["asin"] == "MY-ASIN-1")].iloc[0]
    assert current["hyy_asin"] == 1
    assert current["price_net"] == pytest.approx(17.991, rel=1e-3)
    assert current["content_score"] == pytest.approx(0.10, rel=1e-3)
    assert 0.0 <= current["rank_score"] <= 1.0
    assert 0.0 <= current["social_proof"] <= 1.0
    assert current["ad_ratio"] == pytest.approx(0.45, rel=1e-3)
    assert current["kw_brand_share_7d_avg"] == pytest.approx(0.35, rel=1e-3)
    assert current["kw_days_covered"] == 7


def test_build_pairs_and_deltas_respect_scoring_rules() -> None:
    snapshots = build_competition_snapshot_sample().drop(columns=["scene_tag", "base_scene", "morphology"])
    scene_tags = build_scene_tag_sample()
    rules = build_scoring_rules_sample()
    traffic, *_ = _build_traffic_features()

    entities = clean_competition_entities(
        snapshots,
        my_asins=MY_ASINS_SAMPLE,
        scene_tags=scene_tags,
        traffic=traffic,
    )
    pairs, traffic_pairs = build_competition_pairs(entities, scoring_rules=rules)
    pairs_each, traffic_pairs_each = build_competition_pairs_each(entities, scoring_rules=rules)

    assert set(pairs["opp_type"]) == {"leader", "median"}
    assert not pairs_each.empty

    leader = pairs[(pairs["week"] == "2025W10") & (pairs["opp_type"] == "leader")].iloc[0]
    leader_traffic = traffic_pairs[(traffic_pairs["week"] == "2025W10") & (traffic_pairs["opp_type"] == "leader")].iloc[0]
    assert leader["price_gap_leader"] == pytest.approx(1.50, rel=1e-3)
    assert leader["price_index_med"] == pytest.approx(19.99 / 20.49, rel=1e-3)
    assert leader["pressure"] == pytest.approx(0.847, rel=1e-3)
    assert leader["intensity_band"] == "C4"
    assert leader_traffic["ad_ratio_gap_leader"] == pytest.approx(0.45 - 0.55, rel=1e-3)
    assert leader_traffic["t_pressure"] is not None
    assert leader_traffic["t_intensity_band"] in {"C1", "C2", "C3", "C4"}

    median = pairs[(pairs["week"] == "2025W10") & (pairs["opp_type"] == "median")].iloc[0]
    median_traffic = traffic_pairs[(traffic_pairs["week"] == "2025W10") & (traffic_pairs["opp_type"] == "median")].iloc[0]
    assert median["price_gap_leader"] == pytest.approx(-0.50, rel=1e-3)
    assert median["pressure"] == pytest.approx(0.499, rel=1e-3)
    assert median["intensity_band"] == "C2"
    assert median_traffic["t_confidence"] is not None

    deltas = build_competition_delta(
        entities,
        pairs_current=pairs[pairs["week"] == "2025W10"],
        pairs_previous=pairs[pairs["week"] == "2025W09"],
        week="2025W10",
        previous_week="2025W09",
    )

    assert len(deltas) == 2
    leader_delta = deltas[deltas["opp_type"] == "leader"].iloc[0]
    assert leader_delta["d_price_gap_leader"] == pytest.approx(-1.51, rel=1e-3)
    assert leader_delta["d_price_net"] == pytest.approx(-1.959, rel=1e-3)
    assert leader_delta["badge_change"] == 1
    assert leader_delta["delta_pressure"] == pytest.approx(-0.0901, abs=5e-5)
    assert leader_delta["d_t_pressure"] is not None


def test_build_pairs_uses_yaml_defaults_when_rules_missing() -> None:
    snapshots = build_competition_snapshot_sample().drop(columns=["scene_tag", "base_scene", "morphology"])
    scene_tags = build_scene_tag_sample()
    traffic, *_ = _build_traffic_features()

    entities = clean_competition_entities(
        snapshots,
        my_asins=MY_ASINS_SAMPLE,
        scene_tags=scene_tags,
        traffic=traffic,
    )

    pairs, traffic_pairs = build_competition_pairs(entities)

    leader = pairs[(pairs["week"] == "2025W10") & (pairs["opp_type"] == "leader")].iloc[0]
    assert leader["pressure"] == pytest.approx(0.847, rel=1e-3)
    assert leader["intensity_band"] == "C4"
    leader_traffic = traffic_pairs[(traffic_pairs["week"] == "2025W10") & (traffic_pairs["opp_type"] == "leader")].iloc[0]
    assert leader_traffic["t_pressure"] is not None

    median = pairs[(pairs["week"] == "2025W10") & (pairs["opp_type"] == "median")].iloc[0]
    assert median["pressure"] == pytest.approx(0.499, rel=1e-3)
    assert median["intensity_band"] == "C2"


def test_build_tables_and_compute_competition_features() -> None:
    snapshots = build_competition_snapshot_sample().drop(columns=["scene_tag", "base_scene", "morphology"])
    scene_tags = build_scene_tag_sample()
    rules = build_scoring_rules_sample()
    traffic, *_ = _build_traffic_features()

    tables = build_competition_tables(
        snapshots,
        week="2025W10",
        previous_week="2025W09",
        my_asins=MY_ASINS_SAMPLE,
        scene_tags=scene_tags,
        scoring_rules=rules,
        traffic=traffic,
    )

    assert not tables.entities.empty
    assert not tables.traffic_entities.empty
    assert not tables.pairs.empty
    assert not tables.traffic_pairs.empty
    assert not tables.pairs_each.empty
    assert not tables.traffic_pairs_each.empty
    assert not tables.delta.empty
    assert not tables.summary.empty
    pairs_each = tables.pairs_each
    traffic_pairs_each = tables.traffic_pairs_each

    summary_row = tables.summary.iloc[0]
    assert summary_row["my_asin_cnt"] == 1
    assert summary_row["comp_cnt"] == 2
    assert summary_row["moves_coupon_up"] == 1
    assert summary_row["moves_price_down"] == 1
    assert summary_row["moves_new_video"] == 1
    assert summary_row["moves_badge_gain"] == 1
    assert summary_row["pressure_p90"] == pytest.approx(0.812, rel=1e-3)
    assert summary_row["traffic"]["lagging_pairs"] >= 0

    result = compute_competition_features(
        snapshots=snapshots,
        week="2025W10",
        previous_week="2025W09",
        my_asins=MY_ASINS_SAMPLE,
        scene_tags=scene_tags,
        scoring_rules=rules,
        traffic=traffic,
    )

    payload = result.as_dict()
    assert not payload["insufficient_data"]
    assert payload["metadata"]["previous_week"] == "2025W09"
    assert payload["metadata"]["previous_sunday"].startswith("2025-03-02")
    assert len(payload["pairs"]) == 2

    leader_pair = next(pair for pair in payload["pairs"] if pair["opp_type"] == "leader")
    assert leader_pair["current_gap"]["price_gap_leader"] == pytest.approx(1.50, rel=1e-3)
    assert leader_pair["delta_gap"]["price_gap_leader"] == pytest.approx(-1.51, rel=1e-3)
    assert leader_pair["my_change"]["price_net"] == pytest.approx(-1.959, rel=1e-3)
    assert leader_pair["my_change"]["badge_change"] == 1
    assert leader_pair["delta_pressure"] == pytest.approx(-0.0901, abs=5e-5)
    assert leader_pair["primary_competitor"]["price_gap_each"] == pytest.approx(-0.499, rel=1e-3)
    assert leader_pair["traffic"]["gap"]["mix"]["ad_ratio_gap"] == pytest.approx(0.45 - 0.55, rel=1e-3)
    assert leader_pair["traffic"]["scores"]["pressure"] is not None
    assert leader_pair["traffic"]["confidence"]["overall"] >= 0.0

    assert payload["summary"]["traffic"]["lagging_pairs"] >= 0

    summary = payload["summary"]
    assert summary["moves"]["moves_coupon_up"] == 1
    assert summary["worsen_ratio"] == pytest.approx(0.0, abs=1e-4)
    assert summary["traffic"]["pressure_p50"] is not None
    assert len(payload["top_opponents"]) == 1
    top_entry = payload["top_opponents"][0]
    assert top_entry["my_asin"] == "MY-ASIN-1"
    assert top_entry["top_competitors"][0]["opp_asin"] == leader_pair["opp_asin"]
    primary = pairs_each[(pairs_each["week"] == "2025W10") & (pairs_each["my_asin"] == "MY-ASIN-1")]
    assert not primary.empty
    leader_each = primary[primary["opp_asin"] == leader_pair["opp_asin"]].iloc[0]
    assert leader_each["price_gap_each"] == pytest.approx(-0.499, rel=1e-3)

    traffic_primary = traffic_pairs_each[
        (traffic_pairs_each["week"] == "2025W10")
        & (traffic_pairs_each["my_asin"] == "MY-ASIN-1")
        & (traffic_pairs_each["opp_asin"] == leader_pair["opp_asin"])
    ]
    assert not traffic_primary.empty
    leader_traffic_each = traffic_primary.iloc[0]
    assert leader_traffic_each["t_pressure"] is not None
