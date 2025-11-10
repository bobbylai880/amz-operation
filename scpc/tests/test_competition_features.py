import math

import pandas as pd
import pytest

from scpc.etl.competition_features import (
    build_competition_delta,
    build_competition_pairs,
    build_competition_pairs_each,
    build_competition_tables,
    build_competition_tables_from_entities,
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


def _prepare_entities() -> pd.DataFrame:
    snapshots = build_competition_snapshot_sample().drop(
        columns=["scene_tag", "base_scene", "morphology"]
    )
    scene_tags = build_scene_tag_sample()
    traffic, *_ = _build_traffic_features()
    return clean_competition_entities(
        snapshots,
        my_asins=MY_ASINS_SAMPLE,
        scene_tags=scene_tags,
        traffic=traffic,
    )


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
    rules = build_scoring_rules_sample()
    entities = _prepare_entities()
    pairs, traffic_pairs = build_competition_pairs(entities, scoring_rules=rules)
    pairs_each, traffic_pairs_each = build_competition_pairs_each(entities, scoring_rules=rules)

    assert set(pairs["opp_type"]) == {"leader", "median"}
    assert not pairs_each.empty
    assert "opp_type" in pairs_each.columns
    assert {"leader", "asin"}.issubset(set(pairs_each["opp_type"]))

    leader_each = pairs_each[
        (pairs_each["week"] == "2025W10") & (pairs_each["opp_type"] == "leader")
    ].iloc[0]

    leader = pairs[(pairs["week"] == "2025W10") & (pairs["opp_type"] == "leader")].iloc[0]
    leader_traffic = traffic_pairs[(traffic_pairs["week"] == "2025W10") & (traffic_pairs["opp_type"] == "leader")].iloc[0]
    assert leader["price_gap_leader"] == pytest.approx(1.50, rel=1e-3)
    assert leader["price_index_med"] == pytest.approx(19.99 / 20.49, rel=1e-3)
    assert leader["pressure"] == pytest.approx(0.847, rel=1e-3)
    assert leader["intensity_band"] == "C4"
    assert leader_traffic["ad_ratio_gap_leader"] == pytest.approx(0.45 - 0.55, rel=1e-3)
    assert leader_traffic["t_pressure"] is not None
    assert leader_traffic["t_intensity_band"] in {"C1", "C2", "C3", "C4"}
    assert leader_each["opp_asin"] == leader["opp_asin"]

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


def test_leader_selection_prefers_lowest_rank_leaf() -> None:
    entities = _prepare_entities()
    pairs, _ = build_competition_pairs(entities)
    leader = pairs[(pairs["week"] == "2025W10") & (pairs["opp_type"] == "leader")].iloc[0]
    assert leader["opp_asin"] == "COMP-ASIN-1"


def test_leader_selection_falls_back_to_rank_score_when_rank_leaf_missing() -> None:
    entities = _prepare_entities().copy()
    mask = (entities["hyy_asin"] == 0) & (entities["week"] == "2025W10")
    entities.loc[mask, "rank_leaf"] = math.nan
    entities.loc[mask & (entities["asin"] == "COMP-ASIN-1"), "rank_score"] = 0.9
    entities.loc[mask & (entities["asin"] == "COMP-ASIN-2"), "rank_score"] = 0.2

    pairs, _ = build_competition_pairs(entities)
    leader = pairs[(pairs["week"] == "2025W10") & (pairs["opp_type"] == "leader")].iloc[0]
    assert leader["opp_asin"] == "COMP-ASIN-1"


def test_leader_selection_falls_back_to_rank_pos_pct_when_available() -> None:
    entities = _prepare_entities().copy()
    mask = (entities["hyy_asin"] == 0) & (entities["week"] == "2025W10")
    entities.loc[mask, "rank_leaf"] = math.nan
    entities.loc[mask, "rank_score"] = math.nan
    entities.loc[mask & (entities["asin"] == "COMP-ASIN-1"), "rank_pos_pct"] = 0.72
    entities.loc[mask & (entities["asin"] == "COMP-ASIN-2"), "rank_pos_pct"] = 0.54

    pairs, _ = build_competition_pairs(entities)
    leader = pairs[(pairs["week"] == "2025W10") & (pairs["opp_type"] == "leader")].iloc[0]
    assert leader["opp_asin"] == "COMP-ASIN-1"


def test_leader_selection_falls_back_to_rank_root_when_scores_missing() -> None:
    entities = _prepare_entities().copy()
    mask = (entities["hyy_asin"] == 0) & (entities["week"] == "2025W10")
    entities.loc[mask, "rank_leaf"] = math.nan
    entities.loc[mask, "rank_score"] = math.nan

    pairs, _ = build_competition_pairs(entities)
    leader = pairs[(pairs["week"] == "2025W10") & (pairs["opp_type"] == "leader")].iloc[0]
    assert leader["opp_asin"] == "COMP-ASIN-1"


def test_leader_selection_uses_tie_breaker_rating_when_rank_equal() -> None:
    entities = _prepare_entities().copy()
    mask = (entities["hyy_asin"] == 0) & (entities["week"] == "2025W10")
    entities.loc[mask, "rank_leaf"] = 40
    entities.loc[mask, "rank_score"] = 0.5
    entities.loc[mask & (entities["asin"] == "COMP-ASIN-1"), "rating"] = 4.6
    entities.loc[mask & (entities["asin"] == "COMP-ASIN-2"), "rating"] = 4.9

    pairs, _ = build_competition_pairs(entities)
    leader = pairs[(pairs["week"] == "2025W10") & (pairs["opp_type"] == "leader")].iloc[0]
    assert leader["opp_asin"] == "COMP-ASIN-2"


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


def test_build_tables_from_entities_matches_snapshot_pipeline() -> None:
    snapshots = build_competition_snapshot_sample().drop(columns=["scene_tag", "base_scene", "morphology"])
    scene_tags = build_scene_tag_sample()
    rules = build_scoring_rules_sample()
    traffic, *_ = _build_traffic_features()

    entities_full = clean_competition_entities(
        snapshots,
        my_asins=MY_ASINS_SAMPLE,
        scene_tags=scene_tags,
        traffic=traffic,
    )

    tables_from_entities = build_competition_tables_from_entities(
        entities_full,
        week="2025W10",
        previous_week="2025W09",
        scoring_rules=rules,
    )

    tables_from_snapshots = build_competition_tables(
        snapshots,
        week="2025W10",
        previous_week="2025W09",
        my_asins=MY_ASINS_SAMPLE,
        scene_tags=scene_tags,
        scoring_rules=rules,
        traffic=traffic,
    )

    assert len(tables_from_entities.pairs) == len(tables_from_snapshots.pairs)

    leader_entity = tables_from_entities.pairs[
        (tables_from_entities.pairs["week"] == "2025W10")
        & (tables_from_entities.pairs["opp_type"] == "leader")
    ].iloc[0]
    leader_snapshot = tables_from_snapshots.pairs[
        (tables_from_snapshots.pairs["week"] == "2025W10")
        & (tables_from_snapshots.pairs["opp_type"] == "leader")
    ].iloc[0]

    assert leader_entity["price_gap_leader"] == pytest.approx(leader_snapshot["price_gap_leader"], rel=1e-6)
    assert leader_entity["pressure"] == pytest.approx(leader_snapshot["pressure"], rel=1e-6)

    assert len(tables_from_entities.delta) == len(tables_from_snapshots.delta)
    entity_delta = tables_from_entities.delta[
        tables_from_entities.delta["opp_type"] == "leader"
    ].iloc[0]
    snapshot_delta = tables_from_snapshots.delta[
        tables_from_snapshots.delta["opp_type"] == "leader"
    ].iloc[0]
    assert entity_delta["d_price_gap_leader"] == pytest.approx(
        snapshot_delta["d_price_gap_leader"], rel=1e-6
    )

    entity_summary = tables_from_entities.summary.iloc[0]
    snapshot_summary = tables_from_snapshots.summary.iloc[0]
    assert entity_summary["pressure_p90"] == pytest.approx(snapshot_summary["pressure_p90"], rel=1e-6)

def test_competition_pipeline_two_level_judgement() -> None:
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

    result = compute_competition_features(
        entities=tables.entities,
        traffic_entities=tables.traffic_entities,
        pairs=tables.pairs,
        traffic_pairs=tables.traffic_pairs,
        pairs_each=tables.pairs_each,
        traffic_pairs_each=tables.traffic_pairs_each,
        deltas=tables.delta,
        week="2025W10",
        previous_week="2025W09",
    )

    payload = result.as_dict()
    assert len(payload["pairs"]) == 2

    leader_pair = next(pair for pair in payload["pairs"] if pair["opp_type"] == "leader")
    assert leader_pair["current_gap"]["price_gap_leader"] is not None
    assert leader_pair["score_components"]["score_price"] is not None
    assert leader_pair["traffic"]["gap"]["mix"]["ad_ratio_gap"] is not None
    assert leader_pair["traffic"]["scores"]["pressure"] is not None
    assert leader_pair["traffic"]["confidence"]["overall"] is not None
    assert leader_pair["my_snapshot"]["ad_ratio"] is not None
    assert leader_pair["opp_snapshot"]["ad_ratio"] is not None
    assert leader_pair["primary_competitor"]["traffic_scores"]["pressure"] is not None

    summary = payload["summary"]
    assert summary["avg_scores"]["score_price"] is not None
    assert summary["traffic"]["lagging_pairs"] >= 0
    assert payload["top_opponents"]
