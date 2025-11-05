from __future__ import annotations

import pytest

from scpc.etl.competition_features import (
    build_competition_delta,
    build_competition_pairs,
    build_competition_tables,
    clean_competition_entities,
    compute_competition_features,
)
from scpc.tests.data import (
    MY_ASINS_SAMPLE,
    build_competition_snapshot_sample,
    build_scene_tag_sample,
    build_scoring_rules_sample,
)


def test_clean_competition_entities_derives_expected_features() -> None:
    snapshots = build_competition_snapshot_sample().drop(columns=["scene_tag", "base_scene", "morphology"])
    scene_tags = build_scene_tag_sample()

    entities = clean_competition_entities(
        snapshots,
        my_asins=MY_ASINS_SAMPLE,
        scene_tags=scene_tags,
    )

    current = entities[(entities["week"] == "2025W10") & (entities["asin"] == "MY-ASIN-1")].iloc[0]
    assert current["hyy_asin"] == 1
    assert current["price_net"] == pytest.approx(17.991, rel=1e-3)
    assert current["content_score"] == pytest.approx(0.10, rel=1e-3)
    assert 0.0 <= current["rank_score"] <= 1.0
    assert 0.0 <= current["social_proof"] <= 1.0


def test_build_pairs_and_deltas_respect_scoring_rules() -> None:
    snapshots = build_competition_snapshot_sample().drop(columns=["scene_tag", "base_scene", "morphology"])
    scene_tags = build_scene_tag_sample()
    rules = build_scoring_rules_sample()

    entities = clean_competition_entities(
        snapshots,
        my_asins=MY_ASINS_SAMPLE,
        scene_tags=scene_tags,
    )
    pairs = build_competition_pairs(entities, scoring_rules=rules)

    assert set(pairs["opp_type"]) == {"leader", "median"}

    leader = pairs[(pairs["week"] == "2025W10") & (pairs["opp_type"] == "leader")].iloc[0]
    assert leader["price_gap_leader"] == pytest.approx(1.50, rel=1e-3)
    assert leader["price_index_med"] == pytest.approx(19.99 / 20.49, rel=1e-3)
    assert leader["pressure"] == pytest.approx(0.847, rel=1e-3)
    assert leader["intensity_band"] == "C4"

    median = pairs[(pairs["week"] == "2025W10") & (pairs["opp_type"] == "median")].iloc[0]
    assert median["price_gap_leader"] == pytest.approx(-0.50, rel=1e-3)
    assert median["pressure"] == pytest.approx(0.499, rel=1e-3)
    assert median["intensity_band"] == "C2"

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


def test_build_tables_and_compute_competition_features() -> None:
    snapshots = build_competition_snapshot_sample().drop(columns=["scene_tag", "base_scene", "morphology"])
    scene_tags = build_scene_tag_sample()
    rules = build_scoring_rules_sample()

    tables = build_competition_tables(
        snapshots,
        week="2025W10",
        previous_week="2025W09",
        my_asins=MY_ASINS_SAMPLE,
        scene_tags=scene_tags,
        scoring_rules=rules,
    )

    assert not tables.entities.empty
    assert not tables.pairs.empty
    assert not tables.delta.empty
    assert not tables.summary.empty

    summary_row = tables.summary.iloc[0]
    assert summary_row["my_asin_cnt"] == 1
    assert summary_row["comp_cnt"] == 2
    assert summary_row["moves_coupon_up"] == 1
    assert summary_row["moves_price_down"] == 1
    assert summary_row["moves_new_video"] == 1
    assert summary_row["moves_badge_gain"] == 1
    assert summary_row["avg_score_price"] == pytest.approx(0.613, rel=1e-3)
    assert summary_row["pressure_p90"] == pytest.approx(0.812, rel=1e-3)

    result = compute_competition_features(
        snapshots=snapshots,
        week="2025W10",
        previous_week="2025W09",
        my_asins=MY_ASINS_SAMPLE,
        scene_tags=scene_tags,
        scoring_rules=rules,
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

    summary = payload["summary"]
    assert summary["moves"]["moves_coupon_up"] == 1
    assert summary["avg_scores"]["score_price"] == pytest.approx(0.613, rel=1e-3)
    assert summary["worsen_ratio"] == pytest.approx(0.0, abs=1e-4)
