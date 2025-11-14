import json
from pathlib import Path

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine, text

from scpc.llm.competition_config import load_competition_llm_config
from scpc.llm.competition_workflow import CompetitionLLMOrchestrator

from .test_competition_llm_workflow import StubLLM


@pytest.fixture()
def stage3_engine():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE vw_amz_comp_llm_overview (
                    scene_tag TEXT,
                    base_scene TEXT,
                    morphology TEXT,
                    marketplace_id TEXT,
                    week TEXT,
                    sunday TEXT,
                    my_parent_asin TEXT,
                    my_asin TEXT,
                    opp_type TEXT,
                    asin_priority INTEGER,
                    price_net REAL,
                    rank_score REAL,
                    rank_pos_pct REAL,
                    content_score REAL,
                    social_proof REAL,
                    price_gap_leader REAL,
                    price_index_med REAL,
                    content_gap REAL,
                    social_gap REAL,
                    badge_delta_sum REAL,
                    pressure REAL,
                    confidence REAL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE vw_amz_comp_llm_overview_traffic (
                    scene_tag TEXT,
                    base_scene TEXT,
                    morphology TEXT,
                    marketplace_id TEXT,
                    week TEXT,
                    sunday TEXT,
                    my_parent_asin TEXT,
                    my_asin TEXT,
                    opp_type TEXT,
                    asin_priority INTEGER,
                    traffic_gap REAL,
                    ad_ratio_index_med REAL,
                    ad_to_natural_gap REAL,
                    sp_share_in_ad_gap REAL,
                    kw_top3_share_gap REAL,
                    kw_brand_share_gap REAL,
                    kw_competitor_share_gap REAL,
                    t_confidence REAL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE bi_amz_comp_llm_packet (
                    scene_tag TEXT,
                    base_scene TEXT,
                    morphology TEXT,
                    marketplace_id TEXT,
                    week TEXT,
                    sunday TEXT,
                    my_asin TEXT,
                    lag_type TEXT,
                    opp_type TEXT,
                    evidence_json TEXT
                )
                """
            )
        )
    try:
        yield engine
    finally:
        engine.dispose()


def _create_orchestrator(engine, tmp_path):
    config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    return CompetitionLLMOrchestrator(
        engine=engine,
        llm_orchestrator=StubLLM(),
        config=config,
        storage_root=tmp_path,
    )


def test_classify_delta_direction(stage3_engine, tmp_path):
    orchestrator = _create_orchestrator(stage3_engine, tmp_path)
    result_higher = orchestrator._classify_delta(
        "rank_score",
        "rank",
        0.6,
        0.4,
        tolerance=0.0,
    )
    assert result_higher["direction"] == "improve"
    assert pytest.approx(result_higher["delta"], rel=1e-6) == 0.2

    result_lower = orchestrator._classify_delta(
        "price_net",
        "price",
        19.0,
        21.0,
        tolerance=0.0,
    )
    assert result_lower["direction"] == "improve"

    result_unknown = orchestrator._classify_delta(
        "rank_score",
        "rank",
        None,
        0.5,
        tolerance=0.0,
    )
    assert result_unknown["direction"] == "unknown"


def test_build_stage3_leader_index(stage3_engine, tmp_path):
    orchestrator = _create_orchestrator(stage3_engine, tmp_path)
    packets = [
        {
            "scene_tag": "SCN",
            "base_scene": "Kitchen",
            "morphology": "standard",
            "marketplace_id": "US",
            "week": "2025-W02",
            "my_asin": "ASIN1",
            "lag_type": "pricing",
            "evidence": {
                "top_competitors": [
                    {"asin": "LEADER-B", "rank_leaf": 2},
                    {"asin": "LEADER-A", "rank_leaf": 1},
                ]
            },
        },
        {
            "scene_tag": "SCN",
            "base_scene": "Kitchen",
            "morphology": "standard",
            "marketplace_id": "US",
            "week": "2025-W01",
            "my_asin": "ASIN1",
            "lag_type": "pricing",
            "evidence": {
                "top_competitors": [
                    {"asin": "LEADER-OLD", "rank_root": 1},
                    {"asin": "OTHER", "rank_leaf": 3},
                ]
            },
        },
    ]
    index = orchestrator._build_stage3_leader_index(packets)
    assert index[("SCN", "Kitchen", "standard", "US", "ASIN1", "2025-W02")] == "LEADER-A"
    assert index[("SCN", "Kitchen", "standard", "US", "ASIN1", "2025-W01")] == "LEADER-OLD"


def test_run_stage3_generates_results(stage3_engine, tmp_path):
    orchestrator = _create_orchestrator(stage3_engine, tmp_path)
    context = {
        "scene_tag": "SCN-STAGE3",
        "base_scene": "Kitchen",
        "morphology": "standard",
        "marketplace_id": "US",
        "my_parent_asin": "PARENT1",
        "my_asin": "ASIN123",
    }

    def insert_overview(week: str, sunday: str, opp_type: str, asin_priority: int, values: dict):
        payload = dict(context)
        payload.update({"week": week, "sunday": sunday, "opp_type": opp_type, "asin_priority": asin_priority})
        payload.update(values)
        with stage3_engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO vw_amz_comp_llm_overview (
                        scene_tag, base_scene, morphology, marketplace_id, week, sunday,
                        my_parent_asin, my_asin, opp_type, asin_priority,
                        price_net, rank_score, rank_pos_pct, content_score, social_proof,
                        price_gap_leader, price_index_med, content_gap, social_gap,
                        badge_delta_sum, pressure, confidence
                    ) VALUES (
                        :scene_tag, :base_scene, :morphology, :marketplace_id, :week, :sunday,
                        :my_parent_asin, :my_asin, :opp_type, :asin_priority,
                        :price_net, :rank_score, :rank_pos_pct, :content_score, :social_proof,
                        :price_gap_leader, :price_index_med, :content_gap, :social_gap,
                        :badge_delta_sum, :pressure, :confidence
                    )
                    """
                ),
                payload,
            )

    def insert_traffic(week: str, sunday: str, opp_type: str, asin_priority: int, values: dict):
        payload = dict(context)
        payload.update({"week": week, "sunday": sunday, "opp_type": opp_type, "asin_priority": asin_priority})
        payload.update(values)
        with stage3_engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO vw_amz_comp_llm_overview_traffic (
                        scene_tag, base_scene, morphology, marketplace_id, week, sunday,
                        my_parent_asin, my_asin, opp_type, asin_priority,
                        traffic_gap, ad_ratio_index_med, ad_to_natural_gap, sp_share_in_ad_gap,
                        kw_top3_share_gap, kw_brand_share_gap, kw_competitor_share_gap, t_confidence
                    ) VALUES (
                        :scene_tag, :base_scene, :morphology, :marketplace_id, :week, :sunday,
                        :my_parent_asin, :my_asin, :opp_type, :asin_priority,
                        :traffic_gap, :ad_ratio_index_med, :ad_to_natural_gap, :sp_share_in_ad_gap,
                        :kw_top3_share_gap, :kw_brand_share_gap, :kw_competitor_share_gap, :t_confidence
                    )
                    """
                ),
                payload,
            )

    def insert_packet(week: str, sunday: str, lag_type: str, top_comp: list[dict]):
        with stage3_engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO bi_amz_comp_llm_packet (
                        scene_tag, base_scene, morphology, marketplace_id,
                        week, sunday, my_asin, lag_type, opp_type, evidence_json
                    ) VALUES (
                        :scene_tag, :base_scene, :morphology, :marketplace_id,
                        :week, :sunday, :my_asin, :lag_type, :opp_type, :evidence_json
                    )
                    """
                ),
                {
                    "scene_tag": context["scene_tag"],
                    "base_scene": context["base_scene"],
                    "morphology": context["morphology"],
                    "marketplace_id": context["marketplace_id"],
                    "week": week,
                    "sunday": sunday,
                    "my_asin": context["my_asin"],
                    "lag_type": lag_type,
                    "opp_type": "page",
                    "evidence_json": json.dumps({"top_competitors": top_comp}),
                },
            )

    insert_overview(
        "2025-W01",
        "2025-01-05",
        "self",
        1,
        {
            "price_net": 25.0,
            "rank_score": 0.42,
            "rank_pos_pct": 0.62,
            "content_score": 0.55,
            "social_proof": 80,
            "price_gap_leader": 1.2,
            "price_index_med": 1.05,
            "content_gap": 0.18,
            "social_gap": 0.12,
            "badge_delta_sum": 0.0,
            "pressure": 0.32,
            "confidence": 0.9,
        },
    )
    insert_overview(
        "2025-W01",
        "2025-01-05",
        "leader",
        2,
        {
            "price_net": 19.5,
            "rank_score": 0.65,
            "rank_pos_pct": 0.38,
            "content_score": 0.68,
            "social_proof": 160,
            "price_gap_leader": 0.0,
            "price_index_med": 0.98,
            "content_gap": -0.05,
            "social_gap": -0.08,
            "badge_delta_sum": 0.0,
            "pressure": 0.20,
            "confidence": 0.9,
        },
    )
    insert_overview(
        "2025-W02",
        "2025-01-12",
        "self",
        1,
        {
            "price_net": 23.0,
            "rank_score": 0.52,
            "rank_pos_pct": 0.48,
            "content_score": 0.62,
            "social_proof": 120,
            "price_gap_leader": 0.6,
            "price_index_med": 1.02,
            "content_gap": 0.10,
            "social_gap": 0.07,
            "badge_delta_sum": 1.0,
            "pressure": 0.26,
            "confidence": 0.92,
        },
    )
    insert_overview(
        "2025-W02",
        "2025-01-12",
        "leader",
        2,
        {
            "price_net": 19.0,
            "rank_score": 0.70,
            "rank_pos_pct": 0.34,
            "content_score": 0.70,
            "social_proof": 175,
            "price_gap_leader": 0.0,
            "price_index_med": 0.96,
            "content_gap": -0.08,
            "social_gap": -0.10,
            "badge_delta_sum": 0.0,
            "pressure": 0.18,
            "confidence": 0.93,
        },
    )

    insert_traffic(
        "2025-W01",
        "2025-01-05",
        "self",
        1,
        {
            "traffic_gap": -0.08,
            "ad_ratio_index_med": 0.85,
            "ad_to_natural_gap": -0.12,
            "sp_share_in_ad_gap": -0.05,
            "kw_top3_share_gap": -0.07,
            "kw_brand_share_gap": -0.03,
            "kw_competitor_share_gap": 0.04,
            "t_confidence": 0.8,
        },
    )
    insert_traffic(
        "2025-W01",
        "2025-01-05",
        "leader",
        2,
        {
            "traffic_gap": 0.10,
            "ad_ratio_index_med": 1.05,
            "ad_to_natural_gap": 0.15,
            "sp_share_in_ad_gap": 0.02,
            "kw_top3_share_gap": 0.08,
            "kw_brand_share_gap": 0.03,
            "kw_competitor_share_gap": -0.02,
            "t_confidence": 0.82,
        },
    )
    insert_traffic(
        "2025-W02",
        "2025-01-12",
        "self",
        1,
        {
            "traffic_gap": -0.02,
            "ad_ratio_index_med": 0.92,
            "ad_to_natural_gap": -0.08,
            "sp_share_in_ad_gap": -0.03,
            "kw_top3_share_gap": -0.04,
            "kw_brand_share_gap": -0.01,
            "kw_competitor_share_gap": 0.02,
            "t_confidence": 0.85,
        },
    )
    insert_traffic(
        "2025-W02",
        "2025-01-12",
        "leader",
        2,
        {
            "traffic_gap": 0.12,
            "ad_ratio_index_med": 1.08,
            "ad_to_natural_gap": 0.18,
            "sp_share_in_ad_gap": 0.03,
            "kw_top3_share_gap": 0.09,
            "kw_brand_share_gap": 0.04,
            "kw_competitor_share_gap": -0.03,
            "t_confidence": 0.88,
        },
    )

    insert_packet(
        "2025-W01",
        "2025-01-05",
        "pricing",
        [{"asin": "LEADER-OLD", "rank_leaf": 1}],
    )
    insert_packet(
        "2025-W02",
        "2025-01-12",
        "pricing",
        [
            {"asin": "LEADER-NEW", "rank_leaf": 1},
            {"asin": "LEADER-ALT", "rank_leaf": 2},
        ],
    )

    results = orchestrator.run_stage3(None, marketplace_id="US")
    assert len(results) == 1

    summary = orchestrator.stage3_last_summary
    assert summary.reason is None
    assert summary.week_w0 == "2025-W02"
    assert summary.week_w1 == "2025-W01"
    assert summary.scene_count == 1
    assert summary.record_count > 0

    result = results[0]
    assert result.context["scene_tag"] == context["scene_tag"]
    assert len(result.self_entities) == 2
    assert len(result.leader_entities) == 2
    assert result.gap_deltas
    assert result.dimensions

    page_gap = next(
        gap for gap in result.gap_deltas if gap.channel == "page"
    )
    price_gap = page_gap.gap_deltas["price_index_med"]
    assert price_gap["direction"] == "improve"
    assert price_gap["lag_type"] == "price"

    leader_page = next(
        entity for entity in result.leader_entities if entity.channel == "page"
    )
    assert leader_page.leader_changed is True
    assert leader_page.entity_asin == "LEADER-NEW"

    dimension = next(dim for dim in result.dimensions if dim.lag_type == "price")
    assert dimension.aggregates["total_entities"] >= 1
    assert len(dimension.top_changes) <= orchestrator._config.stage_3.top_n_per_dimension

    stage3_file = tmp_path / "2025-W02" / "stage3" / "SCN-STAGE3_Kitchen_standard.json"
    assert stage3_file.exists()
