import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine, text

from scpc.llm.competition_config import load_competition_llm_config
from scpc.llm.competition_workflow import (
    CompetitionLLMOrchestrator,
    StageOneLLMResult,
)


class StubLLM:
    def __init__(self) -> None:
        self.stage1_requests = []
        self.stage1_overrides: dict[str, dict] = {}

    def set_stage1_override(self, asin: str, response: dict) -> None:
        self.stage1_overrides[asin] = response

    def run(self, config, *, retry=True, fallback=None):  # noqa: D401 - signature mirrors orchestrator
        facts = config.facts
        self.stage1_requests.append(facts)
        context = facts["context"]
        asin = context.get("my_asin")
        if asin in self.stage1_overrides:
            override = self.stage1_overrides[asin]
            override.setdefault("context", context)
            return override
        return {
            "context": context,
            "summary": "整体诊断：该 ASIN 在价格维度落后，需要关注定价策略。",
            "dimensions": [
                {
                    "lag_type": "pricing",
                    "status": "lag",
                    "severity": "high",
                    "source_opp_type": "page",
                    "source_confidence": 0.8,
                    "notes": "价格差距扩大到 12%，我方明显劣势。",
                }
            ],
        }


@pytest.fixture()
def sqlite_engine():
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
                    price_gap REAL,
                    price_gap_leader REAL,
                    price_index_med REAL,
                    price_z REAL,
                    rank_pos_pct REAL,
                    content_gap REAL,
                    social_gap REAL,
                    badge_delta_sum REAL,
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
        conn.execute(
            text(
                """
                CREATE TABLE bi_amz_comp_lag_insights (
                    scene_tag TEXT,
                    base_scene TEXT,
                    morphology TEXT,
                    marketplace_id TEXT,
                    week TEXT,
                    sunday TEXT,
                    my_asin TEXT,
                    lag_type TEXT,
                    opp_type TEXT,
                    reason_code TEXT,
                    severity TEXT,
                    reason_detail TEXT,
                    top_opp_asins_csv TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE bi_amz_comp_pairs (
                    scene_tag TEXT,
                    base_scene TEXT,
                    morphology TEXT,
                    marketplace_id TEXT,
                    week TEXT,
                    sunday TEXT,
                    my_parent_asin TEXT,
                    my_asin TEXT,
                    opp_type TEXT,
                    opp_asin TEXT,
                    price_gap_leader REAL,
                    price_index_med REAL,
                    price_z REAL,
                    rank_pos_pct REAL,
                    content_gap REAL,
                    social_gap REAL,
                    badge_delta_sum REAL,
                    badge_diff TEXT,
                    pressure REAL,
                    intensity_band TEXT,
                    confidence REAL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE bi_amz_comp_delta (
                    scene_tag TEXT,
                    base_scene TEXT,
                    morphology TEXT,
                    marketplace_id TEXT,
                    window_id TEXT,
                    my_asin TEXT,
                    opp_type TEXT,
                    week_w0 TEXT,
                    week_w1 TEXT,
                    my_parent_asin TEXT,
                    d_price_net REAL,
                    d_rank_score REAL,
                    d_social_proof REAL,
                    d_content_score REAL,
                    badge_change REAL,
                    d_price_gap_leader REAL,
                    d_price_index_med REAL,
                    d_rank_pos_pct REAL,
                    d_content_gap REAL,
                    d_social_gap REAL,
                    delta_pressure REAL
                )
                """
            )
        )
    return engine


def test_competition_workflow_end_to_end(sqlite_engine, tmp_path):
    evidence = {
        "context": {
            "my": {"brand": "OurBrand", "asin": "B012345"},
            "ref": {"brand": "CompBrand", "asin": "B099999"},
        },
        "metrics": {"price_gap_pct": 0.12},
        "drivers": [{"name": "price_gap_pct", "value": 0.12}],
        "top_competitors": [{"brand": "CompBrand", "asin": "B099999"}],
    }
    with sqlite_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO vw_amz_comp_llm_overview
                (scene_tag, base_scene, morphology, marketplace_id, week, sunday, my_parent_asin, my_asin, opp_type, asin_priority, price_gap, confidence)
                VALUES (:scene_tag, :base_scene, :morphology, :marketplace_id, :week, :sunday, :my_parent_asin, :my_asin, :opp_type, :asin_priority, :price_gap, :confidence)
                """
            ),
            {
                "scene_tag": "SCN-1",
                "base_scene": "base",
                "morphology": "standard",
                "marketplace_id": "US",
                "week": "2025-W01",
                "sunday": "2025-01-05",
                "my_parent_asin": "PARENT1",
                "my_asin": "B012345",
                "opp_type": "page",
                "asin_priority": 1,
                "price_gap": 0.12,
                "confidence": 0.82,
            },
        )
        conn.execute(
            text(
                """
                UPDATE vw_amz_comp_llm_overview
                SET price_gap_leader=:price_gap_leader,
                    price_index_med=:price_index_med,
                    price_z=:price_z,
                    rank_pos_pct=:rank_pos_pct,
                    content_gap=:content_gap,
                    social_gap=:social_gap,
                    badge_delta_sum=:badge_delta_sum
                WHERE my_asin=:my_asin AND week=:week
                """
            ),
            {
                "price_gap_leader": 2.0,
                "price_index_med": 1.2,
                "price_z": 0.65,
                "rank_pos_pct": 0.45,
                "content_gap": -0.1,
                "social_gap": -0.05,
                "badge_delta_sum": 1.0,
                "my_asin": "B012345",
                "week": "2025-W01",
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO vw_amz_comp_llm_overview_traffic
                (scene_tag, base_scene, morphology, marketplace_id, week, sunday, my_parent_asin, my_asin, opp_type, asin_priority, traffic_gap, t_confidence)
                VALUES (:scene_tag, :base_scene, :morphology, :marketplace_id, :week, :sunday, :my_parent_asin, :my_asin, :opp_type, :asin_priority, :traffic_gap, :t_confidence)
                """
            ),
            {
                "scene_tag": "SCN-1",
                "base_scene": "base",
                "morphology": "standard",
                "marketplace_id": "US",
                "week": "2025-W01",
                "sunday": "2025-01-05",
                "my_parent_asin": "PARENT1",
                "my_asin": "B012345",
                "opp_type": "page",
                "asin_priority": 1,
                "traffic_gap": 0.05,
                "t_confidence": 0.76,
            },
        )
        conn.execute(
            text(
                """
                UPDATE vw_amz_comp_llm_overview_traffic
                SET ad_ratio_index_med=:ad_ratio_index_med,
                    ad_to_natural_gap=:ad_to_natural_gap,
                    sp_share_in_ad_gap=:sp_share_in_ad_gap,
                    kw_top3_share_gap=:kw_top3_share_gap,
                    kw_brand_share_gap=:kw_brand_share_gap,
                    kw_competitor_share_gap=:kw_competitor_share_gap
                WHERE my_asin=:my_asin AND week=:week
                """
            ),
            {
                "ad_ratio_index_med": 0.95,
                "ad_to_natural_gap": -0.1,
                "sp_share_in_ad_gap": -0.05,
                "kw_top3_share_gap": -0.05,
                "kw_brand_share_gap": -0.02,
                "kw_competitor_share_gap": 0.03,
                "my_asin": "B012345",
                "week": "2025-W01",
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO bi_amz_comp_llm_packet
                (scene_tag, base_scene, morphology, marketplace_id, week, sunday, my_asin, lag_type, opp_type, evidence_json)
                VALUES (:scene_tag, :base_scene, :morphology, :marketplace_id, :week, :sunday, :my_asin, :lag_type, :opp_type, :evidence_json)
                """
            ),
            {
                "scene_tag": "SCN-1",
                "base_scene": "base",
                "morphology": "standard",
                "marketplace_id": "US",
                "week": "2025-W01",
                "sunday": "2025-01-05",
                "my_asin": "B012345",
                "lag_type": "pricing",
                "opp_type": "page",
                "evidence_json": json.dumps(evidence),
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO bi_amz_comp_lag_insights
                (scene_tag, base_scene, morphology, marketplace_id, week, sunday, my_asin, lag_type, opp_type, reason_code, severity, reason_detail, top_opp_asins_csv)
                VALUES (:scene_tag, :base_scene, :morphology, :marketplace_id, :week, :sunday, :my_asin, :lag_type, :opp_type, :reason_code, :severity, :reason_detail, :top_opp_asins_csv)
                """
            ),
            {
                "scene_tag": "SCN-1",
                "base_scene": "base",
                "morphology": "standard",
                "marketplace_id": "US",
                "week": "2025-W01",
                "sunday": "2025-01-05",
                "my_asin": "B012345",
                "lag_type": "pricing",
                "opp_type": "page",
                "reason_code": "pricing_misalignment",
                "severity": "high",
                "reason_detail": "价差扩大",
                "top_opp_asins_csv": "B099999",
            },
        )
    config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    stub_llm = StubLLM()
    orchestrator = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=stub_llm,
        config=config,
        storage_root=tmp_path,
    )

    result = orchestrator.run("2025-W01", marketplace_id="US")

    assert result.stage1_processed == 1

    stage1_file = tmp_path / "2025-W01" / "stage1" / "B012345_page.json"
    assert stage1_file.exists()

    stage1_payload = json.loads(stage1_file.read_text(encoding="utf-8"))
    assert stage1_payload["context"]["my_asin"] == "B012345"
    assert stage1_payload["dimensions"][0]["status"] == "lag"

    assert len(stub_llm.stage1_requests) == 1


def test_stage1_missing_context_fields_raise(sqlite_engine, tmp_path):
    config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    orchestrator = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=StubLLM(),
        config=config,
        storage_root=tmp_path,
    )

    with sqlite_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO vw_amz_comp_llm_overview
                (scene_tag, base_scene, morphology, marketplace_id, week, sunday, my_parent_asin, my_asin, opp_type, asin_priority, price_gap, confidence)
                VALUES (:scene_tag, :base_scene, :morphology, :marketplace_id, :week, :sunday, :my_parent_asin, :my_asin, :opp_type, :asin_priority, :price_gap, :confidence)
                """
            ),
            {
                "scene_tag": "SCN-ERR",
                "base_scene": "base",
                "morphology": "standard",
                "marketplace_id": "US",
                "week": "2025-W05",
                "sunday": "2025-02-02",
                "my_parent_asin": None,
                "my_asin": "B0MISSING",
                "opp_type": "leader",
                "asin_priority": 1,
                "price_gap": 0.2,
                "confidence": 0.7,
            },
        )

    with pytest.raises(ValueError, match="Stage-1 context missing required fields"):
        orchestrator.run("2025-W05", marketplace_id="US")










def test_stage1_outputs_summary_for_leading_dimensions(sqlite_engine, tmp_path):
    config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    stub_llm = StubLLM()
    orchestrator = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=stub_llm,
        config=config,
        storage_root=tmp_path,
    )

    with sqlite_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO vw_amz_comp_llm_overview
                (scene_tag, base_scene, morphology, marketplace_id, week, sunday, my_parent_asin, my_asin, opp_type, asin_priority, price_gap, confidence)
                VALUES (:scene_tag, :base_scene, :morphology, :marketplace_id, :week, :sunday, :my_parent_asin, :my_asin, :opp_type, :asin_priority, :price_gap, :confidence)
                """
            ),
            {
                "scene_tag": "SCN-LEAD",
                "base_scene": "base",
                "morphology": "standard",
                "marketplace_id": "US",
                "week": "2025-W03",
                "sunday": "2025-01-19",
                "my_parent_asin": "PARENT2",
                "my_asin": "B0LEADASIN",
                "opp_type": "leader",
                "asin_priority": 1,
                "price_gap": -0.05,
                "confidence": 0.9,
            },
        )
        conn.execute(
            text(
                """
                UPDATE vw_amz_comp_llm_overview
                SET price_gap_leader=:price_gap_leader,
                    price_index_med=:price_index_med,
                    price_z=:price_z,
                    rank_pos_pct=:rank_pos_pct,
                    content_gap=:content_gap,
                    social_gap=:social_gap,
                    badge_delta_sum=:badge_delta_sum
                WHERE my_asin=:my_asin AND week=:week
                """
            ),
            {
                "price_gap_leader": -1.5,
                "price_index_med": 0.95,
                "price_z": -0.4,
                "rank_pos_pct": 0.3,
                "content_gap": 0.2,
                "social_gap": 0.1,
                "badge_delta_sum": 1.0,
                "my_asin": "B0LEADASIN",
                "week": "2025-W03",
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO vw_amz_comp_llm_overview_traffic
                (scene_tag, base_scene, morphology, marketplace_id, week, sunday, my_parent_asin, my_asin, opp_type, asin_priority, traffic_gap, t_confidence)
                VALUES (:scene_tag, :base_scene, :morphology, :marketplace_id, :week, :sunday, :my_parent_asin, :my_asin, :opp_type, :asin_priority, :traffic_gap, :t_confidence)
                """
            ),
            {
                "scene_tag": "SCN-LEAD",
                "base_scene": "base",
                "morphology": "standard",
                "marketplace_id": "US",
                "week": "2025-W03",
                "sunday": "2025-01-19",
                "my_parent_asin": "PARENT2",
                "my_asin": "B0LEADASIN",
                "opp_type": "leader",
                "asin_priority": 1,
                "traffic_gap": 0.12,
                "t_confidence": 0.85,
            },
        )
        conn.execute(
            text(
                """
                UPDATE vw_amz_comp_llm_overview_traffic
                SET ad_ratio_index_med=:ad_ratio_index_med,
                    ad_to_natural_gap=:ad_to_natural_gap,
                    sp_share_in_ad_gap=:sp_share_in_ad_gap,
                    kw_top3_share_gap=:kw_top3_share_gap,
                    kw_brand_share_gap=:kw_brand_share_gap,
                    kw_competitor_share_gap=:kw_competitor_share_gap
                WHERE my_asin=:my_asin AND week=:week
                """
            ),
            {
                "ad_ratio_index_med": 1.1,
                "ad_to_natural_gap": 0.2,
                "sp_share_in_ad_gap": 0.15,
                "kw_top3_share_gap": 0.12,
                "kw_brand_share_gap": 0.08,
                "kw_competitor_share_gap": -0.02,
                "my_asin": "B0LEADASIN",
                "week": "2025-W03",
            },
        )

    result = orchestrator.run("2025-W03", marketplace_id="US")

    stage1_paths = [p for p in result.storage_paths if "stage1" in str(p)]
    assert stage1_paths, "Stage-1 output should be persisted even without Stage-2"
    target_path = next(p for p in stage1_paths if "B0LEADASIN" in p.name)
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    assert payload["summary"] == "未检测到落后维度"
    assert payload["dimensions"] == []




def test_build_page_evidence_includes_objective_values(sqlite_engine, tmp_path):
    config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    orchestrator = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=StubLLM(),
        config=config,
        storage_root=tmp_path,
    )

    ctx = {
        "scene_tag": "SCN-OBJ",
        "base_scene": "base",
        "morphology": "standard",
        "marketplace_id": "US",
        "week": "2025-W10",
        "sunday": "2025-03-09",
        "my_asin": "B0MYASIN",
        "opp_type": "leader",
    }

    with sqlite_engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_amz_comp_pairs (
                    scene_tag TEXT,
                    base_scene TEXT,
                    morphology TEXT,
                    marketplace_id TEXT,
                    week TEXT,
                    sunday TEXT,
                    my_asin TEXT,
                    opp_type TEXT,
                    price_index_med REAL,
                    price_gap_leader REAL,
                    price_z REAL,
                    rank_pos_pct REAL,
                    content_gap REAL,
                    social_gap REAL,
                    badge_diff TEXT,
                    badge_delta_sum REAL,
                    pressure REAL,
                    intensity_band TEXT,
                    confidence REAL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_amz_comp_pairs_each (
                    scene_tag TEXT,
                    base_scene TEXT,
                    morphology TEXT,
                    marketplace_id TEXT,
                    week TEXT,
                    sunday TEXT,
                    my_asin TEXT,
                    opp_asin TEXT,
                    opp_type TEXT,
                    opp_parent_asin TEXT,
                    price_gap_each REAL,
                    price_ratio_each REAL,
                    rank_pos_delta REAL,
                    content_gap_each REAL,
                    social_gap_each REAL,
                    badge_delta_sum REAL,
                    my_price_net REAL,
                    opp_price_net REAL,
                    my_price_current REAL,
                    opp_price_current REAL,
                    my_rank_pos_pct REAL,
                    opp_rank_pos_pct REAL,
                    my_content_score REAL,
                    opp_content_score REAL,
                    my_social_proof REAL,
                    opp_social_proof REAL,
                    score_price REAL,
                    score_rank REAL,
                    score_cont REAL,
                    score_soc REAL,
                    score_badge REAL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_amz_comp_entities_clean (
                    scene_tag TEXT,
                    base_scene TEXT,
                    morphology TEXT,
                    marketplace_id TEXT,
                    week TEXT,
                    sunday TEXT,
                    asin TEXT,
                    price_current REAL,
                    price_list REAL,
                    coupon_pct REAL,
                    price_net REAL,
                    rank_leaf INTEGER,
                    rank_root INTEGER,
                    rank_score REAL,
                    image_cnt INTEGER,
                    video_cnt INTEGER,
                    bullet_cnt INTEGER,
                    title_len INTEGER,
                    aplus_flag INTEGER,
                    content_score REAL,
                    rating REAL,
                    reviews INTEGER,
                    social_proof REAL,
                    badge_json TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_amz_asin_product_snapshot (
                    marketplace_id TEXT,
                    week TEXT,
                    asin TEXT,
                    brand TEXT,
                    payload TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_amz_comp_payload_path (
                    field_name TEXT,
                    json_path TEXT,
                    path TEXT,
                    field_path TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO bi_amz_comp_pairs
                (scene_tag, base_scene, morphology, marketplace_id, week, sunday, my_asin, opp_type,
                 price_index_med, price_gap_leader, price_z, rank_pos_pct, content_gap, social_gap,
                 badge_diff, badge_delta_sum, pressure, intensity_band, confidence)
                VALUES (:scene_tag, :base_scene, :morphology, :marketplace_id, :week, :sunday, :my_asin, :opp_type,
                        :price_index_med, :price_gap_leader, :price_z, :rank_pos_pct, :content_gap, :social_gap,
                        :badge_diff, :badge_delta_sum, :pressure, :intensity_band, :confidence)
                """
            ),
            {
                "scene_tag": ctx["scene_tag"],
                "base_scene": ctx["base_scene"],
                "morphology": ctx["morphology"],
                "marketplace_id": ctx["marketplace_id"],
                "week": ctx["week"],
                "sunday": ctx["sunday"],
                "my_asin": ctx["my_asin"],
                "opp_type": "leader",
                "price_index_med": 1.8,
                "price_gap_leader": 6.0,
                "price_z": 0.4,
                "rank_pos_pct": 0.7,
                "content_gap": 0.2,
                "social_gap": -0.1,
                "badge_diff": "{}",
                "badge_delta_sum": -1,
                "pressure": 0.5,
                "intensity_band": "C2",
                "confidence": 0.9,
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO bi_amz_comp_pairs_each
                (scene_tag, base_scene, morphology, marketplace_id, week, sunday, my_asin, opp_asin, opp_type, opp_parent_asin,
                 price_gap_each, price_ratio_each, rank_pos_delta, content_gap_each, social_gap_each, badge_delta_sum,
                 my_price_net, opp_price_net, my_price_current, opp_price_current,
                 my_rank_pos_pct, opp_rank_pos_pct, my_content_score, opp_content_score,
                 my_social_proof, opp_social_proof, score_price, score_rank, score_cont, score_soc, score_badge)
                VALUES (:scene_tag, :base_scene, :morphology, :marketplace_id, :week, :sunday, :my_asin, :opp_asin, :opp_type, :opp_parent_asin,
                        :price_gap_each, :price_ratio_each, :rank_pos_delta, :content_gap_each, :social_gap_each, :badge_delta_sum,
                        :my_price_net, :opp_price_net, :my_price_current, :opp_price_current,
                        :my_rank_pos_pct, :opp_rank_pos_pct, :my_content_score, :opp_content_score,
                        :my_social_proof, :opp_social_proof, :score_price, :score_rank, :score_cont, :score_soc, :score_badge)
                """
            ),
            {
                "scene_tag": ctx["scene_tag"],
                "base_scene": ctx["base_scene"],
                "morphology": ctx["morphology"],
                "marketplace_id": ctx["marketplace_id"],
                "week": ctx["week"],
                "sunday": ctx["sunday"],
                "my_asin": ctx["my_asin"],
                "opp_asin": "B0OPPASIN",
                "opp_parent_asin": "PARENT-OPP",
                "opp_type": "leader",
                "price_gap_each": 7.0,
                "price_ratio_each": 1.54,
                "rank_pos_delta": 0.1,
                "content_gap_each": -0.05,
                "social_gap_each": -0.02,
                "badge_delta_sum": -1,
                "my_price_net": 19.99,
                "opp_price_net": 12.99,
                "my_price_current": 21.99,
                "opp_price_current": 13.49,
                "my_rank_pos_pct": 0.75,
                "opp_rank_pos_pct": 0.55,
                "my_content_score": 0.62,
                "opp_content_score": 0.78,
                "my_social_proof": 2.3,
                "opp_social_proof": 2.9,
                "score_price": 0.9,
                "score_rank": 0.5,
                "score_cont": 0.4,
                "score_soc": 0.3,
                "score_badge": 0.2,
            },
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_amz_asin_product_snapshot (
                    marketplace_id TEXT,
                    week TEXT,
                    asin TEXT,
                    brand TEXT,
                    payload TEXT
                )
                """
            )
        )
        conn.execute(text("DELETE FROM bi_amz_asin_product_snapshot"))
        for asin, price_current, price_net, content_score, social_proof, badge_json in (
            ("B0MYASIN", 21.99, 19.99, 0.62, 2.3, '{"badges": ["Prime"]}'),
            ("B0OPPASIN", 13.49, 12.99, 0.78, 2.9, '{"badges": []}')
        ):
            conn.execute(
                text(
                    """
                    INSERT INTO bi_amz_comp_entities_clean
                    (scene_tag, base_scene, morphology, marketplace_id, week, sunday, asin,
                     price_current, price_list, coupon_pct, price_net, rank_leaf, rank_root, rank_score,
                     image_cnt, video_cnt, bullet_cnt, title_len, aplus_flag, content_score,
                     rating, reviews, social_proof, badge_json)
                    VALUES (:scene_tag, :base_scene, :morphology, :marketplace_id, :week, :sunday, :asin,
                            :price_current, :price_list, :coupon_pct, :price_net, :rank_leaf, :rank_root, :rank_score,
                            :image_cnt, :video_cnt, :bullet_cnt, :title_len, :aplus_flag, :content_score,
                            :rating, :reviews, :social_proof, :badge_json)
                    """
                ),
                {
                    "scene_tag": ctx["scene_tag"],
                    "base_scene": ctx["base_scene"],
                    "morphology": ctx["morphology"],
                    "marketplace_id": ctx["marketplace_id"],
                    "week": ctx["week"],
                    "sunday": ctx["sunday"],
                    "asin": asin,
                    "price_current": price_current,
                    "price_list": price_current + 1,
                    "coupon_pct": 0.1,
                    "price_net": price_net,
                    "rank_leaf": 5,
                    "rank_root": 10,
                    "rank_score": 0.45,
                    "image_cnt": 6,
                    "video_cnt": 1,
                    "bullet_cnt": 5,
                    "title_len": 80,
                    "aplus_flag": 1,
                    "content_score": content_score,
                    "rating": 4.5,
                    "reviews": 1200,
                    "social_proof": social_proof,
                    "badge_json": badge_json,
                },
            )
            conn.execute(
                text(
                    """
                    INSERT INTO bi_amz_asin_product_snapshot
                    (marketplace_id, week, asin, brand, payload)
                    VALUES (:marketplace_id, :week, :asin, :brand, :payload)
                    """
                ),
                {
                    "marketplace_id": ctx["marketplace_id"],
                    "week": ctx["week"],
                    "asin": asin,
                    "brand": "MyBrand" if asin == "B0MYASIN" else "OppBrand",
                    "payload": '{"brand": "MyBrand"}' if asin == "B0MYASIN" else '{"brand": "OppBrand"}',
                },
            )

    evidence = orchestrator._build_page_evidence(ctx, "price", "leader")
    assert evidence["top_opps"]
    row = evidence["top_opps"][0]
    assert row["my_price_net"] == pytest.approx(19.99)
    assert row["opp_price_net"] == pytest.approx(12.99)
    assert row["my_price_current"] == pytest.approx(21.99)
    assert row["opp_price_current"] == pytest.approx(13.49)
    assert row["my_badge_json"] == '{"badges": ["Prime"]}'
    assert row["opp_badge_json"] == '{"badges": []}'
    assert row["my_brand"] == "MyBrand"
    assert row["opp_brand"] == "OppBrand"

    lag_data = {"top_opps": [row]}
    entries = orchestrator._extract_pairwise_evidence("price", lag_data, limit=6)
    metrics = {entry["metric"] for entry in entries}
    assert "price_net" in metrics
    assert "price_current" in metrics


def test_brand_fallback_snapshot_column(sqlite_engine, tmp_path):
    config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    orchestrator = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=StubLLM(),
        config=config,
        storage_root=tmp_path,
    )
    ctx = {"marketplace_id": "US", "week": "2024-W10"}
    asin = "B0BRANDSNAP"
    with sqlite_engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_amz_asin_product_snapshot (
                    marketplace_id TEXT,
                    week TEXT,
                    asin TEXT,
                    brand TEXT,
                    payload TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_amz_comp_payload_path (
                    field_name TEXT,
                    json_path TEXT,
                    path TEXT,
                    field_path TEXT
                )
                """
            )
        )
        conn.execute(text("DELETE FROM bi_amz_asin_product_snapshot"))
        conn.execute(text("DELETE FROM bi_amz_comp_payload_path"))
        conn.execute(
            text(
                """
                INSERT INTO bi_amz_asin_product_snapshot
                (marketplace_id, week, asin, brand, payload)
                VALUES (:marketplace_id, :week, :asin, :brand, :payload)
                """
            ),
            {
                "marketplace_id": ctx["marketplace_id"],
                "week": ctx["week"],
                "asin": asin,
                "brand": "SnapshotBrand",
                "payload": '{"brand": "SnapshotBrand"}',
            },
        )

    brand = orchestrator._lookup_brand(ctx, asin)
    assert brand == "SnapshotBrand"


def test_brand_fallback_payload_json(sqlite_engine, tmp_path):
    config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    orchestrator = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=StubLLM(),
        config=config,
        storage_root=tmp_path,
    )
    ctx = {"marketplace_id": "US", "week": "2024-W11"}
    asin = "B0BRANDJSON"
    payload = json.dumps({"product": {"meta": {"brand_name": "PayloadBrand"}}})
    with sqlite_engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_amz_asin_product_snapshot (
                    marketplace_id TEXT,
                    week TEXT,
                    asin TEXT,
                    brand TEXT,
                    payload TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_amz_comp_payload_path (
                    field_name TEXT,
                    json_path TEXT,
                    path TEXT,
                    field_path TEXT
                )
                """
            )
        )
        conn.execute(text("DELETE FROM bi_amz_asin_product_snapshot"))
        conn.execute(text("DELETE FROM bi_amz_comp_payload_path"))
        conn.execute(
            text(
                """
                INSERT INTO bi_amz_comp_payload_path
                (field_name, json_path, path, field_path)
                VALUES (:field_name, :json_path, NULL, NULL)
                """
            ),
            {"field_name": "brand", "json_path": "$.product.meta.brand_name"},
        )
        conn.execute(
            text(
                """
                INSERT INTO bi_amz_asin_product_snapshot
                (marketplace_id, week, asin, brand, payload)
                VALUES (:marketplace_id, :week, :asin, :brand, :payload)
                """
            ),
            {
                "marketplace_id": ctx["marketplace_id"],
                "week": ctx["week"],
                "asin": asin,
                "brand": None,
                "payload": payload,
            },
        )

    brand = orchestrator._lookup_brand(ctx, asin)
    assert brand == "PayloadBrand"


def test_missing_brand_no_fail(sqlite_engine, tmp_path):
    config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    orchestrator = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=StubLLM(),
        config=config,
        storage_root=tmp_path,
    )
    ctx = {"marketplace_id": "US", "week": "2024-W12"}
    asin = "B0UNKNOWN"
    with sqlite_engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_amz_asin_product_snapshot (
                    marketplace_id TEXT,
                    week TEXT,
                    asin TEXT,
                    brand TEXT,
                    payload TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS bi_amz_comp_payload_path (
                    field_name TEXT,
                    json_path TEXT,
                    path TEXT,
                    field_path TEXT
                )
                """
            )
        )
        conn.execute(text("DELETE FROM bi_amz_asin_product_snapshot"))
        conn.execute(text("DELETE FROM bi_amz_comp_payload_path"))

    brand = orchestrator._lookup_brand(ctx, asin)
    assert brand is None
    assert orchestrator._lookup_brand(ctx, asin) is None


def test_pairwise_evidence_notes_and_units(sqlite_engine, tmp_path):
    config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    orchestrator = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=StubLLM(),
        config=config,
        storage_root=tmp_path,
    )

    rank_row = {
        "opp_asin": "B0OPP1",
        "opp_brand": "BrandX",
        "my_rank_leaf": 35,
        "opp_rank_leaf": 12,
        "my_rank_pos_pct": 0.42,
        "opp_rank_pos_pct": 0.18,
    }
    rank_entries = orchestrator._extract_pairwise_evidence(
        "rank", {"top_opps": [rank_row]}, limit=2
    )
    assert rank_entries[0]["metric"] == "rank_leaf"
    assert rank_entries[0]["unit"] == "rank"
    assert "排名" in rank_entries[0]["note"]
    assert rank_entries[0]["direction"] == "lower_better"
    assert rank_entries[0]["worse"] is True
    assert rank_entries[1]["metric"] == "rank_pos_pct"
    assert rank_entries[1]["unit"] == "pct"
    assert "排名百分位" in rank_entries[1]["note"]
    assert rank_entries[1]["direction"] == "lower_better"
    assert rank_entries[1]["delta"] == pytest.approx(0.24, rel=1e-6)
    assert rank_entries[1]["worse"] is True

    content_row = {
        "opp_asin": "B0OPP2",
        "my_image_cnt": 5,
        "opp_image_cnt": 8,
        "my_video_cnt": 1,
        "opp_video_cnt": 3,
        "my_content_score": 0.62,
        "opp_content_score": 0.78,
    }
    content_entries = orchestrator._extract_pairwise_evidence(
        "content", {"top_opps": [content_row]}, limit=3
    )
    assert {entry["metric"] for entry in content_entries} == {
        "image_cnt",
        "video_cnt",
        "content_score",
    }
    assert all(entry.get("note") for entry in content_entries)

    social_row = {
        "opp_asin": "B0OPP3",
        "my_rating": 4.3,
        "opp_rating": 4.7,
        "my_reviews": 1200,
        "opp_reviews": 2000,
        "my_social_proof": 2.31,
        "opp_social_proof": 2.46,
    }
    social_entries = orchestrator._extract_pairwise_evidence(
        "social", {"top_opps": [social_row]}, limit=3
    )
    assert {entry["metric"] for entry in social_entries} == {
        "rating",
        "reviews",
        "social_proof",
    }
    assert all(entry.get("note") for entry in social_entries)

    keyword_row = {
        "opp_asin": "B0OPP4",
        "keyword_pairs": [
            {
                "keyword": "bath bag",
                "my_rank": None,
                "opp_rank": 1,
                "my_share": 0.05,
                "opp_share": 0.22,
                "impact": 0.22,
            }
        ],
    }
    keyword_entries = orchestrator._extract_pairwise_evidence(
        "keyword", {"top_opps": [keyword_row]}, limit=2
    )
    assert keyword_entries
    first_keyword = keyword_entries[0]
    assert first_keyword["metric"] == "keyword_rank"
    assert first_keyword["unit"] == "rank"
    assert first_keyword["my_value"] == "无"
    assert "关键词" in first_keyword["note"]
    assert "%" in first_keyword["note"]
    assert first_keyword["direction"] == "lower_better"
    assert first_keyword["worse"] is None
