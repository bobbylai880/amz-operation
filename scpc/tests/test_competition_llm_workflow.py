import json
from pathlib import Path

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
        self.stage2_requests = []

    def run(self, config, *, retry=True, fallback=None):  # noqa: D401 - signature mirrors orchestrator
        facts = config.facts
        if "machine_json_schema" in facts:
            self.stage2_requests.append(facts)
            context = facts["first_round_item"]["context"]
            lag_type = facts["first_round_item"].get("lag_type", "")
            machine_json = {
                "context": context,
                "lag_type": lag_type,
                "why_chain": [
                    {
                        "why": "Why is the ASIN lagging?",
                        "answer": "Pricing gap is wider than leader.",
                        "evidence_refs": ["metrics.price_gap_pct"],
                    }
                ],
                "root_causes": [
                    {
                        "description": "pricing misalignment",
                        "evidence_refs": ["metrics.price_gap_pct"],
                        "is_root": True,
                        "is_partial": False,
                        "root_cause_code": "pricing_misalignment",
                        "owner": "pricing",
                    }
                ],
                "recommended_actions": [
                    {
                        "action": "调整售价与竞品差距",
                        "owner": "pricing",
                        "expected_impact": "提升买盒概率",
                        "validation_metric": "price_gap_pct",
                        "action_code": "price_adjust",
                        "evidence_refs": ["metrics.price_gap_pct"],
                    }
                ],
            }
            return {"machine_json": machine_json, "human_markdown": "- 调整售价以收窄差距"}
        self.stage1_requests.append(facts)
        return {
            "context": facts["context"],
            "dimensions": [
                {
                    "lag_type": "pricing",
                    "status": "lag",
                    "severity": "high",
                    "source_opp_type": "page",
                    "source_confidence": 0.8,
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
    assert result.stage2_candidates == 1
    assert result.stage2_processed == 1

    stage1_file = tmp_path / "2025-W01" / "stage1" / "B012345_page.json"
    stage2_file = tmp_path / "2025-W01" / "stage2" / "B012345_pricing_page.json"
    assert stage1_file.exists()
    assert stage2_file.exists()

    stage1_payload = json.loads(stage1_file.read_text(encoding="utf-8"))
    assert stage1_payload["context"]["my_asin"] == "B012345"
    assert stage1_payload["dimensions"][0]["status"] == "lag"

    stage2_payload = json.loads(stage2_file.read_text(encoding="utf-8"))
    assert stage2_payload["machine_json"]["lag_type"] == "pricing"
    assert "调整售价" in stage2_payload["human_markdown"]

    assert stub_llm.stage1_requests
    assert stub_llm.stage2_requests


def test_stage2_candidate_requires_confidence(sqlite_engine, tmp_path):
    config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    stub_llm = StubLLM()
    orchestrator = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=stub_llm,
        config=config,
        storage_root=tmp_path,
    )
    context = {
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
    }
    stage1_results = (
        StageOneLLMResult(
            context=context,
            dimensions=(
                {
                    "lag_type": "pricing",
                    "status": "lag",
                    "severity": "high",
                    "source_opp_type": "page",
                    "source_confidence": 0.2,
                },
            ),
        ),
    )
    candidates = orchestrator._prepare_stage2_candidates(stage1_results)
    assert candidates == ()


def test_stage2_validation_enforces_allowed_codes(sqlite_engine, tmp_path):
    config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    stub_llm = StubLLM()
    orchestrator = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=stub_llm,
        config=config,
        storage_root=tmp_path,
    )
    payload = {
        "context": {
            "scene_tag": "SCN-1",
            "base_scene": "base",
            "morphology": "standard",
            "marketplace_id": "US",
            "week": "2025-W01",
            "sunday": "2025-01-05",
            "my_parent_asin": "PARENT1",
            "my_asin": "B012345",
            "opp_type": "page",
        },
        "lag_type": "pricing",
        "why_chain": [
            {
                "why": "Why?",
                "answer": "Because price gap",
                "evidence_refs": ["metrics.price_gap_pct"],
            }
        ],
        "root_causes": [
            {
                "description": "pricing issue",
                "evidence_refs": ["metrics.price_gap_pct"],
                "is_root": True,
                "is_partial": False,
                "root_cause_code": "pricing_misalignment",
            }
        ],
        "recommended_actions": [
            {
                "action": "dummy",
                "owner": "pricing",
                "expected_impact": "fix",
                "validation_metric": "price_gap_pct",
                "action_code": "invalid_code",
                "evidence_refs": ["metrics.price_gap_pct"],
            }
        ],
    }
    with pytest.raises(ValueError):
        orchestrator._validate_stage2_machine_json(payload)
