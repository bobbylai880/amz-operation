import json
from dataclasses import replace
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
        self.stage1_overrides: dict[str, dict] = {}

    def set_stage1_override(self, asin: str, response: dict) -> None:
        self.stage1_overrides[asin] = response

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
                        "summary": "价格指数偏高，净价落后",
                        "evidence": [
                            {
                                "metric": "price_index_med",
                                "against": "median",
                                "my_value": 1.2,
                                "opp_value": 1.0,
                                "unit": "ratio",
                                "source": "page.overview",
                            }
                        ],
                        "is_root": True,
                        "is_partial": False,
                        "root_cause_code": "pricing_misalignment",
                        "priority": 1,
                        "owner": "pricing",
                    }
                ],
                "actions": [
                    {
                        "code": "price_adjust",
                        "why": "price_gap_pct 明显高于竞品，需要调整定价",
                        "how": "审查竞品促销与我方定价，制定降价阶梯并评估毛利",
                        "expected_impact": "提升买盒概率并缩小价格差距",
                        "owner": "pricing",
                        "due_weeks": 1,
                        "priority": 1,
                    }
                ],
            }
            return {"machine_json": machine_json, "human_markdown": "- 调整售价以收窄差距"}
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


def test_stage2_packet_lookup_handles_blank_morphology(sqlite_engine, tmp_path):
    evidence = {
        "context": {
            "my": {"brand": "OurBrand", "asin": "B012346"},
            "ref": {"brand": "CompBrand", "asin": "B099998"},
        },
        "metrics": {"price_gap_pct": 0.15},
        "drivers": [{"name": "price_gap_pct", "value": 0.15}],
        "top_competitors": [{"brand": "CompBrand", "asin": "B099998"}],
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
                "scene_tag": "SCN-2",
                "base_scene": "base",
                "morphology": "",
                "marketplace_id": "US",
                "week": "2025-W02",
                "sunday": "2025-01-12",
                "my_parent_asin": "PARENT2",
                "my_asin": "B012346",
                "opp_type": "leader",
                "asin_priority": 1,
                "price_gap": 0.15,
                "confidence": 0.9,
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
                "scene_tag": "SCN-2",
                "base_scene": "base",
                "morphology": "",
                "marketplace_id": "US",
                "week": "2025-W02",
                "sunday": "2025-01-12",
                "my_parent_asin": "PARENT2",
                "my_asin": "B012346",
                "opp_type": "leader",
                "asin_priority": 1,
                "traffic_gap": 0.08,
                "t_confidence": 0.85,
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
                "scene_tag": "SCN-2",
                "base_scene": "base",
                "morphology": None,
                "marketplace_id": "US",
                "week": "2025-W02",
                "sunday": "2025-01-12",
                "my_asin": "B012346",
                "lag_type": "pricing",
                "opp_type": "leader",
                "evidence_json": json.dumps(evidence),
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

    result = orchestrator.run("2025-W02", marketplace_id="US")

    assert result.stage2_candidates == 1
    assert result.stage2_processed == 1


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
            summary="整体诊断：置信度不足，暂不进入二轮。",
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


def test_stage2_candidates_allow_missing_or_string_confidence(sqlite_engine, tmp_path):
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
            summary="整体诊断：价格与流量均存在问题。",
            dimensions=(
                {
                    "lag_type": "pricing",
                    "status": "lag",
                    "severity": "mid",
                    "source_opp_type": "page",
                    "source_confidence": None,
                },
                {
                    "lag_type": "traffic",
                    "status": "lag",
                    "severity": "high",
                    "source_opp_type": "traffic",
                    "source_confidence": "0.72 (traffic)",
                },
            ),
        ),
    )

    candidates = orchestrator._prepare_stage2_candidates(stage1_results)
    assert len(candidates) == 2


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

    stub_llm.set_stage1_override(
        "B0LEADASIN",
        {
            "summary": "整体诊断：所有关键维度领先或与竞对持平，请保持现有策略。",
            "dimensions": [
                {
                    "lag_type": "pricing",
                    "status": "lead",
                    "severity": "low",
                    "source_opp_type": "page",
                    "source_confidence": 0.92,
                    "notes": "定价相对竞对低 5%，具备价格优势。",
                },
                {
                    "lag_type": "traffic",
                    "status": "lead",
                    "severity": "low",
                    "source_opp_type": "traffic",
                    "source_confidence": 0.88,
                    "notes": "流量较竞对高出 12%，持续扩大曝光。",
                },
            ],
        },
    )

    result = orchestrator.run("2025-W03", marketplace_id="US")

    assert result.stage2_candidates == 2
    assert result.stage2_processed == 0

    stage1_paths = [p for p in result.storage_paths if "stage1" in str(p)]
    assert stage1_paths, "Stage-1 output should be persisted even without Stage-2"
    target_path = next(p for p in stage1_paths if "B0LEADASIN" in p.name)
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    assert payload["summary"].startswith("整体诊断")
    for dim in payload["dimensions"]:
        assert dim["status"] in {"lead", "parity"}
        assert dim.get("notes")


def test_stage2_trigger_status_configuration(sqlite_engine, tmp_path):
    base_config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    stub_llm = StubLLM()
    context = {
        "scene_tag": "SCN-CONFIG",
        "base_scene": "base",
        "morphology": "standard",
        "marketplace_id": "US",
        "week": "2025-W04",
        "sunday": "2025-01-26",
        "my_parent_asin": "PARENT3",
        "my_asin": "B0CONFIG",
        "opp_type": "leader",
        "asin_priority": 1,
    }
    lead_dimension = {
        "lag_type": "pricing",
        "status": "lead",
        "severity": "low",
        "source_opp_type": "page",
        "source_confidence": 0.9,
    }
    parity_dimension = {
        "lag_type": "traffic",
        "status": "parity",
        "severity": "low",
        "source_opp_type": "traffic",
        "source_confidence": 0.95,
    }
    stage1_results = (
        StageOneLLMResult(
            context=context,
            summary="整体诊断：我方优势明显，需总结经验。",
            dimensions=(lead_dimension, parity_dimension),
        ),
    )

    orchestrator_all = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=stub_llm,
        config=base_config,
        storage_root=tmp_path,
    )

    candidates_all = orchestrator_all._prepare_stage2_candidates(stage1_results)
    assert len(candidates_all) == 2

    lag_only_config = replace(
        base_config,
        stage_2=replace(base_config.stage_2, trigger_status=("lag",)),
    )
    orchestrator_lag_only = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=stub_llm,
        config=lag_only_config,
        storage_root=tmp_path,
    )
    candidates_lag_only = orchestrator_lag_only._prepare_stage2_candidates(stage1_results)
    assert candidates_lag_only == ()


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
                "summary": "价格指数偏高",
                "evidence": [
                    {
                        "metric": "price_index_med",
                        "against": "median",
                        "my_value": 1.3,
                        "opp_value": 1.0,
                        "unit": "ratio",
                    }
                ],
                "is_root": True,
                "is_partial": False,
                "root_cause_code": "pricing_misalignment",
                "priority": 1,
            }
        ],
        "actions": [
            {
                "code": "invalid_code",
                "why": "price gap remains high",
                "how": "调整促销",
                "expected_impact": "fix",
                "owner": "pricing",
                "due_weeks": 1,
            }
        ],
    }
    with pytest.raises(ValueError):
        orchestrator._validate_stage2_machine_json(payload)


def test_stage2_validation_drops_missing_code_actions(sqlite_engine, tmp_path):
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
        "lag_type": "mixed",
        "root_causes": [
            {
                "root_cause_code": "pricing_misalignment",
                "summary": "价格与竞品差距显著",
                "evidence": [
                    {
                        "metric": "price_index_med",
                        "against": "median",
                        "my_value": 1.25,
                        "opp_value": 1.0,
                        "unit": "ratio",
                    }
                ],
                "priority": 1,
            }
        ],
        "actions": [
            {
                "code": "",
                "why": "",
                "how": "",
                "expected_impact": "",
                "owner": "pricing",
                "due_weeks": 1,
            }
        ],
    }

    orchestrator._validate_stage2_machine_json(payload)

    assert payload["actions"] == []


def test_materialize_evidence_converts_legacy_refs(sqlite_engine, tmp_path):
    config = load_competition_llm_config(Path("configs/competition_llm.yaml"))
    orchestrator = CompetitionLLMOrchestrator(
        engine=sqlite_engine,
        llm_orchestrator=StubLLM(),
        config=config,
        storage_root=tmp_path,
    )

    facts = {
        "lag_items": [
            {
                "lag_type": "pricing",
                "overview": {"median": {"price_index_med": 1.8}},
                "top_opps": [
                    {
                        "opp_asin": "B099999",
                        "my_price_net": 16.99,
                        "opp_price_net": 12.99,
                    }
                ],
            }
        ]
    }
    machine_json = {
        "context": {"my_asin": "B012345"},
        "lag_type": "mixed",
        "root_causes": [
            {
                "root_cause_code": "pricing_misalignment",
                "summary": "价格指数偏高",
                "priority": 1,
                "evidence_refs": [
                    {
                        "lag_type": "price",
                        "opp_type": "median",
                        "metric": "price_index_med",
                        "value": 1.8,
                    }
                ],
            }
        ],
        "actions": [],
    }

    result = orchestrator._materialize_evidence(machine_json, facts)
    root_causes = result["root_causes"]
    assert root_causes
    evidence = root_causes[0].get("evidence")
    assert evidence
    first = evidence[0]
    assert first["metric"] == "price_index_med"
    assert first["against"] == "median"
    assert first["opp_value"] == 1.0
    assert "evidence_refs" not in root_causes[0]
