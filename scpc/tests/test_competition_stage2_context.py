import pytest


pytest.importorskip("yaml")


from scpc.llm.competition_workflow import _ensure_stage2_context


def test_ensure_stage2_context_preserves_existing_context():
    machine_json = {
        "context": {"my_asin": "B001"},
        "lag_type": "mixed",
        "root_causes": [],
        "actions": [],
    }
    fallback = {"my_asin": "B002"}

    result = _ensure_stage2_context(machine_json, fallback)

    assert result["context"] == machine_json["context"]
    assert result is not machine_json


def test_ensure_stage2_context_injects_fallback_when_missing():
    machine_json = {
        "lag_type": "mixed",
        "root_causes": [],
        "actions": [],
    }
    fallback = {"my_asin": "B003", "week": "2025-W01"}

    result = _ensure_stage2_context(machine_json, fallback)

    assert result["context"] == fallback
    assert "context" not in machine_json


def test_ensure_stage2_context_handles_non_mapping_fallback():
    machine_json = {
        "lag_type": "mixed",
        "root_causes": [],
        "actions": [],
    }

    result = _ensure_stage2_context(machine_json, None)

    assert result["context"] == {}
