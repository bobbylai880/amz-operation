"""Sanity tests for the LLM orchestrator."""
from __future__ import annotations

from typing import Any

import pytest

import json

from scpc.llm.orchestrator import LLMOrchestrator, LLMRunConfig


class StubClient:
    """Minimal stub implementing the ``DeepSeekClient`` interface."""

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls = 0

    def generate(self, **_: Any) -> Any:  # pragma: no cover - signature simplified
        self.calls += 1
        return type("Resp", (), {"content": self._payload, "usage": {}})()


def test_orchestrator_validates_schema(tmp_path: Any) -> None:
    payload = json.dumps({"status": "ok"})
    schema = {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]}
    client = StubClient(payload)
    orchestrator = LLMOrchestrator(client)  # type: ignore[arg-type]
    config = LLMRunConfig(
        prompt="You are a test",
        facts={"hello": "world"},
        schema=schema,
        model="deepseek",
    )
    result = orchestrator.run(config)
    assert result["status"] == "ok"
    assert client.calls == 1


def test_orchestrator_raises_without_fallback() -> None:
    schema = {"type": "object"}

    class ErrorClient(StubClient):
        def generate(self, **_: Any) -> Any:  # pragma: no cover - deterministic
            raise RuntimeError("boom")

    orchestrator = LLMOrchestrator(ErrorClient({}))  # type: ignore[arg-type]
    config = LLMRunConfig(prompt="", facts={}, schema=schema, model="deepseek")
    with pytest.raises(RuntimeError):
        orchestrator.run(config, retry=False)
