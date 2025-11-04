from __future__ import annotations

import json

import pytest

from scpc.llm.deepseek_client import DeepSeekClient


class _DummyResponse:
    def __init__(self) -> None:
        self._payload = json.dumps(
            {
                "choices": [
                    {"message": {"content": "{}"}},
                ],
                "usage": {},
            }
        ).encode("utf-8")

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - no cleanup
        return None

    def read(self) -> bytes:
        return self._payload


@pytest.mark.parametrize(
    ("base_url", "expected"),
    (
        ("https://api.deepseek.com", "https://api.deepseek.com/v1/chat/completions"),
        ("https://api.deepseek.com/", "https://api.deepseek.com/v1/chat/completions"),
        ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/chat/completions"),
        (
            "https://api.deepseek.com/v1/chat/completions",
            "https://api.deepseek.com/v1/chat/completions",
        ),
    ),
)
def test_generate_respects_base_url(monkeypatch, base_url: str, expected: str) -> None:
    captured: list[str] = []

    def fake_urlopen(request, timeout):  # type: ignore[override]
        captured.append(request.full_url)
        return _DummyResponse()

    monkeypatch.setattr("scpc.llm.deepseek_client.urlopen", fake_urlopen)

    client = DeepSeekClient(base_url=base_url, api_key="token", timeout=12)

    client.generate(
        prompt="system prompt",
        facts={"foo": "bar"},
        model="deepseek-chat",
        temperature=0.1,
        response_format="json_object",
    )

    assert captured == [expected]
