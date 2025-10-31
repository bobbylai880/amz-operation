"""LLM orchestration utilities for S-C-P-C flows."""
from __future__ import annotations

from dataclasses import dataclass
from json import loads
from typing import Any, Callable, Mapping

from .deepseek_client import DeepSeekClient, DeepSeekError


def _validate_schema(schema: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Very small subset of JSON Schema validation.

    The function checks required keys and basic type compatibility.  It is not a
    full JSON Schema implementation but is sufficient for unit tests and
    low-risk sanity validation.  A complete validator (e.g. ``jsonschema``)
    should be plugged in once dependencies are available.
    """

    required = schema.get("required", [])
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing required key: {key}")
    properties: Mapping[str, Any] = schema.get("properties", {})
    for key, value in payload.items():
        expected = properties.get(key)
        if not expected:
            continue
        expected_type = expected.get("type")
        if expected_type is None:
            continue
        if isinstance(expected_type, list):
            types = tuple(tp for name in expected_type for tp in _python_types(name))
        else:
            types = _python_types(expected_type)
        if types and not isinstance(value, types):
            raise ValueError(f"Field {key} expected {types}, got {type(value)!r}")


def _python_types(name: str) -> tuple[type, ...]:
    mapping: dict[str, tuple[type, ...]] = {
        "object": (dict,),
        "array": (list,),
        "string": (str,),
        "number": (int, float),
        "boolean": (bool,),
        "null": (type(None),),
    }
    return mapping.get(name, (object,))


@dataclass(slots=True)
class LLMRunConfig:
    """Configuration bundle for invoking a single DeepSeek segment."""

    prompt: str
    facts: Mapping[str, Any]
    schema: Mapping[str, Any]
    model: str
    temperature: float = 0.1
    response_format: str = "json_object"
    top_p: float = 0.9


class LLMOrchestrator:
    """Coordinates prompt execution, validation, retry and fallback."""

    def __init__(self, client: DeepSeekClient) -> None:
        self._client = client

    def run(self, config: LLMRunConfig, *, retry: bool = True, fallback: Callable[[], Mapping[str, Any]] | None = None) -> Mapping[str, Any]:
        """Execute a single LLM step and validate against the provided schema."""

        attempt = 0
        last_error: Exception | None = None
        while True:
            attempt += 1
            try:
                response = self._client.generate(
                    prompt=config.prompt,
                    facts=config.facts,
                    model=config.model,
                    temperature=config.temperature,
                    response_format=config.response_format,
                    top_p=config.top_p,
                )
                payload = loads(response.content)
                _validate_schema(config.schema, payload)
                return payload
            except (DeepSeekError, ValueError) as exc:
                last_error = exc
                if retry and attempt == 1:
                    continue
                break
        if fallback is None:
            raise RuntimeError("LLM invocation failed") from last_error
        payload = fallback()
        _validate_schema(config.schema, payload)
        return payload


__all__ = ["LLMOrchestrator", "LLMRunConfig"]
