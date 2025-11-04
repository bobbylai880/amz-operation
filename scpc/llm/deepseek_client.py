"""Client wrapper around the DeepSeek API.

The client is intentionally thin; the orchestrator is responsible for retries
and schema validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from json import dumps
from typing import Any, Mapping

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from scpc.settings import DeepSeekSettings, get_deepseek_settings


class DeepSeekError(RuntimeError):
    """Raised when the DeepSeek API responds with an error payload."""


@dataclass(slots=True)
class DeepSeekResponse:
    """Structured response from the DeepSeek API."""

    content: str
    usage: Mapping[str, Any]


class DeepSeekClient:
    """Small helper around ``httpx`` to keep the integration contained."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def generate(
        self,
        prompt: str,
        facts: Mapping[str, Any],
        *,
        model: str,
        temperature: float,
        response_format: Mapping[str, Any] | str,
        top_p: float = 0.9,
    ) -> DeepSeekResponse:
        """Send a structured generation request to DeepSeek."""

        if isinstance(response_format, str):
            response_format_payload: Mapping[str, Any] = {"type": response_format}
        else:
            response_format_payload = response_format

        payload = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "response_format": response_format_payload,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": facts},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        request = Request(
            f"{self._base_url}/v1/chat/completions",
            data=dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:  # pragma: no cover - network failure path
            body: str
            try:
                body = exc.read().decode("utf-8")
            except Exception:  # pragma: no cover - defensive guard
                body = ""
            if not body:
                reason = getattr(exc, "reason", "")
                status = getattr(exc, "code", "")
                body = " ".join(str(part) for part in (status, reason) if part).strip()
            raise DeepSeekError(f"DeepSeek error: {body}") from exc
        except URLError as exc:  # pragma: no cover - network failure path
            raise DeepSeekError(f"DeepSeek network error: {exc.reason}") from exc
        import json

        data = json.loads(raw)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive guard
            raise DeepSeekError(f"Malformed DeepSeek response: {data}") from exc
        usage = data.get("usage", {})
        return DeepSeekResponse(content=content, usage=usage)

    def close(self) -> None:
        """Provided for API parity; no persistent connections are kept."""


def create_client_from_env(*, settings: DeepSeekSettings | None = None) -> DeepSeekClient:
    """Factory that instantiates ``DeepSeekClient`` using ``.env`` settings."""

    if settings is None:
        settings = get_deepseek_settings()
    return DeepSeekClient(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout,
    )


__all__ = [
    "DeepSeekClient",
    "DeepSeekError",
    "DeepSeekResponse",
    "create_client_from_env",
]
