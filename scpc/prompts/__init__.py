"""Utilities for loading prompt templates used by the LLM pipelines."""
from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """Return the contents of a prompt template located in ``scpc/prompts``.

    Parameters
    ----------
    name:
        File name relative to the prompts package. The ``.md`` suffix should
        be included (e.g. ``"competition_stage1.md"``).
    """

    path = _PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {name}")
    return path.read_text(encoding="utf-8")


__all__ = ["load_prompt"]
