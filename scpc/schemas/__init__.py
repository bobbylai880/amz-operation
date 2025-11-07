"""Helper utilities for working with local JSON Schema definitions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

_SCHEMAS_DIR = Path(__file__).resolve().parent


def load_schema(name: str) -> Mapping[str, Any]:
    """Load a JSON Schema definition bundled with the package."""

    path = _SCHEMAS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {name}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["load_schema"]
