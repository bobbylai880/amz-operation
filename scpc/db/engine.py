"""SQLAlchemy engine helpers for Doris/StarRocks connectivity."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


DEFAULT_POOL_KWARGS: dict[str, Any] = {
    "pool_pre_ping": True,
    "pool_recycle": 1800,
}


def _get_db_uri() -> str:
    """Return the database URI from the environment."""

    db_uri = os.getenv("DB_URI")
    if not db_uri:
        raise RuntimeError("Environment variable DB_URI must be configured")
    return db_uri


@lru_cache(maxsize=1)
def create_doris_engine(**overrides: Any) -> Engine:
    """Create (or reuse) a SQLAlchemy engine configured for Doris."""

    db_uri = _get_db_uri()
    kwargs = {**DEFAULT_POOL_KWARGS, **overrides}
    return create_engine(db_uri, **kwargs)


__all__ = ["create_doris_engine"]
