"""SQLAlchemy engine helpers for Doris/StarRocks connectivity."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


DEFAULT_POOL_KWARGS: dict[str, Any] = {
    "pool_pre_ping": True,
    "pool_recycle": 1800,
}


def _load_env_file() -> None:
    """Populate ``os.environ`` from a local ``.env`` if present.

    The loader is intentionally lightweight so we can avoid importing
    ``python-dotenv``.  It looks in a handful of reasonable roots (the current
    working directory as well as the repository parents) and only populates
    keys that are not already present in the environment.
    """

    resolved = Path(__file__).resolve()
    parent_chain = list(resolved.parents)
    candidate_roots: list[Path] = [Path.cwd(), resolved.parent]
    # Extend with up to two higher-level parents (typically the package root
    # and repository root) while avoiding duplicates.
    for parent in parent_chain[:2]:
        if parent not in candidate_roots:
            candidate_roots.append(parent)

    for root in candidate_roots:
        env_path = root / ".env"
        if not env_path.exists() or not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip('"').strip("'")
        # Stop at the first .env discovered to avoid surprising overrides.
        break


def _build_db_uri_from_components() -> tuple[str | None, list[str]]:
    """Attempt to construct a Doris connection URI from split environment variables."""

    required_keys: Iterable[str] = ("DORIS_HOST", "DORIS_USER", "DORIS_DATABASE")
    resolved = {key: os.getenv(key, "").strip() for key in required_keys}
    missing = [key for key, value in resolved.items() if not value]
    if missing:
        return None, missing

    driver = os.getenv("DORIS_DRIVER", "mysql+pymysql")
    host = resolved["DORIS_HOST"]
    user = resolved["DORIS_USER"]
    database = resolved["DORIS_DATABASE"]
    port = os.getenv("DORIS_PORT", "9030").strip()
    password = os.getenv("DORIS_PASSWORD", "").strip()

    credentials = user if not password else f"{user}:{password}"
    host_segment = host if not port else f"{host}:{port}"
    uri = f"{driver}://{credentials}@{host_segment}/{database}"
    return uri, []


def _get_db_uri() -> str:
    """Return the database URI from the environment or a local ``.env``."""

    _load_env_file()
    db_uri = os.getenv("DB_URI")
    if db_uri:
        return db_uri

    constructed, missing = _build_db_uri_from_components()
    if constructed:
        return constructed

    raise RuntimeError(
        "Environment variable DB_URI must be configured or provide "
        "DORIS_HOST/DORIS_USER/DORIS_DATABASE (missing: "
        + ", ".join(missing)
        + ")"
    )


@lru_cache(maxsize=1)
def create_doris_engine(**overrides: Any) -> Engine:
    """Create (or reuse) a SQLAlchemy engine configured for Doris."""

    db_uri = _get_db_uri()
    kwargs = {**DEFAULT_POOL_KWARGS, **overrides}
    return create_engine(db_uri, **kwargs)


__all__ = ["create_doris_engine"]
