"""Runtime configuration helpers for environment-driven settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def load_dotenv(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> None:
    """Load key-value pairs from a ``.env`` file into ``os.environ``.

    Parameters
    ----------
    path:
        Location of the ``.env`` file.  Defaults to the repository root.
    override:
        When ``True`` existing environment variables will be overwritten;
        otherwise only missing keys are populated.  The default keeps runtime
        overrides intact.
    """

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if override or key not in os.environ:
            os.environ[key] = value


@dataclass(slots=True)
class DeepSeekSettings:
    """Settings required to interact with the DeepSeek API."""

    base_url: str
    model: str
    api_key: str
    timeout: float


def get_deepseek_settings(*, env_paths: Iterable[str | os.PathLike[str]] = (".env",)) -> DeepSeekSettings:
    """Return DeepSeek credentials sourced from environment variables."""

    for candidate in env_paths:
        load_dotenv(candidate)
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    timeout_raw = os.getenv("DEEPSEEK_TIMEOUT", "30")
    if not model:
        raise RuntimeError("Environment variable DEEPSEEK_MODEL must be configured")
    if not api_key:
        raise RuntimeError("Environment variable DEEPSEEK_API_KEY must be configured")
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise RuntimeError("DEEPSEEK_TIMEOUT must be a numeric value") from exc
    return DeepSeekSettings(base_url=base_url, model=model, api_key=api_key, timeout=timeout)


@dataclass(slots=True)
class DatabaseSettings:
    """Settings used to establish SQLAlchemy connections to MySQL."""

    host: str
    port: int
    database: str
    user: str
    password: str

    def sqlalchemy_url(self, *, driver: str = "mysql+pymysql") -> str:
        """Render a SQLAlchemy connection URL."""

        return f"{driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


def get_database_settings(*, env_paths: Iterable[str | os.PathLike[str]] = (".env",)) -> DatabaseSettings:
    """Return database connection information from environment variables."""

    for candidate in env_paths:
        load_dotenv(candidate)
    host = os.getenv("MYSQL_HOST")
    port_raw = os.getenv("MYSQL_PORT", "3306")
    database = os.getenv("MYSQL_DATABASE")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    missing = [
        name
        for name, value in (
            ("MYSQL_HOST", host),
            ("MYSQL_DATABASE", database),
            ("MYSQL_USER", user),
            ("MYSQL_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    try:
        port = int(port_raw)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise RuntimeError("MYSQL_PORT must be an integer") from exc
    return DatabaseSettings(host=host, port=port, database=database, user=user, password=password)


__all__ = [
    "DeepSeekSettings",
    "DatabaseSettings",
    "get_deepseek_settings",
    "get_database_settings",
    "load_dotenv",
]
