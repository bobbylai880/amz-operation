"""Tests for the Doris engine helpers."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def reload_engine_module() -> None:
    """Reload ``scpc.db.engine`` ensuring caches/environment reset."""

    import scpc.db.engine as engine_module

    importlib.reload(engine_module)


def test_db_uri_loaded_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``DB_URI`` should be read directly when provided in ``.env``."""

    monkeypatch.delenv("DB_URI", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("DB_URI=mysql://example\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    reload_engine_module()

    import scpc.db.engine as engine_module

    assert engine_module._get_db_uri() == "mysql://example"


def test_db_uri_constructed_from_components(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Split Doris settings should assemble into a SQLAlchemy URI."""

    for key in (
        "DB_URI",
        "DORIS_HOST",
        "DORIS_PORT",
        "DORIS_USER",
        "DORIS_PASSWORD",
        "DORIS_DATABASE",
    ):
        monkeypatch.delenv(key, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "DORIS_HOST=127.0.0.1",
                "DORIS_PORT=9130",
                "DORIS_USER=test_user",
                "DORIS_PASSWORD=topsecret",
                "DORIS_DATABASE=bi_amz",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    reload_engine_module()

    import scpc.db.engine as engine_module

    assert (
        engine_module._get_db_uri()
        == "mysql+pymysql://test_user:topsecret@127.0.0.1:9130/bi_amz"
    )


def test_missing_db_uri_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A clear error should surface when neither URI nor components exist."""

    for key in ("DB_URI", "DORIS_HOST", "DORIS_USER", "DORIS_DATABASE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    reload_engine_module()

    import scpc.db.engine as engine_module

    with pytest.raises(RuntimeError) as err:
        engine_module._get_db_uri()

    message = str(err.value)
    assert "DB_URI" in message
    assert "DORIS_HOST" in message
