from __future__ import annotations

from types import SimpleNamespace

from scpc.utils import dependencies


def test_ensure_packages_installs_missing(monkeypatch):
    calls: list[str] = []
    pip_commands: list[list[str]] = []

    def fake_import(name: str):
        if name == "missing_pkg" and not calls:
            calls.append("first")
            raise ModuleNotFoundError
        calls.append("import")
        return SimpleNamespace()

    def fake_run(cmd, check, stdout=None, stderr=None, text=None):  # noqa: ANN001 - signature matches subprocess
        pip_commands.append(cmd)
        return SimpleNamespace(stdout="installed")

    monkeypatch.setattr(dependencies, "importlib", SimpleNamespace(import_module=fake_import))
    monkeypatch.setattr(dependencies.subprocess, "run", fake_run)
    dependencies.reset_dependency_cache()

    dependencies.ensure_packages([("missing_pkg", "missing-pkg")])

    assert pip_commands == [[dependencies.sys.executable, "-m", "pip", "install", "missing-pkg"]]
    assert calls[0] == "first"


def test_ensure_packages_caches(monkeypatch):
    imports: list[str] = []

    def fake_import(name: str):
        imports.append(name)
        if name == "another_pkg" and len(imports) == 1:
            raise ModuleNotFoundError
        return SimpleNamespace()

    def fake_run(cmd, check, stdout=None, stderr=None, text=None):  # noqa: ANN001
        return SimpleNamespace(stdout="installed")

    monkeypatch.setattr(dependencies, "importlib", SimpleNamespace(import_module=fake_import))
    monkeypatch.setattr(dependencies.subprocess, "run", fake_run)
    dependencies.reset_dependency_cache()

    dependencies.ensure_packages([("another_pkg", "another-pkg")])
    dependencies.ensure_packages([("another_pkg", "another-pkg")])

    assert imports.count("another_pkg") == 2  # one failure + one success
