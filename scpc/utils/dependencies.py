"""Runtime dependency helpers."""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PackageSpec:
    """Description of a dependency that might require installation."""

    module: str
    package: str


_ENSURED: set[str] = set()


def _normalise_packages(packages: Iterable[str | tuple[str, str] | PackageSpec]) -> list[PackageSpec]:
    normalised: list[PackageSpec] = []
    for item in packages:
        if isinstance(item, PackageSpec):
            normalised.append(item)
        elif isinstance(item, tuple):
            module, package = item
            normalised.append(PackageSpec(module=module, package=package))
        else:
            normalised.append(PackageSpec(module=item, package=item))
    return normalised


def ensure_packages(
    packages: Iterable[str | tuple[str, str] | PackageSpec],
    *,
    quiet: bool = True,
) -> None:
    """Ensure that optional dependencies are importable.

    ``packages`` can include either a module name string, a ``(module, package)``
    tuple when the pip package name differs from the importable module, or a
    :class:`PackageSpec`. When a module is missing this helper invokes
    ``pip install`` to fetch it and raises :class:`RuntimeError` if the module is
    still unavailable afterwards.
    """

    for spec in _normalise_packages(packages):
        if spec.module in _ENSURED:
            continue
        try:
            importlib.import_module(spec.module)
        except ModuleNotFoundError:
            LOGGER.info("Installing missing dependency", extra={"module": spec.module, "package": spec.package})
            cmd = [sys.executable, "-m", "pip", "install", spec.package]
            if quiet:
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                LOGGER.debug(
                    "pip install output", extra={"module": spec.module, "package": spec.package, "output": result.stdout}
                )
            else:
                subprocess.run(cmd, check=True)
        try:
            importlib.import_module(spec.module)
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(
                f"Unable to import required package '{spec.module}' even after attempting installation."
            ) from exc
        _ENSURED.add(spec.module)


def reset_dependency_cache() -> None:
    """Reset the internal ensure cache (useful for tests)."""

    _ENSURED.clear()
