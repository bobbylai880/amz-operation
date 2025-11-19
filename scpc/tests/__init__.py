"""Test package bootstrap utilities."""

from scpc.utils.dependencies import PackageSpec, ensure_packages

# Ensure heavy optional dependencies required by the test-suite are available
# in the CI environment. The versions align with requirements-dev.
ensure_packages(
    [
        PackageSpec(module="numpy", package="numpy>=2.3,<3"),
        PackageSpec(module="pandas", package="pandas>=2.3,<3"),
    ],
    quiet=True,
)
