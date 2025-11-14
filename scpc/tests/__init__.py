"""Test package bootstrap utilities."""

from scpc.utils.dependencies import PackageSpec, ensure_packages

# Make sure heavy optional dependencies used by the competition ETL tests are
# available in the CI environment. The versions align with requirements-dev.
ensure_packages(
    [
        PackageSpec(module="numpy", package="numpy>=2.3,<3"),
        PackageSpec(module="pandas", package="pandas>=2.3,<3"),
    ],
    quiet=True,
)
