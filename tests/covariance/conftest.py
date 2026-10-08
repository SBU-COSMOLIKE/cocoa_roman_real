"""Locate the project interface for the optional covariance test sector."""

from pathlib import Path
import sys

# parents[2] of this file is the project folder; its interface/ folder
# holds the compiled module.
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/"interface"))

import pytest
import cosmolike_roman_real_interface as ci


@pytest.fixture(scope="session", autouse=True)
def covariance_build():
    """Skip this sector when covariance generation was intentionally omitted.

    A fixture is a function pytest runs around tests. scope="session" runs
    it once, and autouse=True applies it to every test under this folder
    without the tests naming it, so pytest.skip here skips them all when
    the compiled module reports has_covariance = False.
    """
    if not getattr(ci, "has_covariance", False):
        pytest.skip(
            "Covariance generation is disabled. Unset "
            "IGNORE_COSMOLIKE_ROMAN_REAL_COVARIANCE after start_cocoa.sh, "
            "then recompile this project."
        )
