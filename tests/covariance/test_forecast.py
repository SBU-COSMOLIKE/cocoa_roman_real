"""Check this project's galaxy/shear covariance interface and catalog inputs.

Run separately from tests/data_vector; the shared check uses small numerical
settings and a measured subset, so it does not certify survey convergence.
"""

from pathlib import Path
import sys

# parents[2] of this file is the project folder and parents[1] of the
# project is cocoa/Cocoa: the shared cosmolike_core helpers, the compiled
# module (interface/) and the survey adapter (covariance/) go first on the
# module search path.
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project.parents[1]/"external_modules/code/cosmolike_core"))
sys.path.insert(0, str(project/"interface"))
sys.path.insert(0, str(project/"covariance"))

from cocoa_covariance_testing import check_project_forecast
import cosmolike_roman_real_interface as ci
import roman_real_covariance as survey


def test_forecast_adapter(tmp_path):
    """Check that the real and Fourier covariance components repeat and save intact.

    check_project_forecast computes the components at one and eight OpenMP
    threads, requires them to agree, and checks the saved archive and the
    layout sizes, 2115 (real) and 1575 (Fourier).

    Arguments:
      tmp_path = a fresh temporary folder that pytest creates for this test
                 (a built-in fixture, requested by naming the argument)
    """
    check_project_forecast(
        interface=ci, survey=survey, expected_sizes=(2115, 1575), directory=tmp_path,
    )
