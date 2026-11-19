"""Unit test: the non-Limber early-exit tolerance, the FKEM hand-off
and the noise of the low-rank (SVD) engine.

The non-Limber real-space C_l switch a lens bin (gg) or a lens-source
pair (gs) to Limber once they agree with Limber within a tolerance; the
switch multipole moves with the parameters and puts steps into
finite-difference derivatives. init_nonlimber_tolerance sets the
tolerances (core defaults 0.002 for gg and 0.01 for gs; 0 = no early
exit). This project's likelihood keys nonlimber_tolerance_gg and
nonlimber_tolerance_gs are 0: production runs without the early
exit, and tests 2-4 and 8 switch it on with the core defaults. The
shared factory (cosmolike_core/cocoa_testing_suites.py,
nonlimber_tolerance_suite) documents the checks. On example2:

    This project's 3x2pt data vector is in real space: w(theta)
    (and gamma_t where adopt_limber_gs = 0) runs the non-Limber
    path with the early exit, so tests 3 and 4 measure what the
    tolerance does here.

The whitened fourth difference of five evaluations is measured with
cosmolike_notebook_utils.fisher.get_stencil_noise, using h = 0.025 for
w0, wa and roman_B1_8. This statistic includes smooth fourth-order
curvature as well as possible numerical noise.

The FKEM/SVD comparison sets both the clustering and galaxy-shear caps
to the same requested value, 150 or 200, for each engine. Otherwise a
changed galaxy-shear cap can contaminate the clustering-engine comparison.
The production state is restored after the measurements. The empirical
bounds below are unchanged; the 2026-11-19 run with six OpenMP workers
passed all these checks, as recorded in refreeze_record_20261119.json.
Passing these diagnostics does not by itself establish Fisher convergence
with respect to endpoint, sampling density or step size.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_real/tests/data_vector/test_nonlimber_tolerance.py -s
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = os.environ.get("COCOA_TEST_OMP_THREADS", "4")

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# These empirical bounds were retained during the endpoint refreeze;
# the module docstring explains what their diagnostics do and do not test.
TestNonLimberTolerance = u.nonlimber_tolerance_suite(
    u, name="TestNonLimberTolerance", example="example2",
    bias_parameter="roman_B1_8",
    expected_defaults=(0.0, 0.0),
    change_bound=0.06,
    noise_bound=0.002,
    handoff_margin=0.1,
    svd_rule='not_larger',
    svd_margin=0.1)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
