"""Unit test: the scale-cut diagnostic functions (cosmo2D_scuts).

These are the notebook-facing derivative diagnostics of
arXiv:2011.06469, eq. 17: the normalized Fourier derivative
dlnC_ss/dlnk, its real-space sibling dlnxi_pm/dlnk, and the response
functions rf_C_ss and rf_xi that accumulate |dln X/dlnk| up to a cutoff
wavenumber (normalized by the full integral). They run on the batch
_work machinery (dC_ss_dlnk_tomo_limber_work and the Gauss-Legendre
node arrays of the RF functions), and this test guards that machinery.

The low multipoles (l = 3 and 10) are load-bearing: at l <= 20 an
exact-scalar low-l evaluation underflows k to 0 inside the
normalization integrand, where a k > 0 guard would call exit(1) and
also kill a jupyter kernel running the call. The test requires finite
values there.

Checks, all in one process on the frozen TATT cosmic-shear fiducial:
  1. every scalar and array overload returns finite values;
  2. the scalar overloads agree with the matching array-overload
     entries (they run the same batch engines);
  3. the response functions lie in [0, 1] up to quadrature slack and
     grow with the cutoff wavenumber.
"""
import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load; setdefault
# keeps a value the user exported and sets 4 otherwise.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pytest

# tests/conftest.py puts tests/ on the module search path, so the harness
# imports directly when pytest collects this file.
import cocoa_test_utils as u

# RTOL: the scalar and array overloads run the same batch engines, so
# they agree to rounding.
KK = np.array([0.05, 0.2, 1.0])       # wavenumbers in (Mpc/h)^-1
ELL = np.array([3.0, 10.0, 100.0, 1000.0])  # includes the old fatal l <= 20
RTOL = 1e-10


@pytest.fixture(scope="module")
def shear_state():
    """Evaluate the frozen TATT cosmic-shear fiducial once; return ci.

    A pytest fixture with scope="module" runs once for this file, and
    pytest hands its return value to every test that names shear_state
    as an argument. The evaluation leaves cosmolike holding the
    fiducial state that the diagnostics read.

    Returns:
      the compiled module cosmolike_roman_real_interface
    """
    info = u.load_frozen_info("example1", tatt=True)
    model = u.make_model(info)
    point = u.build_point(model, "example1", tatt=True)
    u.evaluate_chi2(model, point)
    import cosmolike_roman_real_interface as ci
    return ci


def test_dlnC_ss_dlnk(shear_state):
    """dlnC_ss/dlnk: finite, and the scalar call matches the array entry.

    The array overload returns (EE, BB) cubes indexed [k, l, ni, nj].
    """
    ci = shear_state
    (EE, BB) = ci.dlnC_ss_dlnk_tomo_limber(k=KK, l=ELL)
    EE, BB = np.asarray(EE), np.asarray(BB)
    assert np.isfinite(EE).all() and np.isfinite(BB).all()
    (ee, bb) = ci.dlnC_ss_dlnk_tomo_limber(k=float(KK[1]), l=float(ELL[2]),
                                           ni=0, nj=1)
    assert np.isclose(ee, EE[1, 2, 0, 1], rtol=RTOL)
    assert np.isclose(bb, BB[1, 2, 0, 1], rtol=RTOL)


def test_rf_C_ss(shear_state):
    """rf_C_ss: finite, inside [0, 1] up to quadrature slack, growing
    with the cutoff, and scalar-equal to the array entry at l = 3."""
    ci = shear_state
    (EE, tmp) = ci.rf_C_ss_tomo_limber(k=KK, l=ELL)
    EE = np.asarray(EE)
    assert np.isfinite(EE).all()
    # a normalized cumulative fraction, monotone in the cutoff
    assert (EE > -1e-6).all() and (EE < 1.0 + 1e-2).all()
    filled = np.abs(EE[-1]) > 1e-12
    assert (EE[-1][filled] >= EE[0][filled] - 1e-6).all()
    (ee, bb) = ci.rf_C_ss_tomo_limber(k=float(KK[1]), l=float(ELL[0]),
                                      ni=0, nj=1)  # l = 3: the old fatal
    assert np.isclose(ee, EE[1, 0, 0, 1], rtol=RTOL)


def test_dlnxi_dlnk(shear_state):
    """dlnxi_pm/dlnk: finite, and the scalar call matches the array row."""
    ci = shear_state
    (XP, XM) = ci.dlnxi_dlnk_pm_tomo_limber(k=KK)
    XP, XM = np.asarray(XP), np.asarray(XM)
    assert np.isfinite(XP).all() and np.isfinite(XM).all()
    (xp, xm) = ci.dlnxi_dlnk_pm_tomo_limber(k=float(KK[1]))
    assert np.allclose(np.asarray(xp), XP[1], rtol=RTOL)
    assert np.allclose(np.asarray(xm), XM[1], rtol=RTOL)


def test_rf_xi(shear_state):
    """rf_xi: finite, inside [0, 1] up to slack, and scalar-equal to the
    array entry [k, theta bin, ni, nj]."""
    ci = shear_state
    (XP, XM) = ci.rf_xi_tomo_limber(k=KK[:2])
    XP, XM = np.asarray(XP), np.asarray(XM)
    assert np.isfinite(XP).all() and np.isfinite(XM).all()
    assert (XP > -1e-6).all() and (XP < 1.0 + 1e-2).all()
    (xp, xm) = ci.rf_xi_tomo_limber(k=float(KK[0]), nt=5, ni=0, nj=1)
    assert np.isclose(xp, XP[0, 5, 0, 1], rtol=RTOL)
    assert np.isclose(xm, XM[0, 5, 0, 1], rtol=RTOL)
