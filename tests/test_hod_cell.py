"""HOD galaxy power in the Limber C_l^gg (cosmo2D.c include_HOD_GX).

With include_HOD_GX = 1 the batched gg engine replaces the perturbative
galaxy bias with the halo-model spectra of halo.c,

  C_l^gg = int da (dchi/da)/f_K^2 [ W_d^2 p_gg + 2 W_d W_m p_gm
                                    + W_m^2 P_delta ],

with W_d = W_gal (no bias factor; the bias lives inside the HOD) and
W_m = W_mag ell_prefactor b_mag. The checks:

1. C_l(b_mag) is exactly quadratic in the magnification amplitude
   (the three terms above and nothing else), at fixed nonzero b_mag
   support (b_mag = 0 flips the a-range gate of redshift_spline.c, a
   discrete quadrature change, so the alphas here stay nonzero).
2. The b_mag^2 coefficient, isolated by a second difference, must
   agree with the SAME second difference of the standard engine: both
   are int W_m^2 P_delta over identical nodes and weights, so this
   pins the HOD branch's quadrature against the production engine to
   floating-point cancellation noise.
3. Turning the flag off returns the standard C_l bitwise (the cache
   keys on the flag in both directions).
4. At the lowest multipole and near-zero b_mag (1e-3, so the a-range
   gate of redshift_spline.c stays on its magnification-active
   branch), p_gg -> bgal^2 P_delta, so C^HOD/C^std(b1 = 1) ~
   bgal(ni, <a>)^2 (loose: bgal varies over the bin and the HMx
   2-halo correction enters).

Slow (a model build plus several C_l table rebuilds): runs only with
COCOA_HALO_SLOW=1, like the spectrum tier of test_halo.py.
"""
import os
import numpy as np
import pytest
import test_halo as th

RUN_SLOW = os.environ.get("COCOA_HALO_SLOW", "0") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_SLOW,
    reason="slow HOD C_l checks; set COCOA_HALO_SLOW=1 to run")

ELLS = np.array([31.0, 97.0, 313.0, 1021.0, 3163.0])
QUADRATIC_RTOL = 1.0e-12
ORACLE_RTOL = 1.0e-8
BIAS_LIMIT_RTOL = 0.15
TINY_BMAG = 1.0e-3


@pytest.fixture(scope="module")
def cells():
    """Standard and HOD C_l^gg cubes at b_mag = alpha * bmag0."""
    state = th.build_halo_state()
    ci = state["ci"]
    nbin = state["nbin"]
    Z = np.zeros(nbin)
    O = np.ones(nbin)
    bmag0 = 0.6 + 0.05*np.arange(nbin)

    def cube():
        return np.asarray(ci.C_gg_tomo_limber(l=ELLS)).copy()

    def set_bias(alpha):
        ci.set_nuisance_bias(B1=O, B2=Z, B_MAG=alpha*bmag0, B3nl=Z, BK=Z)

    std = {}
    for alpha in (1.0, 2.0, 3.0):
        set_bias(alpha)
        std[alpha] = cube()
    set_bias(TINY_BMAG)
    std_tiny = cube()

    ci.init_include_HOD_GX(1)
    hod = {}
    for alpha in (1.0, 2.0, 3.0, 4.0):
        set_bias(alpha)
        hod[alpha] = cube()
    set_bias(TINY_BMAG)
    hod_tiny = cube()

    ci.init_include_HOD_GX(0)
    set_bias(1.0)
    back = cube()

    return {"state": state, "std": std, "hod": hod, "back": back,
            "std_tiny": std_tiny, "hod_tiny": hod_tiny, "nbin": nbin}


def test_hod_cell_quadratic_in_bmag(cells):
    hod = cells["hod"]
    d = np.arange(cells["nbin"])
    third = hod[4.0] - 3*hod[3.0] + 3*hod[2.0] - hod[1.0]
    rel = np.max(np.abs(third[:, d, d])/np.abs(hod[1.0][:, d, d]))
    assert rel < QUADRATIC_RTOL


def test_hod_cell_magnification_oracle(cells):
    std, hod = cells["std"], cells["hod"]
    d = np.arange(cells["nbin"])
    tmm_hod = hod[3.0] - 2*hod[2.0] + hod[1.0]
    tmm_std = std[3.0] - 2*std[2.0] + std[1.0]
    rel = np.abs(tmm_hod - tmm_std)[:, d, d]/np.abs(tmm_std)[:, d, d]
    assert np.max(rel) < ORACLE_RTOL


def test_hod_cell_flag_returns_bitwise(cells):
    assert np.array_equal(cells["back"], cells["std"][1.0])


def test_hod_cell_low_ell_bias_limit(cells):
    state = cells["state"]
    ci = state["ci"]
    d = np.arange(cells["nbin"])
    ratio = cells["hod_tiny"][0, d, d]/cells["std_tiny"][0, d, d]
    bgal2 = np.array(
        [ci.bgal(ni=ni, a=th.lens_bin_mean_scale_factor(state, ni))**2
         for ni in range(cells["nbin"])])
    assert np.all(np.abs(ratio/bgal2 - 1.0) < BIAS_LIMIT_RTOL)
