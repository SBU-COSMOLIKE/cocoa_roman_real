"""HOD galaxy power in the Limber C_l^gg and C_l^gs (include_HOD_GX).

With include_HOD_GX = 1 the batched engines replace the perturbative
galaxy bias with the halo-model spectra of halo.c,

  C_l^gg = int da (dchi/da)/f_K^2 [ W_d^2 p_gg + 2 W_d W_m p_gm
                                    + W_m^2 P_delta ],
  C_l^gs = int da (dchi/da) ep2/f_K^2 [ W_d p_gm + W_m P_delta ]
                                      (W_kappa - W_source C1),

with W_d = W_gal (no bias factor; the bias lives inside the HOD) and
W_m = W_mag ell_prefactor b_mag. The checks per probe:

1. Structure in the magnification amplitude: C_l^gg(b_mag) is exactly
   quadratic and C_l^gs(b_mag) exactly linear, the terms above and
   nothing else. (b_mag = 0 flips the a-range gate of
   redshift_spline.c, a discrete quadrature change, so every b_mag
   here stays nonzero; the "magnification-free" arm uses 1e-3.)
2. The pure-magnification coefficient, isolated by a second (gg) or
   first (gs) difference, must agree with the same difference of the
   standard engine: both integrate an identical W_m-power P_delta
   term over identical nodes and weights, so this pins the HOD
   quadrature against the production engine to floating-point
   cancellation noise.
3. Turning the flag off returns the standard C_l bitwise (the caches
   key on the flag in both directions).
4. At the lowest multipole and near-zero b_mag, p_gg -> bgal^2
   P_delta and p_gm -> bgal P_delta, so C^HOD/C^std(b1 = 1) tracks
   bgal^2 (gg) and bgal (gs). Loose: bgal varies over the bin, the
   HMx 2-halo correction enters, and weak-signal ggl pairs amplify
   the model difference, so the gs bound is on the median.

Here b_mag = alpha * bmag0 with bmag0 = 0.6 + 0.05 i for lens bin i, so
a function of b_mag is sampled at alpha = 1, 2, 3 (4). For a polynomial
in alpha the finite differences

  f(4) - 3 f(3) + 3 f(2) - f(1) = 0   (any quadratic)
  f(3) - 2 f(2) + f(1)          = 0   (any linear function)

test the structure, while f(3) - 2 f(2) + f(1) of a quadratic (twice its
alpha^2 coefficient) and f(2) - f(1) of a linear function (its slope)
isolate the magnification terms compared in check 2.

The state is the frozen NLA 3x2pt model (example2): the gs HOD path
supports NLA only and aborts under TATT (test_halo's own halo state is
example1 + TATT, which is fine for the gg-only checks but not here).

Slow (a model build plus several C_l table rebuilds): runs only with
COCOA_HALO_SLOW=1, like the spectrum tier of test_halo.py.
"""
import os
import numpy as np
import pytest
# tests/conftest.py puts tests/ on the module search path (the harness);
# test_halo, in this folder, supplies the pinned HOD parameters.
import cocoa_test_utils as u
import test_halo as th

# pytestmark is read by pytest: the skipif mark applies to every test of
# this file, which then runs only when COCOA_HALO_SLOW=1.
RUN_SLOW = os.environ.get("COCOA_HALO_SLOW", "0") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_SLOW,
    reason="slow HOD C_l checks; set COCOA_HALO_SLOW=1 to run")

# Five multipoles, log spaced from l = 31 to 3163; the low-l bias limit
# (check 4) reads the first one.
# STRUCTURE_RTOL: the polynomial structure is exact, so the finite
#   differences vanish to rounding.
# ORACLE_RTOL: HOD and standard engines integrate the same magnification
#   term on the same nodes; only cancellation noise separates them.
# GG_BIAS_LIMIT_RTOL, GS_BIAS_LIMIT_MEDIAN: the loose bounds of check 4.
# TINY_BMAG: the "magnification-free" amplitude (b_mag = 0 would change
#   the quadrature, see check 1).
ELLS = np.array([31.0, 97.0, 313.0, 1021.0, 3163.0])
STRUCTURE_RTOL = 1.0e-12
ORACLE_RTOL = 1.0e-8
GG_BIAS_LIMIT_RTOL = 0.15
GS_BIAS_LIMIT_MEDIAN = 0.10
TINY_BMAG = 1.0e-3


@pytest.fixture(scope="module")
def cells():
    """Standard and HOD C_l cubes at b_mag = alpha * bmag0 (NLA state).

    Builds the frozen example2 model, evaluates its fiducial, pins the
    HOD of every lens bin (the five Coupon et al. 2012 rows of test_halo,
    reused cyclically over the eight bins by ni % 5), then computes the
    C_gg and C_gs cubes [n_ell, n_lens, n_lens or n_source] with the
    flag off (alpha = 1, 2, 3 and TINY_BMAG), with the flag on (alpha = 1
    to 4 and TINY_BMAG), and with the flag off again (alpha = 1).

    Returns:
      dict of the cubes (keyed by alpha), the compiled module, the
      likelihood options and the number of lens bins
    """
    info = u.load_frozen_info("example2", tatt=False)
    name = u.EXAMPLES["example2"]["likelihood"]
    model = u.make_model(info)
    point = u.build_point(model, "example2", tatt=False)
    u.evaluate_chi2(model, point)
    import cosmolike_roman_real_interface as ci

    nbin = int(model.likelihood[name].lens_ntomo)
    for ni in range(nbin):
        ci.set_nuisance_hod(ni=ni,
                            hod=np.array(th.hod_of_bin(ni % 5), dtype=float),
                            gc=th.GALAXY_CONCENTRATION_FACTOR)
    state = {"like": info["likelihood"][name]}

    Z = np.zeros(nbin)
    O = np.ones(nbin)
    bmag0 = 0.6 + 0.05*np.arange(nbin)

    def cubes():
        """Return copies of the current (C_gg, C_gs) cubes at ELLS."""
        return (np.asarray(ci.C_gg_tomo_limber(l=ELLS)).copy(),
                np.asarray(ci.C_gs_tomo_limber(l=ELLS)).copy())

    def set_bias(alpha):
        """Set b1 = 1, b_mag = alpha * bmag0 and the other biases to 0."""
        ci.set_nuisance_bias(B1=O, B2=Z, B_MAG=alpha*bmag0, B3nl=Z, BK=Z)

    std_gg, std_gs = {}, {}
    for alpha in (1.0, 2.0, 3.0):
        set_bias(alpha)
        std_gg[alpha], std_gs[alpha] = cubes()
    set_bias(TINY_BMAG)
    std_gg_tiny, std_gs_tiny = cubes()

    ci.init_include_HOD_GX(1)
    hod_gg, hod_gs = {}, {}
    for alpha in (1.0, 2.0, 3.0, 4.0):
        set_bias(alpha)
        hod_gg[alpha], hod_gs[alpha] = cubes()
    set_bias(TINY_BMAG)
    hod_gg_tiny, hod_gs_tiny = cubes()

    ci.init_include_HOD_GX(0)
    set_bias(1.0)
    back_gg, back_gs = cubes()

    return {"ci": ci, "state": state, "nbin": nbin,
            "std_gg": std_gg, "std_gs": std_gs,
            "hod_gg": hod_gg, "hod_gs": hod_gs,
            "std_gg_tiny": std_gg_tiny, "std_gs_tiny": std_gs_tiny,
            "hod_gg_tiny": hod_gg_tiny, "hod_gs_tiny": hod_gs_tiny,
            "back_gg": back_gg, "back_gs": back_gs}


def test_hod_cell_gg_quadratic_in_bmag(cells):
    """Check 1 (gg): the third difference in alpha vanishes on the auto
    spectra (the diagonal [:, d, d] of the cube)."""
    hod = cells["hod_gg"]
    d = np.arange(cells["nbin"])
    third = hod[4.0] - 3*hod[3.0] + 3*hod[2.0] - hod[1.0]
    rel = np.max(np.abs(third[:, d, d])/np.abs(hod[1.0][:, d, d]))
    assert rel < STRUCTURE_RTOL


def test_hod_cell_gs_linear_in_bmag(cells):
    """Check 1 (gs): the second difference in alpha vanishes on every
    pair the standard engine fills (excluded pairs are zero)."""
    hod = cells["hod_gs"]
    mask = cells["std_gs"][1.0] != 0
    second = hod[3.0] - 2*hod[2.0] + hod[1.0]
    rel = np.max(np.abs(second[mask])/np.abs(hod[1.0][mask]))
    assert rel < STRUCTURE_RTOL


def test_hod_cell_gg_magnification_oracle(cells):
    """Check 2 (gg): the second differences of the HOD and standard
    engines (the pure-magnification term) agree to ORACLE_RTOL."""
    std, hod = cells["std_gg"], cells["hod_gg"]
    d = np.arange(cells["nbin"])
    tmm_hod = hod[3.0] - 2*hod[2.0] + hod[1.0]
    tmm_std = std[3.0] - 2*std[2.0] + std[1.0]
    rel = np.abs(tmm_hod - tmm_std)[:, d, d]/np.abs(tmm_std)[:, d, d]
    assert np.max(rel) < ORACLE_RTOL


def test_hod_cell_gs_magnification_oracle(cells):
    """Check 2 (gs): the first differences of the two engines (the
    magnification slope) agree to ORACLE_RTOL."""
    std, hod = cells["std_gs"], cells["hod_gs"]
    mask = std[1.0] != 0
    mag_hod = hod[2.0] - hod[1.0]
    mag_std = std[2.0] - std[1.0]
    rel = np.abs(mag_hod - mag_std)[mask]/np.abs(mag_std)[mask]
    assert np.max(rel) < ORACLE_RTOL


def test_hod_cell_flag_returns_bitwise(cells):
    """Check 3: flag off again reproduces the standard cubes bitwise."""
    assert np.array_equal(cells["back_gg"], cells["std_gg"][1.0])
    assert np.array_equal(cells["back_gs"], cells["std_gs"][1.0])


def test_hod_cell_gg_low_ell_bias_limit(cells):
    """Check 4 (gg): at l = 31 and tiny b_mag, C^HOD/C^std tracks bgal^2
    at the lens bin's mean redshift, within GG_BIAS_LIMIT_RTOL."""
    ci = cells["ci"]
    state = cells["state"]
    d = np.arange(cells["nbin"])
    ratio = cells["hod_gg_tiny"][0, d, d]/cells["std_gg_tiny"][0, d, d]
    bgal2 = np.array(
        [ci.bgal(ni=ni, a=th.lens_bin_mean_scale_factor(state, ni))**2
         for ni in range(cells["nbin"])])
    assert np.all(np.abs(ratio/bgal2 - 1.0) < GG_BIAS_LIMIT_RTOL)


def test_hod_cell_gs_low_ell_bias_limit(cells):
    """Check 4 (gs): at l = 31 and tiny b_mag, the median deviation of
    C^HOD/C^std from bgal stays below GS_BIAS_LIMIT_MEDIAN. np.argwhere
    lists the (lens, source) index of every filled pair; column 0 is
    the lens bin."""
    ci = cells["ci"]
    state = cells["state"]
    mask0 = cells["std_gs_tiny"][0] != 0
    ratio = cells["hod_gs_tiny"][0][mask0]/cells["std_gs_tiny"][0][mask0]
    lens_of_entry = np.argwhere(mask0)[:, 0]
    bgal = np.array(
        [ci.bgal(ni=int(ni), a=th.lens_bin_mean_scale_factor(state, int(ni)))
         for ni in lens_of_entry])
    dev = np.abs(ratio/bgal - 1.0)
    assert float(np.median(dev)) < GS_BIAS_LIMIT_MEDIAN
