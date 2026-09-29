"""Unit test: the halo model (cosmolike halo.c) through its Python bindings.

halo.c computes cosmolike's halo model: the halo mass function and
halo bias (Tinker et al. 2010), halo concentrations (Bhattacharya et
al. 2013) and density profiles (NFW), the gas pressure profile
(Komatsu-Seljak), the HOD galaxy integrals, and the power spectra
built from them (p_mm, p_my, p_yy, p_gm, p_gg). The compiled interface
exposes each function under its halo.c name through
cosmolike/halo_wrapper.cpp, so ci.p_mm(k, a) here runs halo.c p_mm.

Units are halo.c's code units: k in (c/H0)^-1 (k[h/Mpc] * COVERH0),
power spectra in (c/H0)^3, masses in M_sun/h, number densities in
(c/H0)^-3.

Four groups of checks, all in ONE pytest process on the frozen
cosmic-shear example at its fiducial point, with the HOD and gas
parameters pinned below:

  1. TestFrozenReferences - every probe reproduces, point by point,
     the values frozen from the current (GSL fixed-quadrature)
     implementation: the ground truth that every later rewrite of
     halo.c is compared against.
  2. TestPhysicsInvariants - properties the physics itself demands
     (published fits, normalizations, large-scale limits, bounds);
     they hold for ANY correct implementation, so they survive every
     rewrite unchanged.
  3. TestCacheConsistency - halo.c caches its tables behind keys
     (cosmology.random, Ntable.random, nuisance.random_*): a change
     must be seen, and undoing it must reproduce the first values bit
     for bit.
  4. TestDeterminism - the same inputs give the same bits: on repeated
     calls, and with 1, 4 or 8 OpenMP threads.

Blocked probes. HEAD_DEFECTS below lists what the current build
cannot evaluate (today: the Compton-y spectra, waiting on Omega_b
through the cobaya glue): those tests are marked xfail(run=False) -
reported, never executed, since halo.c aborts the process - and the
generator leaves them out of the frozen file. The ticket that
unblocks one removes its entry and regenerates the frozen file.

Slow tests. Building a spectrum table costs one 1000-node mass
integral per (a, k) node, about a minute at 4 threads for p_mm alone;
those tests run only with COCOA_HALO_SLOW=1. The default run takes a
model build plus a few seconds of halo integrals.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_real/tests/test_halo.py
    COCOA_HALO_SLOW=1 python -m pytest ./projects/roman_real/tests/test_halo.py

The frozen values (frozen/halo_reference.json) come from

    python ./projects/roman_real/tests/generate_frozen_reference.py --halo

and until that file exists the frozen tests skip.
"""

import functools
import json
import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys

import numpy as np
import pytest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

# =============================================================================
# CONFIGURATION
# =============================================================================

# The configuration the halo quantities are evaluated on: the frozen
# cosmic-shear example (its likelihood also reads the lens n(z), which
# the HOD integrals need). test_scale_cut_diagnostics.py builds the
# same configuration in-process, so the two modules can share one
# pytest process: cosmolike aborts when a configuration of different
# data-vector dimensions initializes after another (cocoa_test_utils).
EXAMPLE = "example1"
TATT = True

# The ground truth written by generate_frozen_reference.py --halo.
HALO_REFERENCE_FILE = os.path.join(u.FROZEN_DIR, "halo_reference.json")

# The spectrum tables are slow to build (see the module docstring).
# skipif evaluates its condition once, at import.
RUN_SLOW = os.environ.get("COCOA_HALO_SLOW", "0") == "1"
slow = pytest.mark.skipif(
    not RUN_SLOW,
    reason="slow halo-model spectrum table; set COCOA_HALO_SLOW=1 to run")

# Constants of halo.c and structs.c the independent recomputations
# below need (mirrored: the C code has no binding for them).
DELTA_C = 1.686         # halo.c delta_c: linear collapse threshold
TINKER_DELTA = 200.0    # halo.c Delta: halo overdensity / mean density
HALO_M_MIN = 1.0e6      # structs.c limits.halo_m_min in M_sun/h
HALO_M_MAX = 1.0e17     # structs.c limits.halo_m_max in M_sun/h
COVERH0 = 2997.92458    # structs.c cosmology.coverH0 = c/H0 in Mpc/h
RHO_CRIT = 7.4775e+21   # structs.c cosmology.rho_crit (c/H0 units)
SIGMA2_N_M = 1024       # structs.c Ntable.N_M: the sigma2 ln M table nodes

# ---- the HOD and gas parameters the tests pin -------------------------------

# HOD of each lens bin, {lg M_min, sigma_lgM, lg M_1, lg M_0, alpha,
# f_c}: the Coupon et al. (2012, 1107.0616, Table B.1) fits for all
# galaxies with M_g - 5 log h < -21.8 that halo.c set_HOD hard-codes,
# one row per redshift slice (0.2-0.4, ..., 1.0-1.2), copied here so
# no test depends on set_HOD or on struct defaults. Lens bins 0-4 get
# rows 0-4; every later bin repeats row 4. Every lens bin must be
# set: halo.c aborts on a bin whose lg M_min is outside
# [10, 16], and its ngal/bgal tables cover all bins at once.
HOD_COUPON_2012 = (
    (13.17, 0.39, 14.53, 11.09, 1.27, 1.00),
    (13.18, 0.30, 14.47, 10.93, 1.36, 1.00),
    (12.96, 0.38, 14.10, 12.47, 1.28, 1.00),
    (12.80, 0.33, 13.94, 12.15, 1.52, 1.00),
    (12.62, 0.30, 13.79, 8.67, 1.50, 1.00),
)

# Galaxy concentration factor f_g of every bin (galaxy profile
# concentration = f_g x halo concentration); 1 = the galaxies trace the
# dark matter. set_HOD does not set it.
GALAXY_CONCENTRATION_FACTOR = 1.0

# Gas (Compton-y) parameters in the structs.h layout. A representative
# physical point, not a fit: Gamma = 1.17 (the Komatsu-Seljak exponent
# 1/(Gamma - 1) needs Gamma > 1), beta = 0.6, lg M_0 = 14,
# eps1 = eps2 = 0 (unread by halo.c), alpha = 1 (bound gas at the
# virial temperature), A_star = 0.03, lg M_star = 12.5,
# sigma_star = 1.2, lg T_w = 6.5, f_H = 0.752 (the primordial hydrogen
# mass fraction).
GAS_PARAMS = (1.17, 0.6, 14.0, 0.0, 0.0, 1.0, 0.03, 12.5, 1.2, 6.5, 0.752)

# ---- probe grids ------------------------------------------------------------
# np.logspace(p, q, n) = n points from 10^p to 10^q, uniform in log

NU_GRID = np.logspace(-1.0, np.log10(5.0), 16)  # peak heights nu
TINKER_A = (0.2, 0.5, 0.9)      # 0.2 < 0.25 exercises fnu's z <= 3 freeze
# Scale factors of the Eq. 7 normalization check (0.2 < 0.25 again).
TINKER_NORM_A = (0.2, 0.25, 0.4, 0.7, 0.99)
HB1NU_A = 0.5                   # the Delta = 200 bias fit does not evolve
CONC_M = np.logspace(8.0, 16.0, 9)             # M_sun/h
CONC_GROWFAC = (1.0, 0.7, 0.4)                 # D(a)
DLOGNU_M = np.logspace(7.0, 16.0, 40)          # M_sun/h
BIAS_NORM_A = np.linspace(0.05, 0.9995, 40)
# The bias_norm table grid (halo.c bias_norm): Ntable.N_a nodes
# uniform in a over [limits.a_min, 0.9999999]; N_a is 256 times the
# accuracy boost (init_accuracy_boost, rounded up). Nodes tested
# exactly, as fractions of the node range.
BIAS_NORM_A_MIN = 1.0/41.0     # structs.c limits.a_min (z = 40)
BIAS_NORM_A_END = 0.9999999    # halo.c bias_norm last node
BIAS_NORM_N_A_BASE = 256       # structs.c Ntable.N_a
BIAS_NORM_NODE_FRAC = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
U_NFW_C = (3.0, 8.0)
U_NFW_M = (1.0e11, 1.0e14)                     # M_sun/h
U_NFW_K = np.logspace(0.0, 5.0, 6)             # 3.3e-4 to 33 h/Mpc
U_NFW_A = 0.7                   # unused by the NFW form (halo.c signature)
U_KS_C = (2.0, 5.0, 10.0)       # inside [halo_uks_cmin, halo_uks_cmax]
U_KS_RV = 3.0e-4                # c/H0 (0.9 Mpc/h)
U_KS_K = np.logspace(0.0, 5.0, 6)
HOD_TEST_BINS = (0, 1)          # lens bins probed (mean z 0.32 and 0.55)
HOD_A_FACTORS = np.linspace(0.98, 1.02, 8)     # times 1/(1 + <z>_bin)
PK_K = np.logspace(-1.0, 6.0, 20)              # 3.3e-5 to 330 h/Mpc
PK_A = (0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 0.99)

# ---- known defects of the current halo.c ------------------------------------

# Probes the current build cannot evaluate. They are never called
# (xfail(run=False)) and never frozen; the ticket that unblocks one
# deletes its entry here and regenerates the frozen file (--halo).
OMEGAB_GLUE = (
    "cosmology.Omega_b is not wired through the cobaya glue yet: "
    "set_cosmological_parameters carries omega_baryon and the "
    "set_cosmology binding defaults omegab to 0, so the Compton-y "
    "spectra abort until the glue passes a positive value")
HEAD_DEFECTS = {
    "p_my": OMEGAB_GLUE,
    "p_yy": OMEGAB_GLUE,
}

# Probes whose table build is slow (module docstring).
SLOW_PROBES = ("p_mm", "p_my", "p_yy", "p_gm", "p_gg")

# ---- tolerances -------------------------------------------------------------

# Frozen references: the implementation is unchanged since the freeze,
# so only libm/compiler rounding (~1e-15 relative) may differ.
FROZEN_RTOL = 1.0e-12
# A conversion ticket enters its measured, justified bound per probe
# here, in the same commit that changes the implementation.
FROZEN_RTOL_BY_PROBE = {}

# Closed-form fits re-evaluated in numpy: pow/exp differ by a few ulp.
TINKER_RTOL = 1.0e-12
# fnu carries alpha(z) from halo.c's table: exact at 140 nodes, cubic-
# upsampled to 4096 nodes in a, read linearly; measured in numpy
# (2026-09-28) to 4.9e-8 at most, set by the linear read.
FNU_RTOL = 2.0e-7
CONC_RTOL = 1.0e-12

# u_nfw_c at k -> 0: u = 1 - k^2 <r^2>/6 + ..., <r^2> < r_Delta^2 <=
# (2.5 Mpc/h)^2 for m <= 1e15 M_sun/h; at U_NFW_K0 (3.3e-4 h/Mpc) the
# deviation is below 2e-7.
U_NFW_K0 = 1.0          # (c/H0)^-1
U_NFW_K0_ATOL = 1.0e-6
# |u| <= 1 is exact for a positive density; allow the rounding of the
# Si/Ci differences at small k r.
U_NFW_BOUND_ATOL = 1.0e-12

# A correct d ln nu/d ln M matches a central difference of the same
# sigma2 table (step DLOGNU_FD_STEP) to well under 1%; a lost factor
# (e.g. the 1/2 of ln sigma) would be >= 50%.
DLOGNU_RTOL = 1.0e-2
# ln M step of the reference central difference: two sigma2 table cells
# (0.025 in ln M) on either side.
DLOGNU_FD_STEP = 0.05
# Masses whose +-DLOGNU_FD_STEP stencil stays inside the sigma2 table
# [HALO_M_MIN, HALO_M_MAX] for both halo.c and the reference here.
DLOGNU_TEST_M = np.logspace(8.0, 15.0, 15)

# sigma2 vs an independent numpy integral of the same p_lin (dense
# trapezoid in ln x; 2e6 and 4e6 nodes agree). The masses sit on the
# table's own ln M nodes, so the table read is exact and only the lobe
# quadrature is tested. Measured 2026-09-29 at hdi = 0: 6.2e-6 max
# (N_M_internal = 192) and 5.0e-6 (exact branch); a head segment
# integrated uniformly in x instead of ln x would be off by ~6e-3.
SIGMA2_REF_RTOL = 2.0e-5
SIGMA2_REF_NODES = (0, 128, 256, 384, 512, 640, 768, 896, 1023)
SIGMA2_REF_NX = 2_000_001

# bias_norm at its own table nodes vs the numpy integral. numpy's
# leggauss rules are
# exact to rounding and converged here (1000 vs 2000 nodes agree to
# 1e-11). halo.c integrates with GSL's tabulated 128-node rule at
# hdi = 0, converged to 3e-15, but its f(nu) carries alpha(z) from the
# alpha table (FNU_RTOL) while the numpy side computes alpha exactly;
# a wrong fit or bound would be >= 1e-3.
BIAS_NORM_GL_NODES = 2000
BIAS_NORM_QUAD_RTOL = FNU_RTOL
# bias_norm table vs the numpy integral between nodes: linear
# interpolation over
# da ~ 4e-3 of a smooth function, error ~ da^2/8 |f''/f| ~ 1e-5.
BIAS_NORM_INTERP_RTOL = 1.0e-4
# Off-node test points, below the pinned last table cell.
BIAS_NORM_INTERP_A = np.linspace(0.3, 0.98, 12)
# Queries past the table's last node (0.9999999): constant extrapolation
# returns the endpoint value itself.
BIAS_NORM_PINNED_A = (0.99999995, 0.999999999)
# int b f dnu = 1 over all nu (Tinker et al. 2010 Eq. 7, which sets
# alpha); the 1e6 M_sun/h floor of the tabulated range misses the
# nu < ~0.3 tail, about 20% for the low-nu slope f ~ nu^-0.49, and the
# integrand is positive, so the tabulated part stays below 1.
BIAS_NORM_TODAY_A = 0.99
BIAS_NORM_TODAY_RANGE = (0.6, 1.0)

# ngal in (h/Mpc)^3: halos above ~1e13 M_sun/h at z < 1 have abundances
# 1e-6 to 1e-3 (h/Mpc)^3; the bracket catches only unit or
# normalization errors (coverH0^3 ~ 3e10).
NGAL_BRACKET_H3 = (1.0e-7, 1.0e-2)
# Linear bias of red galaxies in >~1e13 M_sun/h halos at z < 1.
BGAL_RANGE = (1.0, 5.0)
# The Coupon HOD puts roughly 5-20% of its galaxies in satellites; the
# floor flags only a vanishing satellite term.
FSAT_FLOOR = 1.0e-3

# Large-scale limits: at the linear P(k) peak (k = 0.01-0.02 h/Mpc) the
# 1-halo term (~<M>/rho_m, a few 1e2 (Mpc/h)^3) is a few percent of the
# 2-halo term (~1e4 (Mpc/h)^3).
P_2H_K_HMPC = (0.01, 0.02)
P_2H_A = (0.5, 0.8, 0.99)
P_2H_RTOL = 0.1
# p_my^2 <= p_mm p_yy holds node by node (Cauchy-Schwarz on the mass
# integrals); bilinear interpolation of the three ln P tables with
# shared weights preserves it, up to rounding.
CAUCHY_SCHWARZ_RTOL = 1.0e-10
# u_KS <= u_KS(k -> 0) <= 1 holds node by node; allow the rounding of
# the table and of the Gauss-Legendre sums.
U_KS_BOUND_ATOL = 1.0e-8
U_KS_K0 = 1.0e-3        # (c/H0)^-1; k rv/c ~ 1e-7, inside the table

# Cache checks: an omegam step of 0.02 moves sigma(M) at the percent
# level; a relative change below the floor means a stale table.
OMEGAM_STEP = 0.02
HOD_LGMMIN_STEP = 0.1
GAS_GAMMA_STEP = 0.03
CACHE_CHANGE_FLOOR = 1.0e-6

# Thread counts of the determinism check (the default is 4).
DETERMINISM_THREADS = (1, 8)

# =============================================================================
# PUBLISHED FITS, RECOMPUTED IN NUMPY
# =============================================================================
def tinker_bias(nu):
    """Tinker et al. 2010 halo bias, Eq. 6 with the Table 2 coefficients.

    b(nu) = 1 - A nu^a/(nu^a + delta_c^a) + B nu^b + C nu^c, with
    y = log10(Delta):
      A = 1 + 0.24 y exp(-(4/y)^4),  a = 0.44 y - 0.88
      B = 0.183,                     b = 1.5
      C = 0.019 + 0.107 y + 0.19 exp(-(4/y)^4),  c = 2.4

    Arguments:
      nu = peak height(s), float or numpy array.

    Returns:
      b(nu), same shape as nu.
    """
    y = np.log10(TINKER_DELTA)
    big_a = 1.0 + 0.24*y*np.exp(-(4.0/y)**4)
    small_a = 0.44*y - 0.88
    big_c = 0.019 + 0.107*y + 0.19*np.exp(-(4.0/y)**4)
    return (1.0 - big_a*nu**small_a/(nu**small_a + DELTA_C**small_a)
            + 0.183*nu**1.5 + big_c*nu**2.4)


def tinker_shape(nu, a):
    """Tinker et al. 2010 multiplicity function f(nu) with alpha = 1.

    [1 + (beta nu)^(-2 phi)] nu^(2 eta) exp(-gamma nu^2/2), with the
    Delta = 200 parameters of Table 4 (beta0 = 0.589, gamma0 = 0.864,
    phi0 = -0.729, eta0 = -0.243) evolving as beta0 (1+z)^0.20,
    phi0 (1+z)^-0.08, eta0 (1+z)^0.27, gamma0 (1+z)^-0.01; the evolution
    is frozen at z = 3, the edge of the fitted range (discussion after
    Eq. 12).

    Arguments:
      nu = peak height(s), float or numpy array.
      a  = scale factor.

    Returns:
      the alpha = 1 shape, same shape as nu.
    """
    onepz = 1.0/max(a, 0.25)   # 1 + z, capped at z = 3
    beta = 0.589*onepz**0.20
    phi = -0.729*onepz**-0.08
    eta = -0.243*onepz**0.27
    gamma = 0.864*onepz**-0.01
    return ((1.0 + (beta*nu)**(-2.0*phi))*nu**(2.0*eta)
            * np.exp(-gamma*nu*nu/2.0))


@functools.lru_cache(maxsize=None)
def tinker_alpha(a):
    """alpha(z) of Tinker et al. 2010: the normalization for which matter
    is unbiased with respect to itself, int_0^inf b(nu) f(nu) dnu = 1
    (Eq. 7), at every z (frozen at z = 3 with the shape).

    Trapezoid in s = ln nu (dnu = nu ds) over [-120, 4] with step 0.02:
    the integrand decays at both ends, so the rule converges fast
    (1e-12; the Table 4 value alpha = 0.368 at z = 0 comes out 0.36841).

    Arguments:
      a = scale factor.

    Returns:
      alpha at a.
    """
    ds = 0.02
    s = np.arange(-120.0, 4.0 + 0.5*ds, ds)
    nu = np.exp(s)
    g = tinker_bias(nu)*tinker_shape(nu, a)*nu
    return 1.0/(ds*(g.sum() - 0.5*(g[0] + g[-1])))


def bias_norm_integral(ci, a):
    """bias_norm(a) = int b(nu) f(nu) dnu between nu(M_min) and nu(M_max),
    recomputed in numpy from the published fits, sigma2 and D(a)
    (BIAS_NORM_GL_NODES Gauss-Legendre nodes in nu). sigma falls with M,
    so M_min gives the small nu and M_max the large one.

    Arguments:
      ci = the cosmolike interface module (sigma2, growfac).
      a  = scale factor.

    Returns:
      the integral at a.
    """
    x, w = np.polynomial.legendre.leggauss(BIAS_NORM_GL_NODES)
    d = ci.growfac(a=a)
    nu_lo = DELTA_C/(np.sqrt(ci.sigma2(M=HALO_M_MIN))*d)
    nu_hi = DELTA_C/(np.sqrt(ci.sigma2(M=HALO_M_MAX))*d)
    # map the Legendre nodes from [-1, 1] onto [nu_lo, nu_hi]
    half = 0.5*(nu_hi - nu_lo)
    nu = half*x + 0.5*(nu_hi + nu_lo)
    return half*np.sum(w*tinker_bias(nu)*tinker_multiplicity(nu, a))


def bias_norm_nodes(halo):
    """The scale factors of the bias_norm table nodes that
    BIAS_NORM_NODE_FRAC selects (same arithmetic as halo.c bias_norm)."""
    n_a = int(np.ceil(BIAS_NORM_N_A_BASE*float(halo["like"]["accuracyboost"])))
    da = (BIAS_NORM_A_END - BIAS_NORM_A_MIN)/(n_a - 1.0)
    return [BIAS_NORM_A_MIN + int(round(f*(n_a - 1)))*da
            for f in BIAS_NORM_NODE_FRAC]


def tinker_multiplicity(nu, a):
    """Tinker et al. 2010 multiplicity function f(nu), Eqs. 7-12:
    alpha(z) (Eq. 7) times the alpha = 1 shape.

    Arguments:
      nu = peak height(s), float or numpy array.
      a  = scale factor.

    Returns:
      f(nu), same shape as nu.
    """
    return tinker_alpha(float(a))*tinker_shape(nu, a)


# =============================================================================
# STATE: THE MODEL, THE PINNED PARAMETERS, THE PROBES
# =============================================================================
def hod_of_bin(ni):
    """The pinned HOD of lens bin ni (HOD_COUPON_2012; bins past the
    table repeat its last row)."""
    return HOD_COUPON_2012[min(ni, len(HOD_COUPON_2012) - 1)]


def apply_halo_parameters(state):
    """Write the pinned HOD (every lens bin) and gas parameters.

    Arguments:
      state = the dictionary from build_halo_state.

    Returns:
      nothing; the setters draw new cache keys when a value changed.
    """
    ci = state["ci"]
    for ni in range(state["nbin"]):
        ci.set_nuisance_hod(ni=ni,
                            hod=np.array(hod_of_bin(ni), dtype=float),
                            gc=GALAXY_CONCENTRATION_FACTOR)
    ci.set_nuisance_gas(gas=np.array(GAS_PARAMS, dtype=float))


def lens_bin_mean_scale_factor(state, ni):
    """1/(1 + <z>) of lens bin ni, from the frozen lens n(z) file.

    Arguments:
      state = the dictionary from build_halo_state.
      ni    = lens bin.

    Returns:
      the scale factor at the bin's mean redshift, a float.
    """
    from getdist import IniFile

    like = state["like"]
    # the dataset descriptor names the n(z) file; relativeFileName
    # resolves it against the descriptor's own folder (frozen/data)
    ini = IniFile(os.path.normpath(os.path.join(like["path"],
                                                like["data_file"])))
    table = np.loadtxt(ini.relativeFileName("nz_lens_file"))
    # column 0 = z on a uniform grid, column 1 + ni = bin ni; on a
    # uniform grid the mean is a plain weighted sum
    z, nz = table[:, 0], table[:, 1 + ni]
    return float(1.0/(1.0 + np.sum(z*nz)/np.sum(nz)))


def probe_inputs(state):
    """The input grid of every probe, as json-ready lists.

    Arguments:
      state = the dictionary from build_halo_state (the HOD probes need
              the lens n(z)).

    Returns:
      {probe name: {argument name: list of values}}; for the per-bin
      probes, "a" holds one list per entry of "ni".
    """
    def floats(values):
        return [float(v) for v in values]

    a_bins = [floats(HOD_A_FACTORS*lens_bin_mean_scale_factor(state, ni))
              for ni in HOD_TEST_BINS]
    hod = {"ni": list(HOD_TEST_BINS), "a": a_bins}
    spectra = {"k": floats(PK_K), "a": floats(PK_A)}
    spectra_bins = {"ni": list(HOD_TEST_BINS), "k": floats(PK_K),
                    "a": a_bins}
    return {
        "hb1nu": {"nu": floats(NU_GRID), "a": [HB1NU_A]},
        "fnu": {"nu": floats(NU_GRID), "a": floats(TINKER_A)},
        "conc": {"m": floats(CONC_M), "growfac_a": floats(CONC_GROWFAC)},
        "dlognudlogm": {"M": floats(DLOGNU_M)},
        "bias_norm": {"a": floats(BIAS_NORM_A)},
        "u_nfw_c": {"c": floats(U_NFW_C), "m": floats(U_NFW_M),
                    "k": floats(U_NFW_K), "a": [U_NFW_A]},
        "u_KS": {"c": floats(U_KS_C), "k": floats(U_KS_K),
                 "rv": [U_KS_RV]},
        "ngal_nointerp": hod,
        "bgal_nointerp": hod,
        "ngal": hod,
        "bgal": hod,
        "mmean_nointerp": hod,
        "fsat_nointerp": hod,
        "p_mm": spectra,
        "p_my": spectra,
        "p_yy": spectra,
        "p_gm": spectra_bins,
        "p_gg": spectra_bins,
    }


def _per_bin(function, x):
    """function(ni, a) for each probed bin, then that bin's a values."""
    return [function(ni=ni, a=a)
            for ni, a_list in zip(x["ni"], x["a"]) for a in a_list]


def _spectrum(function, x):
    """The array overload function(k array, a), for each a in turn."""
    k = np.asarray(x["k"], dtype=float)
    # np.ravel flattens the (nk, 1) column carma returns
    return [v for a in x["a"] for v in np.ravel(function(k=k, a=a))]


def _spectrum_per_bin(function, x, auto):
    """The per-bin array overload, bin outer, then a, then k; auto =
    True passes nj = ni (p_gg)."""
    k = np.asarray(x["k"], dtype=float)
    out = []
    for ni, a_list in zip(x["ni"], x["a"]):
        # ** unpacks the dictionary into keyword arguments: nj = ni for
        # the auto spectrum, nothing for p_gm
        extra = {"nj": ni} if auto else {}
        for a in a_list:
            out.extend(np.ravel(function(k=k, a=a, ni=ni, **extra)))
    return out


# One evaluator per probe: (compiled interface, inputs) -> values. The
# comprehensions run their "for" clauses left to right, outer loop
# first, which fixes the order of the flat value list.
EVALUATORS = {
    "hb1nu": lambda ci, x: [ci.hb1nu(nu=nu, a=a)
                            for a in x["a"] for nu in x["nu"]],
    "fnu": lambda ci, x: [ci.fnu(nu=nu, a=a)
                          for a in x["a"] for nu in x["nu"]],
    "conc": lambda ci, x: [ci.conc(m=m, growfac_a=d)
                           for d in x["growfac_a"] for m in x["m"]],
    "dlognudlogm": lambda ci, x: [ci.dlognudlogm(M=m) for m in x["M"]],
    "bias_norm": lambda ci, x: [ci.bias_norm(a=a) for a in x["a"]],
    "u_nfw_c": lambda ci, x: [ci.u_nfw_c(c=c, k=k, m=m, a=a)
                              for a in x["a"] for c in x["c"]
                              for m in x["m"] for k in x["k"]],
    "u_KS": lambda ci, x: [ci.u_KS(c=c, k=k, rv=rv)
                           for rv in x["rv"] for c in x["c"]
                           for k in x["k"]],
    "ngal_nointerp": lambda ci, x: _per_bin(ci.ngal_nointerp, x),
    "bgal_nointerp": lambda ci, x: _per_bin(ci.bgal_nointerp, x),
    "ngal": lambda ci, x: _per_bin(ci.ngal, x),
    "bgal": lambda ci, x: _per_bin(ci.bgal, x),
    "mmean_nointerp": lambda ci, x: _per_bin(ci.mmean_nointerp, x),
    "fsat_nointerp": lambda ci, x: _per_bin(ci.fsat_nointerp, x),
    "p_mm": lambda ci, x: _spectrum(ci.p_mm, x),
    "p_my": lambda ci, x: _spectrum(ci.p_my, x),
    "p_yy": lambda ci, x: _spectrum(ci.p_yy, x),
    "p_gm": lambda ci, x: _spectrum_per_bin(ci.p_gm, x, auto=False),
    "p_gg": lambda ci, x: _spectrum_per_bin(ci.p_gg, x, auto=True),
}
PROBE_NAMES = tuple(EVALUATORS)

# The probes the cache and determinism checks re-evaluate: cheap, and
# callable on the current halo.c.
FAST_PROBES = tuple(name for name in PROBE_NAMES
                    if name not in SLOW_PROBES and name not in HEAD_DEFECTS)

# The fast probes a change of omegam must move (hb1nu and fnu are
# closed forms in nu: no cosmology enters).
COSMOLOGY_PROBES = ("conc", "dlognudlogm", "bias_norm", "u_nfw_c",
                    "ngal_nointerp", "bgal_nointerp")


def evaluate_probe(ci, name, x):
    """Evaluate one probe on its inputs.

    Arguments:
      ci   = the compiled interface module.
      name = a key of EVALUATORS.
      x    = the probe's inputs (probe_inputs, or a frozen entry's).

    Returns:
      the values as a list of plain floats (json-ready).
    """
    return [float(v) for v in EVALUATORS[name](ci, x)]


def evaluate_fast(state):
    """{probe name: numpy array of values} for every FAST_PROBES entry."""
    return {name: np.array(evaluate_probe(state["ci"], name,
                                          state["inputs"][name]))
            for name in FAST_PROBES}


def rebuild_tables(state):
    """Draw a new Ntable.random at unchanged table sizes.

    init_accuracy_boost with the frozen configuration's own settings
    recomputes the same sizes and draws a new key, so every
    Ntable-keyed table rebuilds on its next call.

    Arguments:
      state = the dictionary from build_halo_state.

    Returns:
      nothing.
    """
    like = state["like"]
    state["ci"].init_accuracy_boost(
        accuracy_boost=float(like["accuracyboost"]),
        integration_accuracy=int(like["integration_accuracy"]))


def build_halo_state():
    """Build the configuration the halo quantities are evaluated on.

    Builds the frozen EXAMPLE model, evaluates its fiducial point
    (which hands cosmolike the cosmology: P(k, z), growth, distances)
    and pins the HOD and gas parameters. Shared by the fixture below
    and generate_frozen_reference.py --halo, so the tests and the
    frozen values always describe the same state.

    Returns:
      a dictionary: ci = the compiled interface module, model = the
      cobaya Model, point = the frozen fiducial point, like = the
      likelihood block of the frozen configuration, nbin = the number
      of lens bins, inputs = probe_inputs(state).
    """
    info = u.load_frozen_info(EXAMPLE, tatt=TATT)
    model = u.make_model(info)
    point = u.build_point(model, EXAMPLE, tatt=TATT)
    u.evaluate_chi2(model, point)
    import cosmolike_roman_real_interface as ci

    name = u.EXAMPLES[EXAMPLE]["likelihood"]
    state = {
        "ci": ci,
        "model": model,
        "point": point,
        "like": info["likelihood"][name],
        "nbin": int(model.likelihood[name].lens_ntomo),
    }
    apply_halo_parameters(state)
    state["inputs"] = probe_inputs(state)
    return state


def freeze_probes(state):
    """Evaluate every probe the current halo.c can evaluate.

    Arguments:
      state = the dictionary from build_halo_state.

    Returns:
      {probe name: {"inputs": ..., "values": [...]}} for every probe
      not in HEAD_DEFECTS (slow ones included).
    """
    frozen = {}
    for name in PROBE_NAMES:
        if name in HEAD_DEFECTS:
            continue
        x = state["inputs"][name]
        frozen[name] = {"inputs": x,
                        "values": evaluate_probe(state["ci"], name, x)}
    return frozen


def frozen_meta():
    """The settings the frozen values depend on (stored in _meta)."""
    return {
        "example": EXAMPLE,
        "tatt": TATT,
        "hod_coupon_2012": [list(row) for row in HOD_COUPON_2012],
        "galaxy_concentration_factor": GALAXY_CONCENTRATION_FACTOR,
        "gas": list(GAS_PARAMS),
        "head_defects": dict(HEAD_DEFECTS),
    }


def _max_relative_change(new, old):
    """max |new - old|/|old| over the entries where old != 0."""
    nonzero = old != 0
    return float(np.max(np.abs(new[nonzero] - old[nonzero])
                        / np.abs(old[nonzero])))


# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture(scope="module")
def halo():
    """The halo state, built once for the whole module."""
    u.require_cocoa_environment()
    u.verify_frozen()
    return build_halo_state()


@pytest.fixture(scope="module")
def frozen_reference():
    """The frozen values; skips the frozen tests while the file is absent."""
    if not os.path.isfile(HALO_REFERENCE_FILE):
        pytest.skip("frozen/halo_reference.json does not exist yet: run "
                    "generate_frozen_reference.py --halo")
    with open(HALO_REFERENCE_FILE) as f:
        reference = json.load(f)
    # a json round trip turns the tuples into lists, as in the file
    expected = json.loads(json.dumps(frozen_meta()))
    stored = {key: reference["_meta"].get(key) for key in expected}
    if stored != expected:
        pytest.fail("frozen/halo_reference.json was generated with other "
                    "settings (configuration, HOD, gas or defect list); "
                    "regenerate it with generate_frozen_reference.py "
                    "--halo")
    return reference


def _frozen_params():
    """One pytest parameter per probe, carrying its defect/slow marks."""
    params = []
    for name in PROBE_NAMES:
        marks = []
        if name in HEAD_DEFECTS:
            marks.append(pytest.mark.xfail(run=False,
                                           reason=HEAD_DEFECTS[name]))
        if name in SLOW_PROBES:
            marks.append(slow)
        params.append(pytest.param(name, marks=marks, id=name))
    return params


# =============================================================================
# 1. FROZEN REFERENCES
# =============================================================================
class TestFrozenReferences:
    """Every probe reproduces its frozen values, point by point.

    The frozen file holds the values of the current implementation
    (GSL fixed-order Gauss-Legendre integrals, linear tables) on the
    probe grids, taken before any rewrite of halo.c. While the
    implementation is unchanged they must agree to rounding
    (FROZEN_RTOL). A ticket that rewrites a family replaces the
    tolerance of its probes by the measured accuracy of the new code
    (FROZEN_RTOL_BY_PROBE) in the same commit: the frozen values stay
    the fixed yardstick of the whole campaign. The inputs are read
    from the frozen file itself, so later edits of the grids above
    cannot desynchronize a comparison.
    """

    @pytest.mark.parametrize("name", _frozen_params())
    def test_matches_frozen(self, frozen_reference, halo, name):
        entry = frozen_reference["probes"].get(name)
        if entry is None:
            pytest.skip(f"{name} is not in frozen/halo_reference.json "
                        "(regenerate with --halo)")
        values = evaluate_probe(halo["ci"], name, entry["inputs"])
        rtol = FROZEN_RTOL_BY_PROBE.get(name, FROZEN_RTOL)
        # equal_nan: a frozen NaN must stay NaN (and nothing else may
        # become one); atol = 0 makes the check purely relative
        np.testing.assert_allclose(
            values, entry["values"], rtol=rtol, atol=0.0, equal_nan=True,
            err_msg=f"{name} moved away from its frozen values")


# =============================================================================
# 2. PHYSICS INVARIANTS
# =============================================================================
class TestPhysicsInvariants:
    """Properties the physics demands of any correct implementation.

    Each test states a fact that follows from the definitions - a
    published fitting formula, the mass normalization of a profile, the
    large-scale limit of the halo model, a bound - and checks halo.c
    against it with a tolerance justified next to its constant. None of
    them refers to how halo.c integrates or tabulates, so they stay
    unchanged through every rewrite.
    """

    # ---- profiles ----------------------------------------------------------
    def test_u_nfw_c_mass_normalization(self, halo):
        """u(k -> 0 | m) = 1: on scales much larger than the halo, its
        profile is a point holding the halo mass."""
        ci = halo["ci"]
        for c in U_NFW_C:
            for m in (1.0e8, 1.0e11, 1.0e13, 1.0e15):
                u0 = ci.u_nfw_c(c=c, k=U_NFW_K0, m=m, a=U_NFW_A)
                assert abs(u0 - 1.0) < U_NFW_K0_ATOL, (
                    f"u_nfw_c(c={c}, k={U_NFW_K0}, m={m:.1e}) = {u0!r}")

    def test_u_nfw_c_bounded_by_one(self, halo):
        """|u(k|m)| <= u(0|m) = 1: the Fourier transform of a positive
        density never exceeds its k = 0 value, the total mass."""
        ci = halo["ci"]
        for c in U_NFW_C:
            for m in U_NFW_M:
                for k in np.logspace(0.0, 6.5, 27):
                    value = ci.u_nfw_c(c=c, k=float(k), m=m, a=U_NFW_A)
                    assert abs(value) <= 1.0 + U_NFW_BOUND_ATOL, (
                        f"|u_nfw_c(c={c}, k={k:.2e}, m={m:.1e})| = "
                        f"{abs(value)!r} > 1")

    def test_u_KS_bounded(self, halo):
        """0 < u_KS(k) <= u_KS(k -> 0) <= 1: the pressure profile is a
        positive function (|sin z| <= z bounds its transform by the k = 0
        value), and the pressure-to-density normalization ln(1+x)/x <= 1
        keeps the k = 0 value at or below 1."""
        ci = halo["ci"]
        for c in U_KS_C:
            u0 = ci.u_KS(c=c, k=U_KS_K0, rv=U_KS_RV)
            assert 0.0 < u0 <= 1.0 + U_KS_BOUND_ATOL, f"u_KS(k->0) = {u0}"
            for k in U_KS_K:
                value = ci.u_KS(c=c, k=float(k), rv=U_KS_RV)
                assert value <= u0 + U_KS_BOUND_ATOL, (
                    f"u_KS(c={c}, k={k:.2e}) = {value} > u_KS(k->0)")

    # ---- mass function and bias kernels ------------------------------------
    def test_fnu_matches_tinker2010(self, halo):
        """f(nu) is the published Tinker et al. 2010 multiplicity
        function (recomputed in numpy from the paper's equations)."""
        ci = halo["ci"]
        for a in TINKER_A:
            for nu in NU_GRID:
                np.testing.assert_allclose(
                    ci.fnu(nu=float(nu), a=a), tinker_multiplicity(nu, a),
                    rtol=FNU_RTOL, atol=0.0,
                    err_msg=f"fnu(nu={nu:.3f}, a={a})")

    def test_fnu_leaves_matter_unbiased(self, halo):
        """Eq. 7 of Tinker et al. 2010: int_0^inf b(nu) f(nu) dnu = 1 at
        every z, from the production hb1nu and fnu (trapezoid in ln nu,
        the same rule as tinker_alpha above)."""
        ci = halo["ci"]
        ds = 0.02
        nus = np.exp(np.arange(-120.0, 4.0 + 0.5*ds, ds))
        for a in TINKER_NORM_A:
            g = np.array([ci.hb1nu(nu=float(nu), a=a)*ci.fnu(nu=float(nu), a=a)
                          for nu in nus])*nus
            total = ds*(g.sum() - 0.5*(g[0] + g[-1]))
            np.testing.assert_allclose(
                total, 1.0, rtol=FNU_RTOL, atol=0.0,
                err_msg=f"int b f dnu at a = {a}")

    def test_hb1nu_matches_tinker2010(self, halo):
        """b(nu) is the published Tinker et al. 2010 halo bias
        (recomputed in numpy from the paper's equations)."""
        ci = halo["ci"]
        for nu in NU_GRID:
            np.testing.assert_allclose(
                ci.hb1nu(nu=float(nu), a=HB1NU_A), tinker_bias(nu),
                rtol=TINKER_RTOL, atol=0.0, err_msg=f"hb1nu(nu={nu:.3f})")

    def test_conc_matches_bhattacharya2013(self, halo):
        """c(m) = 9.0 nu^-0.29 D^1.15 (Bhattacharya et al. 2013, Table
        2, Delta = 200 mean), with nu from the same sigma2 table."""
        ci = halo["ci"]
        for d in CONC_GROWFAC:
            for m in CONC_M:
                nu = DELTA_C/(np.sqrt(ci.sigma2(M=float(m)))*d)
                np.testing.assert_allclose(
                    ci.conc(m=float(m), growfac_a=d),
                    9.0*nu**-0.29*d**1.15, rtol=CONC_RTOL, atol=0.0,
                    err_msg=f"conc(m={m:.1e}, D={d})")

    def test_conc_decreases_with_mass(self, halo):
        """Massive halos formed late and are less concentrated: c(m)
        falls monotonically with m at fixed D."""
        ci = halo["ci"]
        for d in CONC_GROWFAC:
            values = np.array([ci.conc(m=float(m), growfac_a=d)
                               for m in CONC_M])
            assert np.all(np.diff(values) < 0), f"D={d}: {values}"

    def test_dlognudlogm_matches_finite_difference(self, halo):
        """d ln nu/d ln M = -(1/2) d ln sigma2/d ln M: compared with a
        central difference of the same sigma2 table (nu = delta_c/sigma,
        so delta_c drops out)."""
        ci = halo["ci"]
        h = DLOGNU_FD_STEP
        for m in DLOGNU_TEST_M:
            m = float(m)
            slope = -0.5*(np.log(ci.sigma2(M=m*np.exp(h)))
                          - np.log(ci.sigma2(M=m*np.exp(-h))))/(2.0*h)
            np.testing.assert_allclose(
                ci.dlognudlogm(M=m), slope, rtol=DLOGNU_RTOL, atol=0.0,
                err_msg=f"dlognudlogm(M={m:.2e})")

    @slow
    def test_sigma2_matches_python_integral(self, halo):
        """sigma2(M) = 1/(2 pi^2 R^3) int P_lin(x/R, a = 1) 9 j1(x)^2 dx,
        R = (3M/(4 pi rho_crit Omega_m))^(1/3): recomputed in numpy from
        the same p_lin over the x range of the C lobe cache (512 lobes)."""
        from scipy.special import spherical_jn
        ci = halo["ci"]
        omegam = float(halo["point"]["omegam"])
        x_end = 513.5*np.pi - 1.0/(513.5*np.pi)  # last lobe edge
        s = np.linspace(np.log(1.0e-4), np.log(x_end), SIGMA2_REF_NX)
        x = np.exp(s)
        w = 9.0*spherical_jn(1, x)**2*x*(s[1] - s[0])  # dx = x ds
        w[0] *= 0.5
        w[-1] *= 0.5
        dlnm = np.log(HALO_M_MAX/HALO_M_MIN)/(SIGMA2_N_M - 1)
        for j in SIGMA2_REF_NODES:
            m = HALO_M_MIN*np.exp(j*dlnm)
            r = (0.75*m/(np.pi*RHO_CRIT*omegam))**(1.0/3.0)
            p = np.array([ci.p_lin(k=float(k), a=1.0) for k in x/r])
            expected = np.dot(w, p)/(2.0*np.pi**2*r**3)
            np.testing.assert_allclose(
                ci.sigma2(M=float(m)), expected, rtol=SIGMA2_REF_RTOL,
                atol=0.0, err_msg=f"sigma2(M={m:.2e})")

    # ---- bias normalization ------------------------------------------------
    def test_bias_norm_endpoint_continuity(self, halo):
        """The last table node (a = 0.9999999) holds the real integral
        like every other node - no sentinel value - and queries past it
        return that endpoint by constant extrapolation."""
        ci = halo["ci"]
        end = ci.bias_norm(a=BIAS_NORM_A_END)
        np.testing.assert_allclose(
            end, bias_norm_integral(ci, BIAS_NORM_A_END),
            rtol=BIAS_NORM_QUAD_RTOL, atol=0.0,
            err_msg="bias_norm table end vs the numpy integral")
        for a in BIAS_NORM_PINNED_A:
            assert ci.bias_norm(a=a) == end, f"bias_norm({a!r})"

    def test_bias_norm_is_tinker_integral(self, halo):
        """At its own nodes the bias_norm table returns the stored
        integral, which must equal the defining integral recomputed in
        numpy (bias_norm_integral): this checks the quadrature."""
        ci = halo["ci"]
        for a in bias_norm_nodes(halo):
            np.testing.assert_allclose(
                ci.bias_norm(a=a), bias_norm_integral(ci, a),
                rtol=BIAS_NORM_QUAD_RTOL, atol=0.0,
                err_msg=f"bias_norm at the node a={a}")

    def test_bias_norm_near_one_today(self, halo):
        """Near a = 1 the tabulated mass range carries most of the
        Tinker normalization int b f dnu = 1 (the missing part is the
        unresolved low-mass tail)."""
        value = halo["ci"].bias_norm(a=BIAS_NORM_TODAY_A)
        low, high = BIAS_NORM_TODAY_RANGE
        assert low < value < high, (
            f"bias_norm({BIAS_NORM_TODAY_A}) = {value}")

    def test_bias_norm_table_matches_direct_integral(self, halo):
        """Between its nodes the bias_norm table reproduces the numpy
        integral to within linear-interpolation error: this checks the
        interpolation."""
        ci = halo["ci"]
        for a in BIAS_NORM_INTERP_A:
            a = float(a)
            np.testing.assert_allclose(
                ci.bias_norm(a=a), bias_norm_integral(ci, a),
                rtol=BIAS_NORM_INTERP_RTOL, atol=0.0,
                err_msg=f"bias_norm table vs the numpy integral at a={a}")

    # ---- HOD integrals -----------------------------------------------------
    def test_ngal_physical(self, halo):
        """The HOD number density is positive and of the abundance of the
        halos the HOD populates (a unit or normalization slip moves it by
        many decades)."""
        ci = halo["ci"]
        x = halo["inputs"]["ngal_nointerp"]
        low, high = NGAL_BRACKET_H3
        for ni, a_list in zip(x["ni"], x["a"]):
            for a in a_list:
                # code units (c/H0)^-3 -> (h/Mpc)^3
                n = ci.ngal_nointerp(ni=ni, a=a)/COVERH0**3
                assert low < n < high, f"ngal(ni={ni}, a={a:.3f}) = {n}"

    def test_bgal_is_a_mean_bias(self, halo):
        """bgal is the number-weighted mean bias of the HOD galaxies: an
        order-unity number, above 1 for red galaxies in group-mass
        halos."""
        ci = halo["ci"]
        x = halo["inputs"]["bgal_nointerp"]
        low, high = BGAL_RANGE
        for ni, a_list in zip(x["ni"], x["a"]):
            for a in a_list:
                b = ci.bgal_nointerp(ni=ni, a=a)
                assert low < b < high, f"bgal(ni={ni}, a={a:.3f}) = {b}"

    def test_fsat_is_a_fraction(self, halo):
        """The satellite fraction lies in (0, 1), and the Coupon HOD
        does have satellites."""
        ci = halo["ci"]
        x = halo["inputs"]["fsat_nointerp"]
        for ni, a_list in zip(x["ni"], x["a"]):
            for a in a_list:
                f = ci.fsat_nointerp(ni=ni, a=a)
                assert FSAT_FLOOR < f < 1.0, f"fsat(ni={ni}, a={a}) = {f}"

    def test_mmean_within_populated_range(self, halo):
        """The mean halo mass of the galaxies lies inside the mass range
        the HOD integrals cover."""
        ci = halo["ci"]
        x = halo["inputs"]["mmean_nointerp"]
        for ni, a_list in zip(x["ni"], x["a"]):
            low = 10.0**(hod_of_bin(ni)[0] - 2.0)
            for a in a_list:
                mean_m = ci.mmean_nointerp(ni=ni, a=a)
                assert low < mean_m < HALO_M_MAX, (
                    f"mmean(ni={ni}, a={a}) = {mean_m:.3e}")

    def test_set_HOD_loads_coupon_values(self, halo):
        """set_HOD(ni) loads the Coupon et al. 2012 HOD that
        HOD_COUPON_2012 copies: ngal after set_HOD equals ngal after
        setting the copied values explicitly, bit for bit, in every
        lens bin set_HOD covers (0-4)."""
        ci = halo["ci"]
        a = 0.6
        for ni in range(min(int(halo["nbin"]), len(HOD_COUPON_2012))):
            try:
                ci.set_HOD(ni=ni)
                via_set_hod = ci.ngal_nointerp(ni=ni, a=a)
            finally:
                apply_halo_parameters(halo)
            assert via_set_hod == ci.ngal_nointerp(ni=ni, a=a), (
                f"lens bin {ni}")

    # ---- spectra: large-scale (2-halo) limits ------------------------------
    @slow
    def test_p_mm_two_halo_limit(self, halo):
        """On large scales P_mm -> P_lin: the 2-halo term is
        I_m(k)^2 P_lin with I_m -> 1 (the HMx additive correction of
        2005.00009 App. A puts the matter of halos below M_min at M_min,
        so I_m(k -> 0) = bias_norm + (1 - bias_norm) = 1), and the
        1-halo term is small."""
        ci = halo["ci"]
        for a in P_2H_A:
            for k_h in P_2H_K_HMPC:
                k = k_h*COVERH0
                ratio = ci.p_mm(k=k, a=a)/ci.p_lin(k=k, a=a)
                assert abs(ratio - 1.0) < P_2H_RTOL, (
                    f"p_mm/p_lin(k={k_h} h/Mpc, a={a}) = {ratio}")

    @slow
    def test_p_gm_two_halo_limit(self, halo):
        """On large scales P_gm -> bgal P: galaxies trace matter with
        their mean bias."""
        ci = halo["ci"]
        for ni in HOD_TEST_BINS:
            a = lens_bin_mean_scale_factor(halo, ni)
            for k_h in P_2H_K_HMPC:
                k = k_h*COVERH0
                expected = ci.bgal(ni=ni, a=a)*ci.Pdelta(k=k, a=a)
                ratio = ci.p_gm(k=k, a=a, ni=ni)/expected
                assert abs(ratio - 1.0) < P_2H_RTOL, (
                    f"p_gm/(bgal P)(ni={ni}, k={k_h} h/Mpc) = {ratio}")

    @slow
    def test_p_gg_two_halo_limit(self, halo):
        """On large scales P_gg -> bgal^2 P."""
        ci = halo["ci"]
        for ni in HOD_TEST_BINS:
            a = lens_bin_mean_scale_factor(halo, ni)
            for k_h in P_2H_K_HMPC:
                k = k_h*COVERH0
                expected = ci.bgal(ni=ni, a=a)**2*ci.Pdelta(k=k, a=a)
                ratio = ci.p_gg(k=k, a=a, ni=ni, nj=ni)/expected
                assert abs(ratio - 1.0) < P_2H_RTOL, (
                    f"p_gg/(bgal^2 P)(ni={ni}, k={k_h} h/Mpc) = {ratio}")

    @slow
    @pytest.mark.xfail(run=False, reason=HEAD_DEFECTS["p_my"])
    def test_p_my_cauchy_schwarz(self, halo):
        """A cross spectrum is bounded by its autos: p_my^2 <= p_mm p_yy
        (Cauchy-Schwarz on each mass integral), and p_yy > 0."""
        ci = halo["ci"]
        k = np.asarray(PK_K, dtype=float)
        for a in P_2H_A:
            pmm = np.ravel(ci.p_mm(k=k, a=a))
            pmy = np.ravel(ci.p_my(k=k, a=a))
            pyy = np.ravel(ci.p_yy(k=k, a=a))
            assert np.all(pyy > 0), f"a={a}: p_yy = {pyy}"
            assert np.all(pmy**2 <= pmm*pyy*(1.0 + CAUCHY_SCHWARZ_RTOL)), (
                f"a={a}: p_my^2/(p_mm p_yy) = {pmy**2/(pmm*pyy)}")


# =============================================================================
# 3. CACHE CONSISTENCY
# =============================================================================
class TestCacheConsistency:
    """Table rebuilds: a change is seen, and undoing it restores the bits.

    halo.c keeps its tables in C statics keyed on the global cache keys
    (cosmology.random, Ntable.random, nuisance.random_galaxy_bias,
    nuisance.random_gas). Each test changes one input, checks that the
    halo quantities move (a stale table would not), changes it back and
    checks that every value returns bit for bit (a rebuild that reads
    leftover state would not). The contract belongs to the caching
    design, not to the integration method, so it survives every
    rewrite. Every test restores the pinned state in a finally block,
    so a failure cannot leak into the tests after it.
    """

    def test_cosmology_round_trip(self, halo):
        """omegam -> omegam + OMEGAM_STEP -> back, through the likelihood
        (so the whole cosmology pipeline, CAMB included, runs again)."""
        before = evaluate_fast(halo)
        moved_point = dict(halo["point"])
        moved_point["omegam"] = halo["point"]["omegam"] + OMEGAM_STEP
        try:
            u.evaluate_chi2(halo["model"], moved_point)
            moved = evaluate_fast(halo)
        finally:
            u.evaluate_chi2(halo["model"], halo["point"])
        after = evaluate_fast(halo)
        for name in COSMOLOGY_PROBES:
            change = _max_relative_change(moved[name], before[name])
            assert change > CACHE_CHANGE_FLOOR, (
                f"{name} did not move with omegam (stale table?)")
        for name in FAST_PROBES:
            assert np.array_equal(after[name], before[name],
                                  equal_nan=True), (
                f"{name} differs after returning to the fiducial cosmology")

    def test_ntable_rebuild_round_trip(self, halo):
        """A new Ntable.random at unchanged sizes rebuilds every table;
        the rebuilt tables must equal the old ones bit for bit."""
        before = evaluate_fast(halo)
        rebuild_tables(halo)
        after = evaluate_fast(halo)
        for name in FAST_PROBES:
            assert np.array_equal(after[name], before[name],
                                  equal_nan=True), (
                f"{name} changed after a same-size table rebuild")

    def test_hod_round_trip(self, halo):
        """lg M_min of one bin -> + HOD_LGMMIN_STEP -> back, through the
        HOD setter (direct integrals: they read the HOD at every call)."""
        ci = halo["ci"]
        names = ("ngal_nointerp", "bgal_nointerp")
        before = evaluate_fast(halo)
        ni = HOD_TEST_BINS[0]
        moved_hod = np.array(hod_of_bin(ni), dtype=float)
        moved_hod[0] += HOD_LGMMIN_STEP
        try:
            ci.set_nuisance_hod(ni=ni, hod=moved_hod,
                                gc=GALAXY_CONCENTRATION_FACTOR)
            moved = evaluate_fast(halo)
        finally:
            apply_halo_parameters(halo)
        after = evaluate_fast(halo)
        for name in names:
            change = _max_relative_change(moved[name], before[name])
            assert change > CACHE_CHANGE_FLOOR, f"{name} ignored the HOD"
        for name in FAST_PROBES:
            assert np.array_equal(after[name], before[name],
                                  equal_nan=True), (
                f"{name} differs after restoring the HOD")

    def test_hod_tables_round_trip(self, halo):
        """The same HOD round trip through the ngal/bgal tables, which
        rebuild on nuisance.random_galaxy_bias."""
        ci = halo["ci"]
        x = halo["inputs"]["ngal"]
        names = ("ngal", "bgal")
        before = {n: np.array(evaluate_probe(ci, n, x)) for n in names}
        ni = HOD_TEST_BINS[0]
        moved_hod = np.array(hod_of_bin(ni), dtype=float)
        moved_hod[0] += HOD_LGMMIN_STEP
        try:
            ci.set_nuisance_hod(ni=ni, hod=moved_hod,
                                gc=GALAXY_CONCENTRATION_FACTOR)
            moved = {n: np.array(evaluate_probe(ci, n, x)) for n in names}
        finally:
            apply_halo_parameters(halo)
        after = {n: np.array(evaluate_probe(ci, n, x)) for n in names}
        for name in names:
            assert (_max_relative_change(moved[name], before[name])
                    > CACHE_CHANGE_FLOOR), f"{name}: stale table"
            assert np.array_equal(after[name], before[name]), (
                f"{name} differs after restoring the HOD")

    def test_gas_round_trip(self, halo):
        """Gamma -> Gamma + GAS_GAMMA_STEP -> back, through the gas
        setter; the u_KS table rebuilds on nuisance.random_gas."""
        ci = halo["ci"]
        x = halo["inputs"]["u_KS"]
        before = np.array(evaluate_probe(ci, "u_KS", x))
        moved_gas = np.array(GAS_PARAMS, dtype=float)
        moved_gas[0] += GAS_GAMMA_STEP
        try:
            ci.set_nuisance_gas(gas=moved_gas)
            moved = np.array(evaluate_probe(ci, "u_KS", x))
        finally:
            apply_halo_parameters(halo)
        after = np.array(evaluate_probe(ci, "u_KS", x))
        assert _max_relative_change(moved, before) > CACHE_CHANGE_FLOOR
        assert np.array_equal(after, before)


# =============================================================================
# 4. DETERMINISM
# =============================================================================
class TestDeterminism:
    """The same inputs give the same bits.

    halo.c fills its tables in OpenMP loops. Every table entry is an
    independent computation (no sums across threads), so the result
    must not depend on how many threads run the loop or on how often a
    function is called. This is a property of the parallel design, not
    of the integration method, so it survives every rewrite. The thread
    counts are switched in-process (set_omp_threads plus a table
    rebuild) rather than in subprocesses: the cosmology inputs from
    CAMB then stay bit-identical, and only cosmolike's own threading
    varies. The slow spectra are left out (p_mm on one thread takes
    several minutes).
    """

    def test_repeated_calls(self, halo):
        """Two evaluations of every fast probe, back to back."""
        first = evaluate_fast(halo)
        second = evaluate_fast(halo)
        for name in FAST_PROBES:
            assert np.array_equal(first[name], second[name],
                                  equal_nan=True), f"{name} not repeatable"

    def test_thread_count(self, halo):
        """Tables rebuilt with 1 and 8 threads equal the 4-thread ones."""
        ci = halo["ci"]
        reference = evaluate_fast(halo)
        default_threads = int(os.environ["OMP_NUM_THREADS"])
        try:
            for threads in DETERMINISM_THREADS:
                ci.set_omp_threads(threads)
                rebuild_tables(halo)
                values = evaluate_fast(halo)
                for name in FAST_PROBES:
                    assert np.array_equal(values[name], reference[name],
                                          equal_nan=True), (
                        f"{name} differs with {threads} threads")
        finally:
            ci.set_omp_threads(default_threads)
            rebuild_tables(halo)
