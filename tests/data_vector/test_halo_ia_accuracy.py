"""Unit test: numerical convergence of the halo-model intrinsic alignments.

With include_halo_IA = 1 the Limber cosmic-shear (ss) and
galaxy-galaxy-lensing (gs) engines of cosmo2D.c replace the NLA
intrinsic-alignment legs by the halo model of Fortuna et al. 2021
(F21), one IA population over the source redshift range:

  matter-intrinsic   P_dI = f_rc(a) C1(a) P_delta(k) f_2h(k) + P_dI^1h
  intrinsic-intrinsic P_II = f_rc(a)^2 C1 C1' P_delta f_2h + P_II^1h

(cosmo2D.c subtracts both legs, as it subtracts the NLA term).
f_rc is the red-central fraction of the IA population, C1 the NLA
amplitude (the likelihood's roman_A1 parameters), f_2h(k) =
exp[-(k/k_2h)^2] the F21 window of the 2-halo term, and P^1h the
satellite 1-halo terms of halo.c: mass integrals over the halo mass
function of the satellite occupation times the Fourier-space
alignment profile gamma_hat(k|M), a multipole sum truncated at
l <= Ntable.halo_ia_lmax, tabulated on (a, ln k).

The halo.c tables carry four numerical knobs:

  - Ntable.halo_ia_na: the a nodes of the IA tables, scaled by
    init_accuracy_boost (the accuracyboost arm below);
  - the mass-node ladder on Ntable.high_def_integration (hdi):
    halo_nm x 1, 2, 4, then 1024 nodes at hdi >= 3;
  - the coarse ln k step of the mass sums on the same ladder:
    halo_nk_step, halo_nk_step/2, then 1 at hdi >= 2;
  - Ntable.halo_ia_lmax (2, 4, 6): the multipole truncation. This is
    a PHYSICS choice (F21 use l <= 6), not a numerical one.

HOW THE ARMS SET THE KNOBS

The frozen configurations carry accuracyboost and
integration_accuracy as likelihood options. cocoa_testing's
high-accuracy path (test_accuracy.py) writes them into the likelihood
block and builds a NEW model, because the likelihood consumes
accuracyboost at initialization for its Python-side grids as well
(z_interp_2D, log10k_interp_2D, and CAMB's k_max = kmax_boltzmann x
accuracyboost) before forwarding both values to
ci.init_accuracy_boost. A yaml arm would therefore move CAMB and the
P(k) tables handed to cosmolike together with cosmolike's own tables.

This test measures cosmolike's numerics only, so every arm builds
the SAME frozen model and then calls ci.init_accuracy_boost directly:
the function rescales every Ntable size from the baseline cached at
its first call (the likelihood's own call at initialization), so
calls never compound, and it redraws Ntable.random, so every table
keyed on it rebuilds at its next use. No model rebuild is needed for
that. Each arm still runs in its OWN worker subprocess (the pattern
of test_halo_cache_consistency.py's fresh worker): cosmolike's tables
and init_accuracy_boost's baseline cache are per-process statics, so
a separate process guarantees every arm starts from the same pristine
state.

THE STATE

The frozen NLA 3x2pt model (example2) with Limber galaxy-galaxy
lensing (include_halo_IA supports Limber gs only), evaluated at its
frozen point, with the IA fiducial

  {a_1h, eta_1h, z_pivot}                   = {0.001, 0, 0.62}
    (F21 red galaxies; no redshift evolution)
  red sigmoids {lgM_cen, w_cen, lgM_sat, w_sat} = {13.0, 0.5, 12.5, 0.7}
    (the student's notebook values)
  IA HOD {lgMmin, sigma_lgM, lgM1, lgM0, alpha, f_c}
        = {13.17, 0.39, 14.53, 11.09, 1.27, 1.00}
    (the student's default: Coupon et al. 2012, red M_r < -21.8)

NO SCALE CUTS. A masked data vector skips the cut points, and the
production mask cuts the small-scale gammat and w points, where the
gs 1-halo term lives; so the data vector is computed WITHOUT scale
cuts: the model loads a scratch dataset (written into a
temporary directory; nothing touches frozen/) whose mask is
ones.mask and whose covariance is the diagonal of the frozen one
(the full covariance is not positive definite without a mask, and
the interface refuses it). Only the model VECTOR is used; the chi2
values below are computed in Python from the frozen covariance.

SCORING. Every judgment is a chi2 with a reference vector INJECTED
AS TRUTH, delta^T C^-1 delta, scored on the most aggressive
positive-definite mask of the frozen covariance (all shear points
plus 85 of the 165 points the production mask cuts;
AGGRESSIVE_MASK_EXCLUDED below), with the production-mask number
reported next to it.

Assertions:

  IA0. Invisible when off: with include_halo_IA = 0 the data vector
       is BITWISE identical to that of a model (in another process)
       that never called the IA setters - both right after setting
       the IA parameters and after switching the flag on, evaluating,
       and switching it back off (the C_ss/C_gs caches key on the
       flag). The flag-on vector must differ, so the check cannot
       pass vacuously.

  IA1-IA5. Convergence of the numerical knobs. Each arm computes, in
       one worker and at identical settings, the flag-on vector and
       the flag-off vector; their difference is the IA INCREMENT,
       the part of the data vector the halo-model IA code produces.
       The reference arm is accuracyboost 3 (test_accuracy.py's
       all-knobs boost) and integration_accuracy 3 (the top of every
       IA ladder: 1024 mass nodes, coarse k step 1). For each arm

         chi2_IA = (inc_arm - inc_ref)^T C^-1 (inc_arm - inc_ref)

       on the aggressive mask must stay below IA_ACCURACY_TOLERANCE.
       The arms: IA1 production (accuracyboost 1, hdi 0), IA2
       accuracyboost 2, IA3-IA5 integration_accuracy 1, 2, 3.
       Why the increment and not the whole vector: accuracyboost and
       integration_accuracy move EVERY cosmolike table (N_ell, N_a,
       the Limber node counts), so the whole-vector difference is the
       whole code's numerical error, which test_accuracy.py reports
       (advisory). Differencing the two vectors of one arm cancels
       the terms the flag does not touch (lensing, clustering,
       magnification: the same tables in both vectors), leaving the
       numerical error of the CHANGE the flag makes: the halo.c IA
       tables and the quadrature of the new IA legs, minus the
       quadrature error of the NLA legs the flag removes. The
       whole-vector chi2 is printed next to it (advisory). As in
       test_accuracy.py, the reference carries its own (smaller)
       error, so arm-vs-reference bounds the arm's error only while
       the ladder converges. A vacuity guard requires the reference
       increment itself to be large (MIN_IA_SIGNAL_CHI2). The
       production arm also prints the size of the signal: the whole
       increment, its 2-halo part (flag on at a_1h = 0 minus flag
       off: mostly the (f_rc f_2h - 1) suppression of the NLA legs)
       and its 1-halo part (a_1h = 0.001 minus a_1h = 0). The three
       chi2 are not additive (cross terms); the 1-halo one sets how
       sensitive this test is to the 1-halo tables.

  IA6. halo_ia_lmax is live: at production settings, the lmax = 2
       and 4 vectors differ from the lmax = 6 vector (a dead-knob
       check), and returning to lmax = 6 reproduces the first vector
       bit for bit (so the difference is the knob, not rebuild
       noise). The chi2 of each truncation against lmax = 6 is
       printed; there is no tolerance, the truncation is physics.
       Needs the init_ntable_halo_ia_lmax binding; without it IA6
       skips and says so.

Slow (nine worker subprocesses, each a model build plus one
evaluation; the reference arm is the expensive one): runs only with
COCOA_HALO_SLOW=1, like the spectrum tier of test_halo.py. To run
(from the Cocoa/ folder, cocoa environment active, start_cocoa.sh
sourced):

    COCOA_HALO_SLOW=1 python -m pytest \\
      ./projects/roman_real/tests/data_vector/test_halo_ia_accuracy.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

# The harness stays in the parent tests/ folder. Add it explicitly so
# direct execution and worker processes resolve this project's stored inputs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

RUN_SLOW = os.environ.get("COCOA_HALO_SLOW", "0") == "1"

THIS_FILE = os.path.abspath(__file__)

EXAMPLE = "example2"
LIKELIHOOD = u.EXAMPLES[EXAMPLE]["likelihood"]

# the scratch dataset (no scale cuts) the workers load
NOCUT_DATASET = "halo_ia_nocut.dataset"
NOCUT_COVARIANCE = "halo_ia_diagonal_cov"

# ---- the IA fiducial ---------------------------------------------------------

# {a_1h, eta_1h, z_pivot}: F21 red galaxies, no redshift evolution
IA_HALO_FIDUCIAL = (0.001, 0.0, 0.62)

# {lgM_cen, width_cen, lgM_sat, width_sat}: the student's notebook values
IA_RED_FIDUCIAL = (13.0, 0.5, 12.5, 0.7)

# {lgMmin, sigma_lgM, lgM1, lgM0, alpha, f_c}: the student's default,
# Coupon et al. 2012 red galaxies with M_r < -21.8
IA_HOD_FIDUCIAL = (13.17, 0.39, 14.53, 11.09, 1.27, 1.00)

# ---- the arms ---------------------------------------------------------------

# (accuracyboost, integration_accuracy) handed to ci.init_accuracy_boost
PRODUCTION_SETTINGS = (1.0, 0)
REFERENCE_SETTINGS = (3.0, 3)

# test number -> (label, accuracyboost, integration_accuracy)
ACCURACY_ARMS = {
    "IA1": ("production", 1.0, 0),
    "IA2": ("accuracyboost 2", 2.0, 0),
    "IA3": ("integration_accuracy 1", 1.0, 1),
    "IA4": ("integration_accuracy 2", 1.0, 2),
    "IA5": ("integration_accuracy 3", 1.0, 3),
}

# the multipole truncations of the alignment profile; the first is
# the default, and the others are compared against it
HALO_IA_LMAX_VALUES = (6, 4, 2)

# ---- tolerances --------------------------------------------------------------

# The whole code's numerical errors must sum to chi2 <~ 0.2 against
# injected truth. The halo-model IA is one of about ten numerical
# components sharing that budget (Limber quadrature, Legendre sums,
# FAST-PT grids, the P(k) tables, photo-z interpolation, the halo
# tables, ...). Uncorrelated component errors add in chi2, so a tenth
# of the budget is 0.02; fully aligned errors add in amplitude
# (sqrt chi2), and sqrt(0.02/0.2) = 0.32 still leaves two thirds of
# the amplitude budget to the rest of the code.
IA_ACCURACY_TOLERANCE = 0.02

# Vacuity guard: the reference IA increment must itself be a large
# signal (here 50 x the tolerance), or a converged-looking chi2 would
# only mean the flag changed nothing.
MIN_IA_SIGNAL_CHI2 = 1.0

# ---- the aggressive positive-definite mask ----------------------------------

# The 3x2pt indices (xi+ 0-539, xi- 540-1079, gammat 1080-1994,
# w 1995-2114) that stay CUT in the most aggressive mask whose
# covariance block is still positive definite. Built from the
# production mask (example1.mask cuts 165 points, all gammat and w)
# by re-admitting the cut points one at a time, largest theta bin
# first, keeping a point only while the smallest eigenvalue of the
# masked CORRELATION matrix stays >= 1e-4: 85 re-admitted, these 80
# remain cut. Stored as data because the greedy search costs minutes.
AGGRESSIVE_MASK_EXCLUDED = (
    1080, 1095, 1096, 1110, 1111, 1125, 1126, 1140, 1141, 1155,
    1156, 1170, 1171, 1185, 1186, 1215, 1216, 1230, 1231, 1245,
    1246, 1260, 1261, 1275, 1276, 1277, 1290, 1291, 1292, 1305,
    1306, 1307, 1335, 1350, 1365, 1380, 1381, 1395, 1396, 1410,
    1411, 1425, 1426, 1455, 1500, 1515, 1530, 1545, 1560, 1575,
    1620, 1650, 1665, 1680, 1695, 1710, 1740, 1755, 1770, 1785,
    1800, 1815, 1860, 1875, 1890, 1905, 1920, 1935, 1980, 1995,
    1996, 1997, 2010, 2011, 2012, 2025, 2026, 2040, 2055, 2070,
)


# =============================================================================
# THE SCORING (parent process: no cosmolike)
# =============================================================================
def load_covariance(path):
    """Read a cosmolike covariance file into a dense symmetric matrix.

    Arguments:
      path = the covariance file: one (i, j, ...) row per entry of
             the lower or upper triangle; 3 columns (i, j, cov), 4
             (i, j, cov_g, cov_ng) or 10 (i, j, theta_i, theta_j, ...,
             cov_g, cov_ng in columns 8 and 9).

    Returns:
      the n x n covariance as a numpy array.
    """
    import numpy as np

    table = np.loadtxt(path, comments="#")
    ncol = table.shape[1]
    if ncol == 3:
        values = table[:, 2]
    elif ncol == 4:
        values = table[:, 2] + table[:, 3]
    else:
        values = table[:, 8] + table[:, 9]

    i = table[:, 0].astype(int)
    j = table[:, 1].astype(int)
    n = int(max(i.max(), j.max())) + 1

    cov = np.zeros((n, n))
    cov[i, j] = values
    cov[j, i] = values
    return cov


def build_scoring(workdir):
    """Write the scratch dataset and factor the two scoring covariances.

    The scratch dataset is the frozen descriptor with every file path
    made absolute, ones.mask as its mask, and the diagonal of the
    frozen covariance as its covariance (see the module docstring).

    Arguments:
      workdir = the temporary directory the workers load the dataset
                from.

    Returns:
      a dictionary: n = data-vector length; "aggressive" and
      "production" each map to (keep, cholesky factor of the kept
      covariance block).

    Raises:
      AssertionError when a mask disagrees with the covariance, the
      aggressive mask does not contain the production mask, or a
      kept block is not positive definite (the Cholesky factorization
      fails).
    """
    import numpy as np
    from getdist import IniFile

    info = u.load_frozen_info(EXAMPLE, tatt=False)
    block = info["likelihood"][LIKELIHOOD]
    base = os.path.normpath(os.path.join(block["path"], block["data_file"]))
    ini = IniFile(base)

    cov_path = ini.relativeFileName("cov_file")
    production_mask_path = ini.relativeFileName("mask_file")
    ones_mask_path = os.path.join(os.path.dirname(base), "ones.mask")

    print("  reading the frozen covariance ...", flush=True)
    cov = load_covariance(cov_path)
    n = cov.shape[0]

    # ---- the scratch dataset: same files, no cuts, diagonal covariance
    diagonal_path = os.path.join(workdir, NOCUT_COVARIANCE)
    index = np.arange(n)
    np.savetxt(diagonal_path,
               np.column_stack([index, index, np.diag(cov)]),
               fmt="%d %d %.12e")

    lines = []
    for key, value in ini.params.items():
        if key in ("mask_file", "cov_file"):
            continue
        text = str(value)
        if key.endswith("_file"):
            # the workers load the descriptor from workdir, so every
            # file it names must be an absolute path
            full = ini.relativeFileName(key)
            if os.path.isfile(full):
                text = full
        lines.append(f"{key} = {text}")
    lines.append(f"mask_file = {ones_mask_path}")
    lines.append(f"cov_file = {diagonal_path}")
    with open(os.path.join(workdir, NOCUT_DATASET), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- the two scoring masks
    ones = np.loadtxt(ones_mask_path)[:, 1] != 0
    production = np.loadtxt(production_mask_path)[:, 1] != 0
    aggressive = np.ones(n, dtype=bool)
    aggressive[list(AGGRESSIVE_MASK_EXCLUDED)] = False

    if len(ones) != n or len(production) != n:
        raise AssertionError(
            f"mask lengths ({len(ones)}, {len(production)}) differ from "
            f"the covariance size {n}")
    if not np.all(ones):
        raise AssertionError(f"{ones_mask_path} is not all ones")
    if not np.all(aggressive[production]):
        raise AssertionError(
            "the aggressive mask cuts a point the production mask keeps: "
            "AGGRESSIVE_MASK_EXCLUDED no longer matches the frozen mask")

    scoring = {"n": n}
    for name, keep in (("aggressive", aggressive), ("production", production)):
        try:
            factor = np.linalg.cholesky(cov[np.ix_(keep, keep)])
        except np.linalg.LinAlgError:
            raise AssertionError(
                f"the {name} mask's covariance block is not positive "
                "definite")
        scoring[name] = (keep, factor)
        print(f"  {name} mask: {int(keep.sum())} of {n} points",
              flush=True)
    return scoring


def chi2_of(delta, scoring, mask):
    """delta^T C^-1 delta on one scoring mask.

    Arguments:
      delta   = a full-length difference of two data vectors.
      scoring = the dictionary from build_scoring.
      mask    = "aggressive" or "production".

    Returns:
      the chi2 as a float (C = L L^T, so the chi2 is |L^-1 delta|^2).
    """
    from scipy.linalg import solve_triangular

    keep, factor = scoring[mask]
    whitened = solve_triangular(factor, delta[keep], lower=True)
    return float(whitened @ whitened)


def report_arm(number, label, scores, tol):
    """Print one convergence arm as a readable block.

    Arguments:
      number = the test number (IA1-IA5).
      label  = the arm's settings.
      scores = {"ia_aggr", "ia_prod", "full_aggr", "full_prod"}: the
               chi2 of the IA increment and of the whole vector,
               against the reference arm, on both masks.
      tol    = the pass limit on scores["ia_aggr"].

    Returns:
      scores["ia_aggr"], the asserted quantity.
    """
    tested = scores["ia_aggr"]
    if tested < tol:
        verdict = "OK"
    else:
        verdict = "EXCEEDS LIMIT"
    print(f"""
{'-' * 66}
TEST {number}: {label} vs reference (accuracyboost {REFERENCE_SETTINGS[0]}, \
integration_accuracy {REFERENCE_SETTINGS[1]})
  IA increment,  aggressive mask  chi2 = {scores['ia_aggr']:.6e}   \
(limit: < {tol})
  IA increment,  production mask  chi2 = {scores['ia_prod']:.6e}
  whole vector,  aggressive mask  chi2 = {scores['full_aggr']:.6e}   \
(advisory: every cosmolike knob)
  whole vector,  production mask  chi2 = {scores['full_prod']:.6e}
  -> {verdict}
{'-' * 66}""", flush=True)
    return tested


# =============================================================================
# THE WORKERS (subprocesses: one model, one set of cosmolike statics)
# =============================================================================
def build_model(workdir, include_halo_IA):
    """The frozen NLA 3x2pt model on the no-cut scratch dataset.

    Arguments:
      workdir         = the directory holding the scratch dataset.
      include_halo_IA = 0 or 1 written into the likelihood block, or
                        None to leave the key out (the likelihood's
                        default applies).

    Returns:
      (model, frozen point, compiled interface).
    """
    import cosmolike_roman_real_interface as ci

    info = u.load_frozen_info(EXAMPLE, tatt=False)
    block = info["likelihood"][LIKELIHOOD]
    block["path"] = workdir
    block["data_file"] = NOCUT_DATASET
    # include_halo_IA supports Limber galaxy-galaxy lensing only
    block["adopt_limber_gs"] = 1
    if include_halo_IA is not None:
        block["include_halo_IA"] = include_halo_IA
    model = u.make_model(info)
    point = u.build_point(model, EXAMPLE, tatt=False)
    return model, point, ci


def set_ia(ci, ia_halo):
    """Set the IA parameters (the red sigmoids and the HOD fiducial)."""
    import numpy as np

    ci.set_nuisance_ia_halo(
        ia_halo=np.array(ia_halo, dtype=float),
        ia_red=np.array(IA_RED_FIDUCIAL, dtype=float),
        ia_hod=np.array(IA_HOD_FIDUCIAL, dtype=float))


def data_vector(ci):
    """The full-length theory vector from cosmolike's current state.

    Under ones.mask every point is kept, so the masked vector is the
    whole vector. The caches make this cheap after an evaluation; a
    flag or parameter change since then rebuilds the affected tables.
    """
    import numpy as np

    return np.array(ci.compute_data_vector_masked(), dtype=float).ravel()


# the binding that sets Ntable.halo_ia_lmax, in the init_ntable_*
# pattern (validate 2/4/6, write the field, redraw Ntable.random)
HALO_IA_LMAX_BINDING = "init_ntable_halo_ia_lmax"


def set_halo_ia_lmax(ci, lmax):
    """Set Ntable.halo_ia_lmax through its binding."""
    setter = getattr(ci, HALO_IA_LMAX_BINDING)
    setter(halo_ia_lmax=lmax)


def worker_main(spec_json, out_path):
    """Subprocess entry: build one arm and save its vectors.

    spec kinds:
      "accuracy" - flag on at (accuracy_boost, integration_accuracy):
                   dv_on, dv_off; with one_halo_signal also dv_2h (flag
                   on, a_1h = 0).
      "lmax"     - flag on at production settings: dv_6, dv_4, dv_2,
                   dv_6_again; binding_missing alone when the
                   interface cannot set Ntable.halo_ia_lmax.
      "pristine" - include_halo_IA left at its default, the IA setters
                   never called: dv.
      "touched"  - include_halo_IA = 0 with the IA parameters set: dv_a;
                   flag on: dv_on; flag back off: dv_b.
    """
    import numpy as np

    u.require_cocoa_environment()
    spec = json.loads(spec_json)
    kind = spec["kind"]
    workdir = spec["workdir"]
    out = {}

    if kind == "accuracy":
        model, point, ci = build_model(workdir, 1)
        ci.init_accuracy_boost(
            accuracy_boost=float(spec["accuracy_boost"]),
            integration_accuracy=int(spec["integration_accuracy"]))
        set_ia(ci, IA_HALO_FIDUCIAL)
        u.evaluate_chi2(model, point)
        out["dv_on"] = data_vector(ci)
        ci.init_include_halo_IA(include_halo_IA=0)
        out["dv_off"] = data_vector(ci)
        if spec.get("one_halo_signal"):
            ci.init_include_halo_IA(include_halo_IA=1)
            no_one_halo = (0.0,) + tuple(IA_HALO_FIDUCIAL[1:])
            set_ia(ci, no_one_halo)
            out["dv_2h"] = data_vector(ci)

    elif kind == "lmax":
        import cosmolike_roman_real_interface as ci

        if not hasattr(ci, HALO_IA_LMAX_BINDING):
            # reported to the parent, which skips the test by name
            out["binding_missing"] = np.array([1.0])
        else:
            model, point, ci = build_model(workdir, 1)
            set_ia(ci, IA_HALO_FIDUCIAL)
            u.evaluate_chi2(model, point)
            for lmax in HALO_IA_LMAX_VALUES:
                set_halo_ia_lmax(ci, lmax)
                out[f"dv_{lmax}"] = data_vector(ci)
            set_halo_ia_lmax(ci, HALO_IA_LMAX_VALUES[0])
            out[f"dv_{HALO_IA_LMAX_VALUES[0]}_again"] = data_vector(ci)

    elif kind == "pristine":
        model, point, ci = build_model(workdir, None)
        u.evaluate_chi2(model, point)
        out["dv"] = data_vector(ci)

    elif kind == "touched":
        model, point, ci = build_model(workdir, 0)
        set_ia(ci, IA_HALO_FIDUCIAL)
        u.evaluate_chi2(model, point)
        out["dv_a"] = data_vector(ci)
        ci.init_include_halo_IA(include_halo_IA=1)
        out["dv_on"] = data_vector(ci)
        ci.init_include_halo_IA(include_halo_IA=0)
        out["dv_b"] = data_vector(ci)

    else:
        raise ValueError(f"unknown worker kind {kind!r}")

    np.savez(out_path, **out)


def run_worker(workdir, label, spec):
    """Spawn one worker subprocess and hand back its vectors.

    Arguments:
      workdir = the temporary directory (scratch dataset, results).
      label   = a short name for the progress lines and the file.
      spec    = the worker's spec dictionary (see worker_main);
                workdir is added here.

    Returns:
      {name: numpy array} as the worker saved it.

    Raises:
      RuntimeError when the worker exits without a result (cosmolike
      aborts the process on an internal inconsistency; its reason is
      printed just above, since the worker's output streams).
    """
    import numpy as np

    spec = dict(spec, workdir=workdir)
    slug = label.replace(" ", "_").replace(",", "")
    out_path = os.path.join(workdir, f"{slug}.npz")
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = u.REQUIRED_OMP_THREADS

    print(f"  worker {label} ...", flush=True)
    start = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, THIS_FILE, "--worker", json.dumps(spec), out_path],
        env=environment)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError(
            f"worker {label} exited with code {completed.returncode} "
            "before writing its vectors (the reason is printed above)")
    print(f"  worker {label}: {elapsed:.1f} s", flush=True)

    with np.load(out_path) as saved:
        return {name: saved[name].copy() for name in saved.files}


# =============================================================================
# THE TESTS
# =============================================================================
@unittest.skipUnless(RUN_SLOW, "slow halo-model IA accuracy arms; set "
                     "COCOA_HALO_SLOW=1 to run")
class TestHaloIAAccuracy(unittest.TestCase):
    """IA0-IA6: flag invisibility, knob convergence, the lmax knob.

    setUpClass verifies the frozen state, writes the scratch dataset
    and factors the scoring covariances once; the worker results are
    cached on the class, so the reference arm is built once for all
    the convergence tests.
    """

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.workdir = tempfile.mkdtemp(prefix="cocoa_halo_ia_accuracy_")
        # unittest skips tearDownClass when setUpClass raises, so the
        # temporary directory is removed here on that path
        try:
            cls.scoring = build_scoring(cls.workdir)
        except Exception:
            shutil.rmtree(cls.workdir, ignore_errors=True)
            raise
        cls.results = {}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workdir, ignore_errors=True)

    def _worker(self, label, spec):
        """The vectors of one worker, run once per class.

        A failed worker is cached too: its exception is raised again
        for every later test that needs it, so a crashing reference
        arm (the expensive one) is not respawned by IA1-IA5 in turn.
        """
        results = type(self).results
        if label not in results:
            try:
                results[label] = run_worker(self.workdir, label, spec)
            except RuntimeError as failure:
                results[label] = failure
        if isinstance(results[label], RuntimeError):
            raise results[label]
        vectors = results[label]
        for name, vector in vectors.items():
            if not name.startswith("dv"):
                continue
            self.assertEqual(
                len(vector), self.scoring["n"],
                f"worker {label}: {name} has {len(vector)} points, the "
                f"covariance {self.scoring['n']}")
        return vectors

    def _accuracy_arm(self, label, accuracy_boost, integration_accuracy,
                      one_halo_signal=False):
        spec = {"kind": "accuracy",
                "accuracy_boost": accuracy_boost,
                "integration_accuracy": integration_accuracy,
                "one_halo_signal": one_halo_signal}
        return self._worker(label, spec)

    def _reference(self):
        """The reference arm, with its vacuity guard and signal report."""
        ref = self._accuracy_arm("reference", *REFERENCE_SETTINGS)
        increment = ref["dv_on"] - ref["dv_off"]
        signal = chi2_of(increment, self.scoring, "aggressive")
        self.assertGreater(
            signal, MIN_IA_SIGNAL_CHI2,
            f"the reference IA increment has chi2 {signal:.3e} on the "
            "aggressive mask: include_halo_IA barely changes the data "
            "vector, so the convergence checks would pass vacuously")
        return ref

    def _check_arm(self, number):
        label, boost, hdi = ACCURACY_ARMS[number]
        ref = self._reference()
        is_production = (boost, hdi) == PRODUCTION_SETTINGS
        arm = self._accuracy_arm(label, boost, hdi,
                                 one_halo_signal=is_production)

        increment_ref = ref["dv_on"] - ref["dv_off"]
        increment_arm = arm["dv_on"] - arm["dv_off"]
        delta_ia = increment_arm - increment_ref
        delta_full = arm["dv_on"] - ref["dv_on"]
        scores = {
            "ia_aggr": chi2_of(delta_ia, self.scoring, "aggressive"),
            "ia_prod": chi2_of(delta_ia, self.scoring, "production"),
            "full_aggr": chi2_of(delta_full, self.scoring, "aggressive"),
            "full_prod": chi2_of(delta_full, self.scoring, "production"),
        }
        settings = (f"{label} (accuracyboost {boost}, "
                    f"integration_accuracy {hdi})")
        tested = report_arm(number, settings, scores, IA_ACCURACY_TOLERANCE)

        if is_production:
            # how large the signal under test is: the whole increment,
            # its 2-halo part (a_1h = 0) and its 1-halo part
            two_halo = arm["dv_2h"] - arm["dv_off"]
            one_halo = arm["dv_on"] - arm["dv_2h"]
            for name, vector in (("whole IA increment", increment_arm),
                                 ("2-halo part (a_1h = 0)", two_halo),
                                 ("1-halo part", one_halo)):
                print(f"  signal, {name:24s} chi2 = "
                      f"{chi2_of(vector, self.scoring, 'aggressive'):.6e}"
                      f" (aggressive)   "
                      f"{chi2_of(vector, self.scoring, 'production'):.6e}"
                      " (production)", flush=True)

        self.assertLess(
            tested, IA_ACCURACY_TOLERANCE,
            f"{number} {settings}: the IA increment differs from the "
            f"reference by chi2 {tested:.4e} on the aggressive mask "
            f"(limit {IA_ACCURACY_TOLERANCE})")

    def test_ia0_flag_off_is_invisible(self):
        """IA0: include_halo_IA = 0 reproduces a never-touched model."""
        import numpy as np

        pristine = self._worker("pristine", {"kind": "pristine"})["dv"]
        touched = self._worker("touched", {"kind": "touched"})

        self.assertFalse(
            np.array_equal(touched["dv_on"], pristine),
            "include_halo_IA = 1 left the data vector unchanged: the "
            "flag is dead, so the flag-off checks would pass vacuously")

        def largest_relative(a, b):
            nonzero = b != 0
            if not nonzero.any():
                return 0.0
            return float(np.max(np.abs(a[nonzero]/b[nonzero] - 1.0)))

        cases = (("dv_a", "IA parameters set, flag never on"),
                 ("dv_b", "flag switched on, evaluated, switched off"))
        for name, what in cases:
            same = np.array_equal(touched[name], pristine)
            if same:
                verdict = "bitwise identical"
            else:
                verdict = (f"DIFFERS (max rel "
                           f"{largest_relative(touched[name], pristine):.2e})")
            print(f"  IA0 {what}: {verdict}", flush=True)
            self.assertTrue(
                same,
                f"include_halo_IA = 0 ({what}) is not bitwise identical "
                "to a model that never touched the IA setters: the IA "
                "code is visible when off")

    def test_ia1_production(self):
        """IA1: production settings vs the reference arm."""
        self._check_arm("IA1")

    def test_ia2_accuracyboost_2(self):
        """IA2: accuracyboost 2 vs the reference arm."""
        self._check_arm("IA2")

    def test_ia3_integration_accuracy_1(self):
        """IA3: integration_accuracy 1 vs the reference arm."""
        self._check_arm("IA3")

    def test_ia4_integration_accuracy_2(self):
        """IA4: integration_accuracy 2 vs the reference arm."""
        self._check_arm("IA4")

    def test_ia5_integration_accuracy_3(self):
        """IA5: integration_accuracy 3 vs the reference arm."""
        self._check_arm("IA5")

    def test_ia6_halo_ia_lmax_is_live(self):
        """IA6: the lmax = 2 and 4 truncations move the data vector."""
        import numpy as np

        vectors = self._worker("lmax", {"kind": "lmax"})
        if "binding_missing" in vectors:
            self.skipTest(
                f"the compiled interface has no {HALO_IA_LMAX_BINDING} "
                "binding, so Ntable.halo_ia_lmax cannot be set from Python")
        default = HALO_IA_LMAX_VALUES[0]
        dv_default = vectors[f"dv_{default}"]

        print(f"\n{'-' * 66}\nTEST IA6: halo_ia_lmax truncations vs lmax = "
              f"{default} (physics choice: no tolerance)", flush=True)
        for lmax in HALO_IA_LMAX_VALUES[1:]:
            delta = vectors[f"dv_{lmax}"] - dv_default
            aggr = chi2_of(delta, self.scoring, "aggressive")
            prod = chi2_of(delta, self.scoring, "production")
            print(f"  lmax = {lmax}:  chi2 = {aggr:.6e} (aggressive)   "
                  f"{prod:.6e} (production)", flush=True)
            self.assertFalse(
                np.array_equal(vectors[f"dv_{lmax}"], dv_default),
                f"halo_ia_lmax = {lmax} gives the lmax = {default} data "
                "vector bit for bit: the knob is dead (or its tables did "
                "not rebuild)")
            self.assertGreater(aggr, 0.0)

        again = np.array_equal(vectors[f"dv_{default}_again"], dv_default)
        if again:
            verdict = "bitwise identical"
        else:
            verdict = "DIFFERS"
        print(f"  back to lmax = {default}: {verdict}\n{'-' * 66}",
              flush=True)
        self.assertTrue(
            again,
            f"returning to halo_ia_lmax = {default} did not reproduce the "
            "first vector bit for bit, so the truncation differences "
            "above are not the knob alone")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], sys.argv[3])
    else:
        unittest.main(verbosity=2)
