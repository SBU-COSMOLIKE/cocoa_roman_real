"""Unit tests 20-21: race and thread-count checks of the halo-model IA.

With include_halo_IA = 1 the Limber C_l^ss and C_l^gs engines add the
halo-model intrinsic alignment of Fortuna et al. (2021): a 1-halo
satellite term, a_1h(a) f_1h(k) S_dI(k, a) and a_1h(a)^2 f_1h(k)
S_II(k, a), plus the NLA 2-halo term of the red centrals weighted by
the red-central fraction f_rc(a). halo.c fills S_dI, S_II and f_rc in
OpenMP loop nests (one table entry per (a row, halo) or (a row, k)
pair) and the C_l engines read them inside their own parallel
regions.

Why a race shows up as nondeterminism. Two threads writing the same
scratch slot, a thread reading a row another thread has not finished,
or a table read before its single-threaded warm-up completed: each of
these makes the result depend on how the threads happened to be
scheduled. Correct code has no such dependence - every table entry is
an independent computation with a fixed summation order - so the same
inputs must give the same BITS on every run and for every thread
count. A race, by contrast, shows up as run-to-run noise (it moves
with scheduling) or as a thread-count dependence (it moves with the
partition of the loops). Both checks below are therefore bitwise: no
tolerance could separate a rare race from float noise, and correct
code needs none.

The state is the frozen NLA 3x2pt model (example2) with Limber gs
(adopt_limber_gs = 1; the halo-IA gs path is Limber-only),
perturbative-bias galaxies (include_HOD_GX = 0; HOD x halo IA aborts)
and the halo-IA fiducial set through ci.set_nuisance_ia_halo:
a_1h = 0.001 (F21 red galaxies), eta_1h = 0, z_pivot = 0.62; the
red-fraction sigmoids {lg M_cen, width_cen, lg M_sat, width_sat} =
{13, 0.5, 12.5, 0.7} (the student's defaults); the IA-population HOD
{lg M_min, sigma_lgM, lg M_1, lg M_0, alpha, f_c} =
{12, 0.3, 13.3, 11.5, 1, 1} (the Zheng et al. 2007 values of the IA
port's table tests). The red-central 2-halo amplitude is the frozen
point's roman_A1.

  20. Ten in a row: on ONE model instance, the fiducial point is
      evaluated ten times (cobaya's cache bypassed, cached = False),
      and before each of evaluations 2-10 the halo-IA tables are
      forced to REFILL: one IA input is moved away (cycling through
      a_1h, the red-satellite sigmoid centre and the IA-HOD lg M_0,
      which changes the set of halos that do kernel work), the IA
      probes are read at the moved state - which refills the tables
      with different numbers and must change the probes, or the
      refill was not forced - and the input is moved back. Every
      evaluation's chi2, masked data vector and IA probe vector must
      equal the first bit for bit. Run at REQUIRED_OMP_THREADS (4)
      threads.
  21. Thread count: the same point evaluated once in fresh worker
      processes at OMP_NUM_THREADS = 1, 4 and 8 (OpenMP reads the
      variable when the compiled library loads, and the likelihood
      re-applies it before every evaluation, so each count needs its
      own process). The IA probe vector, the data vector and the chi2
      must be bitwise identical across the three counts. Two
      attribution vectors are printed, not asserted: the CAMB inputs
      the likelihood hands to cosmolike (a thread-dependent CAMB
      would move everything downstream) and the data vector with
      include_halo_IA = 0 (a flag-off vector that also moves locates
      the nondeterminism outside the halo-IA path).

The IA probes are the production readers at fixed points:
ci.ia_f_red_central(a), ci.ia_p1h_dI(k, a) and ci.ia_p1h_II(k, a)
at a = 0.4, 0.6, 0.8 and k = 0.1, 1, 10, 50 h/Mpc, all inside the
source a range (the readers return 0 outside it; a zero probe would
make the comparison vacuous, so every probe must be nonzero).

Every model build runs in a worker subprocess of this file: the
halo-IA flag and tables are process-wide statics of cosmolike, so the
pytest process itself never initializes them and no later test module
inherits them.

Slow (one model build plus ten evaluations for test 20; three builds
and six evaluations, one set on a single thread, for test 21): runs
only with COCOA_HALO_SLOW=1, like the spectrum tier of test_halo.py.
To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    COCOA_HALO_SLOW=1 python -m pytest \\
      ./projects/roman_real/tests/test_halo_ia_race.py
"""

import os
import sys

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process. A
# worker subprocess of this file (argv[1] == "--worker") keeps the
# value its parent chose: test 21 spawns workers at 1, 4 and 8.
_IS_WORKER = len(sys.argv) > 1 and sys.argv[1] == "--worker"
if not _IS_WORKER:
    os.environ["OMP_NUM_THREADS"] = "4"

import subprocess
import tempfile
import unittest

import numpy as np

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

RUN_SLOW = os.environ.get("COCOA_HALO_SLOW", "0") == "1"

EXAMPLE = "example2"

# ---- the halo-IA fiducial (see the module docstring) ------------------------
IA_HALO = (0.001, 0.0, 0.62)                    # a_1h, eta_1h, z_pivot
IA_RED = (13.0, 0.5, 12.5, 0.7)                 # lgM_cen, w_cen, lgM_sat, w_sat
IA_HOD = (12.0, 0.3, 13.3, 11.5, 1.0, 1.0)      # lgMmin, sigma, lgM1, lgM0,
                                                # alpha, f_c

# The moves that force a refill before evaluations 2-10 of test 20,
# used in turn: (label, set_nuisance_ia_halo argument, index, value).
# Each is a legal input (|a_1h| < 0.3, widths untouched) that changes
# the tables: a_1h enters the truncation radius r_e and the amplitude,
# the sigmoid centre the red-satellite occupation, lg M_0 the list of
# halos with red satellites (the "active" kernel nodes).
AWAY_MOVES = (
    ("a_1h 0.001 -> 0.002", "ia_halo", 0, 0.002),
    ("red-satellite sigmoid centre 12.5 -> 12.8", "ia_red", 2, 12.8),
    ("IA-HOD lg M_0 11.5 -> 12.0", "ia_hod", 3, 12.0),
)
NEVAL = 10

# ---- the IA probe points ----------------------------------------------------
COVERH0 = 2997.92458                            # c/H0 in Mpc/h (structs.c)
PROBE_A = (0.4, 0.6, 0.8)
PROBE_K = np.array([0.1, 1.0, 10.0, 50.0]) * COVERH0   # (c/H0)^-1

# ---- test 21 ----------------------------------------------------------------
THREAD_COUNTS = ("1", "4", "8")

# The smallest positive double: |delta| < BITWISE holds only for
# delta == 0, so report_race_test's verdict matches the bitwise
# assertion.
BITWISE = 5e-324


# =============================================================================
# WORKER SIDE (runs inside the subprocesses)
# =============================================================================
def build_halo_ia_model():
    """The frozen NLA 3x2pt model with the halo-model IA switched on.

    Arguments:
      none.

    Returns:
      (model, fiducial point, compiled interface), with the halo-IA
      fiducial already set.
    """
    import cosmolike_roman_real_interface as ci

    info = u.load_frozen_info(EXAMPLE, tatt=False)
    block = info["likelihood"][u.EXAMPLES[EXAMPLE]["likelihood"]]
    block["adopt_limber_gs"] = 1
    block["include_HOD_GX"] = 0
    block["include_halo_IA"] = 1
    model = u.make_model(info)
    point = u.build_point(model, EXAMPLE, tatt=False)
    set_ia_state(ci, IA_HALO, IA_RED, IA_HOD)
    return model, point, ci


def set_ia_state(ci, ia_halo, ia_red, ia_hod):
    """Set the halo-IA inputs (the setter redraws the cache tag on change)."""
    ci.set_nuisance_ia_halo(ia_halo=np.array(ia_halo, dtype=float),
                            ia_red=np.array(ia_red, dtype=float),
                            ia_hod=np.array(ia_hod, dtype=float))


def ia_probes(ci):
    """f_rc, P_dI^1h and P_II^1h at the probe points, as one flat vector.

    Reading refills the halo-IA tables when their cache keys changed.
    """
    out = []
    for a in PROBE_A:
        out.append(np.atleast_1d(ci.ia_f_red_central(a=a)))
        out.append(np.asarray(ci.ia_p1h_dI(k=PROBE_K, a=a), dtype=float).ravel())
        out.append(np.asarray(ci.ia_p1h_II(k=PROBE_K, a=a), dtype=float).ravel())
    return np.concatenate(out)


def camb_inputs(model):
    """The CAMB products the likelihood reads, as one flat vector.

    The linear and nonlinear P(k) interpolators (read on a fixed (z, k)
    grid) and the comoving distances that set_cosmo_related consumes.
    """
    like = model.likelihood[u.EXAMPLES[EXAMPLE]["likelihood"]]
    out = []
    # the interpolators set_cosmo_related reads, on a fixed (z, k) grid
    z = np.linspace(0.0, 3.0, 16)
    k = np.logspace(-3.0, 1.0, 32)
    for nonlinear in (False, True):
        pk = like.provider.get_Pk_interpolator(
            ("delta_tot", "delta_tot"), nonlinear=nonlinear)
        out.append(np.ravel(pk.P(z, k)))
    out.append(np.ravel(
        like.provider.get_comoving_radial_distance(like.z_interp_1D)))
    return np.concatenate(out)


def evaluate(model, point, ci):
    """One full evaluation; returns (chi2, masked data vector, IA probes)."""
    chi2 = u.evaluate_chi2(model, point)
    dv = np.array(ci.compute_data_vector_masked())
    return chi2, dv, ia_probes(ci)


def race_worker(out_path):
    """Test 20: ten evaluations of the fiducial, refill forced between."""
    model, point, ci = build_halo_ia_model()
    fiducial = {"ia_halo": IA_HALO, "ia_red": IA_RED, "ia_hod": IA_HOD}

    chi2s = []
    dvs = []
    probes = []
    away_changed = []
    for i in range(NEVAL):
        if i > 0:
            label, name, index, value = AWAY_MOVES[(i - 1) % len(AWAY_MOVES)]
            moved = {key: list(val) for key, val in fiducial.items()}
            moved[name][index] = value
            set_ia_state(ci, moved["ia_halo"], moved["ia_red"],
                         moved["ia_hod"])
            away_changed.append(not np.array_equal(ia_probes(ci), probes[0]))
            set_ia_state(ci, IA_HALO, IA_RED, IA_HOD)
        else:
            label = "fresh model"

        chi2, dv, probe = evaluate(model, point, ci)
        chi2s.append(chi2)
        dvs.append(dv)
        probes.append(probe)
        print(f"  evaluation {i + 1:2d}/{NEVAL} (after {label}):  "
              f"chi2 = {chi2:.10f}", flush=True)

    np.savez(out_path, chi2=np.array(chi2s), dv=np.array(dvs),
             probes=np.array(probes), away_changed=np.array(away_changed))


def threads_worker(out_path):
    """Test 21: one evaluation at this process's OMP_NUM_THREADS."""
    print(f"  worker OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}",
          flush=True)
    model, point, ci = build_halo_ia_model()
    chi2, dv, probe = evaluate(model, point, ci)
    camb = camb_inputs(model)

    # attribution: the same point with the halo IA off
    ci.init_include_halo_IA(include_halo_IA=0)
    u.evaluate_chi2(model, point)
    dv_off = np.array(ci.compute_data_vector_masked())

    np.savez(out_path, chi2=np.array([chi2]), dv=dv, probes=probe,
             camb=camb, dv_off=dv_off)


def run_worker(mode, threads):
    """Spawn one worker of this file and load its npz result.

    Arguments:
      mode    = "race" or "threads".
      threads = the OMP_NUM_THREADS text of the worker.

    Returns:
      a dictionary {array name: numpy array}.

    Raises:
      RuntimeError when the worker exits without a result (a
      cosmolike abort prints its reason just above).
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "result.npz")
        environment = dict(os.environ)
        environment["OMP_NUM_THREADS"] = threads
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", mode,
             out_path], env=environment)
        if completed.returncode != 0 or not os.path.isfile(out_path):
            raise RuntimeError(
                f"{mode} worker (OMP_NUM_THREADS={threads}) exited with "
                f"code {completed.returncode} before writing a result")
        with np.load(out_path) as result:
            return {name: result[name] for name in result.files}


# =============================================================================
# PARENT SIDE (pytest)
# =============================================================================
def max_relative_difference(a, b):
    """max |a/b - 1| over b != 0, and max |a - b| where b == 0."""
    nonzero = b != 0
    rel = np.abs(a[nonzero]/b[nonzero] - 1.0)
    absolute = np.abs(a[~nonzero] - b[~nonzero])
    return max(float(np.max(rel, initial=0.0)),
               float(np.max(absolute, initial=0.0)))


def bitwise_verdict(vectors):
    """'bitwise equal' or the largest difference against the first."""
    worst = max(max_relative_difference(v, vectors[0]) for v in vectors[1:])
    if all(np.array_equal(v, vectors[0]) for v in vectors[1:]):
        return "bitwise equal"
    return f"DIFFERS (max rel {worst:.3e})"


def report_thread_test(number, label, result):
    """Print test 21 in the block format of the other race tests.

    Arguments:
      number = the test number shown in the header.
      label  = one line naming the example, probe, and IA model.
      result = {thread count: the worker's arrays}.

    Returns:
      {vector name: verdict text} for the probes, the data vector,
      and the two attribution vectors.
    """
    chi2 = {n: float(result[n]["chi2"][0]) for n in THREAD_COUNTS}
    names = {
        "probes": f"IA probes ({result['1']['probes'].size})",
        "dv": f"data vector ({result['1']['dv'].size})",
        "camb": "CAMB inputs (attribution)",
        "dv_off": "flag-off dv (attribution)",
    }
    verdicts = {
        name: bitwise_verdict([result[n][name] for n in THREAD_COUNTS])
        for name in names}

    lines = []
    for n in THREAD_COUNTS:
        lines.append(f"  chi2 (OMP_NUM_THREADS = {n})".ljust(34)
                     + f"= {chi2[n]:.10f}")
    for name, text in names.items():
        lines.append(f"  {text}".ljust(34) + f"= {verdicts[name]}")

    ok = verdicts["probes"] == "bitwise equal"
    ok = ok and verdicts["dv"] == "bitwise equal"
    ok = ok and len(set(chi2.values())) == 1
    body = "\n".join(lines)
    print(f"""
{'-' * 66}
TEST {number}: {label}
{body}
  -> {'OK' if ok else 'THREAD-COUNT DEPENDENT'}
{'-' * 66}""", flush=True)
    return verdicts


@unittest.skipUnless(RUN_SLOW, "slow halo-IA race checks; set "
                     "COCOA_HALO_SLOW=1 to run")
class TestHaloIARace(unittest.TestCase):
    """Tests 20-21, sharing one frozen-state verification."""

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()

    def test_x20_halo_ia_ten_in_a_row(self):
        """Ten evaluations with forced IA refills agree bit for bit."""
        u.assert_omp_threads()
        result = run_worker("race", u.REQUIRED_OMP_THREADS)
        chi2 = result["chi2"]
        u.report_race_test(
            20, "example2 (3x2pt, NLA + halo IA) race check: 10 "
            "evaluations, IA tables refilled between", float(chi2[0]),
            float(chi2[-1]), BITWISE)

        # the refills were forced and the probes are not vacuous
        self.assertTrue(np.all(result["away_changed"]),
                        "a moved IA input left the IA probes unchanged: "
                        "the tables were not refilled (stale cache key)")
        self.assertTrue(np.all(result["probes"][0] != 0),
                        "an IA probe is zero at the fiducial: a probe point "
                        "is outside the source range, or the flag is off")

        for i in range(1, NEVAL):
            self.assertEqual(
                chi2[i], chi2[0],
                f"evaluation {i + 1}: chi2 = {chi2[i]:.12f} vs first "
                f"{chi2[0]:.12f}")
            self.assertTrue(
                np.array_equal(result["dv"][i], result["dv"][0]),
                f"evaluation {i + 1}: the data vector differs from the "
                "first (max rel "
                f"{max_relative_difference(result['dv'][i], result['dv'][0]):.3e})")
            self.assertTrue(
                np.array_equal(result["probes"][i], result["probes"][0]),
                f"evaluation {i + 1}: the IA probes differ from the "
                "first (max rel "
                f"{max_relative_difference(result['probes'][i], result['probes'][0]):.3e})")

    def test_x21_halo_ia_thread_count(self):
        """1, 4 and 8 threads give the same bits."""
        result = {n: run_worker("threads", n) for n in THREAD_COUNTS}
        verdicts = report_thread_test(
            21, "example2 (3x2pt, NLA + halo IA) determinism: "
            "OMP_NUM_THREADS = 1 / 4 / 8", result)

        attribution = (f"(CAMB inputs: {verdicts['camb']}; flag-off data "
                       f"vector: {verdicts['dv_off']})")
        self.assertTrue(np.all(result["1"]["probes"] != 0),
                        "an IA probe is zero at the fiducial")
        self.assertEqual(verdicts["probes"], "bitwise equal",
                         f"the IA probes depend on the thread count "
                         f"{attribution}")
        self.assertEqual(verdicts["dv"], "bitwise equal",
                         f"the data vector depends on the thread count "
                         f"{attribution}")
        for n in THREAD_COUNTS[1:]:
            self.assertEqual(result[n]["chi2"][0], result["1"]["chi2"][0],
                             f"chi2 at {n} threads differs from 1 thread")


# __name__ is "__main__" when this file runs directly: as a worker
# (--worker <mode> <result path>) or as a plain unittest run
if __name__ == "__main__":
    if _IS_WORKER:
        u.require_cocoa_environment()
        worker = {"race": race_worker, "threads": threads_worker}[sys.argv[2]]
        worker(sys.argv[3])
    else:
        unittest.main(verbosity=2)
