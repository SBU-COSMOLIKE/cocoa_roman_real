"""Exercise the public Roman real notebook route at the frozen fiducial.

EXAMPLE_EVALUATE1 imports cosmolike_roman_real_notebook_wrappers. This
regression calls that same module's configure, init_cosmolike and get_chi2
functions, so a broken wrapper cannot be hidden by a private reimplementation
of its CAMB or native-interface calls. The frozen dataset and nuisance point
keep the comparison reproducible. Its chi-squared is checked against the
frozen Cobaya reference within the existing regression tolerance of 0.2.

CAMB and interpolation settings differ slightly between the two public
routes, so this tolerance is a regression check, not a new convergence claim.
The worker also checks the prepared vector is finite, repeatable and scored
consistently. It runs in its own subprocess to isolate native global state.

Run from activated Cocoa:
    python -m pytest projects/roman_real/tests/data_vector/test_notebook_interface.py
"""

import json
import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = os.environ.get("COCOA_TEST_OMP_THREADS", "4")

import sys
import unittest

# The harness stays in the parent tests/ folder. Add it explicitly so
# direct execution and worker processes resolve this project's stored inputs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

EXAMPLE = "example1"

# =============================================================================
# WORKER LAYER (mirrors the worker layer of cosmolike_core's
# cocoa_testing.py for this file's computation; the shared _worker only
# knows the cobaya evaluations)
# =============================================================================
# __file__ is this file's own path; the worker driver imports the
# file by that absolute path (see _WORKER_DRIVER)
_THIS_FILE = os.path.abspath(__file__)

# A flag distinct from the shared harness's, so a notebook-interface
# worker is never mistaken for a cobaya-evaluation worker.
_WORKER_FLAG = "COCOA_TESTS_NOTEBOOK_WORKER"

# The driver handed to `python -c` inside the worker: load this file
# by path and call its _worker with the result path.
_WORKER_DRIVER = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location("
    "'test_notebook_interface_worker', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "module._worker(sys.argv[2])\n"
)


def _notebook_chi2_impl():
    """Exercise the same public wrappers imported by EXAMPLE_EVALUATE1.

    The frozen dataset fixes the observed vector, mask and covariance.
    Its fiducial parameters and source conventions are passed explicitly;
    the actual notebook wrapper owns CAMB, native initialization and units.
    This test must fail when that public route breaks, even if the Cobaya
    likelihood remains usable. It does not maintain a second CAMB producer.
    """
    import numpy as np
    import cosmolike_roman_real_interface as ci
    import cosmolike_roman_real_notebook_wrappers as nw

    info = u.load_frozen_info(EXAMPLE, tatt=False)
    like = info["likelihood"]["roman_real.cosmic_shear"]
    camb_args = info["theory"]["camb"]["extra_args"]
    point = u.load_frozen_point(EXAMPLE)
    # The wrapper's CAMB prescription is Takahashi with a fixed neutrino
    # mass. Fail explicitly if a future frozen fixture changes that model.
    if camb_args["halofit_version"] != "takahashi":
        raise AssertionError("The notebook comparison requires matched Takahashi power.")
    if point["mnu"] != nw.mnu:
        raise AssertionError("The notebook and frozen neutrino masses differ.")

    nw.configure(
        path=like["path"],
        data_file=like["data_file"],
        lmax=like["lmax"],
        ggl_exclude=like["ggl_exclude"],
        IA_model=0,
        IA_code=0,
        IA_redshift_evolution=like["IA_redshift_evolution"],
        internal_accuracyboost=like["internal_accuracyboost"],
        photoz_interpolation_type=like["photoz_interpolation_type"],
        photoz_zmid_convention=like["photoz_zmid_convention"],
        nonlimber_zmax=like["nonlimber_zmax"],
        non_linear_emul=2,
    )
    nw.init_cosmolike(CLprobe="xi", with_data=True)
    # AccuracyBoost=1 keeps the wrapper's sampling multiplier separate
    # from the explicitly requested CAMB and native accuracy controls.
    chi2 = nw.get_chi2(
        omegam=point["omegam"],
        omegab=point["omegab"],
        H0=point["H0"],
        ns=point["ns"],
        As_1e9=point["As_1e9"],
        w=point["w"],
        w0pwa=point["w0pwa"],
        A1=[point["roman_A1_1"], point["roman_A1_2"], 0, 0, 0, 0, 0, 0],
        A2=[point["roman_A2_1"], point["roman_A2_2"], 0, 0, 0, 0, 0, 0],
        BTA=[point["roman_BTA_1"], 0, 0, 0, 0, 0, 0, 0],
        shear_photoz_bias=[point[f"roman_DZ_S{i}"] for i in range(1, 9)],
        M=[point[f"roman_M{i}"] for i in range(1, 9)],
        AccuracyBoost=1.0,
        CAMBAccuracyBoost=camb_args["AccuracyBoost"],
        CLAccuracyBoost=like["accuracyboost"],
        CLIntegrationAccuracy=like["integration_accuracy"],
        kmax=like["kmax_boltzmann"],
        k_per_logint=camb_args["k_per_logint"],
        non_linear_emul=2,
    )
    vector = np.asarray(ci.compute_data_vector_masked()).copy()
    if not np.isfinite(vector).all() or not np.isfinite(chi2):
        raise AssertionError("The notebook wrapper produced nonfinite results.")
    # Re-reading the already prepared vector must preserve its entries;
    # re-scoring it checks that the public wrapper returns this same state.
    np.testing.assert_array_equal(vector, ci.compute_data_vector_masked())
    np.testing.assert_allclose(chi2, ci.compute_chi2(vector), rtol=0, atol=1.e-12)
    print(f"    notebook-wrapper chi2 = {chi2:.6f}", flush=True)
    return float(chi2)


def _worker(result_path):
    """Worker-side entry: compute the chi2 and save it as json.

    Arguments:
      result_path = file the chi2 is written into; the parent reads
                    it back. Progress prints go to the inherited
                    stdout, so the terminal streams them.

    Returns:
      nothing; the result lands in result_path.
    """
    u.require_cocoa_environment()
    value = _notebook_chi2_impl()
    # json.dump writes the chi2 into the file as json text; the with
    # block closes the file even when the dump fails
    with open(result_path, "w") as f:
        json.dump(value, f)


def notebook_interface_chi2():
    """Return the chi2 of the notebook call path, from a fresh worker.

    Returns:
      the chi2 as a float.

    Raises:
      RuntimeError when the worker dies without writing a result: the
      cosmolike C layer aborts the process on an internal
      inconsistency (a changed binding or a rejected grid) instead of
      raising, and that abort is precisely what this test watches for.
    """
    import subprocess
    import tempfile

    # .get returns None when the flag is absent, so a normal
    # (parent) process falls through and spawns the worker below
    if os.environ.get(_WORKER_FLAG) == "1":
        return _notebook_chi2_impl()
    # delete=False keeps the file when the with block closes it: only
    # a fresh unique name is needed; the worker writes the file and
    # the finally below removes it
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False) as tmp:
        result_path = tmp.name
    # dict(os.environ) is a copy of the environment: the edits below
    # reach only the worker subprocess, never this process
    environment = dict(os.environ)
    environment[_WORKER_FLAG] = "1"
    environment["OMP_NUM_THREADS"] = u.REQUIRED_OMP_THREADS
    # subprocess.run starts the worker and waits until it exits;
    # env=environment hands the child the edited environment copy
    completed = subprocess.run(
        [sys.executable, "-c", _WORKER_DRIVER, _THIS_FILE, result_path],
        env=environment)
    # the finally below runs on every exit from the try, an exception
    # included, so the temporary file never outlives this call
    try:
        if completed.returncode != 0:
            raise RuntimeError(
                "notebook-interface worker exited with code "
                f"{completed.returncode} before writing a result; a "
                "cosmolike-level abort prints its reason (e.g. a grid "
                "rejection from basics.c) just above")
        # json.load parses the worker's file back into the number it
        # dumped; float() pins the type
        with open(result_path) as f:
            return float(json.load(f))
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)


# =============================================================================
# THE TEST
# =============================================================================
class TestNotebookInterface(unittest.TestCase):
    """Test 5, sharing the suite's frozen-state verification.

    setUpClass runs once: it moves to ROOTDIR, verifies every frozen
    file against the SHA-256 manifest (an edited frozen state must
    fail loudly before any physics runs), and loads the frozen
    reference chi2 values.
    """

    # @classmethod hands the class itself in as cls; unittest calls
    # this once, before the first test of the class
    @classmethod
    def setUpClass(cls):
        """Verify the frozen state and load the frozen reference chi2 values."""
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def test_5_notebook_interface_chi2(self):
        """The public notebook-wrapper call path agrees with Cobaya."""
        chi2 = notebook_interface_chi2()
        ref = self.reference[f"{EXAMPLE}_nla"]
        u.report_chi2_test(
            5, "example1 public notebook wrappers (cosmic shear, "
               "NLA) chi2 vs frozen cobaya reference",
            chi2, ref, u.CHI2_TOLERANCE)
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"notebook-wrapper chi2 = {chi2:.6f} vs frozen cobaya "
                f"reference {ref:.6f} (|delta| >= {u.CHI2_TOLERANCE}); "
                "an interface-binding or grid-validation change likely "
                "broke the notebook call path")


if __name__ == "__main__":
    unittest.main(verbosity=2)
