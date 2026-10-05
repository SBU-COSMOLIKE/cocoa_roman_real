"""Unit test: halo-model IA cache invalidation (the IA parameter ladder).

The halo-model intrinsic alignment (Fortuna et al. 2021, one IA
population over the source redshift range) lives in one halo.c table
owner, ia_tables: the 1-halo spectra S_dI and S_II of the red
satellites and the red-central fraction f_rc(a), tabulated on
(a, ln k) over the source a range. Its refill is keyed on the
cosmology, Ntable, the IA-halo nuisance tag (random_ia_halo), the
source n(z) tag and the source photo-z tag - the last because the
source a range the tables live on is re-read at every refill (the
source-range invalidation class fixed in core commit 10c0193).
cosmo2D.c reads those tables into the Limber C_l^ss and C_l^gs when
include_halo_IA = 1, and its C_ss/C_gs caches key on the flag and on
random_ia_halo. A partial-invalidation bug - one sector's update
failing to rebuild a table another sector consumes - produces
silently wrong spectra only in MIXED update sequences, which the
per-point suites never exercise. This is
test_halo_cache_consistency.py's ladder, applied to the IA sectors.

The model: the frozen NLA example2 3x2pt configuration with Limber
C_l^gs (the halo-IA gs engine is Limber-only) and perturbative galaxy
bias (include_HOD_GX = 0: HOD x halo IA aborts), include_halo_IA = 1.
The sampled NLA amplitude roman_A1_* is the red-central (2-halo)
amplitude in this model.

The test walks a deterministic ladder IN ONE PROCESS, evaluating the
model after every step (each sector's later steps keep the earlier
sectors at their last values, so the ladder ends at one well-defined
point):

    3 x cosmology-only steps          (omegam, H0, As_1e9)
    3 x IA-halo-only steps            (set_nuisance_ia_halo; step s
                                       moves subset (s - 1) mod 3:
                                       {a_1h, eta_1h}, the four red
                                       sigmoid parameters, the IA HOD
                                       lg M_min and alpha)
    3 x NLA-amplitude steps           (every sampled roman_A1_*)
    3 x source-photo-z steps          (every DZ_S shift)
    3 x shear-calibration steps       (every M)
    3 x galaxy-bias steps             (every B1, which enters gs)

The IA-halo sector is not sampled: the ladder sets it through the
interface setter, which redraws random_ia_halo only when a value
changes, so every other table stays cached across a step - exactly
the partial-invalidation case under test. The IA-halo fiducial is
a_1h = 0.001 (F21 red), eta_1h = 0, z_pivot = 0.62, with the
student's defaults for the rest (the red sigmoids {13.0, 0.5, 12.5,
0.7} of the student's halo.c and notebook, and the IA HOD row the
student hard-coded, Coupon et al. 2012 red galaxies M_r < -21.8:
{13.17, 0.39, 14.53, 11.09, 1.27, 1.00}). After every step the test
records two vectors:

  dv = the masked 3x2pt data vector;
  ia = the IA probes at fixed points: ia_f_red_central(a),
       ia_p1h_dI(k, a) and ia_p1h_II(k, a) at (k, a) spanning the
       source range (a in [0.2, 0.999] for the frozen n(z)), and
       ia_window_2h(k).

Assertions (test_halo_ia_cache_consistency):
  1. every ladder step changes the data vector; the cosmology and
     IA-halo steps also change the IA probes, and the NLA, shear-
     calibration and galaxy-bias steps leave them bitwise unchanged
     (the tables read none of those parameters) - a dead sector flag
     would pass the later checks vacuously. The photo-z steps are
     checked on dv only: whether they move the IA probes depends on
     whether the shifts move the source a range, which the fresh-
     process check (5) covers either way;
  2. a no-op update (the final point again) leaves both vectors
     bitwise unchanged;
  3. after a SCRAMBLE (every sector moved at once), returning to the
     ladder's final point reproduces both vectors bit for bit;
  4. a second model instance walking the MIRRORED ladder lands on the
     same vectors bit for bit;
  5. a FRESH process (a subprocess: cosmolike's tables are
     per-process statics, so a second model instance in this process
     is not fresh) that evaluates the final point once, with every
     cache built from scratch, reproduces both vectors bit for bit.

Flag flip (test_halo_ia_flag_flip), at the fiducial point: turning
include_halo_IA 1 -> 0 -> 1 -> 0 with nothing else changed must move
the data vector (the flag is live), leave the IA probes bitwise
unchanged (the tables do not read the flag), and return bitwise to
the recorded vectors of each flag value (the C_ss/C_gs caches key on
the flag).

Every evaluation forces a full recomputation (cobaya's cache is
bypassed), so each assertion tests cosmolike's own invalidation.
include_halo_IA is a process-wide static: the class teardown resets
it to 0 for the next test module's models.

Slow (about fifty full 3x2pt evaluations plus the fresh subprocess):
runs only with COCOA_HALO_SLOW=1, like the spectrum tier of
test_halo.py. To run (from the Cocoa/ folder, cocoa environment
active, start_cocoa.sh sourced):

    COCOA_HALO_SLOW=1 python -m pytest \\
      ./projects/roman_real/tests/data_vector/test_halo_ia_cache_consistency.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import json
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

RUN_SLOW = os.environ.get("COCOA_HALO_SLOW", "0") == "1"

EXAMPLE = "example2"

# ---- the pinned IA-halo state (the fiducial of the IA-halo sector) ----------

# {a_1h, eta_1h, z_pivot}: F21 red a_1h, no redshift evolution
IA_HALO_FIDUCIAL = (0.001, 0.0, 0.62)
# red-fraction sigmoids {lg M_cen, width_cen, lg M_sat, width_sat}
# (the student's defaults)
IA_RED_FIDUCIAL = (13.0, 0.5, 12.5, 0.7)
# IA-population HOD {lg M_min, sigma_lgM, lg M_1, lg M_0, alpha, f_c}
# (the Coupon et al. 2012 row the student hard-coded)
IA_HOD_FIDUCIAL = (13.17, 0.39, 14.53, 11.09, 1.27, 1.00)

# The IA-halo subsets, each with its per-move offsets {setter array:
# {column: delta}}. Step s of the sector moves subset (s - 1) mod 3,
# so the three ladder steps move three different subsets.
IA_HALO_SUBSETS = (
    {"ia_halo": {0: 0.0005, 1: 0.3}},                   # a_1h, eta_1h
    {"ia_red": {0: 0.1, 1: 0.05, 2: 0.1, 3: 0.05}},     # red sigmoids
    {"ia_hod": {0: 0.03, 4: 0.02}},                     # lg M_min, alpha
)

# ---- the sampled sectors ----------------------------------------------------

SECTORS = (
    ("cosmo", re.compile(r"^(As_1e9|H0|ns|omegab|omegam|mnu|w|w0pwa)$")),
    ("nla", re.compile(r"_A1_")),
    ("dz_source", re.compile(r"_DZ_S")),
    ("m", re.compile(r"_M[0-9]+$")),
    ("bias", re.compile(r"_B1_")),
    ("other", re.compile(r".")),
)
DELTAS = {
    "cosmo": {"omegam": 0.002, "H0": 0.2, "As_1e9": 0.02},
    "nla": {re.compile(r"_A1_[0-9]+$"): 0.05},
    "dz_source": {re.compile(r"_DZ_S"): 0.001},
    "m": {re.compile(r"_M[0-9]+$"): 0.005},
    "bias": {re.compile(r"_B1_"): 0.05},
}

# ladder order: the IA-halo steps sit between steps of other sectors
PHASES = ("cosmo", "ia_halo", "nla", "dz_source", "m", "bias")
HALO_SECTORS = ("ia_halo",)
NSTEP = 3
SCRAMBLE_STEP = 4

# what each sector's steps must do to the IA probes: "changed",
# "unchanged" (the tables read none of the sector's parameters), or
# None (not asserted)
IA_RESPONSE = {
    "cosmo": "changed",
    "ia_halo": "changed",
    "nla": "unchanged",
    "dz_source": None,
    "m": "unchanged",
    "bias": "unchanged",
}

# fixed IA probe points: k in (c/H0)^-1 (0.01 to 10 h/Mpc) and a
# spanning the source range [0.2, 0.999] of the frozen n(z)
PROBE_K = (30.0, 300.0, 3000.0, 30000.0)
PROBE_A = (0.25, 0.4, 0.55, 0.7, 0.85, 0.95)


def _sector_of(name):
    for sector, pat in SECTORS:
        if pat.search(name):
            return sector
    return "other"


def _deltas_for(sector, names):
    """{parameter: per-step delta} for this sector's sampled names."""
    table = DELTAS.get(sector, {})
    out = {}
    for n in names:
        for key, d in table.items():
            if (key == n) if isinstance(key, str) else key.search(n):
                out[n] = d
                break
    return out


def build_ia_model():
    """The frozen NLA 3x2pt model with the halo-model IA switched on.

    Returns:
      (model, fiducial point, compiled interface)
    """
    import cosmolike_roman_real_interface as ci

    info = u.load_frozen_info(EXAMPLE, tatt=False)
    name = u.EXAMPLES[EXAMPLE]["likelihood"]
    # the halo-IA gs engine is Limber-only
    info["likelihood"][name]["adopt_limber_gs"] = 1
    model = u.make_model(info)
    fid = dict(u.build_point(model, EXAMPLE, tatt=False))
    # perturbative galaxy bias: HOD x halo IA aborts
    ci.init_include_HOD_GX(0)
    ci.init_include_halo_IA(1)
    return model, fid, ci


def point_at(fid, sector_deltas, steps):
    """The sampled point with each sampled sector at its step count."""
    point = dict(fid)
    for sector, step in steps.items():
        for n, d in sector_deltas.get(sector, {}).items():
            point[n] = fid[n] + step * d
    return point


def ia_halo_moves(step):
    """How many times each IA-halo subset has moved after `step`
    steps of the sector (step s moves subset (s - 1) mod 3)."""
    nsub = len(IA_HALO_SUBSETS)
    return tuple((step + nsub - 1 - j) // nsub for j in range(nsub))


def apply_ia_halo_state(ci, steps):
    """Set the IA-halo nuisance at its sector step count (the setter
    redraws random_ia_halo only when a value changes)."""
    import numpy as np

    arrays = {"ia_halo": list(IA_HALO_FIDUCIAL),
              "ia_red": list(IA_RED_FIDUCIAL),
              "ia_hod": list(IA_HOD_FIDUCIAL)}
    for moves, subset in zip(ia_halo_moves(steps["ia_halo"]),
                             IA_HALO_SUBSETS):
        for key, cols in subset.items():
            for col, d in cols.items():
                arrays[key][col] += moves * d
    ci.set_nuisance_ia_halo(
        ia_halo=np.array(arrays["ia_halo"], dtype=float),
        ia_red=np.array(arrays["ia_red"], dtype=float),
        ia_hod=np.array(arrays["ia_hod"], dtype=float))


def ia_probes(ci):
    """The IA tables at fixed points, as one flat vector."""
    import numpy as np

    out = []
    for a in PROBE_A:
        out.append(ci.ia_f_red_central(a=a))
        for k in PROBE_K:
            out.append(ci.ia_p1h_dI(k=k, a=a))
            out.append(ci.ia_p1h_II(k=k, a=a))
    for k in PROBE_K:
        out.append(ci.ia_window_2h(k=k))
    return np.array(out)


def evaluate(model, ci, fid, sector_deltas, steps):
    """One full evaluation at the ladder point; returns (dv, ia)."""
    import numpy as np

    apply_ia_halo_state(ci, steps)
    u.evaluate_chi2(model, point_at(fid, sector_deltas, steps))
    dv = np.array(ci.compute_data_vector_masked())
    return dv, ia_probes(ci)


def sector_deltas_of(fid):
    """{sector: {parameter: per-step delta}} for the sampled sectors."""
    return {s: _deltas_for(s, [n for n in fid if _sector_of(n) == s])
            for s, _ in SECTORS}


def fresh_worker(steps_json, out_path):
    """Subprocess entry: evaluate the given ladder point ONCE, every
    cache built from scratch, and save (dv, ia)."""
    import numpy as np

    steps = json.loads(steps_json)
    model, fid, ci = build_ia_model()
    dv, ia = evaluate(model, ci, fid, sector_deltas_of(fid), steps)
    np.savez(out_path, dv=dv, ia=ia)


def _max_rel(a, b):
    """Largest relative difference over b's nonzero entries."""
    import numpy as np

    nz = b != 0
    return float(np.max(np.abs(a[nz]/b[nz] - 1.0))) if nz.any() else 0.0


@unittest.skipUnless(RUN_SLOW, "slow halo IA cache ladder; set "
                     "COCOA_HALO_SLOW=1 to run")
class TestHaloIACacheConsistency(unittest.TestCase):
    """IA-sector ladder cache-invalidation check, halo-model IA live."""

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()

    @classmethod
    def tearDownClass(cls):
        # the halo-IA gate is a process-wide static: leave it off for
        # the next test module's models
        import cosmolike_roman_real_interface as ci
        ci.init_include_halo_IA(0)

    def _walk(self, order):
        """Walk the ladder in the given order; returns the final state
        and the recorded vectors."""
        import numpy as np

        model, fid, ci = build_ia_model()
        sector_deltas = sector_deltas_of(fid)
        for s in ("cosmo", "nla", "dz_source", "m", "bias"):
            self.assertTrue(sector_deltas[s],
                            f"no sampled parameters in sector {s}")

        active = PHASES
        if order == "mirrored":
            active = tuple(reversed(active))

        steps = {s: 0 for s in list(sector_deltas) + list(HALO_SECTORS)}
        prev_dv, prev_ia = evaluate(model, ci, fid, sector_deltas, steps)
        self.assertTrue(np.all(np.isfinite(prev_ia)),
                        f"{order}: non-finite IA probes at the fiducial")

        for sector in active:
            for r in range(1, NSTEP + 1):
                steps[sector] = r
                dv, ia = evaluate(model, ci, fid, sector_deltas, steps)
                self.assertFalse(
                    np.array_equal(dv, prev_dv),
                    f"{order}: {sector} step {r} left the data vector "
                    "unchanged (dead sector flag or stale cache)")
                if IA_RESPONSE[sector] == "changed":
                    self.assertFalse(
                        np.array_equal(ia, prev_ia),
                        f"{order}: {sector} step {r} left the IA probes "
                        "unchanged (dead sector flag or stale IA table)")
                elif IA_RESPONSE[sector] == "unchanged":
                    self.assertTrue(
                        np.array_equal(ia, prev_ia),
                        f"{order}: {sector} step {r} changed the IA "
                        "probes, which read none of its parameters "
                        f"(max rel {_max_rel(ia, prev_ia):.2e})")
                prev_dv, prev_ia = dv, ia

        final_steps = dict(steps)
        final_dv, final_ia = evaluate(model, ci, fid, sector_deltas,
                                      final_steps)

        # 2. no-op: the same point again, bitwise
        dv, ia = evaluate(model, ci, fid, sector_deltas, final_steps)
        self.assertTrue(np.array_equal(dv, final_dv),
                        f"{order}: a no-op re-evaluation changed the data "
                        "vector")
        self.assertTrue(np.array_equal(ia, final_ia),
                        f"{order}: a no-op re-evaluation changed the IA "
                        "probes")

        # 3. scramble (every sector at once), then return
        scramble = {s: SCRAMBLE_STEP for s in steps}
        evaluate(model, ci, fid, sector_deltas, scramble)
        back_dv, back_ia = evaluate(model, ci, fid, sector_deltas,
                                    final_steps)
        self.assertTrue(np.array_equal(back_dv, final_dv),
                        f"{order}: returning after the scramble did not "
                        "reproduce the data vector bit for bit "
                        f"(max rel {_max_rel(back_dv, final_dv):.2e})")
        self.assertTrue(np.array_equal(back_ia, final_ia),
                        f"{order}: returning after the scramble did not "
                        "reproduce the IA probes bit for bit "
                        f"(max rel {_max_rel(back_ia, final_ia):.2e})")
        print(f"  {order} ladder: {len(final_dv)} data points, "
              f"{len(final_ia)} IA probes", flush=True)
        return final_steps, final_dv, final_ia

    def test_halo_ia_cache_consistency(self):
        import numpy as np

        steps, dv_fwd, ia_fwd = self._walk("forward")
        steps_mir, dv_mir, ia_mir = self._walk("mirrored")
        self.assertEqual(steps, steps_mir)

        # 4. the mirrored order lands on the same vectors
        self.assertTrue(np.array_equal(dv_fwd, dv_mir),
                        "the mirrored ladder landed on a different data "
                        "vector: the answer depends on the invalidation "
                        "history")
        self.assertTrue(np.array_equal(ia_fwd, ia_mir),
                        "the mirrored ladder landed on different IA "
                        "probes")

        # 5. a fresh process, every cache built once, at the final point
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "fresh.npz")
            run = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--fresh",
                 json.dumps(steps), out],
                capture_output=True, text=True, env=dict(os.environ))
            self.assertEqual(run.returncode, 0,
                             f"fresh worker failed:\n{run.stderr[-3000:]}")
            fresh = np.load(out)
            dv_fresh, ia_fresh = fresh["dv"], fresh["ia"]

        self.assertTrue(
            np.array_equal(dv_fwd, dv_fresh),
            "the ladder's final data vector differs from a fresh "
            f"process at the same point (max rel "
            f"{_max_rel(dv_fwd, dv_fresh):.2e}): a table was not rebuilt "
            "when its inputs changed")
        self.assertTrue(
            np.array_equal(ia_fwd, ia_fresh),
            "the ladder's final IA probes differ from a fresh process "
            f"at the same point (max rel {_max_rel(ia_fwd, ia_fresh):.2e})"
            ": an IA table was not rebuilt when its inputs changed")

    def test_halo_ia_flag_flip(self):
        import numpy as np

        model, fid, ci = build_ia_model()
        sector_deltas = sector_deltas_of(fid)
        steps = {s: 0 for s in list(sector_deltas) + list(HALO_SECTORS)}

        dv_on, ia_on = evaluate(model, ci, fid, sector_deltas, steps)

        ci.init_include_halo_IA(0)
        dv_off, ia_off = evaluate(model, ci, fid, sector_deltas, steps)
        self.assertFalse(np.array_equal(dv_on, dv_off),
                         "include_halo_IA 1 -> 0 left the data vector "
                         "unchanged (dead flag or C_ss/C_gs caches not "
                         "keyed on it)")
        self.assertTrue(np.array_equal(ia_on, ia_off),
                        "include_halo_IA 1 -> 0 changed the IA probes, "
                        "which do not read the flag")

        ci.init_include_halo_IA(1)
        dv, ia = evaluate(model, ci, fid, sector_deltas, steps)
        self.assertTrue(np.array_equal(dv, dv_on),
                        "include_halo_IA 1 -> 0 -> 1 did not return to the "
                        "data vector bit for bit "
                        f"(max rel {_max_rel(dv, dv_on):.2e})")
        self.assertTrue(np.array_equal(ia, ia_on),
                        "include_halo_IA 1 -> 0 -> 1 did not return to the "
                        "IA probes bit for bit")

        ci.init_include_halo_IA(0)
        dv, ia = evaluate(model, ci, fid, sector_deltas, steps)
        self.assertTrue(np.array_equal(dv, dv_off),
                        "include_halo_IA 0 -> 1 -> 0 did not return to the "
                        "NLA data vector bit for bit "
                        f"(max rel {_max_rel(dv, dv_off):.2e})")
        self.assertTrue(np.array_equal(ia, ia_off),
                        "include_halo_IA 0 -> 1 -> 0 did not return to the "
                        "IA probes bit for bit")
        print(f"  flag flip: {len(dv_on)} data points, max rel "
              f"halo-IA vs NLA {_max_rel(dv_on, dv_off):.2e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--fresh":
        fresh_worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main(verbosity=2)
