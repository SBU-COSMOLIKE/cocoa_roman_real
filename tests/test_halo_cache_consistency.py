"""Unit test: halo-model cache invalidation (the halo parameter ladder).

halo.c caches every expensive table behind its own keys - the spectra
p_mm, p_gm, p_gg, the gas pressure table u_KS, the HOD tables ngal/bgal,
and the shared
sigma^2(M), dlnnu/dlnM and bias_norm tables - and cosmo2D.c's HOD mode
(include_HOD_GX = 1) reads p_gm/p_gg into the Limber C_l^gg and
C_l^gs. A partial-invalidation bug - one sector's update failing to
rebuild a table another sector consumes - produces silently wrong
spectra only in MIXED update sequences, which the per-point suites
never exercise. This is test_cache_consistency.py's ladder, extended
to the halo sectors, with the HOD C_l live.

The test walks a deterministic ladder IN ONE PROCESS, evaluating the
model after every step (each sector's later steps keep the earlier
sectors at their last values, so the ladder ends at one well-defined
point):

    3 x cosmology-only steps          (omegam, H0, As_1e9)
    3 x HOD-only steps                (every bin's lg M_min and alpha)
    3 x galaxy-concentration steps    (every bin's gc)
    3 x gas-only steps                (the polytropic index Gamma)
    3 x IA-only steps                 (A1, A2, BTA)
    3 x source-photo-z steps          (every DZ_S shift)
    3 x shear-calibration steps       (every M)

The halo sectors are not sampled parameters: the ladder sets them
through the interface (set_nuisance_hod, set_nuisance_gas), which
redraw their cache tags only when a value changes, so every other
table stays cached across a step - exactly the partial-invalidation
case under test. After every step the test records two vectors:

  dv   = the masked HOD 3x2pt data vector (Limber gg and gs, NLA);
  halo = the halo probes at fixed (k, a, bin) points: p_mm, u_KS,
         p_gm, p_gg, ngal, bgal (the gas sector moves only u_KS, which
         no data vector reads).

Assertions:
  1. every ladder step changes the vector its sector feeds (dv, or
     halo for the gas sector) - a dead sector flag would pass the
     later checks vacuously;
  2. a no-op update (the final point again) leaves both vectors
     bitwise unchanged;
  3. after a SCRAMBLE (every sector moved at once, galaxy bias
     included), returning to the ladder's final point reproduces both
     vectors bit for bit;
  4. a second model instance walking the MIRRORED ladder lands on the
     same vectors bit for bit;
  5. a FRESH process (a subprocess: cosmolike's tables are
     per-process statics, so a second model instance in this process
     is not fresh) that evaluates the final point once, with every
     cache built from scratch, reproduces both vectors bit for bit.

Every evaluation forces a full recomputation (cobaya's cache is
bypassed), so each assertion tests cosmolike's own invalidation.

Slow (several dozen full HOD evaluations plus the fresh subprocess):
runs only with COCOA_HALO_SLOW=1, like the spectrum tier of
test_halo.py. To run (from the Cocoa/ folder, cocoa environment
active, start_cocoa.sh sourced):

    COCOA_HALO_SLOW=1 python -m pytest \\
      ./projects/roman_real/tests/test_halo_cache_consistency.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

RUN_SLOW = os.environ.get("COCOA_HALO_SLOW", "0") == "1"

EXAMPLE = "example2"

# ---- the pinned halo state (the fiducial of every sector) -------------------

# Coupon et al. 2012 HOD rows {lg M_min, sigma_lgM, lg M_1, lg M_0,
# alpha, f_c}; bins past the table repeat its last row
HOD_ROWS = (
    (13.17, 0.39, 14.53, 11.09, 1.27, 1.00),
    (13.18, 0.30, 14.47, 10.93, 1.36, 1.00),
    (12.96, 0.38, 14.10, 12.47, 1.28, 1.00),
    (12.80, 0.33, 13.94, 12.15, 1.52, 1.00),
    (12.62, 0.30, 13.79, 8.67, 1.50, 1.00),
)
GC_FIDUCIAL = 1.0
GAS_FIDUCIAL = (1.17, 0.6, 14.0, 0.0, 0.0, 1.0, 0.03, 12.5, 1.2, 6.5,
                0.752)

# per-step offsets of the halo sectors
HOD_DELTA = {0: 0.03, 4: 0.02}   # lg M_min, alpha
GC_DELTA = -0.05                 # galaxy concentration factor
GAS_DELTA = {0: 0.02}            # polytropic index Gamma (u_KS reads it)

# ---- the sampled sectors (as in test_cache_consistency.py) ------------------

SECTORS = (
    ("cosmo", re.compile(r"^(As_1e9|H0|ns|omegab|omegam|mnu|w|w0pwa)$")),
    ("ia", re.compile(r"_A1_|_A2_|_BTA_")),
    ("dz_source", re.compile(r"_DZ_S")),
    ("m", re.compile(r"_M[0-9]+$")),
    ("bias", re.compile(r"_B1_|_B2_|_BMAG_")),
    ("other", re.compile(r".")),
)
DELTAS = {
    "cosmo": {"omegam": 0.002, "H0": 0.2, "As_1e9": 0.02},
    "ia": {re.compile(r"_A1_1$"): 0.05, re.compile(r"_A1_2$"): 0.05,
           re.compile(r"_A2_1$"): 0.05, re.compile(r"_A2_2$"): 0.05,
           re.compile(r"_BTA_1$"): 0.05},
    "dz_source": {re.compile(r"_DZ_S"): 0.001},
    "m": {re.compile(r"_M[0-9]+$"): 0.005},
    "bias": {re.compile(r"_B1_"): 0.05},
}

# ladder order: sampled and halo sectors interleaved, so every halo
# step sits between steps of other sectors
PHASES = ("cosmo", "hod", "gc", "gas", "ia", "dz_source", "m")
HALO_SECTORS = ("hod", "gc", "gas")
NSTEP = 3
SCRAMBLE_STEP = 4

# fixed halo probe points: (k in (c/H0)^-1, a) and the lens bins
PROBE_K = (3.0, 30.0, 300.0, 3000.0)
PROBE_A = (0.5, 0.7, 0.9)
# u_KS(c, k, rv) probes: concentrations, and r_v in c/H0 (0.9 Mpc/h)
PROBE_UKS_C = (2.0, 5.0, 10.0)
PROBE_UKS_RV = 3.0e-4


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


def build_hod_model():
    """The frozen NLA 3x2pt model with the HOD Limber C_l switched on.

    Returns:
      (model, fiducial point, compiled interface, number of lens bins)
    """
    import cosmolike_roman_real_interface as ci

    info = u.load_frozen_info(EXAMPLE, tatt=False)
    name = u.EXAMPLES[EXAMPLE]["likelihood"]
    # HOD C_l are Limber-only
    info["likelihood"][name]["adopt_limber_gg"] = 1
    info["likelihood"][name]["adopt_limber_gs"] = 1
    model = u.make_model(info)
    fid = dict(u.build_point(model, EXAMPLE, tatt=False))
    nbin = int(model.likelihood[name].lens_ntomo)
    ci.init_include_HOD_GX(1)
    return model, fid, ci, nbin


def point_at(fid, sector_deltas, steps):
    """The sampled point with each sampled sector at its step count."""
    point = dict(fid)
    for sector, step in steps.items():
        for n, d in sector_deltas.get(sector, {}).items():
            point[n] = fid[n] + step * d
    return point


def apply_halo_state(ci, nbin, steps):
    """Set HOD, gc and gas at their sector step counts (interface
    setters: each redraws its cache tag only when a value changes)."""
    import numpy as np

    for ni in range(nbin):
        row = list(HOD_ROWS[min(ni, len(HOD_ROWS) - 1)])
        for col, d in HOD_DELTA.items():
            row[col] += steps["hod"] * d
        ci.set_nuisance_hod(ni=ni, hod=np.array(row, dtype=float),
                            gc=GC_FIDUCIAL + steps["gc"] * GC_DELTA)
    gas = list(GAS_FIDUCIAL)
    for col, d in GAS_DELTA.items():
        gas[col] += steps["gas"] * d
    ci.set_nuisance_gas(gas=np.array(gas, dtype=float))


def halo_probes(ci, nbin):
    """The halo tables at fixed points, as one flat vector."""
    import numpy as np

    out = []
    for a in PROBE_A:
        for k in PROBE_K:
            out.append(ci.p_mm(k=k, a=a))
    for c in PROBE_UKS_C:
        for k in PROBE_K:
            out.append(ci.u_KS(c=c, k=k, rv=PROBE_UKS_RV))
    for ni in range(nbin):
        for a in PROBE_A:
            out.append(ci.ngal(ni=ni, a=a))
            out.append(ci.bgal(ni=ni, a=a))
            for k in PROBE_K:
                out.append(ci.p_gm(k=k, a=a, ni=ni))
                out.append(ci.p_gg(k=k, a=a, ni=ni, nj=ni))
    return np.array(out)


def evaluate(model, ci, nbin, fid, sector_deltas, steps):
    """One full evaluation at the ladder point; returns (dv, halo)."""
    import numpy as np

    apply_halo_state(ci, nbin, steps)
    u.evaluate_chi2(model, point_at(fid, sector_deltas, steps))
    dv = np.array(ci.compute_data_vector_masked())
    return dv, halo_probes(ci, nbin)


def fresh_worker(steps_json, out_path):
    """Subprocess entry: evaluate the given ladder point ONCE, every
    cache built from scratch, and save (dv, halo)."""
    import numpy as np

    steps = json.loads(steps_json)
    model, fid, ci, nbin = build_hod_model()
    sector_deltas = {s: _deltas_for(s, [n for n in fid if _sector_of(n) == s])
                     for s, _ in SECTORS}
    dv, halo = evaluate(model, ci, nbin, fid, sector_deltas, steps)
    np.savez(out_path, dv=dv, halo=halo)


@unittest.skipUnless(RUN_SLOW, "slow halo cache ladder; set "
                     "COCOA_HALO_SLOW=1 to run")
class TestHaloCacheConsistency(unittest.TestCase):
    """Halo-sector ladder cache-invalidation check, HOD C_l live."""

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()

    @classmethod
    def tearDownClass(cls):
        # the HOD gate is a process-wide static: leave it off for the
        # next test module's models
        import cosmolike_roman_real_interface as ci
        ci.init_include_HOD_GX(0)

    def _walk(self, order):
        """Walk the ladder in the given order; returns the final state
        and the recorded vectors."""
        import numpy as np

        model, fid, ci, nbin = build_hod_model()
        sector_deltas = {
            s: _deltas_for(s, [n for n in fid if _sector_of(n) == s])
            for s, _ in SECTORS}
        for s in ("cosmo", "dz_source", "m"):
            self.assertTrue(sector_deltas[s],
                            f"no sampled parameters in sector {s}")

        # a sampled sector nothing samples drops out of the ladder
        active = tuple(s for s in PHASES
                       if s in HALO_SECTORS or sector_deltas.get(s))
        if order == "mirrored":
            active = tuple(reversed(active))

        steps = {s: 0 for s in list(sector_deltas) + list(HALO_SECTORS)}
        prev_dv, prev_halo = evaluate(model, ci, nbin, fid,
                                      sector_deltas, steps)

        for sector in active:
            for r in range(1, NSTEP + 1):
                steps[sector] = r
                dv, halo = evaluate(model, ci, nbin, fid,
                                    sector_deltas, steps)
                if sector == "gas":
                    changed = not np.array_equal(halo, prev_halo)
                    what = "halo probes (u_KS)"
                else:
                    changed = not np.array_equal(dv, prev_dv)
                    what = "data vector"
                self.assertTrue(
                    changed,
                    f"{order}: {sector} step {r} left the {what} "
                    "unchanged (dead sector flag or stale cache)")
                prev_dv, prev_halo = dv, halo

        final_steps = dict(steps)
        final_dv, final_halo = evaluate(model, ci, nbin, fid,
                                        sector_deltas, final_steps)

        # 2. no-op: the same point again, bitwise
        dv, halo = evaluate(model, ci, nbin, fid, sector_deltas, final_steps)
        self.assertTrue(np.array_equal(dv, final_dv),
                        f"{order}: a no-op re-evaluation changed the data "
                        "vector")
        self.assertTrue(np.array_equal(halo, final_halo),
                        f"{order}: a no-op re-evaluation changed the halo "
                        "probes")

        # 3. scramble (every sector at once, bias included), then return
        scramble = {s: SCRAMBLE_STEP for s in steps}
        evaluate(model, ci, nbin, fid, sector_deltas, scramble)
        back_dv, back_halo = evaluate(model, ci, nbin, fid,
                                      sector_deltas, final_steps)
        self.assertTrue(np.array_equal(back_dv, final_dv),
                        f"{order}: returning after the scramble did not "
                        "reproduce the data vector bit for bit")
        self.assertTrue(np.array_equal(back_halo, final_halo),
                        f"{order}: returning after the scramble did not "
                        "reproduce the halo probes bit for bit")
        print(f"  {order} ladder: {len(final_dv)} data points, "
              f"{len(final_halo)} halo probes", flush=True)
        return final_steps, final_dv, final_halo

    def test_halo_cache_consistency(self):
        import numpy as np

        steps, dv_fwd, halo_fwd = self._walk("forward")
        steps_mir, dv_mir, halo_mir = self._walk("mirrored")
        self.assertEqual(steps, steps_mir)

        # 4. the mirrored order lands on the same vectors
        self.assertTrue(np.array_equal(dv_fwd, dv_mir),
                        "the mirrored ladder landed on a different data "
                        "vector: the answer depends on the invalidation "
                        "history")
        self.assertTrue(np.array_equal(halo_fwd, halo_mir),
                        "the mirrored ladder landed on different halo "
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
            dv_fresh, halo_fresh = fresh["dv"], fresh["halo"]

        def report(a, b):
            nz = b != 0
            return float(np.max(np.abs(a[nz]/b[nz] - 1.0))) if nz.any() else 0.0

        self.assertTrue(
            np.array_equal(dv_fwd, dv_fresh),
            "the ladder's final data vector differs from a fresh "
            f"process at the same point (max rel {report(dv_fwd, dv_fresh):.2e})"
            ": a table was not rebuilt when its inputs changed")
        self.assertTrue(
            np.array_equal(halo_fwd, halo_fresh),
            "the ladder's final halo probes differ from a fresh process "
            f"at the same point (max rel {report(halo_fwd, halo_fresh):.2e})"
            ": a halo table was not rebuilt when its inputs changed")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--fresh":
        fresh_worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main(verbosity=2)
