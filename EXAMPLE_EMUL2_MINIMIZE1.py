"""Annealed minimization of the roman_real hybrid example 1.

The hybrid examples emulate the distances and the matter power spectra
(use_emulator: 2), and cosmolike computes the survey projections.
Example 1 is roman_real.cosmic_shear (cosmic shear) with NLA on example1.dataset;
its configuration is EXAMPLE_EMUL2_EVALUATE1.yaml (--input selects
another). It searches for the smallest -2 ln(posterior), priors included,
with emcee walkers at decreasing temperatures.

run() of cocoa_hybrid_sampling.py (in cosmolike_core) does the work; its
docstring explains every mode and option. From cocoa/Cocoa:

    python ./projects/roman_real/EXAMPLE_EMUL2_MINIMIZE1.py --check
    mpirun -n 2 --bind-to none python ./projects/roman_real/EXAMPLE_EMUL2_MINIMIZE1.py
        --nstw 200 --outroot hybrid_min1

The mpirun command is one line, wrapped here; --check only evaluates
the fiducial and prints the sampled-parameter order. The project README
covers MPI runs across nodes.
"""

from pathlib import Path
import sys

# project is this file's folder (projects/roman_real), and parents[1] of
# it is cocoa/Cocoa: cosmolike_core, which holds cocoa_hybrid_sampling.py,
# goes first on sys.path, Python's module search list.
project = Path(__file__).resolve().parent
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))

from cocoa_hybrid_sampling import run


# The run starts only when the file is executed as a script, not when
# it is imported.
if __name__ == "__main__":
    run(mode="minimize", project=project, example=1)
