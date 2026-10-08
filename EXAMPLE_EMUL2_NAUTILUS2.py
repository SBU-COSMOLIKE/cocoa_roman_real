"""Nautilus nested sampling of the roman_real hybrid example 2.

The hybrid examples emulate the distances and the matter power spectra
(use_emulator: 2), and cosmolike computes the survey projections.
Example 2 is roman_real.combo_3x2pt (3x2pt) with NLA on example1.dataset;
its configuration is EXAMPLE_EMUL2_EVALUATE2.yaml (--input selects
another). It samples the posterior with Nautilus, a nested sampler that
also estimates the Bayesian evidence, and saves weighted samples and a
JSON convergence record.

run() of cocoa_hybrid_sampling.py (in cosmolike_core) does the work; its
docstring explains every mode and option. From cocoa/Cocoa:

    mpirun -n 2 --bind-to none python ./projects/roman_real/EXAMPLE_EMUL2_NAUTILUS2.py
        --nlive 1000 --neff 10000 --maxfeval 100000 --outroot hybrid_nautilus2

The mpirun command is one line, wrapped here. The project README covers
MPI runs across nodes.
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
    run(mode="nautilus", project=project, example=2)
