"""Sample the Roman real-space cosmic-shear posterior with Nautilus.

This file only binds the shared driver run() of
cosmolike_core/cocoa_hybrid_sampling.py to this project in its general mode
(hybrid=False: the yaml's own theory, here the emulator, is kept); the
driver's docstring explains every option. The model, its priors and its
fiducial point are in EXAMPLE_EMUL_NAUTILUS2.yaml. Nautilus samples each
parameter's own prior (Gaussian or flat); extra prior factors of the yaml
enter the likelihood once, and the evidence is the ordinary one.

From the Cocoa/ folder, with Cocoa activated (start_cocoa.sh), check the
setup without sampling:

    python ./projects/roman_real/EXAMPLE_EMUL_NAUTILUS2.py --check

and run under MPI (the project README lists the full commands):

    mpirun -n 5 --oversubscribe python ./projects/roman_real/EXAMPLE_EMUL_NAUTILUS2.py

Writes chains/EXAMPLE_EMUL_NAUTILUS2.1.txt (weight, -log likelihood, the
sampled parameters), .paramnames, the run record EXAMPLE_EMUL_NAUTILUS2.yaml
and the checkpoint; an existing result is never overwritten (--outroot
picks a new name).
"""

from pathlib import Path
import sys

# project = the folder of this file; core = the cosmolike_core checkout, which
# holds cocoa_hybrid_sampling.py and goes first on the import path
project = Path(__file__).resolve().parent
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))

from cocoa_hybrid_sampling import run


# __name__ is "__main__" only when this file runs as a script
if __name__ == "__main__":

    # The driver's default names are EXAMPLE_NAUTILUS2.yaml and chains/EXAMPLE_NAUTILUS2;
    # this example reads EXAMPLE_EMUL_NAUTILUS2.yaml and writes under that name,
    # unless --input or --outroot say otherwise.
    if "--input" not in sys.argv:
        sys.argv += ["--input", str(project/"EXAMPLE_EMUL_NAUTILUS2.yaml")]
    if "--outroot" not in sys.argv:
        sys.argv += ["--outroot", "EXAMPLE_EMUL_NAUTILUS2"]

    run(
        mode="nautilus",
        project=project,
        example=2,
        hybrid=False,
    )
