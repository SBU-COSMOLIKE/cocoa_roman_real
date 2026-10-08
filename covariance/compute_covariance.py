"""Compute and save the roman_real covariance through the compiled interface.

The covariance of the real-space 3x2pt data vector is saved as its Gaussian
(G), super-sample (SSC) and connected non-Gaussian (cNG) parts and their
sum. The shared runner cosmolike_notebook_utils.covariance.command_line
does the work with the compiled cosmolike module of this project; this file
supplies the project pieces: that module and roman_real_covariance.py (the
survey choices). It is the command-line route of the covariance notebook.

From an activated Cocoa installation:
    python projects/roman_real/covariance/compute_covariance.py \
        projects/roman_real/EXAMPLE_EVALUATE_COVARIANCE.yaml

See --help and covariance/README.md for space and accuracy options.
The numerical model and survey settings are shared with the notebook.
"""

import os
from pathlib import Path
import sys

# Set external numerical libraries to one worker before their first import.
# CosmoLike's own OpenMP team is controlled by OMP_NUM_THREADS.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# This runner evaluates one matrix in one process. Cobaya supplies the YAML
# reader; it does not launch MPI workers or a sampler for this calculation.
os.environ["COBAYA_NOMPI"] = "1"

# parents[1] of this file is the project folder, and parents[1] of the
# project is cocoa/Cocoa. cosmolike_core (the shared Python package) and the
# project's interface/ folder (the compiled module) go first on sys.path,
# Python's module search list; roman_real_covariance is found because Python
# also searches the folder of the script it runs.
project = Path(__file__).resolve().parents[1]
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))
sys.path.insert(0, str(project/"interface"))

import cosmolike_roman_real_interface as ci
import roman_real_covariance as survey
from cosmolike_notebook_utils.covariance.command_line import run_covariance


# The block runs only when the file is executed as a script, not when it is
# imported. joint=False selects the galaxy/shear covariance (True is the
# cluster 6x2pt+N adapter).
if __name__ == "__main__":
    run_covariance(
        interface=ci, survey=survey, default_space="real", joint=False,
    )
