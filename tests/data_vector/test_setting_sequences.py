"""Setting sequences: settings changed in one process vs a fresh process.

A notebook changes cosmolike settings between evaluations without
restarting the kernel. Each sequence below changes one kind of setting
in one process and compares the result, bit for bit, with a process
that saw only the final settings, or expects the named error where the
change is refused. The shared sequences come from common_sequences and
the class from setting_sequence_suite (both in
cosmolike_core/cocoa_testing_suites.py, which documents every sequence
and the two worker legs); the project adds the sequences of its own
features below them.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_real/tests/data_vector/test_setting_sequences.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The shared sequences on this project's examples: example 1 computes
# cosmic shear (8 source bins), example 2 has 8 lens bins.
SEQUENCES = u.common_sequences(
    "real", shear_example="example1", lens_example="example2", nsource=8,
    nlens=8, tatt=True, rebuild_example="example2_2x2pt",
    baryons=("antilles-2", "baryons_logPkR.h5"))

# A new ggl exclusion list of the same length: the pair map behind every
# cached gamma_t spectrum changes while the pair count does not. The list changes
# the data-vector layout, so the fresh leg evaluates the base first (a
# masked vector after the change would stop at the layout check).
GGL_SWAP = {
    "name": "S3_ggl_exclusion_swap", "base": "example2",
    "steps": [[("ggl_exclude", {"pairs": [[0, 7], [1, 7]]}), ("recount", {})],
              [("ggl_exclude", {"pairs": [[2, 7], [3, 7]]}), ("recount", {})]],
    "evaluate": "w_gammat_tomo", "fresh": "base_first", "compare": [1, 2],
    "expect": "bitwise"}

# A new exclusion list with no other call: init_ggl_exclude recounts the
# ggl spectra itself once both samples are loaded (the fresh leg recounts
# explicitly).
GGL_NO_RECOUNT = {
    "name": "S3b_ggl_exclusion_without_recount", "base": "example2",
    "steps": [[("ggl_exclude", {"pairs": [[0, 7], [1, 7], [2, 7], [3, 7]]})]],
    "fresh_steps": [[("ggl_exclude", {"pairs": [[0, 7], [1, 7], [2, 7], [3, 7]]}),
                     ("recount", {})]],
    "evaluate": ["sizes", "w_gammat_tomo"], "fresh": "base_first",
    "expect": "bitwise"}
SEQUENCES.append(GGL_SWAP)
SEQUENCES.append(GGL_NO_RECOUNT)

TestSettingSequences = u.setting_sequence_suite(
    u, name="TestSettingSequences", sequences=SEQUENCES)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
