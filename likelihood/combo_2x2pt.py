"""Cobaya likelihood of the Roman real-space 2x2pt data vector.

The 2x2pt combination holds galaxy-galaxy lensing gamma_t(theta) and galaxy
clustering w(theta), without cosmic shear.

Input yamls select it as roman_real.combo_2x2pt; its default options live in
combo_2x2pt.yaml next to this file. _cosmolike_prototype_base does all the
work; this module only names the probe set. Cocoa links this likelihood/
folder into Cobaya as the package cobaya.likelihoods.roman_real, hence the
import path below.
"""
from cobaya.likelihoods.roman_real._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_real_interface as ci
import numpy as np

class combo_2x2pt(_cosmolike_prototype_base):
  """2x2pt likelihood: probe set "2x2pt" (gamma_t and w)."""
  def initialize(self):
    """Initialize the base class for the probe set "2x2pt".

    super(combo_2x2pt, self).initialize calls the method of the parent class,
    _cosmolike_prototype_base.initialize.
    """
    super(combo_2x2pt,self).initialize(probe="2x2pt")