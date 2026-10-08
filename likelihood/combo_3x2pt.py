"""Cobaya likelihood of the Roman real-space 3x2pt data vector.

The 3x2pt combination holds cosmic shear xi_+(theta) and xi_-(theta),
galaxy-galaxy lensing gamma_t(theta) and galaxy clustering w(theta).

Input yamls select it as roman_real.combo_3x2pt; its default options live in
combo_3x2pt.yaml next to this file. _cosmolike_prototype_base does all the
work; this module only names the probe set. Cocoa links this likelihood/
folder into Cobaya as the package cobaya.likelihoods.roman_real, hence the
import path below.
"""
from cobaya.likelihoods.roman_real._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_real_interface as ci
import numpy as np

class combo_3x2pt(_cosmolike_prototype_base):
  """3x2pt likelihood: probe set "3x2pt" (xi_+, xi_-, gamma_t and w)."""
  def initialize(self):
    """Initialize the base class for the probe set "3x2pt".

    super(combo_3x2pt, self).initialize calls the method of the parent class,
    _cosmolike_prototype_base.initialize.
    """
    super(combo_3x2pt,self).initialize(probe="3x2pt")