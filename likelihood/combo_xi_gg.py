"""Cobaya likelihood of Roman real-space cosmic shear plus galaxy clustering.

The data vector holds cosmic shear xi_+(theta) and xi_-(theta) and galaxy
clustering w(theta), without galaxy-galaxy lensing.

Input yamls select it as roman_real.combo_xi_gg; its default options live in
combo_xi_gg.yaml next to this file. _cosmolike_prototype_base does all the
work; this module only names the probe set. Cocoa links this likelihood/
folder into Cobaya as the package cobaya.likelihoods.roman_real, hence the
import path below.
"""
from cobaya.likelihoods.roman_real._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_real_interface as ci
import numpy as np

class combo_xi_gg(_cosmolike_prototype_base):
  """Shear plus clustering likelihood: probe set "xi_gg" (xi_+, xi_- and w)."""
  def initialize(self):
    """Initialize the base class for the probe set "xi_gg".

    super(combo_xi_gg, self).initialize calls the method of the parent class,
    _cosmolike_prototype_base.initialize.
    """
    super(combo_xi_gg,self).initialize(probe="xi_gg")
