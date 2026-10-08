"""Cobaya likelihood of Roman real-space cosmic shear, xi_+(theta) and xi_-(theta).

Input yamls select it as roman_real.cosmic_shear; its default options live in
cosmic_shear.yaml next to this file. _cosmolike_prototype_base does all the
work; this module only names the probe set. Cocoa links this likelihood/
folder into Cobaya as the package cobaya.likelihoods.roman_real, hence the
import path below.
"""
from cobaya.likelihoods.roman_real._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_real_interface as ci
import numpy as np

class cosmic_shear(_cosmolike_prototype_base):
  """Cosmic-shear likelihood: probe set "xi" (xi_+ and xi_-)."""
  def initialize(self):
    """Initialize the base class for the probe set "xi".

    super(cosmic_shear, self).initialize calls the method of the parent class,
    _cosmolike_prototype_base.initialize.
    """
    super(cosmic_shear,self).initialize(probe="xi")