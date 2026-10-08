"""Loader stub for the compiled module cosmolike_roman_real_interface.

The real module is the shared library cosmolike_roman_real_interface.so,
built in this folder from interface.cpp by scripts/compile_roman_real.sh.
Python's import system tries extension modules (.so) before source files
(.py) in the same folder, so `import cosmolike_roman_real_interface` loads
the .so directly and never runs this file while the .so exists. The stub
follows the pattern setuptools writes for eggs and needs pkg_resources and
the imp module, which Python 3.12 removed.
"""
def __bootstrap__():
   """Load the .so next to this file under this module's name.

   Then delete the helper names __bootstrap__ and __loader__ from the module.
   """
   global __bootstrap__, __loader__, __file__
   import sys, pkg_resources, imp
   __file__ = pkg_resources.resource_filename(__name__,'cosmolike_roman_real_interface.so')
   __loader__ = None; del __bootstrap__, __loader__
   imp.load_dynamic(__name__,__file__)
__bootstrap__()
