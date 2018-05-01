"""pysiaf: Python classes and scripts for JWST SIAF generation, maintenance, and internal validation. Reading and working with HST SIAF is supported.


* `ApertureCollection`: content of a SIAF file (Science Instrument Aperture File,
  listing the defined apertures for a given instrument)
* `Aperture`: a single aperture inside a SIAF

"""

from .version import *


from .aperture import Aperture, HstAperture, JwstAperture
from .constants import JWST_PRD_VERSION, JWST_PRD_DATA_ROOT, JWST_PRD_DATA_ROOT_EXCEL, HST_PRD_VERSION, \
    HST_PRD_DATA_ROOT
from .iando import read, write
from .siaf import Siaf, ApertureCollection
from .tests import test_aperture#, test_polynomial
from .utils import polynomial, rotations, tools#, tools2

__all__ = ['Aperture', 'HstAperture', 'JwstAperture', 'SIAF', 'JWST_PRD_VERSION', 'JWST_PRD_DATA_ROOT', 'HST_PRD_VERSION', 'HST_PRD_DATA_ROOT', '_JWST_STAGING_ROOT', 'test_aperture', 'test_polynomial', 'siaf', 'iando', 'polynomial', 'rotations', 'tools', 'compare', 'JWST_PRD_DATA_ROOT_EXCEL', 'generate', 'tools2']

# import os.path
# with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as f:
#     __version__ = f.read().strip()
