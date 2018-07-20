"""pysiaf: Python classes and scripts for JWST SIAF generation, maintenance, and internal validation. Reading and working with HST SIAF is supported.

"""

from __future__ import absolute_import, print_function, division

from .version import *

from .aperture import Aperture, HstAperture, JwstAperture
from .constants import JWST_PRD_VERSION, JWST_PRD_DATA_ROOT, JWST_PRD_DATA_ROOT_EXCEL, HST_PRD_VERSION, \
    HST_PRD_DATA_ROOT
from .iando import read, write
from .siaf import Siaf, ApertureCollection
# from .tests import test_aperture#, test_polynomial
from .utils import polynomial, rotations, tools, projection

__all__ = ['Aperture', 'HstAperture', 'JwstAperture', 'SIAF', 'JWST_PRD_VERSION', 'JWST_PRD_DATA_ROOT', 'HST_PRD_VERSION', 'HST_PRD_DATA_ROOT', '_JWST_STAGING_ROOT', 'siaf', 'iando', 'polynomial', 'rotations', 'tools', 'compare', 'JWST_PRD_DATA_ROOT_EXCEL', 'generate', 'projection']