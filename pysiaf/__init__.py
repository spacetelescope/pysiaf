"""pysiaf: Python classes and scripts for JWST SIAF generation, maintenance, and internal validation. Reading and working with HST SIAF is supported.

"""

from __future__ import absolute_import, print_function, division

import re
import requests

from .version import *

from .aperture import Aperture, HstAperture, JwstAperture
from .constants import JWST_PRD_VERSION, JWST_PRD_DATA_ROOT, JWST_PRD_DATA_ROOT_EXCEL, HST_PRD_VERSION, \
    HST_PRD_DATA_ROOT
from .iando import read, write
from .siaf import Siaf, ApertureCollection
# from .tests import test_aperture#, test_polynomial
from .utils import polynomial, rotations, tools, projection

__all__ = ['Aperture', 'HstAperture', 'JwstAperture', 'SIAF', 'JWST_PRD_VERSION', 'JWST_PRD_DATA_ROOT', 'HST_PRD_VERSION', 'HST_PRD_DATA_ROOT', '_JWST_STAGING_ROOT', 'siaf', 'iando', 'polynomial', 'rotations', 'tools', 'compare', 'JWST_PRD_DATA_ROOT_EXCEL', 'generate', 'projection']

# Check PRD version is up to date
try:
    req = requests.get('https://github.com/spacetelescope/pysiaf/tree/master/pysiaf/prd_data/JWST').text
    p = re.compile("/spacetelescope/pysiaf/tree/master/pysiaf/prd_data/JWST/(.*?)/SIAFXML")
    prd_list = p.findall(req)
    newest_prd = [x for x in sorted(prd_list, reverse=True)][0]

    if JWST_PRD_VERSION != newest_prd:
        print("**WARNING**: LOCAL JWST PRD VERSION {} IS BEHIND THE CURRENT ONLINE VERSION {}\nPlease "
              "consider updating pysiaf, e.g. pip install --upgrade pysiaf or conda update pysiaf".format(
              JWST_PRD_VERSION, newest_prd))
except requests.exceptions.ConnectionError:
    print("**WARNING**: NO INTERNET CONNECTION\nLOCAL JWST PRD VERSION {} CANNOT BE CHECKED AGAINST "
          "ONLINE VERSION".format(JWST_PRD_VERSION))
    pass
