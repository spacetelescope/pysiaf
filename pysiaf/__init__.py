"""pysiaf: Python classes and scripts for JWST SIAF generation, maintenance, and internal validation. Reading and working with HST SIAF is supported.

"""

from __future__ import absolute_import, print_function, division
import logging
import re
import requests

# Configure logging
logger = logging.getLogger(__name__)

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = 'unknown'

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
    prd_list.sort()
    newest_prd = [prd for i, prd in enumerate(prd_list) if
                  bool(re.match(r"^[A-Z]-\d+", prd_list[i].split("PRDOPSSOC-")[1]))
                  is False][-1]  # choose largest number from PRDs matching format: PRODOSSOC-###

    if JWST_PRD_VERSION != newest_prd:
        logger.warning("**WARNING**: LOCAL JWST PRD VERSION %s DOESN'T MATCH THE CURRENT ONLINE VERSION %s"
                       "\nPlease consider updating pysiaf, e.g. pip install --upgrade pysiaf or conda update pysiaf",
                       JWST_PRD_VERSION, newest_prd)
except:
    logger.warning("**WARNING**: LOCAL JWST PRD VERSION %s CANNOT BE CHECKED AGAINST ONLINE VERSION", JWST_PRD_VERSION)
    pass
