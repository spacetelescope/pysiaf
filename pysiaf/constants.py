"""Module that defines central constants and variables used in pysiaf."""

from __future__ import absolute_import, print_function, division

import glob
import os
import re

__all__ = ['JWST_PRD_VERSION', 'JWST_PRD_DATA_ROOT', 'HST_PRD_VERSION', 'HST_PRD_DATA_ROOT']

_THIS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

_DATA_ROOT = os.path.join(_THIS_DIRECTORY, 'prd_data')

_JWST_TEMPORARY_ROOT = os.path.join(_THIS_DIRECTORY, 'temporary_data')

JWST_SOURCE_DATA_ROOT = os.path.join(_THIS_DIRECTORY, 'source_data')

JWST_TEMPORARY_DATA_ROOT = os.path.join(_THIS_DIRECTORY, 'temporary_data')

JWST_DELIVERY_DATA_ROOT = os.path.join(_THIS_DIRECTORY, 'pre_delivery_data')

# test data directory
TEST_DATA_ROOT = os.path.join(_THIS_DIRECTORY, 'tests', 'test_data')

# directory for reports
REPORTS_ROOT = os.path.join(_THIS_DIRECTORY, 'reports')

AVAILABLE_PRD_JWST_VERSIONS = [os.path.basename(dir_name) for dir_name in
                               glob.glob(os.path.join(_DATA_ROOT, 'JWST', '*'))]
AVAILABLE_PRD_JWST_VERSIONS.sort()

# Remove all PRD with old formatting (PRDOPSSOC-LETTER-###) since they cannot be newest
AVAILABLE_PRD_JWST_VERSIONS = [prd for i, prd in enumerate(AVAILABLE_PRD_JWST_VERSIONS) if
                               bool(re.match(r"^[A-Z]-\d+", AVAILABLE_PRD_JWST_VERSIONS[i].split("PRDOPSSOC-")[1]))
                               is False]

# current version of the JWST PRD containing the SIAF
# by default the used version is the latest one
# JWST_PRD_VERSION = 'PRDOPSSOC-H-014' # as of 2018-02-20 (intermediate releases did not update
# SIAF content)
JWST_PRD_VERSION = AVAILABLE_PRD_JWST_VERSIONS[-1]

JWST_PRD_DATA_ROOT = os.path.join(_DATA_ROOT, 'JWST', JWST_PRD_VERSION, 'SIAFXML', 'SIAFXML')
JWST_PRD_DATA_ROOT_EXCEL = os.path.join(_DATA_ROOT, 'JWST', JWST_PRD_VERSION, 'SIAFXML', 'Excel')

# see helpers.download_latest_hst_siaf()
# HST_PRD_VERSION = 'SCIOPSDB-v1.84'  # as of 2017-11-19
HST_PRD_VERSION = 'Latest'  # refactor JSA 2018-12-05
HST_PRD_DATA_ROOT = os.path.join(_DATA_ROOT, 'HST')


# numerical constants
V3_TO_YAN_OFFSET_DEG = 0.13  # offset between origin of V3 and YAN coordinate systems
