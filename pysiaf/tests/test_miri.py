#!/usr/bin/env python
"""Test MIRI transformations against the test dataset.

Authors
-------

    Johannes Sahlmann

References
----------
    mirim_tools.py at https://github.com/STScI-MIRI/miricoord

"""

import copy
import os
import sys

from astropy.table import Table
from numpy.testing import assert_allclose
import pytest

from ..constants import JWST_SOURCE_DATA_ROOT, JWST_DELIVERY_DATA_ROOT
from ..siaf import Siaf

instrument = 'MIRI'

# directory that holds SIAF XML file
test_data_dir = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, 'delivery')

sys.path.append(test_data_dir)
import mirim_siaf_testdata

def test_against_test_data(siaf=None, verbose=False):
    """MIRI test data comparison.

    Mean and RMS difference between the instrument team computations
    and the pysiaf computations are computed and compared against
    acceptable thresholds.

    """
    if siaf is None:
        # Try to use pre-delivery-data since this should best match the source-data. If no data there, use PRD data
        try:
            pre_delivery_dir = os.path.join(JWST_DELIVERY_DATA_ROOT, instrument)
            siaf = Siaf(instrument, basepath=pre_delivery_dir)
        except OSError:
            siaf = Siaf(instrument)

    else:
        # safeguard against side-effects when running several tests on
        #  a provided siaf, e.g. setting tilt to non-zero value
        siaf = copy.deepcopy(siaf)


    x_test, y_test, v2_test, v3_test = mirim_siaf_testdata.siaf_testdata()

    aperture_name = 'MIRIM_FULL'
    aperture = siaf[aperture_name]

    # v2_pysiaf, v3_pysiaf = aperture.det_to_tel(x_test, y_test)
    # x_pysiaf, y_pysiaf = aperture.tel_to_det(v2_test, v3_test)
    v2_pysiaf, v3_pysiaf = aperture.sci_to_tel(x_test, y_test)
    x_pysiaf, y_pysiaf = aperture.tel_to_sci(v2_test, v3_test)

    t = Table([x_test-x_pysiaf, y_test-y_pysiaf, v2_test-v2_pysiaf, v3_test-v3_pysiaf],
              names=('delta_x', 'delta_y', 'delta_v2', 'delta_v3'))

    if verbose:
        print('')
        t.pprint(max_width=-1)

    absolute_tolerance = 0.04

    assert_allclose(x_test, x_pysiaf, atol=absolute_tolerance)
    assert_allclose(y_test, y_pysiaf, atol=absolute_tolerance)
    assert_allclose(v2_test, v2_pysiaf, atol=absolute_tolerance)
    assert_allclose(v3_test, v3_pysiaf, atol=absolute_tolerance)
