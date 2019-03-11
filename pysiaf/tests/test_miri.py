#!/usr/bin/env python
"""Test MIRI transformations against the test dataset delivered
by instrument team.

Authors
-------

    Johannes Sahlmann


References
----------
    mirim_tools.py at https://github.com/STScI-MIRI/miricoord

"""
import os
import sys

import copy

from numpy.testing import assert_allclose

from ..constants import JWST_SOURCE_DATA_ROOT
from ..siaf import Siaf

instrument = 'MIRI'


def test_against_test_data(siaf=None):
    """MIRI test data comparison.

    Mean and RMS difference between the instrument team computations
    and the pysiaf computations are computed and compared against
    acceptable thresholds.

    """
    if siaf is None:
        siaf = Siaf(instrument)
    else:
        # safeguard against side-effects when running several tests on
        #  a provided siaf, e.g. setting tilt to non-zero value
        siaf = copy.deepcopy(siaf)

    # directory that holds SIAF XML file
    test_data_dir = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, 'delivery')

    sys.path.append(test_data_dir)
    import mirim_siaf_testdata

    x_test, y_test, v2_test, v3_test = mirim_siaf_testdata.siaf_testdata()

    aperture_name = 'MIRIM_FULL'
    aperture = siaf[aperture_name]

    v2_pysiaf, v3_pysiaf = aperture.det_to_tel(x_test, y_test)
    x_pysiaf, y_pysiaf = aperture.tel_to_det(v2_test, v3_test)

    print('')
    print(x_test[0:4], y_test[0:4])
    print(x_pysiaf[0:4], y_pysiaf[0:4])

    assert_allclose(x_test, x_pysiaf, atol=0.05)
    assert_allclose(y_test, y_pysiaf, atol=0.05)
    assert_allclose(v2_test, v2_pysiaf, atol=0.05)
    assert_allclose(v3_test, v3_pysiaf, atol=0.05)
