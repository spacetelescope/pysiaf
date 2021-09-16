#!/usr/bin/env python
"""Tests for the pysiaf iando/ write.py and read.py functions.

Authors
-------

    Shannon Osborne

"""
import os

import pytest

from ..aperture import JwstAperture
from ..siaf import ApertureCollection
from ..iando.write import write_jwst_siaf

ON_GITHUB_ACTIONS = '/home/runner' in os.path.expanduser('~') or '/Users/runner' in os.path.expanduser('~')


@pytest.mark.skipif(ON_GITHUB_ACTIONS, reason="Don't want to write and remove dirs on GHA Server")
def test_write_jwst_siaf_xml(tmpdir):
    """Basic test to check that JWST SIAF XML file is written out"""

    aperture = JwstAperture()
    aperture.AperName = 'MIRIM_FULL_OSS'
    aperture.InstrName = 'MIRI'
    aperture.VIdlParity = 1
    aperture.DetSciYAngle = 0
    aperture.DetSciParity = 1
    aperture_dict = {
        'MIRIM_FULL_OSS': aperture
    }
    aperture_collection = ApertureCollection(aperture_dict)
    filename = os.path.join(tmpdir, 'test_miri.xml')

    write_jwst_siaf(aperture_collection, filename=filename,
                    file_format='xml', verbose=False)

    assert os.path.isfile(filename)

    # Remove temporary directory
    tmpdir.remove()


@pytest.mark.skipif(ON_GITHUB_ACTIONS, reason="Don't want to write and remove dirs on GHA Server")
def test_write_jwst_siaf_xlsx(tmpdir):
    """Basic test to check that JWST SIAF XLSX file is written out"""

    aperture = JwstAperture()
    aperture.AperName = 'MIRIM_FULL_OSS'
    aperture.InstrName = 'MIRI'
    aperture.VIdlParity = 1
    aperture.DetSciYAngle = 0
    aperture.DetSciParity = 1
    aperture_dict = {
        'MIRIM_FULL_OSS': aperture
    }
    aperture_collection = ApertureCollection(aperture_dict)
    filename = os.path.join(tmpdir, 'test_miri.xlsx')

    write_jwst_siaf(aperture_collection, filename=filename,
                    file_format='xlsx', verbose=False)

    assert os.path.isfile(filename)

    # Remove temporary directory
    tmpdir.remove()
