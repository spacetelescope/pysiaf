#!/usr/bin/env python
"""Tests for pysiaf HST functionality.

Authors
-------
    Johannes Sahlmann

"""

from ..aperture import HstAperture
from ..iando import read
from ..siaf import Siaf


def test_hst_aperture_init():
    """Test the initialization of an HstAperture object."""
    hst_aperture = HstAperture()
    hst_aperture.a_v2_ref = -100.
    assert hst_aperture.a_v2_ref == hst_aperture.V2Ref


def test_hst_siaf():
    """Test reading HST SIAF."""
    hst_siaf = Siaf('HST')
    assert len(hst_siaf.apertures) > 1


def test_hst_amudotrep():
    """Test reading amu.rep file containing FGS TVS matrices."""
    amudotrep = read.read_hst_fgs_amudotrep()
    fgs_keys = [key for key in amudotrep if 'fgs' in key]
    assert len(fgs_keys) == 3
