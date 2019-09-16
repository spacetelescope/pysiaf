#!/usr/bin/env python
"""Tests for the pysiaf utils.tools functions.

Authors
-------

    Johannes Sahlmann

"""

import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose

from ..siaf import Siaf
from ..utils.tools import get_grid_coordinates
from ..utils.tools import jwst_fgs_to_fgs_matrix
from ..utils import rotations


def test_jwst_fgs_to_fgs_matrix(verbose=False):
    """Test 3D matrix transformation against planar approximation."""
    rotation_1to2 = jwst_fgs_to_fgs_matrix(direction='fgs1_to_fgs2')

    siaf = Siaf('fgs')
    fgs1 = siaf['FGS1_FULL_OSS']
    fgs2 = siaf['FGS2_FULL_OSS']

    # FGS1 ideal coordinates
    fgs1_x_idl, fgs1_y_idl = get_grid_coordinates(3, (0, 0), 100)

    # transform to FGS ideal using planar approximation
    fgs2_x_idl_planar, fgs2_y_idl_planar = fgs2.tel_to_idl(*fgs1.idl_to_tel(fgs1_x_idl, fgs1_y_idl))

    if verbose:
        print('')
        for aperture in [fgs1, fgs2]:
            for attribute in 'V2Ref V3Ref V3IdlYAngle'.split():
                print('{} {} {}'.format(aperture.AperName, attribute, getattr(aperture, attribute)))

        print('FGS1 idl')
        print(fgs1_x_idl, fgs1_y_idl)
        print('tel')
        print(fgs1.idl_to_tel(fgs1_x_idl, fgs1_y_idl))
        print('FGS2 idl')
        print(fgs2_x_idl_planar, fgs2_y_idl_planar)

    # transform to FGS ideal using 3D rotation matrix
    fgs1_unit_vector_idl = rotations.unit_vector_sky(fgs1_x_idl * u.arcsec, fgs1_y_idl * u.arcsec)
    fgs2_unit_vector_idl = np.dot(rotation_1to2, fgs1_unit_vector_idl)
    fgs2_x_idl, fgs2_y_idl = rotations.polar_angles(fgs2_unit_vector_idl)[0].to(u.arcsec).value, \
                             rotations.polar_angles(fgs2_unit_vector_idl)[1].to(u.arcsec).value
    if verbose:
        print(fgs2_x_idl, fgs2_y_idl)
        np.set_printoptions(precision=15, suppress=True)
        print(rotation_1to2)

    # require agreement within 1.5 mas
    absolute_tolerance = 1.5e-3
    assert_allclose(fgs2_x_idl, fgs2_x_idl_planar, atol=absolute_tolerance)
    assert_allclose(fgs2_y_idl, fgs2_y_idl_planar, atol=absolute_tolerance)
