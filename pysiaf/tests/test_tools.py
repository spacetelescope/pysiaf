#!/usr/bin/env python
"""Tests for the pysiaf utils.tools functions.

Authors
-------

    Johannes Sahlmann

"""

import numpy as np
# import pytest
#
import astropy.units as u

from ..utils.tools import jwst_fgs2_fgs1_matrix
# from ..utils import projection

from ..siaf import Siaf, get_jwst_apertures
from ..utils.tools import get_grid_coordinates


def test_jwst_fgs2_fgs1_matrix():

    rotation_1to2 = jwst_fgs2_fgs1_matrix()

    siaf = Siaf('fgs')
    fgs1 = siaf['FGS1_FULL']
    fgs2 = siaf['FGS2_FULL']

    # FGS1 ideal coordinates
    fgs1_x_idl, fgs1_y_idl = get_grid_coordinates(2, (0, 0), 100)

    # transform to FGS ideal

    fgs2_x_idl, fgs2_y_idl = fgs2.tel_to_idl(*fgs1.idl_to_tel(fgs1_x_idl, fgs1_y_idl))

    print('')
    print(fgs1_x_idl, fgs1_y_idl)
    print(fgs2_x_idl, fgs2_y_idl)

    fgs1_x_idl_rad = fgs1_x_idl * u.arcsec.to(u.rad)
    fgs1_y_idl_rad = fgs1_y_idl * u.arcsec.to(u.rad)
    # fgs1_unit_vector_idl = np.array([fgs1_x_idl_rad, fgs1_y_idl_rad, np.sqrt(1 - (fgs1_x_idl_rad**2+fgs1_y_idl_rad**2))])
    fgs1_unit_vector_idl = np.array([np.sqrt(1 - (fgs1_x_idl_rad**2+fgs1_y_idl_rad**2)), fgs1_x_idl_rad, fgs1_y_idl_rad])
    # fgs2_unit_vector_idl = np.dot(rotation_2to1.T, fgs1_unit_vector_idl)
    fgs2_unit_vector_idl = np.dot(rotation_1to2, fgs1_unit_vector_idl)
    # print(fgs2_unit_vector_idl * u.rad.to(u.arcsec))
    fgs2_x_idl, fgs2_y_idl = fgs2_unit_vector_idl[1:] * u.rad.to(u.arcsec)
    print(fgs2_x_idl, fgs2_y_idl)