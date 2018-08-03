#!/usr/bin/env python
"""Tests for the pysiaf projection functions.

Authors
-------

    Johannes Sahlmann

"""

import numpy as np
import pytest

import astropy.units as u

from ..utils.tools import get_grid_coordinates
from ..utils import projection


@pytest.fixture(scope='module')
def grid_coordinates(centre_deg=(0., 0.)):
    """Return tuple of coordinates in deg mapping out a regular grid."""
    n_side = 50
    span = 20 * u.arcmin
    x_width = span.to(u.deg).value

    return get_grid_coordinates(n_side, centre_deg, x_width)


def test_tangent_plane_projection_roundtrip():
    """Transform from RA/Dec to tangent plane and back. Check that input is recovered."""
    centre_deg = (80., -70.)   # setting this to 0,0 fails because of 0,360 deg wrapping
    ra_deg, dec_deg = grid_coordinates(centre_deg=centre_deg)

    # project to detector pixel coordinates
    x, y = projection.project_to_tangent_plane(ra_deg, dec_deg, centre_deg[0], centre_deg[1])

    # deproject
    ra, dec = projection.deproject_from_tangent_plane(x, y, centre_deg[0], centre_deg[1])

    difference_modulus = np.sqrt((ra_deg - ra) ** 2 + (dec_deg - dec) ** 2)

    assert np.std(difference_modulus) < 1.e-13 #, 'Problem with tangent plane projection.')


def test_project_to_tangent_plane():
    """Compare projection code built with astropy functions with independent implementation."""
    centre_deg = (80., -70.)  # setting this to 0,0 fails because of 0,360 deg wrapping
    ra_deg, dec_deg = grid_coordinates(centre_deg=centre_deg)

    x_1, y_1 = projection.project_to_tangent_plane(ra_deg, dec_deg, centre_deg[0], centre_deg[1])
    x_2, y_2 = tangent_plane_projection(ra_deg, dec_deg, centre_deg[0], centre_deg[1])

    difference_modulus = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

    assert np.max(difference_modulus) < 1e-13


def test_deproject_from_tangent_plane():
    """Compare projection code built with astropy functions with independent implementation."""
    centre_deg = (80., -70.)  # setting this to 0,0 fails because of 0,360 deg wrapping
    ra_deg, dec_deg = grid_coordinates(centre_deg=centre_deg)

    x, y = projection.project_to_tangent_plane(ra_deg, dec_deg, centre_deg[0], centre_deg[1])

    ra_1, dec_1 = projection.deproject_from_tangent_plane(x, y, centre_deg[0], centre_deg[1])
    ra_2, dec_2 = tangent_plane_deprojection(x, y, centre_deg[0], centre_deg[1])

    difference_modulus = np.sqrt((ra_1 - ra_2) ** 2 + (dec_1 - dec_2) ** 2)

    assert np.max(difference_modulus) < 1e-13


def tangent_plane_projection(alpha, delta, alpha_ref, delta_ref):
    """Project alpha, delta to tangent plane with reference point at alpha_ref, delta_ref.

    Alternative, direct implementation based on Colin Cox's tanproj function in tancalc.py

    Parameters
    ----------
    alpha : float
        angle (e.g. RA) in decimal deg
    delta : float
        angle (e.g. Dec) in decimal deg
    alpha_ref: float
        angle
    delta_ref: float
        angle

    Returns
    -------
    (e, n) : tuple
        Tangent plane coordinates

    """
    a0 = np.deg2rad(alpha_ref)
    d0 = np.deg2rad(delta_ref)
    a = np.deg2rad(alpha)
    d = np.deg2rad(delta)

    cosrho = np.cos(a-a0)*np.cos(d)*np.cos(d0) + np.sin(d)*np.sin(d0)

    if np.any(cosrho < 0.01): # angle > 89 deg
        raise RuntimeError('Too far from tangent point')

    e_rad = np.cos(d)*np.sin(a-a0)/cosrho
    n_rad = (np.sin(d)/cosrho - np.sin(d0))/np.cos(d0)

    e = np.rad2deg(e_rad)
    n = np.rad2deg(n_rad)

    return (e, n)


def tangent_plane_deprojection(e, n, alpha_ref, delta_ref):
    """Deproject e, n from tangent plane with reference point at alpha_ref, delta_ref.

    Alternative, direct implementation based on Colin Cox's invproj function in tancalc.py

    Parameters
    ----------
    e : float
        tangent plane coordinate
    n : float
        tangent plane coordinate
    alpha_ref
    delta_ref

    Returns
    -------
    (alpha, delta) : tuple
        RA and Dec

    """
    d0 = np.deg2rad(delta_ref)

    s = np.hypot(e, n)
    rho = np.arctan(np.deg2rad(s))
    B = np.arctan2(e, n)

    dalpha = np.arctan2(np.sin(B)*np.sin(rho), np.cos(rho)*np.cos(d0)
                        - np.sin(rho)*np.sin(d0)*np.cos(B))
    alpha = alpha_ref + np.rad2deg(dalpha)

    if np.any(alpha < 0.0):
        index = np.where(alpha < 0.0)
        alpha[index] += 360.0    # set to range 0 to 360 deg

    delta_rad = np.arcsin(np.sin(d0)*np.cos(rho) + np.cos(d0)*np.sin(rho)*np.cos(B))
    delta = np.rad2deg(delta_rad)

    return (alpha, delta)
