#!/usr/bin/env python
"""Tests for the rotations module.

Authors
-------
    Colin Cox
    Johannes Sahlmann

"""
from math import sin, cos, acos, pi
import numpy as np

import astropy.units as u

from pysiaf.utils import rotations, tools

# Some values used in both tests
#  Set up some arbitrary values
ra = 20.0
dec = 70.0
roll = 15.0
v2 = 200.0
v3 = 300.0
ra2 = 21.0
dec2 = 69.0
ra_array = np.array([20.0, 100.0, 200.0, 280.0])
dec_array = np.array([70.0, -20.0, -70.0, 50.0])
v2_array = np.array([200.0, -500.0, 300.0, -400.0])
v3_array = np.array([300.0, -600.0, -400.0, 500.0])

ra_deg = 20.0
dec_deg = 70.0
pa_deg = 15.0
v2_arcsec = 200.0
v3_arcsec = 300.0



def test_attitude(verbose=False):
    """Test the properties of the attitude matrix in calculating positions and roll angles.

    Incidentally tests underlying matrix generation and vector construction and conversions to
    Euler angles.

    Parameters
    ----------
    verbose : bool
        set to true only if detailed print-out needed

    """
    a = rotations.attitude(v2, v3, ra, dec, roll)
    if verbose:
        print('attitude\n', a)

    #  Show that attitude matrix correctly connects given inputs in both directions
    (ra_test1, dec_test1) = rotations.pointing(a, v2, v3)
    if verbose:
        print('RA and Dec %10.6f %10.6f %10.3e %10.3e ' % (ra_test1, dec_test1, ra_test1-ra, dec_test1 - dec))
    assert abs(ra_test1 - ra) < 1.0e-10 and abs(dec_test1 - dec) < 1.0e-10, 'Miscalculated RA or Dec'

    (v2_test1, v3_test1) = rotations.getv2v3(a, ra, dec)
    if verbose:
        print('V2 V3 %10.6f %10.6f %10.3e %10.3e' % (v2_test1, v3_test1, v2_test1 - v2, v3_test1 - v3))
    assert abs(v2_test1 - v2) < 1.0e-10 and abs(v3_test1 - v3) < 1.0e-10, 'Miscalculated V2 or V3'

    #  Show that points away from defining position are correctly handled
    (v2_test2, v3_test2) = rotations.getv2v3(a, ra2, dec2)
    (ra_test2, dec_test2) = rotations.pointing(a, v2_test2, v3_test2)
    if verbose:
        print('Test 2 %10.6f %10.6f' % (v2_test2, v3_test2))
        print('Test2 RA and Dec %10.6f %10.6f' % (ra_test2, dec_test2))
        print('Test2 %10.3e %10.3e' % (ra_test2 - ra2, dec_test2 - dec2))
    assert abs(ra_test2 - ra2) < 1.0e10 and abs(dec_test2 - dec2) < 1.0e-10, 'Miscalculated RA or Dec'

    #  Position angles at reference point
    pa1 = rotations.posangle(a, v2, v3)
    pa2 = rotations.sky_posangle(a, ra, dec)
    #  and at displaced point
    pa3 = rotations.posangle(a, v2_test2, v3_test2)
    pa4 = rotations.sky_posangle(a, ra_test2, dec_test2)
    if verbose:
        print('PA tests')
        print('%10.6f %10.6f %10.6f %10.3e' % (roll, pa1, pa2, pa1 - pa2))
        print('%10.6f %10.6f %10.3e' % (pa3, pa4, pa3-pa4))
    assert abs(pa1 - pa2) < 1.0e-10, 'Disagreement for reference point position angles'
    assert abs(pa3 - pa4) < 1.0e-10, 'Disagreement for displaced point position angles'

    # Test some functions with arrays as input
    unit_vectors = rotations.unit(ra_array, dec_array)
    rd = rotations.radec(unit_vectors, positive_ra=True)
    rd_test = rotations.pointing(a, v2_array, v3_array)
    if verbose:
        print(unit_vectors)
        print('RD\n', rd)
        print(rd_test[0])
        print(rd_test[1])

    v_test = rotations.getv2v3(a, ra_array, dec_array)
    if verbose:
        for i in range(len(ra_array)):
            print(v_test[0][i], v_test[1][i])

    # Check leading values agree
    assert abs(v_test[0][0] - v2_array[0] < 1.0e-10), 'V2 values do not match'
    assert abs(v_test[1][0] - v3_array[0] < 1.0e-10), 'V3 values do not match'


def test_attitude_matrix():
    """Compare original and new attitude matrix generator functions."""

    ra = ra_deg * u.deg
    dec = dec_deg * u.deg
    pa = pa_deg * u.deg
    v2 = v2_arcsec * u.arcsec
    v3 = v3_arcsec * u.arcsec

    attitude = rotations.attitude(v2_arcsec, v3_arcsec, ra_deg, dec_deg, pa_deg)
    attitude_matrix = rotations.attitude_matrix(v2, v3, ra, dec, pa)

    assert np.all(attitude==attitude_matrix)


def test_sky_to_tel():
    """Test application of the attitude matrix"""

    # test with quantities
    ra = ra_deg * u.deg
    dec = dec_deg * u.deg
    pa = pa_deg * u.deg
    v2 = v2_arcsec * u.arcsec
    v3 = v3_arcsec * u.arcsec

    attitude = rotations.attitude_matrix(v2, v3, ra, dec, pa)
    ra_2, dec_2 = rotations.tel_to_sky(attitude, *rotations.sky_to_tel(attitude, ra, dec))
    assert np.abs((ra - ra_2).to(u.milliarcsecond).value) < 1e-6
    assert np.abs((dec - dec_2).to(u.milliarcsecond).value) < 1e-6

    # test without quantities
    attitude = rotations.attitude_matrix(v2_arcsec, v3_arcsec, ra_deg, dec_deg, pa_deg)
    ra_2, dec_2 = rotations.tel_to_sky(attitude, *rotations.sky_to_tel(attitude, ra_deg, dec_deg))
    assert np.abs(ra_deg - ra_2.to(u.deg).value)*u.deg.to(u.milliarcsecond) < 1e-6
    assert np.abs(dec_deg - dec_2.to(u.deg).value)*u.deg.to(u.milliarcsecond) < 1e-6

    # test array inputs
    n_side = 3
    span = 2 * u.arcmin
    x_width = span.to(u.deg).value
    centre_deg = (ra_deg, dec_deg)
    ra_array_deg, dec_array_deg = tools.get_grid_coordinates(n_side, centre_deg, x_width)

    ra_array_2, dec_array_2 = rotations.tel_to_sky(attitude, *rotations.sky_to_tel(attitude, ra_array_deg*u.deg, dec_array_deg*u.deg))
    assert np.all(np.abs(ra_array_deg*u.deg - ra_array_2) < 1e-6 * u.milliarcsecond)
    assert np.all(np.abs(dec_array_deg*u.deg - dec_array_2) < 1e-6 * u.milliarcsecond)

    ra_array_2, dec_array_2 = rotations.tel_to_sky(attitude, *rotations.sky_to_tel(attitude, ra_array_deg, dec_array_deg))
    assert np.all(np.abs(ra_array_deg*u.deg - ra_array_2) < 1e-6 * u.milliarcsecond)
    assert np.all(np.abs(dec_array_deg*u.deg - dec_array_2) < 1e-6 * u.milliarcsecond)



def test_axial_rotation(verbose=False):
    """Compare vector transformation using the attitude matrix with a single rotation about an axis.

    In the process validate the method of deriving the axis and rotation angle from the attitude
    matrix.

    Parameters
    ----------
    verbose : bool
        set to true if detailed print-out needed

    """
    #  Set up an arbitrary vector
    np.random.seed(seed=100)
    values = np.random.rand(2)
    alpha = 2*pi*values[0]
    beta = acos(2*values[1] - 1.0)
    vector = np.array([cos(alpha)*cos(beta), sin(alpha)*cos(beta), sin(beta)])

    a = rotations.attitude(v2, v3, ra, dec, roll)
    va = np.dot(a, vector)    # Transform using attitude matrix

    (axis, phi, quaternion) = rotations.rodrigues(a)  # obtain single rotation parameters equivalent to attitude matrix
    vb = rotations.axial_rotation(axis, phi, vector)  # Transform using axial rotation
    dot_product = np.dot(va, vb)

    if verbose:
        print('Axis rotation test')
        print(axis)
        print(phi)

    if verbose:
        print('VA  ', va)
        print('VB  ', vb)
        print('DIFF', va-vb)
        print('Dot product', dot_product)

    assert abs(dot_product - 1.0) < 1.0e-12, 'Transforms do not agree'


def test_unit_vector_from_cartesian():
    """Test unit vector construction."""

    # scalar inputs
    x = 0.1
    y = 0.2
    unit_vector = rotations.unit_vector_from_cartesian(x=x, y=y)
    assert (np.linalg.norm(unit_vector) - 1) < 1e-14

    # scalar inputs with unit
    x = 200 * u.arcsec
    y = -300 * u.arcsec
    unit_vector = rotations.unit_vector_from_cartesian(x=x, y=y)
    assert (np.linalg.norm(unit_vector) - 1) < 1e-14

    # array inputs with unit
    x = np.linspace(-100, 100, 10) * u.arcsec
    y = np.linspace(-500, -100, 10) * u.arcsec
    unit_vector = rotations.unit_vector_from_cartesian(x=x, y=y)
    assert np.all(np.abs(np.linalg.norm(unit_vector, axis=0) - 1)) < 1e-14
