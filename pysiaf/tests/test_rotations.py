import numpy as np
from math import sin, cos, acos, pi
from pysiaf.utils import rotations as rt
import pylab as pl

# Some values used in both tests
#  Set up some arbitrary values
ra = 20.0
dec = 70.0
roll = 15.0
v2 = 200.0
v3 = 300.0
ra2 = 21.0
dec2 = 69.0


def test_attitude(verbose=False):
    """ Tests the properties of the attitude matrix in calculating positions and roll angles. Incidentally tests underlying
    matrix generation and vector construction  and conversions to Euler angles

    Parameters
    ----------
    verbose     logical - set to true if detailed print-out needed

    Returns
    -------
    None    unless assert statement fails
    """


    a = rt.attitude(v2, v3, ra, dec, roll)
    if verbose: print('attitude\n', a)

    #  Show that attitude matrix correctly connects given inputs in both directions
    (ra_test1, dec_test1) = rt.pointing(a, v2, v3)
    if verbose: print('RA and Dec %10.6f %10.6f %10.3e %10.3e '%(ra_test1, dec_test1, ra_test1-ra, dec_test1 - dec))
    assert abs(ra_test1 - ra) < 1.0e-10 and abs(dec_test1 - dec) < 1.0e-10, 'Miscalculated RA or Dec'

    (v2_test1, v3_test1) = rt.getv2v3(a, ra, dec)
    if verbose: print('V2 V3 %10.6f %10.6f %10.3e %10.3e' %(v2_test1, v3_test1, v2_test1 - v2, v3_test1 - v3))
    assert abs(v2_test1 - v2) < 1.0e-10 and abs(v3_test1 - v3) < 1.0e-10, 'Miscalculated V2 or V3'

    #  Show that points away from defining position are correctly handled
    (v2_test2, v3_test2) = rt.getv2v3(a, ra2, dec2)
    print('Test 2 %10.6f %10.6f' %(v2_test2, v3_test2))
    (ra_test2, dec_test2) = rt.pointing(a, v2_test2, v3_test2)
    print('Test2 RA and Dec %10.6f %10.6f' %(ra_test2, dec_test2))
    print('Test2 %10.3e %10.3e' %(ra_test2 - ra2, dec_test2 - dec2))
    assert abs(ra_test2 - ra2) < 1.0e10 and abs(dec_test2 - dec2) < 1.0e-10, 'Miscalculated RA or Dec'

    #  Position angles at reference point
    pa1 = rt.posangle(a, v2, v3)
    pa2 = rt.sky_posangle(a, ra, dec)
    #  and at displaced point
    pa3 = rt.posangle(a, v2_test2, v3_test2)
    pa4 = rt.sky_posangle(a, ra_test2, dec_test2)
    if verbose:
        print('PA tests')
        print('%10.6f %10.6f %10.6f %10.3e' %(roll, pa1, pa2, pa1 - pa2))
        print('%10.6f %10.6f %10.3e' %(pa3, pa4, pa3-pa4))

def test_axial(verbose=False):

    """ Compares vector transformations using the attitude matrix with a single rotation about an axis.
    In the process validates the method of deriving the axis and rotation angle from the attitude matrix

    Parameters
    ----------
    verbose     logical - set to true if detailed print-out needed

    Returns
    -------
    None    unless assert statement fails
    -------

    """
    #  Set up an arbitrary vector
    np.random.seed(seed=100)
    values = np.random.rand(2)
    alpha = 2*pi*values[0]
    beta = acos(2*values[1] -1.0)
    u = np.array([cos(alpha)*cos(beta), sin(alpha)*cos(beta), sin(beta)])

    a = rt.attitude(v2, v3, ra, dec, roll)
    va = np.dot(a,u)    # Transform using attitude matrix

    (axis, phi, quaternion) = rt.rodrigues(a) # obtain single rotation parameters equivalent to attitude matrix
    vb = rt.axial(axis, phi, u)     #  Transform using axial rotation
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

    assert abs(dot_product -1.0) < 1.0e-12, 'Transforms do not agree'