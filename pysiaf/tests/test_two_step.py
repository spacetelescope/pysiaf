"""
Authors
-------

    Colin Cox

"""

import numpy as np
from pysiaf.utils import polynomial
import makeup_polynomial


def test_two_step(verbose=False):
    A = np.array([10.0, 2.0, 0.1, 0.01, -0.02, 0.03])
    B = np.array([4.0, 1.8, 0.2, 0.02, 0.03, -0.02])
    a = np.array([1.0, 0.5, 0.1])
    b = np.array([2.0, 0.2, 0.6])

    # make up random polynomials of order 5 with terms which decrease strongly with power.
    A = makeup_polynomial.makeup_polynomial()
    B = makeup_polynomial.makeup_polynomial()
    (A2, B2) = polynomial.two_step(A, B, a, b, 5)
    if verbose:
        print('\nA')
        polynomial.triangle(A,5)#     print('B')
        print('B')
        polynomial.triangle(B,5)
        print('\nLinear terms')
        print('a',a)
        print('b', b)
        print('\nA2')
        polynomial.triangle(A2,5)
        print('B2')
        polynomial.triangle(B2,5)

    # Now do a test calculation
    (x,y) = (10,5)
    xp = a[0] + a[1]*x + a[2]*y
    yp = b[0] + b[1]*x + b[2]*y
    u = polynomial.poly(A, xp, yp, 5)
    v = polynomial.poly(B, xp, yp, 5)
    up = polynomial.poly(A2, x, y,5)
    vp = polynomial.poly(B2, x, y,5)

    if verbose:
        print('x,y', x,y)
        print('xp,yp', xp,yp)
        print('Two step', u, v)
        print('One step', up, vp)
    assert abs(u-up) < 1.0e-12 and abs(v-vp) < 1.0e-12, 'Inaccurate transformation'
    return
