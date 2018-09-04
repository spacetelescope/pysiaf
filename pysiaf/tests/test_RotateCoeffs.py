#!/usr/bin/env python


"""
Authors
-------

    Colin Cox

"""

import numpy as np
from pysiaf.utils import polynomial
import makeup_polynomial


def test_RotateCoeffs(verbose=False):
    """Test accuracy of inversion method"""

    # First invent a random but plausible polynomial array
    a = makeup_polynomial.makeup_polynomial()
    order = 5
    if verbose:
        print('A')
        polynomial.triangle(a, order)

    # Random point within 2048 square with origin at the center
    (x, y) = 2048.0*np.random.rand(2) - 1024.0
    u = polynomial.poly(a, x, y, order)

    # Random angle
    theta = 360*np.random.rand(1)
    if verbose: print('Angle', theta)
    thetar = np.radians(theta)
    xp = x*np.cos(thetar) - y*np.sin(thetar)
    yp = x*np.sin(thetar) + y*np.cos(thetar)
    ap = polynomial.RotateCoeffs(a, theta, order)
    u = polynomial.poly(a, x, y, order)
    up = polynomial.poly(ap, xp, yp, order) # using transformed point and polynomial
    if verbose:
        print('X Y', x, y)
        print('U', u)
        print('XP YP', xp, yp)
        print('UP', up)
        print('UP-U', up-u)
    assert abs(up-u) < 1.0e-12, 'Inaccurate transformation conversion'
    return
