#!/usr/bin/env python


"""
Authors
-------

    Colin Cox

"""

import numpy as np
from pysiaf.utils import polynomial


def test_invert(verbose=False):
    """Test accuracy of inversion method"""

    # First invent a plausible pair of polynomial arrays
    order = 5
    terms = (order+1)*(order+2)//2
    a = np.zeros(terms)
    b = np.zeros(terms)
    (a[1], b[2]) = 0.05 + 0.01*np.random.rand(2)
    (a[2], b[1]) = 0.0001*np.random.rand(2)
    a[3:6] = 1.0e-7*np.random.rand(3)
    b[3:6] = 1.0e-7*np.random.rand(3)
    a[6:10] = 1.0e-10*np.random.rand(4)
    b[6:10] = 1.0e-10*np.random.rand(4)
    a[10:15] = 1.0e-13*np.random.rand(5)
    b[10:15] = 1.0e-13*np.random.rand(5)
    a[15:21] = 1.0e-15*np.random.rand(6)
    b[15:21] = 1.0e-15*np.random.rand(6)
    if verbose:
        print('A')
        polynomial.triangle(a, 5)
        print('B')
        polynomial.triangle(b, 5)

    # Random point within 2048 square with origin at the center
    (x, y) = 2048.0*np.random.rand(2) - 1024.0
    u = polynomial.poly(a, x, y, order)
    v = polynomial.poly(b, x, y, order)
    if verbose:
        print('X Y', x, y)
        print('U V', u, v)

    (x2, y2, error, iterations) = polynomial.invert(a, b, u, v, order, verbose=verbose)

    if verbose:
        print('Error', error, ' after',  iterations, ' iterations')
        print('X2 Y2', x2, y2)
        print('Dx Dy', x2-x, y2-y)
    assert abs(x2-x) < 1.0e-12 and abs(y2-y) < 1.0e-12, 'Error too large'

    return
