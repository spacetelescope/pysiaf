
import numpy as np
from pysiaf.utils import polynomial
import pylab as pl


def makeup_polynomial(order = 5):
    """Invent a random but plausible polynomial array.

    designed to be similar to usual SIAF polynomials in which leading coefficients
    are approximately 0.05 and successive power coefficients are smaller by a a factor of about 1000.

    parameters
    return: a - randomly generated polynomial array """

    terms = polynomial.number_of_coefficients(order)
    a = np.zeros(terms)

    np.random.seed(seed=1)
    a[1] = 0.05 + 0.01 * np.random.rand(1)
    np.random.seed(seed=2)
    a[2] = 0.0001 * np.random.rand(1)
    np.random.seed(seed=3)
    a[3:6] = 1.0e-7 * np.random.rand(3)
    np.random.seed(seed=4)
    a[6:10] = 1.0e-10 * np.random.rand(4)
    np.random.seed(seed=5)
    a[10:15] = 1.0e-13 * np.random.rand(5)
    np.random.seed(seed=6)
    a[15:21] = 1.0e-15 * np.random.rand(6)

    return a


def test_poly(verbose=False):
    """ Tests polynomial evaluation by calculating a 9 by 9 array across a 2048 pixel grid
    Then polyfit is used to fit the calculated points to generate a new pair of polynomials
    Finally the original x,y points are used in the new polynomials and the outputs compared
    This incidentally provides a robust test of polyfit

    parameters
    verbose: logical value. If True, print statements and graph will be output
            if False or unassigned there will be no output unless the assert statement issues
            a Polynomial inaccuracy message.
    return: None """

    [x, y] = np.mgrid[0:9, 0:9]
    xp = 256.0*x - 1024.0
    yp = 256.0*y - 1024.0
    #u = np.zeros((9, 9))
    #v = np.zeros((9, 9))

    # Random polynomials
    a = makeup_polynomial()
    b = makeup_polynomial()
    # Switch linear terms for b coefficients so that b[2] is approximate scale
    btemp = b[1]
    b[1] = b[2]
    b[2] = btemp

    if verbose:
        print('A coefficients')
        polynomial.print_triangle(a)
        print('B coefficients')
        polynomial.print_triangle(b)

    # Evaluate polynomials acting on x,y arrays
    u = polynomial.poly(a, x, y, 5)
    v = polynomial.poly(b, x, y, 5)
    # Fit new polynomials to calculated positions
    s1 = polynomial.polyfit(u, x, y, 5)
    s2 = polynomial.polyfit(v, x, y, 5)
    # Evaluate new polynomials
    uc = polynomial.poly(s1, x, y, 5)
    vc = polynomial.poly(s2, x, y, 5)
    # Compare outputs
    du = uc - u
    dv = vc - v
    u_std = np.std(du)
    v_std = np.std(dv)

    if verbose:
        print('Fitted polynomials')
        print('S1')
        polynomial.print_triangle(s1)
        print('S2')
        polynomial.print_triangle(s2)
        print ('Fit comparison STDs {:10.2e} {:10.2e}'.format(u_std, v_std))
        pl.figure(1)
        pl.clf()
        pl.grid(True)
        pl.plot(u, v, 'gx')
        pl.plot(uc, vc, 'r+')

    assert u_std < 1.0e-12 and v_std < 1.0e-12, 'Polynomial inaccuracy'
    return None


def test_RotateCoeffs(verbose=False):
    """Test accuracy of inversion method"""

    # First invent a random but plausible polynomial array
    a = makeup_polynomial()
    order = 5
    if verbose:
        print('A')
        polynomial.print_triangle(a)

    # Random point within 2048 square with origin at the center
    np.random.seed(seed=1)
    [x, y] = 2048.0*np.random.rand(2) - 1024.0
    u = polynomial.poly(a, x, y, order)

    # Random angle

    theta = 360*np.random.rand(1)
    if verbose:
        print('Angle', theta)
    thetar = np.radians(theta)
    xp = x*np.cos(thetar) - y*np.sin(thetar)
    yp = x*np.sin(thetar) + y*np.cos(thetar)
    ap = polynomial.prepend_rotation_to_polynomial(a, theta, order)
    u = polynomial.poly(a, x, y, order)
    up = polynomial.poly(ap, xp, yp, order) # using transformed point and polynomial
    if verbose:
        print('X Y', x, y)
        print('U', u)
        print('XP YP', xp, yp)
        print('UP', up)
        print('UP-U', up-u)
    assert abs(up-u) < 1.0e-12, 'Inaccurate transformation conversion'


def test_two_step(verbose=False):
    # make up random polynomials of order 5 with terms which decrease strongly with power.
    A = makeup_polynomial()
    B = makeup_polynomial()
    a = np.array([1.0, 0.5, 0.1])
    b = np.array([2.0, 0.2, 0.6])

    (A2, B2) = polynomial.two_step(A, B, a, b)
    if verbose:
        print('\nA')
        polynomial.print_triangle(A)#     print('B')
        print('B')
        polynomial.print_triangle(B)
        print('\nLinear terms')
        print('a',a)
        print('b', b)
        print('\nA2')
        polynomial.print_triangle(A2)
        print('B2')
        polynomial.print_triangle(B2)

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


def test_invert(verbose=True):
    """Test accuracy of inversion method"""

    # First invent a plausible pair of polynomial arrays
    order = 5
    a = makeup_polynomial()
    b = makeup_polynomial()
    # Modify linear terms in b array so sclae term is b[2]
    btemp = b[1]
    b[1] = b[2]
    b[2] = btemp

    if verbose:
        print('A')
        polynomial.print_triangle(a)
        print('B')
        polynomial.print_triangle(b)

    # Random point within 2048 square with origin at the center
    np.random.seed(seed=1)
    (x, y) = 2048.0*np.random.rand(2) - 1024.0
    u = polynomial.poly(a, x, y, order)
    v = polynomial.poly(b, x, y, order)
    if verbose:
        print('X Y', x, y)
        print('U V', u, v)

    (x2, y2, error, iterations) = polynomial.invert(a, b, u, v, verbose=verbose)

    if verbose:
        print('Error', error, ' after',  iterations, ' iterations')
        print('X2 Y2', x2, y2)
        print('Dx Dy', x2-x, y2-y)
    assert abs(x2-x) < 1.0e-12 and abs(y2-y) < 1.0e-12, 'Error too large'

    return

def test_ShiftCoeffs(verbose=False):
    """ Test accuracy of shift_coefficients method"""

    # First invent a plausible polynomial
    order = 5
    a = makeup_polynomial()

    if verbose:
        print('A')
        polynomial.print_triangle(a)

    # Shift by a random step
    np.random.seed(seed=1)
    [xshift, yshift] = 1024.0 * np.random.rand(2) - 512.0

    ashift = polynomial.shift_coefficients(a, xshift, yshift, verbose)
    if verbose:
        print('AS')
        polynomial.print_triangle(ashift)

    # Choose a random point
    [x, y] = 2048 * np.random.rand(2) - 1024.0
    u1 = polynomial.poly(a, x, y, order)
    u2 = polynomial.poly(ashift, x-xshift, y-yshift, order)

    if verbose:
        print('XY', x, y)
        print('Shift', xshift, yshift)
        print('U values', u1, u2, u1-u2)

    assert abs(u1-u2) < 1.0e-12, 'Inaccurate shift transformation'

    return None