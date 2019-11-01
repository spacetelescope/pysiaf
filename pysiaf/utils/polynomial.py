"""A collection of functions to manipulate polynomials and their coefficients.

Authors
-------
    - Colin Cox
    - Johannes Sahlmann

"""

from __future__ import absolute_import, print_function, division

from collections import OrderedDict

import numpy as np
from scipy import linalg


def add_rotation(A, B, theta_deg):
    """Add rotation after polynomial transformation.

    Use when a distortion transformation using polynomials A and B is followed by a rotation.

    u = A(x,y) v = B(x,y) followed by
    u2 = u*cos(theta) + v*sin(theta)
    v2 = -u*sin(theta) + v*cos(theta)
    This routine supplies a modified pair of polynomial which combine both steps
    i.e u2 = A2(x,y), v2 = B2(x,y)

    Parameters
    ----------
    A : array
        Set of polynomial coefficients converting from (x,y) to a variable u
    B : array
        set of polynomial coefficients converting from(x,y) to  avariable v
    theta_deg : float
        The angle in degrees of a rotationin the (u,v) plane

    Returns
    -------
    A2 : array
        set of polynomial coefficiients providing combined steps from (x,y) to u2
    B2 : array
        set of polynomial coefficients providing combined steps from (x,y) to v2

    Notes
    -----
    Function formerly named Rotate or rotate_coefficients.
    Ported from makeSIAF.py by J. Sahlmann 2018-01-03.

    """
    theta = np.deg2rad(theta_deg)

    A2 = +A*np.cos(theta) + B*np.sin(theta)
    B2 = -A*np.sin(theta) + B*np.cos(theta)

    return A2, B2


def choose(n, r):
    """Return number of ways of choosing r items from an array with n items.

    Parameters
    ----------
    n : int
        number of items to choose from
    r : int
        number of items to choose

    Returns
    -------
    combinations : int
        The number if ways of making the choice

    """
    if n < 0 or r < 0:
        raise ValueError('Negative values not allowed')
    if r > n:
        raise ValueError('r must not be greater than n')

    combinations = 1
    r1 = min(r, n-r)
    for k in range(r1):
        combinations = combinations * (n - k) // (k + 1)

    return combinations


def dpdx(a, x, y):
    """Differential with respect to x.

    The polynomial is defined as p(x,y) = a[i,j] * x**(i-j) * y**j summed over i and j.
    Then dp/dx = (i-j) * a[i,j] * x**(i-j-1) * y**j.


    Parameters
    ----------
    a : array
        a linear array of polynomial coefficients in JWST order.
    x : array
        an integer or float variable(or an array of same) representing pixel x positions
    y : array
        a variable (or an array) representing  pixel y positions. x and y must be of same shape.

    Returns
    -------
    differential : array
        float values of dp/dx for the given (x,y) point(s)

    """
    poly_degree = polynomial_degree(len(a))

    differential = 0.0
    k = 1  # index for coefficients
    for i in range(1, poly_degree + 1):
        for j in range(i + 1):
            if i - j > 0:
                differential = differential + (i - j) * a[k] * x**(i - j - 1) * y**j
            k += 1
    return differential


def dpdy(a, x, y):
    """Differential with respect to y.

    The polynomial is defined as p(x,y) = a[i,j] * x**(i-j) * y**j, summed over i and j
    Then dp/dy = j * a[i,j] * x**(i-j) * y**(j-1)

    Parameters
    ----------
    a : array
        an array of polynomial coefficients in JWST arrangement.
        The number of coefficients must be (order+1)(order+2)/2
    x : array
        an integer or float variable(or an array of same) representing  pixel x positions
    y : array
        a variable (or an array) representing  pixel y positions

    Returns
    -------
    differential  : array
        float value of dp/dy for the given (x,y) point(s) where p(x,y) is the value of the
        polynomial

    """
    poly_degree = polynomial_degree(len(a))
    differential = 0.0
    k = 1  # index for coefficients
    for i in range(1, poly_degree + 1):
        for j in range(i + 1):
            if j > 0:
                differential = differential + j * a[k] * x**(i - j) * y**(j - 1)
            k += 1
    return differential


def flatten(coefficients):
    """Convert triangular layout to linear array.

    For many of the polynomial operations the coefficients A[i,j] are contained in an
    array of dimension (order+1, order+1) but with all elements where j > i set equal to zero.
    This is what is called the triangular layout.

    The flattened layout is a one-dimensional array containing only the elements where j <= i.

    Parameters
    ----------
    coefficients : array
        a two-dimensional float array of shape (order+1, order+1) supplying the polynomial
        coefficients in JWST order. The coefficient at position [i,j] multiplies the value
        of x**(i-j) * y**j.

    Returns
    -------
    flate_coefficients : array
        a one-dimensional array including only those terms where i <= j

    """
    poly_degree = coefficients.shape[0]-1
    n_coefficients = number_of_coefficients(poly_degree)

    flat_coefficients = np.zeros(n_coefficients)
    k = 0
    for i in range(poly_degree+1):
        for j in range(i+1):
            flat_coefficients[k] = coefficients[i, j]
            k += 1
    return flat_coefficients


def flip_x(A):
    """Change sign of all coefficients with odd x power.

    Used when we have a polynomial expansion in terms of variables x and y and we wish to obtain one
    in which the sign of x is reversed

    Parameters
    ----------
    A : array
        A set of polynomial coefficients given in the triangular layout as described in poly

    Returns
    -------
    AF : array
        Modified or flipped set of coefficients matching negated x values.

    """
    poly_degree = polynomial_degree(len(A))
    AF = np.zeros(len(A))
    k = 0
    for i in range(poly_degree+1):
        for j in range(i+1):
            AF[k] = (-1)**(i-j) * A[k]
            k += 1
    return AF


def flip_y(A):
    """Change sign of all coefficients with odd y power.

    Used when we have a polynomial expansion in terms of variables x and y and we wish to obtain one
    in which the sign of y is reversed

    Parameters
    ----------
    A : array
        A set of polynomial coefficients given in the triangular layout as described in the
        function poly

    Returns
    -------
    AF : array
        Modified or flipped set of coefficients matching negated y values.

    """
    poly_degree = polynomial_degree(len(A))
    order = poly_degree

    terms = (order+1)*(order+2) // 2
    AF = np.zeros(terms)
    k = 0
    for i in range(order+1):
        for j in range(i+1):
            AF[k] = (-1)**(j) * A[k]
            k += 1
    return AF


def flip_xy(A):
    """Change sign for coeffs where sum of x and y powers is odd.

    Used when we have a polynomial expansion in terms of variables x and y and we wish to obtain one
    in which the signs of x and y are reversed

    Parameters
    ----------
    A : array
        A set of polynomial coefficients given in the triangular layout as described in the
        function poly

    Returns
    -------
    AF : array
        Modified or flipped set of coefficients matching negated x and y values.

    """
    poly_degree = polynomial_degree(len(A))
    order = poly_degree

    terms = (order+1)*(order+2) // 2
    AF = np.zeros(terms)
    k = 0
    for i in range(order+1):
        for j in range(i+1):
            AF[k] = (-1)**(i) * A[k]
            k += 1
    return AF


def invert(A, B, u, v, verbose=False):
    """Newton Raphson method in two dimensions.

    Given that u = A[i,j] * x**(i-j) * y**j and v = B[i,j] * x**(i-j) * y**j
    find the values of x and y from the values of u and v


    Parameters
    ----------
    A : array
        A set of polynomial coefficients given in the linear layout as described in the function
        poly converting (x,y) to u
    B : array
        A set of polynomial coefficients given in the linear layout as described in the function
        poly converting (x,y) to v
    u : array
        The result of applying the A coefficients to the (x,y) position
    v : array
        The result of applying the B coefficients to the (x, y)position
    verbose : bool
        Logical variable, set True if full text output required

    Returns
    -------
    x, y  : tuple of arrays
        The pair of values which transform to (u,v)
    err : float
        the standard deviation of the fit
    iteration : int
        the number of iterations taken to determine the solution

    """
    poly_degree = polynomial_degree(len(A))
    order = poly_degree

    tol = 1.0e-6
    err = 1.0
    # Initial guesses - Linear approximation
    det = A[1] * B[2] - A[2] * B[1]
    x0 = (B[2] * (u - A[0]) - A[2] * (v - B[0])) / det
    y0 = (-B[1] * (u - A[0]) + A[1] * (v - B[0])) / det
    if verbose:
        print('Initial guesses', x0, y0)
    x = x0
    y = y0
    X = np.array([x, y])
    iteration = 0
    while err > tol:
        f1 = np.array([poly(A, x, y, order) - u, poly(B, x, y, order) - v])
        j = np.array([[dpdx(A, x, y), dpdy(A, x, y)], [dpdx(B, x, y), dpdy(B, x, y)]])
        invj = np.linalg.inv(j)
        X = X - np.dot(invj, f1)
        if verbose:
            print('[X1,Y1]', X)
        x1 = X[0]
        y1 = X[1]
        err = np.hypot(x - x1, y - y1)
        if verbose:
            print('Error %10.2e' % err)
        [x, y] = [x1, y1]
        iteration += 1

    return x, y, err, iteration


def jacob(a, b, x, y):
    """Calculate relative area using the Jacobian.

               | da_dx   db_dx |
    Jacobian = |               |
               | da_dy   db_dy |
    Then the relative area is the absolute value of the determinant of the Jacobian.
    x and y will usually be Science coordinates while u and v are Ideal coordinates

    Parameters
    ----------
    a : array
        set of polynomial coefficients converting from (x,y) to u
    b : array
        set of polynomial coefficients converting from (x,y) to v
    x : array
        x pixel position or array of x positions
    y : array
        y pixel position or array of y positions matching the y positions

    Returns
    -------
    area : array
        area in (u,v) coordinates matching unit area in the (x,y) coordinates.

    """
    j = dpdx(a, x, y)*dpdy(b, x, y) - dpdx(b, x, y)*dpdy(a, x, y)
    area = np.fabs(j)
    return area


def number_of_coefficients(poly_degree):
    """Return number of coefficients corresponding to polynomial degree."""
    if type(poly_degree) == int:
        n_coefficients = np.int((poly_degree + 1) * (poly_degree + 2) / 2)
        return n_coefficients
    else:
        raise TypeError('Argument has to be of type int')


def poly(a, x, y, order=4):
    """Polynomial evaluation.

    pol = a[i,j] * x**(i-j) * y**j summed over i and j, where i runs from 0 to order.
    Then for each value of i, j runs from 0 to i.
    For many of the polynomial operations the coefficients A[i,j] are contained in an
    array of dimension (order+1, order+1) but with all elements where j > i set equal to zero.
    This is called the triangular layout.
    The flattened layout is a one-dimensional array containing copies of only the elements
    where j <= i.

    The JWST layout is a[0,0] a[1,0] a[1,1] a[2,0] a[2,1] a[2,2] ...
    The number of coefficients will be (n+1)(n+2)/2

    Parameters
    ----------
    a : array
        float array of polynomial coefficients in flattened arrangement
    x : array
        x pixel position. Can be integer or float or an array of integers or floats
    y : array
        y pixel position in same layout as x positions.
    order : int
        integer polynomial order

    Returns
    -------
    pol : float
        result as described above

    """
    pol = 0.0
    k = 0  # index for coefficients
    for i in range(order+1):
        for j in range(i+1):
            pol = pol + a[k] * x**(i-j) * y**j
            k += 1
    return pol


def polyfit(u, x, y, order):
    """Fit polynomial to a set of u values on an x,y grid.

    u is a function u(x,y) being a polynomial of the form
    u = a[i, j] x**(i-j) y**j. x and y can be on a grid or be arbitrary values
    This version uses scipy.linalg.solve instead of matrix inversion.
    u, x and y must have the same shape and may be 2D grids of values.

    Parameters
    ----------
    u : array
        an array of values to be the results of applying the sought after
        polynomial to the values (x,y)
    x : array
        an array of x values
    y : array
        an array of y values
    order : int
        the polynomial order

    Returns
    -------
    coeffs: array
        polynomial coefficients being the solution to the fit.

    """
    # First set up x and y powers for each coefficient
    px = []
    py = []
    for i in range(order + 1):
        for j in range(i + 1):
            px.append(i - j)
            py.append(j)
    terms = len(px)

    # Make up matrix and vector
    vector = np.zeros((terms))
    mat = np.zeros((terms, terms))
    for i in range(terms):
        vector[i] = (u * x ** px[i] * y ** py[i]).sum()  # Summing over all x,y
        for j in range(terms):
            mat[i, j] = (x ** px[i] * y ** py[i] * x ** px[j] * y ** py[j]).sum()

    coeffs = linalg.solve(mat, vector)
    return coeffs


def polynomial_degree(number_of_coefficients):
    """Return degree of the polynomial that has number_of_coefficients.

    Parameters
    ----------
    number_of_coefficients : int
        Number of polynomial coefficients

    Returns
    -------
    polynomial_degree : int
        Degree of the polynomial

    """
    poly_degree = (np.sqrt(8 * number_of_coefficients + 1) - 3) / 2
    if not poly_degree.is_integer():
        raise ValueError('Number of coefficients does not match a valid polynomial degree.')
    else:
        return np.int(poly_degree)


def prepend_rotation_to_polynomial(a, theta, verbose=False):
    """Rotate axes of coefficients by theta degrees.

    Used when a distortion transformation using polynomials A and B is preceded by a rotation.
    The set of polynomial coefficients a[i,j] transform (x,y) as  u = a[i,j] * x**(i-j) * y**j
    Summation over repeated indices is implied.
    If now we have a set of variables (xp,yp) rotated from (x,y) so that
    xp = x * cos(theta) - y * sin(theta)
    yp = x * sin(theta) + y * cos(theta)
    find a set of polynomial coefficients ap so that the same value of u is obtained from (xp,yp)
    i.e, u = ap[i,j] * xp**(i-j) * yp**j
    The rotation is opposite to the usual rotation as this routine was designed for the inverse
    transformation between Ideal and V2V3 or tel. Effectively the angle is reversed


    Parameters
    ----------
    a : array
        Set of polynomial coefficients
    theta : float
        rotation angle in degrees
    verbose : bool
        logical variable set True only if print-out of coefficient factors is desired.

    Returns
    -------
    arotate : array
        set of coefficients derived as described above.

    Notes
    -----
    Function was formerly named RotateCoeffs.

    """
    poly_degree = polynomial_degree(len(a))

    c = np.cos(np.deg2rad(theta))
    s = np.sin(np.deg2rad(theta))

    # First place in triangular layout
    at = triangular_layout(a)

    # Apply rotation
    atrotate = np.zeros([poly_degree+1, poly_degree+1])
    # arotate = np.zeros([len(a)]) # Copy shape of a
    for m in range(poly_degree+1):
        for n in range(m+1):
            for mu in range(0, m-n+1):
                for j in range(m-n-mu, m-mu+1):
                    factor = (-1)**(m-n-mu) * choose(m-j, mu) * choose(j, m-n-mu)
                    cosSin = c**(j+2*mu-m+n) * s**(2*m-2*mu-j-n)
                    atrotate[m, n] = atrotate[m, n] + factor * cosSin * at[m, j]
                    if verbose:
                        print(m, n, j, factor, 'cos^', j+2*mu-m+n, 'sin^', 2*m-2*mu-j-n, ' A', m, j)
    # Put back in linear layout
    arotate = flatten(atrotate)

    return arotate


def print_triangle(coefficients):
    """Print coefficients in triangular layout.

    A[0]
    A[1]  A[2]
    A[3]  A[4]  A[5]
    ...
    equivalent to
    A[0,0]
    A[1,0] A[1,1]
    A[2,0] A[2,1] A[2,2]
    ...
    in [i,j] terms.

    See method poly for details.
    This is just to display the coefficients. No calculation performed.

    Parameters
    ----------
    coefficients : array
        polynomial float array in linear layout

    """
    poly_degree = polynomial_degree(len(coefficients))

    k = 0
    for i in range(poly_degree + 1):
        for j in range(i + 1):
            print('%12.5e' % coefficients[k], end=' ')
            k += 1
        print()


def reorder(A, B):
    """Change coefficient order from y**2 xy x**2 to x**2 xy y**2 in both A and B.

    Parameters
    ----------
    A : array
        polynomial coefficients
    B : array
        polynomial coefficients

    Returns
    -------
    A2, B2: numpy arrays
        coefficients with changed order

    """
    poly_degree = polynomial_degree(len(A))
    A2 = np.zeros((len(A)))
    B2 = np.zeros((len(B)))
    for i in range(poly_degree + 1):
        ti = i * (i + 1) // 2
        for j in range(i + 1):
            A2[ti + j] = A[ti + i - j]
            B2[ti + j] = B[ti + i - j]

    return A2, B2


def rescale(A, B, C, D, scale):
    """Apply a scale to forward and inverse polynomial coefficients.

    Parameters
    ----------
    A : array
        Polynomial coefficients
    B : array
        Polynomial coefficients
    C : array
        Polynomial coefficients
    D : array
        Polynomial coefficients
    scale : float
        Scale factor to apply

    Returns
    -------
    A_scaled, B_scaled, C_scaled, D_scaled : tuple of numpy arrays
        the scales coefficients

    Notes
    -----
    Ported from makeSIAF.py by J. Sahlmann 2018-01-03.
    J. Sahlmann 2018-01-04: fixed side-effect on ABCD variables

    """
    A_scaled = scale*A
    B_scaled = scale*B

    poly_degree = polynomial_degree(len(A))
    number_of_coefficients = len(A)

    C_scaled = np.zeros(number_of_coefficients)
    D_scaled = np.zeros(number_of_coefficients)

    k = 0
    for i in range(poly_degree+1):
        factor = scale**i
        for j in range(i+1):
            C_scaled[k] = C[k]/factor
            D_scaled[k] = D[k]/factor
            k += 1

    return A_scaled, B_scaled, C_scaled, D_scaled

def scale_from_derivatives(pc1, pc2):
    """Return scale estimate."""
    return np.sqrt(pc1 ** 2 + pc2 ** 2)


def rotation_from_derivatives(pc1, pc2):
    """Return rotation estimate."""
    return np.rad2deg(np.arctan2(pc1, pc2))


def rotation_scale_skew_from_derivatives(b, c, e, f):
    """Compute rotations, scales, and skews from polynomial derivatives.

    The four partial derivatives of coefficients that transform between two
    frames E and R are:

    Parameters
    ----------
    b : float
        (dx_E)/(dx_R )
    c : float
        (dx_E)/(dy_R )
    e : float
        (dy_E)/(dx_R )
    f : float
        (dy_E)/(dy_R )

    Returns
    -------

    """
    # compute scales
    scale_x = scale_from_derivatives(b, e)
    scale_y = scale_from_derivatives(c, f)
    scale_global = np.sqrt(b * f - c * e)

    # compute rotations
    rotation_x = rotation_from_derivatives(-e, b)
    rotation_y = rotation_from_derivatives(c, f)
    rotation_global = (rotation_x + rotation_y) / 2.
    rotation_global_2 = np.rad2deg(np.arctan2(c - e, b + f))

    # compute skews
    skew = rotation_y - rotation_x
    skew_onaxis = (b - f) / 2.  # difference in x/y scale
    skew_offaxis = (c + e) / 2.  # non-perpendicularity between the axes

    results = OrderedDict()
    results['scale_x'] = scale_x
    results['scale_y'] = scale_y
    results['scale_global'] = scale_global
    results['rotation_x'] = rotation_x
    results['rotation_y'] = rotation_y
    results['rotation_global'] = rotation_global
    results['rotation_global_2'] = rotation_global_2
    results['skew'] = skew
    results['skew_onaxis'] = skew_onaxis
    results['skew_offaxis'] = skew_offaxis

    return results



def shift_coefficients(a, xshift, yshift, verbose=False):
    """Calculate coefficients of polynomial when shifted to new origin.

    Given a polynomial function such that u = a[i,j] * x**[i-j] * y**[j] summed over i and j.
    Find the polynomial function ashift centered at xshift, yshift
    i.e the same value of u = ashift[i,j] * (x-xshift)**(i-j) * (y-yshift)**j.

    Parameters
    ----------
    a : array
        Set of coefficients for a polynomial of the given order in JWST order
    xshift : float
        x position in pixels of new solution center
    yshift : float
        y position in pixels of new solution center
    verbose : bool
        logical variable to choose print-out of coefficient table - defaults to False

    Returns
    -------
    ashift : array
        shifted version of the polynomial coefficients.

    """
    poly_degree = polynomial_degree(len(a))

    # place in triangular layout
    at = triangular_layout(a)

    # Apply shift
    atshift = np.zeros((poly_degree+1, poly_degree+1))
    for p in range(poly_degree + 1):
        for q in range(p + 1):
            if verbose:
                print("A'%1d%1d" % (p, q))
            for i in range(p, poly_degree + 1):
                for j in range(q, i + 1 - (p - q)):
                    f = choose(j, q) * choose(i - j, p - q)
                    atshift[p, q] = atshift[p, q] + f * xshift**((i - j) - (p - q)) \
                                                    * yshift**(j - q) * at[i, j]
                    if verbose:
                        print('%2d A(%1d,%1d) x^%1d y^%1d' % (f, i, j, i - j - (p - q), (j - q)))
            if verbose:
                print()

    # Put back in linear layout
    ashift = flatten(atshift)

    return ashift


def transform_coefficients(A, a, b, c, d, verbose=False):
    """Transform polynomial coefficients.

    This allows for
    xp = a*x + b*y
    yp = c*x + d*y


    Parameters
    ----------
    A : array
        Polynomial coefficients
    a : float
        factor
    b : float
        factor
    c : float
        factor
    d : float
        factor
    verbose : bool
        verbosity

    Returns
    -------
    AT : array
        Transformed coefficients

    Notes
    -----
    Designed to work with Sabatke solutions which included a linear transformation of the pixel
    coordinates before the polynomial distortion solution was calculated.
    `transform_coefficients` combines the two steps into a single polynomial.

    """
    poly_degree = polynomial_degree(len(A))

    A1 = np.zeros((poly_degree + 1, poly_degree + 1))
    A2 = np.zeros((poly_degree + 1, poly_degree + 1))

    ncoeffs = (poly_degree + 1) * (poly_degree + 2) // 2
    if verbose:
        print(ncoeffs, 'coefficients for poly_degree', poly_degree)
    AT = np.zeros((ncoeffs))

    # First place A in triangular layout
    k = 0
    for i in range(poly_degree + 1):
        for j in range(i + 1):
            A1[i, j] = A[k]
            k += 1

    for m in range(poly_degree + 1):
        for n in range(m + 1):
            if verbose:
                print('\nM,N', m, n)
            for mu in range(m - n + 1):
                for j in range(m - n - mu, m - mu + 1):
                    if verbose:
                        print('J, MU', j, mu)
                    if verbose:
                        print('Choose', m - j, mu, 'and', j, m - n - mu)
                    factor = choose(m - j, mu) * choose(j, m - n - mu)
                    A2[m, n] += factor * a**mu * b**(m - j - mu) * c**(m - n - mu) \
                                * d**(mu + j - m + n) * A1[m, j]
                    if verbose:
                        print(m, j, ' Factor', factor)

    # Restore A2 to flat layout in AT
    k = 0
    for m in range(poly_degree + 1):
        for n in range(m + 1):
            AT[k] = A2[m, n]
            k += 1
    return AT


def triangular_layout(coefficients):
    """Convert linear array to 2-D array with triangular coefficient layout.

    This is the reverse of the flatten method.

    Parameters
    ----------
    coefficients : array
        float array of polynomial coefficients. Must be of dimension (order+1)(order+2)/2.

    Returns
    -------
    triangular_coefficients : array
        coefficients in triangular layout as described in poly.

    """
    poly_degree = polynomial_degree(len(coefficients))
    triangular_coefficients = np.zeros((poly_degree + 1, poly_degree + 1))

    k = 0
    for i in range(poly_degree + 1):
        for j in range(i + 1):
            triangular_coefficients[i, j] = coefficients[k]
            k += 1

    return triangular_coefficients


def two_step(A, B, a, b):
    """Combine linear step followed by a polynomial step into a single polynomial.

    Designed to process Sabatke polynomials which had a linear transformation
    ahead of the polynomial fits.

    Starting from a pair of polynomial arrays A and B such that
    u = A[i,j[ * xp**(i-j) * yp**j
    v = B[i,j] * xp**(i-j) * yp**j
    in which
    xp = a[0] + a[1].x + a[2].y
    yp = b[0] + b[1].x + b[2].y
    find AP and BP such that the same u and v values are given by
    u = AP[i,j] * x**(i-j) * y**j
    v = BP[i,j] * x**(i-j) * y**j
    All input and output polynomials are flattened arrays of dimension (order+1)(order+2)/2
    Internally they are processed as equivalent two dimensional arrays as described in the poly
    documentation.

    Parameters
    ----------
    A : array
        polynomial converting from secondary xp and yp pixel positions to final coordinates u
    B : array
        polynomial converting from secondary xp and yp pixel positions to final coordinates v
    Aflat : array
        set of linear coefficients converting (x,y) to xp
    Bflat : array
        set of linear coefficients converting (x,y) to yp

    Returns
    -------
    Aflat, Bflat : tuple of arrays
        polynomial coefficients as calculated

    """
    poly_degree = polynomial_degree(len(A))
    order = poly_degree

    A2 = np.zeros((poly_degree+1, poly_degree+1))
    B2 = np.zeros((poly_degree+1, poly_degree+1))
    k = 0
    for i in range(order+1):
        for j in range(i+1):
            for alpha in range(i-j+1):
                for beta in range(i-j-alpha+1):
                    f1 = choose(i-j, alpha)*choose(i-j-alpha, beta)*a[0]**(i-j-alpha-beta) \
                         * a[1]**alpha*a[2]**beta
                    for gamma in range(j+1):
                        for delta in range(j-gamma+1):
                            f2 = choose(j, gamma)*choose(j-gamma, delta)*b[0]**(j-gamma-delta) \
                                 * b[1]**gamma*b[2]**delta
                            A2[alpha+beta+gamma+delta, beta+delta] += A[k]*f1*f2
                            B2[alpha+beta+gamma+delta, beta+delta] += B[k]*f1*f2
            k += 1
    # Flatten A2 and B2
    Aflat = flatten(A2)
    Bflat = flatten(B2)

    return Aflat, Bflat
