"""A collection of functions to manipulate polynomials and their coefficients.

Authors
-------

    - Colin Cox
    - Johannes Sahlmann

"""

from __future__ import absolute_import, print_function, division
import numpy as np
import scipy as sp
from scipy import linalg


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

    The polynomial is defined as p(x,y) = a[i,j] * x**(i-j) * y**j summed over i and j
    Then dp/dx = (i-j) * a[i,j] * x**(i-j-1) * y**j


    Parameters
    ----------
    a      a linear array of polynomial coefficients in JWST order.
    x      an integer or float variable(or an array of same) representing pixel x positions
    y      a variable (or an array) representing  pixel y positions. x and y must be of same shape.

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
    a      an array of polynomial coefficients in JWST arrangement.
            The number of coefficients must be (order+1)(order+2)/2
    x      an integer or float variable(or an array of same) representing  pixel x positions
    y      a variable (or an array) representing  pixel y positions
    order  an integer, the polynomal order

    Returns
    -------
    differential    float value of dp/dy for the given (x,y) point(s)
            where p(x,y) is the value of the polynomial

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


def flatten(A, order):
    """Convert triangular layout to linear array.

    For many of the polynomial operations the coefficients A[i,j] are contained in an
    array of dimension (order+1, order+1) but with all elements where j > i set equal to zero.
    This is what is called the triangular layout.

    The flattened layout is a one-dimensional array containing only the elements where j <= i.

    Parameters
    ----------
    A       a two-dimensional float array of shape (order+1, order+1) supplying the polynomial coefficients
            in JWST order. The coefficient at position [i,j] multiplies the value of x**(i-j) * y**j
    order   an integer,  the order of the polynomial

    Returns
    -------
    AF : array
        a one-dimensional array including only those terms where i <= j

    """
    terms = (order+1)*(order+2) // 2
    AF = sp.zeros(terms)
    k = 0
    for i in range(order+1):
        for j in range(i+1):
            AF[k] = A[i, j]
            k += 1
    return AF


def FlipX(A, order=4):
    """Change sign of all coefficients with odd x power.

    Used when we have a polynomial expansion in terms of variables x and y and we wish to obtain one
    in which the sign of x is reversed

    Parameters
    ----------
    A : array
        A set of polynomial coefficients given in the triangular layout as described in poly
    order  The polynomial order

    Returns
    -------
    AF      Modified or flipped set of coefficients matching negated x values.

    """
    terms = (order+1)*(order+2) // 2
    AF = sp.zeros(terms)
    k = 0
    for i in range(order+1):
        for j in range(i+1):
            AF[k] = (-1)**(i-j) * A[k]
            k += 1
    return  AF


def FlipY(A):
    """Change sign of all coefficients with odd y power.

    Used when we have a polynomial expansion in terms of variables x and y and we wish to obtain one
    in which the sign of y is reversed

    Parameters
    ----------
    A      A set of polynomial coefficients given in the triangular layout as described in the function poly
    order  The polynomial order

    Returns
    -------
    AF      Modified or flipped set of coefficients matching negated y values.

    """
    poly_degree = polynomial_degree(len(A))
    order = poly_degree

    terms = (order+1)*(order+2) // 2
    AF = sp.zeros(terms)
    k = 0
    for i in range(order+1):
        for j in range(i+1):
            AF[k] = (-1)**(j) * A[k]
            k += 1
    return AF


def FlipXY(A):
    """Change sign for coeffs where sum of x and y powers is odd.

    Used when we have a polynomial expansion in terms of variables x and y and we wish to obtain one
    in which the signs of x and y are reversed

    Parameters
    ----------
    A      A set of polynomial coefficients given in the triangular layout as described in the function poly
    order  The polynomial order

    Returns
    -------
    AF      Modified or flipped set of coefficients matching negated x and y values.

    """
    poly_degree = polynomial_degree(len(A))
    order = poly_degree

    terms = (order+1)*(order+2) // 2
    AF = sp.zeros(terms)
    k = 0
    for i in range(order+1):
        for j in range(i+1):
            AF[k] = (-1)**(i) * A[k]
            k += 1
    return AF


def invert(A, B, u, v, order, verbose=False):
    """Newton Raphson method in two dimensions.

    Given that u = A[i,j] * x**(i-j) * y**j and v = B[i,j] * x**(i-j) * y**j
    find the values of x and y from the values of u and v


    Parameters
    ----------
    A      A set of polynomial coefficients given in the linear layout as described in the function poly
            converting (x,y) to u
    B      A set of polynomial coefficients given in the linear layout as described in the function poly
            converting (x,y) to v
    u      The result of applyng the A coefficients to the (x,y) position
    v      The result of applyng the B coefficients to the (x, y)position
    order:  The polynomial order
    verbose:    Logical variable, set True if full text output required

    Returns
    -------
    x, y - The pair of values which transform to (u,v)
    err - the standard deviation of the fit
    iter - the number of iterations taken to determine the solution

    """
    tol = 1.0e-6
    err = 1.0
    # Initial guesses - Linear approximation
    det = A[1] * B[2] - A[2] * B[1]
    x0 = (B[2] * (u - A[0]) - A[2] * (v - B[0]))/ det
    y0 = (-B[1] * (u - A[0]) + A[1] * (v - B[0]))/ det
    if verbose:
        print('Initial guesses', x0, y0)
    x = x0
    y = y0
    X = sp.array([x, y])
    iter = 0
    while err > tol:
        f1 = sp.array([poly(A, x, y, order) - u, poly(B, x, y, order) - v])
        j = sp.array([[dpdx(A, x, y), dpdy(A, x, y)], [dpdx(B, x, y), dpdy(B, x, y)]])
        invj = sp.linalg.inv(j)
        X = X - sp.dot(invj, f1)
        if verbose:
            print('[X1,Y1]', X)
        x1 = X[0]
        y1 = X[1]
        err = sp.hypot(x - x1, y - y1)
        if verbose:
            print('Error %10.2e' % err)
        [x, y] = [x1, y1]
        iter += 1

    return x, y, err, iter


def jacob(a, b, x, y, order=4):
    """Calculate relative area using the Jacobian.

               | da_dx   db_dx |
    Jacobian = |               |
               | da_dy   db_dy |

    Then the relative area is the absolute value of the determinant of the Jacobian.

    Parameters
    ----------
    a      set of polynomial coefficients converting from (x,y) to u
    b      set of polynomial coefficients converting from (x,y) to v
    x      x pixel position or array of x positions
    y      y pixel position or array of y positions matching the y positions
    x and y will usually be Science coordinates while u and v are Ideal coordinates
    order:  order of the polynomials

    Returns
    -------
    area    area in (u,v) coordinates matching unit area in the (x,y) coordinates.

    """
    j = dpdx(a, x, y)*dpdy(b, x, y) - dpdx(b, x, y)*dpdy(a, x, y)
    area = sp.fabs(j)
    return area


def nircam_reorder(A, B, order):
    """Change coefficient order from y**2 xy x**2 to x**2 xy y**2.

    Parameters
    ----------
    A
    B
    order

    Returns
    -------
    A2, B2: numpy arrays

    """
    terms = (order + 1) * (order + 2) // 2
    A2 = np.zeros((terms))
    B2 = np.zeros((terms))
    for i in range(order + 1):
        ti = i * (i + 1) // 2
        for j in range(i + 1):
            A2[ti + j] = A[ti + i - j]
            B2[ti + j] = B[ti + i - j]

    return A2, B2


def poly(a, x, y, order=4):
    """Polynomial evaluation.

    pol = a[i,j] * x**(i-j) * y**j summed over i and j
    i runs from 0 to order. Then for each value of i, j runs from 0 to i
    For many of the polynomial operations the coefficients A[i,j] are contained in an
    array of dimension (order+1, order+1) but with all elements where j > i set equal to zero.
    This we call the triangular layout.
    The flattened layout is a one-dimensional array containing copies of only the elements where j <= i.
    The JWST layout is a[0,0] a[1,0] a[1,1] a[2,0] a[2,1] a[2,2] ...
    The number of coefficients will be (n+1)(n+2)/2

    Parameters
    ----------
    a       float array of polynomial coefficients in flattened arrangement
    x       x pixel position. Can be integer or float or an array of integers or floats
    y       y pixel position in same layout as x positions.
    order   integer polynomial order

    Returns
    -------
    pol     result as described above

    """
    pol = 0.0
    k = 0 # index for coefficients
    for i in range(order+1):
        for j in range(i+1):
            pol = pol + a[k] * x**(i-j) * y**j
            k += 1
    return pol


def polyfit(u, x, y, order):
    """SUPERSEDE BY POLYFIT2.

    Fit polynomial to a set of u values on an x,y grid
    u is a function u(x,y) being a polynomial of the form
    u = a[i, j] x**(i-j) y**j. x and y can be on a grid or be arbitrary values
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
    vector = sp.zeros((terms))
    mat = sp.zeros((terms, terms))
    for i in range(terms):
        vector[i] = (u * x ** px[i] * y ** py[i]).sum()
        for j in range(terms):
            mat[i, j] = (x ** px[i] * y ** py[i] * x ** px[j] * y ** py[j]).sum()

    imat = linalg.inv(mat)
    # Check that inversion worked
    # print sp.dot(mat,imat)
    coeffs = sp.dot(imat, vector)
    return coeffs


def polyfit2(u, x, y, order):
    """Fit polynomial to a set of u values on an x,y grid.

    u is a function u(x,y) being a polynomial of the form
    u = a[i, j] x**(i-j) y**j. x and y can be on a grid or be arbitrary values
    This version uses scipy.linalg.solve instead of matrix inversion


    Parameters
    ----------
    u      an array of values to be the results of applying the sought after
            polynomial to the values (x,y)
    x      an array of x values
    y      an array of y values
    order  an integer, the polynomial order
    u, x and y must have the same shape and may be 2D grids of values.

    Returns
    -------
    coeffs: an array of polynomial coefficients being the solution to the fit.

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
    vector = sp.zeros((terms))
    mat = sp.zeros((terms, terms))
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
    poly_degree = np.int((np.sqrt(8 * number_of_coefficients + 1) - 3) / 2)

    return poly_degree


def reorder(A, B, verbose=False):
    """Reorder Sabatke coefficients to Cox convention.

    Changes coefficient order from y**2 xy x**2 to x**2 xy y**2
    Parameters
    ----------
    A
    B
    verbose

    Returns
    -------
    A2, B2 : numpy arrays

    """
    order = 5
    terms = (order+1)*(order+2)//2
    Aarray = sp.zeros((order+1, order+1))
    Barray = sp.zeros((order+1, order+1))

    k1 = 0
    for i in range(order+1):
        for j in range(order+1-i):
            Aarray[j, i] = A[k1]
            Barray[j, i] = B[k1]
            k1 += 1

    A2 = sp.zeros((terms))
    B2 = sp.zeros((terms))
    k2 = 0
    for i in range(order+1):
        for j in range(i+1):
            A2[k2] = Aarray[j, i-j]
            B2[k2] = Barray[j, i-j]
            k2 += 1

    if verbose:
        print('A')
        print_triangle(A2, order)
        print('\nB')
        print_triangle(B2, order)

    return A2, B2


def rescale(A, B, C, D, order, scale):
    """Change coefficients to arcsec scale.

    Ported here from makeSIAF.py
    J. Sahlmann 2018-01-03
    J. Sahlmann 2018-01-04: fixed side-effect on ABCD variables

    Parameters
    ----------
    A
    B
    C
    D
    order
    scale

    Returns
    -------
    A_scaled, B_scaled, C_scaled, D_scaled : numpy arrays

    """
    A_scaled = scale*A
    B_scaled = scale*B
    number_of_coefficients = np.int((order + 1) * (order + 2) / 2)
    C_scaled = np.zeros(number_of_coefficients)
    D_scaled = np.zeros(number_of_coefficients)
    k = 0
    for i in range(order+1):
        factor = scale**i
        for j in range(i+1):
            C_scaled[k] = C[k]/factor
            D_scaled[k] = D[k]/factor
            k += 1

    return A_scaled, B_scaled, C_scaled, D_scaled


def Rotate(A, B, theta):
    """Use when a distortion transformation using polynomials A and B is followed by a rotation.

    u = A(x,y) v = B(x,y) followed by
    u2 = u*cos(theta) + v*sin(theta)
    v2 = -u*sin(theta) + v*cos(theta)
    This routine supplies a modified pair of polynomial which combine both steps
    i.e u2 = A2(x,y), v2 = B2(x,y)

    Ported to here from makeSIAF.py
    J. Sahlmann 2018-01-03

    Parameters
    ----------
    A   set of polynomial coefficients converting from (x,y) to a variable u
    B   set of polynomial coefficients converting from(x,y) to  avariable v
    theta   The angle in radians of a rotationin the (u,v) plane
    WE SHOULD REALLY CHANGE THIS SO THE INPUT IS IN DEGREES TO BE CONSISTENT WITH OTHER ROUTINES
    CONVERSION TO RADIANS SHOULD BE DONE INSIDE THIS METHOD

    Returns
    -------
    A2  set of polynomial coefficiients providing combined steps from (x,y) to u2
    B2  set of polynomial coefficients providing combined steps from (x,y) to v2

    """
    A2 = +A*np.cos(theta) + B*np.sin(theta)
    B2 = -A*np.sin(theta) + B*np.cos(theta)

    return A2, B2


def rotate_coefficients(A, B, angle_deg):
    """Version of rotate_coeffs used in nircam_get_polynomial_both.

    Parameters
    ----------
    A
    B
    angle_deg

    Returns
    -------
    AR, BR : numpy arrays

    """
    AR = A * np.cos(np.deg2rad(angle_deg)) - B * np.sin(np.deg2rad(angle_deg))
    BR = A * np.sin(np.deg2rad(angle_deg)) + B * np.cos(np.deg2rad(angle_deg))

    return AR, BR


def RotateCoeffs(a, theta, order=4, verbose=False):
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
    a          Set of polynomial coefficients
    theta      rotation angle in degrees
    order      polynomial order
    verbose    logical variable set True only if print-out of coefficient factors is desired.

    Returns
    -------
    arotate     set of coefficients derived as described above.

    """
    c = np.cos(np.deg2rad(theta))
    s = np.sin(np.deg2rad(theta))

    # First place in triangular layout
    at = triangular_layout(a)

    # Apply rotation
    atrotate = sp.zeros([order+1, order+1])
    arotate = sp.zeros([len(a)]) # Copy shape of a
    for m in range(order+1):
        for n in range(m+1):
            for mu in range(0, m-n+1):
                for j in range(m-n-mu, m-mu+1):
                    factor = (-1)**(m-n-mu) * choose(m-j, mu) * choose(j, m-n-mu)
                    cosSin = c**(j+2*mu-m+n) * s**(2*m-2*mu-j-n)
                    atrotate[m, n] = atrotate[m, n] + factor * cosSin * at[m, j]
                    if verbose: print(m, n, j, factor, 'cos^', j+2*mu-m+n, 'sin^', 2*m-2*mu-j-n, ' A', m, j)
    # Put back in linear layout
    arotate = flatten(atrotate, order)

    return arotate


def ShiftCoeffs(a, xshift, yshift, order=4, verbose=False):
    """Calculate coefficients of polynomial when shifted to new origin.

    Given a polynomial function such that u = a[i,j] * x**[i-j] * y**[j] summed over i and j
    Find the polynomial function ashift centered at xshift, yshift
    i.e the same value of u = ashift[i,j] * (x-xshift)**(i-j) * (y-yshift)**j

    Parameters
    ----------
    a:      Set of coefficients for a polynomial of the given order in JWST order
    xshift  x position in pixels of new solution center
    yshift  y position in pixels of new solution center
    order   order of the polynomial
    verbose logical variable to choose print-out of coefficient table - defaults to False

    Returns
    -------
    ashift - shifted version of the polynomial coefficients.

    """
    # First place in triangular layout
    at = triangular_layout(a)

    # Apply shift
    atshift = np.zeros((order+1, order+1))
    for p in range(order + 1):
        for q in range(p + 1):
            if verbose:
                print("A'%1d%1d" % (p, q))
            for i in range(p, order + 1):
                for j in range(q, i + 1 - (p - q)):
                    f = choose(j, q) * choose(i - j, p - q)
                    atshift[p, q] = atshift[p, q] + f * xshift**((i - j) - (p - q)) * yshift**(j - q) * at[i, j]
                    if verbose:
                        print('%2d A(%1d,%1d) x^%1d y^%1d' % (f, i, j, i - j - (p - q), (j - q)))
            if verbose:
                print()

    # Put back in linear layout
    ashift = flatten(atshift, order)

    return ashift


def TransCoeffs(A, a, b, c, d, order=4, verbose=False):
    """SUPERSEDED BY two_step.

    Transform polynomial coefficients to allow for
    xp = a*x + b*y
    yp = c*x + d*y

    Designed to work with Sabatke solutions which included a linear transformation of the pixel coordinates
    before the polynomial dostortion solution was calculated.
    TransCoeffs combines the two steps into a single polynomial

    """
    A1 = sp.zeros((order + 1, order + 1))
    A2 = sp.zeros((order + 1, order + 1))
    ncoeffs = (order + 1) * (order + 2) // 2
    if verbose:
        print(ncoeffs, 'coefficients for order', order)
    AT = sp.zeros((ncoeffs))

    # First place A in triangular layout
    k = 0
    for i in range(order + 1):
        for j in range(i + 1):
            A1[i, j] = A[k]
            k += 1

    for m in range(order + 1):
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
                    A2[m, n] += factor * a**mu * b**(m - j - mu) * c**(m - n - mu) * d**(mu + j - m + n) * A1[m, j]
                    if verbose:
                        print(m, j, ' Factor', factor)

    # Restore A2 to flat layout in AT
    k = 0
    for m in range(order + 1):
        for n in range(m + 1):
            AT[k] = A2[m, n]
            k += 1
    return AT


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


def triangular_layout(coefficients):
    """Convert linear array to 2-D array with triangular coefficient layout.

    This is the reverse of the flatten method.

    Parameters
    ----------
    a       float array of polynomial coefficients. A must be of dimension (order+1)(order+2)/2

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


def two_step(A, B, a, b, order):
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
    Internally they are processed as equivalent two dimensional arrays as described in the poly documentation

    Parameters
    ----------
    A       polynomial array converting from secondary xp and yp pixel positions to final coordinates u
    B       polynomial array converting from secondary xp and yp pixel positions to final coordinates v
    a       set of linear coefficients converting (x,y) to xp
    b       set of linear coefficients converting (x,y) to yp
    order   polynomial order

    Returns
    -------
    Aflat, Bflat    arrays of polynomials as calculated

    """
    # terms = (order+1)*(order+2)//2
    A2 = sp.zeros((order+1, order+1))
    B2 = sp.zeros((order+1, order+1))
    
    k = 0
    for i in range(order+1):
        for j in range(i+1):
            for alpha in range(i-j+1):
                for beta in range(i-j-alpha+1):
                    f1 = choose(i-j, alpha)*choose(i-j-alpha, beta)*a[0]**(i-j-alpha-beta)*a[1]**alpha*a[2]**beta
                    for gamma in range(j+1):
                        for delta in range(j-gamma+1):
                            f2 = choose(j, gamma)*choose(j-gamma, delta)*b[0]**(j-gamma-delta)*b[1]**gamma*b[2]**delta
                            A2[alpha+beta+gamma+delta, beta+delta] += A[k]*f1*f2
                            B2[alpha+beta+gamma+delta, beta+delta] += B[k]*f1*f2
            k += 1
    
    # Flatten A2 and B2
    Aflat = flatten(A2, order)
    Bflat = flatten(B2, order)

    return Aflat, Bflat
