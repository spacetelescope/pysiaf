"""A collection of functions to manipulate polynomials and their coefficients


Authors
-------

    - Colin Cox
    - Johannes Sahlmann (minor contributions and fixes)

References
----------

"""

import numpy as np
import pylab as pl
import scipy as sp
from scipy import linalg


def choose(n, r):
    """The number of ways of choosing r items from n"""
    if n < 0 or r < 0:
        print('Negative values not allowed')
        return 0
    if r > n:
        print('r must not be greater than n')
        return 0

    combin = 1
    if r > n / 2:
        r1 = n - r
    else:
        r1 = r
    for k in range(r1):
        combin = combin * (n - k) // (k + 1)

    return combin


def dpdx(a, x, y, order=4):
    """Differential with respect to x

    :param a:
    :param x:
    :param y:
    :param order:
    :return:
    """
    dpdx = 0.0
    k = 1  # index for coefficients
    for i in range(1, order + 1):
        for j in range(i + 1):
            if i - j > 0:
                dpdx = dpdx + (i - j) * a[k] * x ** (i - j - 1) * y ** j
            k += 1
    return dpdx


def dpdy(a, x, y, order=4):
    """Differential with respect to y

    :param a:
    :param x:
    :param y:
    :param order:
    :return:
    """
    dpdy = 0.0
    k = 1  # index for coefficients
    for i in range(1, order + 1):
        for j in range(i + 1):
            if j > 0:
                dpdy = dpdy + j * a[k] * x ** (i - j) * y ** (j - 1)
            k += 1
    return dpdy


def flatten(A, order):
    """Convert triangular layout to linear array"""
    terms = (order+1)*(order+2)//2
    AF = sp.zeros(terms)
    k = 0
    for i in range(order+1):
        for j in range(i+1):
            AF[k] = A[i, j]
            k += 1
    return AF        

def FlipX(A, order=4):
    """Change sign of all coefficients with odd x power"""

    terms = (order+1)*(order+2)//2
    AF = sp.zeros(terms)
    k = 0
    for i in range(order+1):
        for j in range(i+1):
            AF[k] = (-1)**(i-j)*A[k]
            k += 1
    return  AF

def FlipXY(A, order=4):
    "Change sign for coeffs where sum of x and y powers is odd"
    terms = (order+1)*(order+2)//2
    AF = sp.zeros(terms)
    k = 0
    for i in range(order+1):
        for j in range(i+1):
            AF[k] = (-1)**(i)*A[k]
            k += 1
    return AF

def FlipY(A, order = 4):
    """Change sign of all coefficients with odd y power"""

    terms = (order+1)*(order+2)//2
    AF = sp.zeros(terms)
    k = 0
    for i in range(order+1):
        for j in range(i+1):
            AF[k] = (-1)**(j)*A[k]
            k += 1
    return AF

def invert(a, b, u, v, n, verbose=False):
    """Given that order n polynomials of (x,y) have the result (u,v), find (x,y)
    Newton Raphson method in two dimensions"""

    tol = 1.0e-6
    err = 1.0
    # Initial guesses - Linear approximation
    det = a[1] * b[2] - a[2] * b[1]
    x0 = (b[2] * u - a[2] * v) / det
    y0 = (-b[1] * u + a[1] * v) / det
    if verbose:
        print('Initial guesses', x0, y0)
    x = x0
    y = y0
    X = sp.array([x, y])
    iter = 0
    while err > tol:
        f1 = sp.array([poly(a, x, y, n) - u, poly(b, x, y, n) - v])
        j = sp.array([[dpdx(a, x, y, n), dpdy(a, x, y, n)], [dpdx(b, x, y, n), dpdy(b, x, y, n)]])
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
    """Calculation of Jacobean, or relative area"""
    j = dpdx(a, x, y,order)*dpdy(b, x, y,order) - dpdx(b, x, y,order)*dpdy(a, x, y,order)
    j = sp.fabs(j)
    return j



def nircam_reorder(A, B, order):
    """Changes coefficient order from y**2 xy x**2 to x**2 xy y**2

    :param A:
    :param B:
    :param order:
    :return:
    """
    terms = (order + 1) * (order + 2) // 2
    A2 = np.zeros((terms))
    B2 = np.zeros((terms))
    for i in range(order + 1):
        ti = i * (i + 1) // 2
        for j in range(i + 1):
            A2[ti + j] = A[ti + i - j]
            B2[ti + j] = B[ti + i - j]

    return (A2, B2)

def poly(a, x, y, order=4):
    """Return polynomial

    :param a:
    :param x:
    :param y:
    :param order:
    :return:
    """

    pol = 0.0
    k = 0 # index for coefficients
    for i in range(order+1):
        for j in range(i+1):
            pol = pol + a[k]*x**(i-j)*y**j
            k+=1
    return pol

def polyfit(u, x, y, order):
    """Fit polynomial to a set of u values on an x,y grid
    u is a function u(x,y) being a polynomial of the form
    u = a[i, j] x**(i-j) y**j. x and y can be on a grid or be arbitrary values"""

    # First set up x and y powers for each coefficient
    px = []
    py = []
    for i in range(order + 1):
        for j in range(i + 1):
            px.append(i - j)
            py.append(j)
    terms = len(px)
    # print terms, ' terms for order ', order
    # print px
    # print py

    # Make up matrix and vector
    vector = sp.zeros((terms))
    mat = sp.zeros((terms, terms))
    for i in range(terms):
        vector[i] = (u * x ** px[i] * y ** py[i]).sum()
        for j in range(terms):
            mat[i, j] = (x ** px[i] * y ** py[i] * x ** px[j] * y ** py[j]).sum()

    # print 'Vector', vector
    # print 'Matrix'
    # print mat
    imat = linalg.inv(mat)
    # print 'Inverse'
    # print imat
    # Check that inversion worked
    # print sp.dot(mat,imat)
    coeffs = sp.dot(imat, vector)
    return coeffs

def polyfit2(u, x, y, order):
    """Fit polynomial to a set of u values on an x,y grid
    u is a function u(x,y) being a polynomial of the form
    u = a[i, j]x**(i-j)y**j. x and y can be on a grid or be arbitrary values
    This version uses solve instead of matrix inversion"""

    # First set up x and y powers for each coefficient
    px = []
    py = []
    for i in range(order + 1):
        for j in range(i + 1):
            px.append(i - j)
            py.append(j)
    terms = len(px)
    # print terms, ' terms for order ', order
    # print px
    # print py

    # Make up matrix and vector
    vector = sp.zeros((terms))
    mat = sp.zeros((terms, terms))
    for i in range(terms):
        vector[i] = (u * x ** px[i] * y ** py[i]).sum()  # Summing over all x,y
        for j in range(terms):
            mat[i, j] = (x ** px[i] * y ** py[i] * x ** px[j] * y ** py[j]).sum()

    coeffs = linalg.solve(mat, vector)
    return coeffs



def reorder(A, B, verbose=False) :
    """Reorder Sabatke coefficients to Cox convention"""

    order = 5
    terms = (order+1)*(order+2)//2
    Aarray = sp.zeros((order+1,order+1))
    Barray = sp.zeros((order+1,order+1))

    k1 = 0
    for i in range(order+1):
        for j in range(order+1-i):
            Aarray[j,i] = A[k1]
            Barray[j,i] = B[k1]
            k1 += 1

    A2 = sp.zeros((terms))
    B2 = sp.zeros((terms))
    k2 = 0
    for i in range(order+1):
        for j in range(i+1):
            A2[k2] = Aarray[j,i-j]
            B2[k2] = Barray[j,i-j]
            k2 += 1

    if verbose:
        print('A')
        triangle(A2, order)
        print('\nB')
        triangle(B2, order)

    return (A2, B2)

def rescale(A, B, C, D, order, scale):
    """
    Change coefficients to arcsec scale

    Ported here from makeSIAF.py
    J. Sahlmann 2018-01-03
    J. Sahlmann 2018-01-04: fixed side-effect on ABCD variables

    :param A:
    :param B:
    :param C:
    :param D:
    :param order:
    :param scale:
    :return:
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

def Rotate(A,B,theta):
    """
    Ported to here from makeSIAF.py
    J. Sahlmann 2018-01-03

    :param A:
    :param B:
    :param theta:
    :return:
    """
    A2 =   A*np.cos(theta) + B*np.sin(theta)
    B2 = - A*np.sin(theta) + B*np.cos(theta)
    return (A2,B2)

def rotate_coefficients(A, B, angle_deg):
    """ J. Sahlmann: this version of rotate_coeffs is used in nircam_get_polynomial_both

    :param A:
    :param B:
    :param angle_deg:
    :return:
    """
    AR = A * np.cos(np.deg2rad(angle_deg)) - B * np.sin(np.deg2rad(angle_deg))
    BR = A * np.sin(np.deg2rad(angle_deg)) + B * np.cos(np.deg2rad(angle_deg))
    return AR, BR

def RotateCoeffs(a, theta, order=4, verbose=False):
    """Rotate axes of coefficients by theta degrees"""
    c = np.cos(np.deg2rad(theta))
    s = np.sin(np.deg2rad(theta))

    # First place in triangular layout
    at = sp.zeros([order+1,order+1])
    k = 0
    for m in range(order+1):
        for n in range(m+1):
            at[m, n] = a[k]
            k+=1

    # Apply rotation
    atrotate = sp.zeros([order+1,order+1])
    arotate = sp.zeros([len(a)]) # Copy shape of a
    for m in range(order+1):
        for n in range(m+1):
            for mu in range(0,m-n+1):
                for j in range(m-n-mu, m-mu+1):
                    factor = (-1)**(m-n-mu)*choose(m-j, mu)*choose(j, m-n-mu)
                    cosSin = c**(j+2*mu-m+n)*s**(2*m-2*mu-j-n)
                    atrotate[m, n] = atrotate[m, n] + factor*cosSin*at[m, j]
                    if verbose: print(m, n, j, factor, 'cos^', j+2*mu-m+n, 'sin^',2*m-2*mu-j-n, ' A',m, j)
    # Put back in linear layout
    k = 0
    for m in range(order+1):
        for n in range(m+1):
            arotate[k] = atrotate[m, n]
            k+=1

    return arotate


def ShiftCoeffs(a, xshift, yshift, order=4, verbose=False):
    """Calculate coefficients of polynomial when shifted to new origin"""

    # First place in triangular layout
    at = sp.zeros([order + 1, order + 1])
    atshift = sp.zeros([order + 1, order + 1])
    ashift = sp.zeros([len(a)])  # Copy shape of a
    k = 0
    for p in range(order + 1):
        for q in range(p + 1):
            at[p, q] = a[k]
            k += 1

    # Apply shift
    for p in range(order + 1):
        for q in range(p + 1):
            if verbose:
                print("A'%1d%1d" % (p, q))
            for i in range(p, order + 1):
                for j in range(q, i + 1 - (p - q)):
                    f = choose(j, q) * choose(i - j, p - q)
                    atshift[p, q] = atshift[p, q] + f * xshift ** ((i - j) - (p - q)) * yshift ** (
                    j - q) * at[i, j]
                    if verbose:
                        print('%2d A(%1d,%1d) x^%1d y^%1d' % (f, i, j, i - j - (p - q), (j - q)))
            if verbose:
                print()

    # Put back in linear layout
    k = 0
    for p in range(order + 1):
        for q in range(p + 1):
            ashift[k] = atshift[p, q]
            k += 1

    return ashift

def testpoly():
    [x, y] = sp.mgrid[0:10, 0:10]
    # print 'X'
    # print x
    # print 'Y'
    # print y
    u = sp.zeros((10, 10))
    v = sp.zeros((10, 10))
    # Random polynomials
    a0 = sp.random.rand(1)
    a1 = 0.1 * (sp.random.rand(2) - 0.5)
    a2 = 0.01 * (sp.random.rand(3) - 0.5)
    a = sp.concatenate((a0, a1))
    a = sp.concatenate((a, a2))
    a[2] = 0.01 * a[2]
    print('A coefficients')
    print(a)
    b0 = sp.random.rand(1)
    b1 = 0.1 * (sp.random.rand(2) - 0.5)
    b2 = 0.01 * (sp.random.rand(3) - 0.5)
    b = sp.concatenate((b0, b1))
    b = sp.concatenate((b, b2))
    b[1] = 0.01 * b[1]
    print('B coeffcicients')
    print(b)
    for i in range(10):
        for j in range(10):
            u[i, j] = poly(a, x[i, j], y[i, j], 2)  # + sp.random.normal(0.0, 0.01)
            v[i, j] = poly(b, x[i, j], y[i, j], 2)  # + sp.random.normal(0.0,0.01)
    # print z
    s1 = polyFit2(u, x, y, 2)
    s2 = polyFit2(v, x, y, 2)
    print('S1', s1)
    print('S2', s2)
    uc = poly(s1, x, y, 2)
    vc = poly(s2, x, y, 2)

    pl.figure(1)
    pl.clf()
    pl.grid(True)
    pl.plot(u, v, 'gx')
    pl.plot(uc, vc, 'r+')


def TransCoeffs(A, a, b, c, d, order=4, verbose=False):
    """Transform polynomial coefficients to allow for
    xp = a*x + b*y
    yp = c*x + d*y"""

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
                    A2[m, n] += factor * a ** mu * b ** (m - j - mu) * c ** (m - n - mu) * d ** (
                    mu + j - m + n) * A1[m, j]
                    if verbose:
                        print(m, j, ' Factor', factor)
    # Restore A2 to flat layout in AT
    k = 0
    for m in range(order + 1):
        for n in range(m + 1):
            AT[k] = A2[m, n]
            k += 1
    return AT


def triangle(A, order=4):
    """Print coefficients in triangular layout"""
    k = 0
    for i in range(order + 1):
        for j in range(i + 1):
            print('%12.5e' % A[k], end=' ')
            k += 1
        print()

def triangulate(A, order):
    """Convert linear array to 2-D array with triangular coefficient layout"""
    AT = sp.zeros((order + 1, order + 1))
    k = 0
    for i in range(order + 1):
        for j in range(i + 1):
            AT[i, j] = A[k]
            k += 1
    return AT

def two_step(A, B, a, b, order):
    """
    Change coefficients when
    xp = a[0] + a[1].x + a[2].y
    yp = b[0] + b[1].x + b[2].y

    :param A:
    :param B:
    :param a:
    :param b:
    :param order:
    :return:
    """

    terms = (order+1)*(order+2)//2
    A2 = sp.zeros((order+1,order+1))
    B2 = sp.zeros((order+1,order+1))
    
    k=0
    for i in range(order+1):
        for j in range(i+1):
            for alpha in range(i-j+1):
                for beta in range(i-j-alpha+1):
                    f1 = choose(i-j,alpha)*choose(i-j-alpha, beta)*a[0]**(i-j-alpha-beta)*a[1]**alpha*a[2]**beta
                    for gamma in range(j+1):
                        for delta in range(j-gamma+1):
                            f2 = choose(j,gamma)*choose(j-gamma,delta)*b[0]**(j-gamma-delta)*b[1]**gamma*b[2]**delta
                            A2[alpha+beta+gamma+delta, beta+delta] += A[k]*f1*f2
                            B2[alpha+beta+gamma+delta, beta+delta] += B[k]*f1*f2
            k += 1
    
    # Flatten A@ and B2                    
    k = 0
    Aflat = sp.zeros(terms)
    Bflat = sp.zeros(terms)
    for i in range(order+1):
        for j in range(i+1):
            Aflat[k] = A2[i, j]
            Bflat[k] = B2[i, j]
            k += 1                      
    return (Aflat, Bflat)     
    
# def TestTwoStep():
#     A = sp.array([10.0, 2.0, 0.1, 0.01, -0.02, 0.03])
#     B = sp.array([4.0, 1.8, 0.2, 0.02, 0.03, -0.02])
#     a = sp.array([1.0, 0.5, 0.1])
#     b = sp.array([2.0, 0.2, 0.6])
#     print('\nA')
#     triangle(A,2)
#     print('B')
#     triangle(B,2)
#     print('a\n',a)
#     print('b\n', b)
#     (A2, B2) = TwoStep(A,B,a, b,2)
#     print('\nA2')
#     triangle(A2,2)
#     print('B2')
#     triangle(B2,2)
#
#     # Now do a test calculation
#     (x,y) = (10,5)
#     xp = a[0] + a[1]*x + a[2]*y
#     yp = b[0] + b[1]*x + b[2]*y
#     print('x,y', x,y)
#     print('xp,yp', xp,yp)
#
#     u = poly(A, xp, yp, 2)
#     v = poly(B, xp, yp, 2)
#     up = poly(A2, x, y,2)
#     vp = poly(B2, x, y,2)
#     print('Two step', u, v)
#     print('One step', up, vp)
#     return