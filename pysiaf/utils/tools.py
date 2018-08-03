"""A collection of helper functions to support pysiaf calculations and functionalities

Authors
-------

    - Johannes Sahlmann

References
----------

    Some functions were adapted from Colin Cox' nircamtrans.py


"""
from __future__ import absolute_import, print_function, division
import copy
import math
<<<<<<< HEAD
import sys
import pylab as pl
from math import sin, cos, radians, degrees, atan2
=======
>>>>>>> bf40d17a6056c2a50d0671625ff19bcdbaff08b3

# from astropy.table import Table
import numpy as np

# from ..aperture import PRD_REQUIRED_ATTRIBUTES_ORDERED
from ..constants import V3_TO_YAN_OFFSET_DEG
from ..iando import read
from .polynomial import ShiftCoeffs, FlipY, FlipX, rotate_coefficients, RotateCoeffs, poly, triangle


def an_to_tel(xan_arcsec, yan_arcsec):
    """Convert from XAN, YAN to V2, V3."""

    v2_arcsec = xan_arcsec
    v3_arcsec = -1*yan_arcsec - V3_TO_YAN_OFFSET_DEG * 3600

    return v2_arcsec, v3_arcsec

def tel_to_an(v2_arcsec, v3_arcsec):
    """Convert from V2, V3 to XAN, YAN."""

    xan_arcsec = v2_arcsec
    yan_arcsec = -1*v3_arcsec - V3_TO_YAN_OFFSET_DEG * 3600

    return xan_arcsec, yan_arcsec

<<<<<<< HEAD
def compute_roundtrip_error(A: object, B: object, C: object, D: object, verbose: object = False, instrument: object = None) -> object:
    """Test whether the forward and inverse transformations are consistent.
=======
def compute_roundtrip_error(A, B, C, D, offset_x=0., offset_y=0., verbose=False, instrument=''):
    """Test whether the forward and inverse idl-sci transformations are consistent.
>>>>>>> bf40d17a6056c2a50d0671625ff19bcdbaff08b3

    Adapted from Cox' checkinv

    Parameters
    ----------
    A
    B
    C
    D
    order

    Returns
    -------

    """
    number_of_coefficients = len(A)
    polynomial_degree = np.int((np.sqrt(8 * number_of_coefficients + 1) - 3) / 2)
    order = polynomial_degree

<<<<<<< HEAD
    # regular grid of points in the full frame science frame
    if instrument is not None and instrument.lower() =='miri':
        grid_amplitude = 1024
    else:
        grid_amplitude = 2048
    x, y = get_grid_coordinates(9, (0,0), grid_amplitude)

    # transform in one direction
    u = poly(A, x, y, order)
    v = poly(B, x, y, order)

    # transform in the opposite direction
    x2 = poly(C, u, v, order)
    y2 = poly(D, u, v, order)
    dx = x2-x
    dy = y2-y
    length = np.hypot(dx, dy)
    maxlength = length.max()
=======
    # regular grid of points (in science pixel coordinates) in the full frame science frame
    # if instrument is None:
    grid_amplitude = 2048
    if instrument.lower() == 'miri':
        grid_amplitude = 1024
    x, y = get_grid_coordinates(10, (grid_amplitude/2+1,grid_amplitude/2+1), grid_amplitude)

    x_in = x - offset_x
    y_in = y - offset_y

    # transform in one direction
    u = poly(A, x_in, y_in, order)
    v = poly(B, x_in, y_in, order)

    # transform back the opposite direction
    x_out = poly(C, u, v, order)
    y_out = poly(D, u, v, order)

    x2 = x_out + offset_x
    y2 = y_out + offset_y
>>>>>>> bf40d17a6056c2a50d0671625ff19bcdbaff08b3

    if verbose:
        print ('\nInverse Check')
        for p in range(len(x)):
<<<<<<< HEAD
            print (8*'%10.3f' %(x[p],y[p], u[p],v[p], x2[p],y2[p], dx[p], dy[p]))

        # Make a distortion plot
        pl.figure()
        pl.axis('equal')
        pl.grid(True)
        pl.quiver(x, y, dx, dy)
        pl.xlabel('Longest arrow%6.3f pixels' %maxlength)

    # coordinate differences
    # h = np.hypot(dx,dy)
    # rms_deviation = np.sqrt((h**2).mean())
=======
            print (8*'%10.3f' %(x[p], y[p], u[p], v[p], x2[p], y2[p], x2[p]-x[p], y2[p]-y[p]))


    data = {}
    data['x'] = x
    data['y'] = y
    data['x2'] = x2
    data['y2'] = y2

    # absolute coordinate differences
    dx = np.abs(x2-x)
    dy = np.abs(y2-y)

>>>>>>> bf40d17a6056c2a50d0671625ff19bcdbaff08b3
    if verbose:
        print(4*'%12.3e' %(dx.mean(), dy.mean(), dx.std(), dy.std()))
        #print ('RMS deviation %5.3f' %rms_deviation)

    # compute one number that indicates if something may be wrong
    error_estimation_metric = np.abs(dx.mean()/dx.std()) + np.abs(dx.mean()/dx.std())
    return error_estimation_metric, dx.mean(), dy.mean(), dx.std(), dy.std(), data


def convert_polynomial_coefficients(A_in, B_in, C_in, D_in, oss=False, inverse=False,
                                    parent_aperture=None, verbose=False):
    """Emulate some transformation made in nircam_get_polynomial_both.
    Written by Johannes Sahlmann 2018-02-18, structure largely based on nircamtrans.py code
    written by Colin Cox.

    Parameters
    ----------
    A_in
    B_in
    C_in
    D_in
    oss
    inverse
    parent_aperture
    verbose

    Returns
    -------

    """
    if inverse is False:
        # forward direction
        V2Ref = A_in[0]
        V3Ref = B_in[0]
        A_in[0] = 0.0
        B_in[0] = 0.0

        V3SciXAngle = np.rad2deg(np.arctan2(A_in[1], B_in[1]))  # V3SciXAngle
        V3SciYAngle = np.rad2deg(np.arctan2(A_in[2], B_in[2]))

        V3Angle = V3SciYAngle  # V3SciYAngle
        if abs(V3Angle) > 90.0:
            V3Angle = V3Angle - math.copysign(180.0, V3Angle)

        AR, BR = rotate_coefficients(A_in, B_in, V3Angle)

        CS = ShiftCoeffs(C_in, V2Ref, V3Ref, 5)
        DS = ShiftCoeffs(D_in, V2Ref, V3Ref, 5)

        CR = RotateCoeffs(CS, V3Angle, 5)
        DR = RotateCoeffs(DS, V3Angle, 5)

        if oss:
            # OSS apertures
            V3Angle = copy.deepcopy(V3SciYAngle)
        else:
            # non-OSS apertures
            if abs(V3SciYAngle) > 90.0:  # e.g. NRCA2_FULL
                # print 'Reverse Y axis direction'
                AR = -FlipY(AR, 5)
                BR = FlipY(BR, 5)
                CR = FlipX(CR, 5)
                DR = -FlipX(DR, 5)

            else:  # e.g NRCA1_FULL
                # print 'Reverse X axis direction'
                AR = -FlipX(AR, 5)
                BR = FlipX(BR, 5)
                CR = -FlipX(CR, 5)
                DR = FlipX(DR, 5)
                V3SciXAngle = V3SciXAngle - math.copysign(180.0, V3SciXAngle)
                # V3Angle = betaY   # Cox: Changed 4/29 - might affect rotated polynomials

        V3SciYAngle = V3Angle

        return AR, BR, CR, DR, V3SciXAngle, V3SciYAngle, V2Ref, V3Ref

    else:
        siaf_detector_layout = read.read_siaf_detector_layout()
        master_aperture_names = siaf_detector_layout['AperName'].data
        if parent_aperture.AperName not in master_aperture_names:
            raise RuntimeError
        polynomial_degree = parent_aperture.Sci2IdlDeg
        V3SciYAngle = copy.deepcopy(parent_aperture.V3SciYAngle)  # betaY
        V3SciXAngle = parent_aperture.V3SciXAngle  # betaX

        betaY = V3SciYAngle + parent_aperture.DetSciYAngle

        # master aperture is never OSS
        if abs(betaY) > 90.0:  # e.g. NRCA2_FULL
            # print 'Reverse Y axis direction'
            AR = -FlipY(A_in, polynomial_degree)
            BR = FlipY(B_in, polynomial_degree)
            CR = FlipX(C_in, polynomial_degree)
            DR = -FlipX(D_in, polynomial_degree)

        else:  # e.g NRCA1_FULL
            # print 'Reverse X axis direction'
            AR = -FlipX(A_in, polynomial_degree)
            BR = FlipX(B_in, polynomial_degree)
            CR = -FlipX(C_in, polynomial_degree)
            DR = FlipX(D_in, polynomial_degree)
            V3SciXAngle = revert_correct_V3SciXAngle(V3SciXAngle)

        # rotate the other way
        A, B = rotate_coefficients(AR, BR, -V3SciYAngle)

        A[0] = parent_aperture.V2Ref
        B[0] = parent_aperture.V3Ref

        # now invert the last part of nircam_get_polynomial_forward
        AFS = A
        BFS = B

        # shift by parent aperture reference point
        AF = ShiftCoeffs(AFS, -parent_aperture.XDetRef, -parent_aperture.YDetRef, polynomial_degree)
        BF = ShiftCoeffs(BFS, -parent_aperture.XDetRef, -parent_aperture.YDetRef, polynomial_degree)

        CS = RotateCoeffs(CR, -V3SciYAngle, polynomial_degree)
        DS = RotateCoeffs(DR, -V3SciYAngle, polynomial_degree)

        C = ShiftCoeffs(CS, -parent_aperture.V2Ref, -parent_aperture.V3Ref, polynomial_degree)
        D = ShiftCoeffs(DS, -parent_aperture.V2Ref, -parent_aperture.V3Ref, polynomial_degree)

        C[0] += parent_aperture.XDetRef
        D[0] += parent_aperture.YDetRef

        return AF, BF, C, D


def correct_V3SciXAngle(V3SciXAngle_deg):
    """Correct input angle.

    Parameters
    ----------
    V3SciXAngle_deg

    Returns
    -------
    V3SciXAngle_deg : float

    """
    V3SciXAngle_deg = V3SciXAngle_deg - math.copysign(180.0, V3SciXAngle_deg)
    return V3SciXAngle_deg


def correct_V3SciYAngle(V3SciYAngle_deg):
    """Correct input angle.

    Parameters
    ----------
    V3SciYAngle_deg

    Returns
    -------
    V3SciYAngle_deg_corrected : float

    """
    if np.abs(V3SciYAngle_deg) > 90.0:
        V3SciYAngle_deg_corrected = V3SciYAngle_deg - math.copysign(180.0, V3SciYAngle_deg)
    return V3SciYAngle_deg_corrected


def get_grid_coordinates(n_side, centre, x_width, y_width=None):
    """Return tuple of arrays that contain the coordinates on a regular grid.

    Parameters
    ----------
    n_side: int
        Number of points per side. The returned arrays have n_side**2 entries.
    centre: tuple of floats
        Center coordinate
    x_width: float
        Extent of the grid in the first dimension
    t_width: float
        Extent of the grid in the second dimension

    Returns
    -------
    x : array
    y : array

    """

    if y_width is None:
        y_width = x_width

    x_linear = np.linspace(centre[0] - x_width/2, centre[0] + x_width/2, n_side)
    y_linear = np.linspace(centre[1] - y_width/2, centre[1] + y_width/2, n_side)

    # coordinates on grid
    x_mesh, y_mesh = np.meshgrid(x_linear, y_linear)
    x = x_mesh.flatten()
    y = y_mesh.flatten()

    return x, y


def revert_correct_V3SciYAngle(V3SciYAngle_deg):
    """ Only correct if the original V3SciYAngle in [0,180) deg

    :param V3SciYAngle_deg:
    :return:
    """
    if V3SciYAngle_deg < 0.:
        V3SciYAngle_deg += 180.
    return V3SciYAngle_deg


def revert_correct_V3SciXAngle(V3SciXAngle_deg):
    """

    :param V3SciXAngle_deg:
    :return:
    """
    # if V3SciXAngle_deg < 0.:
    V3SciXAngle_deg += 180.
    return V3SciXAngle_deg


def set_reference_point_and_distortion(instrument, aperture, parent_aperture):
    """Compute V2Ref and V3ref and distortion polynomial for an aperture that depends on a parent_aperture

    :param aperture:
    :param parent_aperture:
    :return:
    """

    polynomial_degree = parent_aperture.Sci2IdlDeg
    number_of_coefficients = np.int((polynomial_degree + 1) * (polynomial_degree + 2) / 2)

    k = 0
    sci2idlx_coefficients = np.zeros(number_of_coefficients)
    sci2idly_coefficients = np.zeros(number_of_coefficients)
    idl2scix_coefficients = np.zeros(number_of_coefficients)
    idl2sciy_coefficients = np.zeros(number_of_coefficients)
    for i in range(polynomial_degree + 1):
        for j in np.arange(i + 1):
            sci2idlx_coefficients[k] = getattr(parent_aperture, 'Sci2IdlX{:d}{:d}'.format(i, j))
            sci2idly_coefficients[k] = getattr(parent_aperture, 'Sci2IdlY{:d}{:d}'.format(i, j))
            idl2scix_coefficients[k] = getattr(parent_aperture, 'Idl2SciX{:d}{:d}'.format(i, j))
            idl2sciy_coefficients[k] = getattr(parent_aperture, 'Idl2SciY{:d}{:d}'.format(i, j))
            k += 1

    if instrument in ['NIRISS', 'FGS']:
        # see calc worksheet in SIAFEXCEL (e.g. column E)
        xsci_offset = (aperture.XDetRef - parent_aperture.XDetRef) * np.cos(np.deg2rad(aperture.DetSciYAngle))
        ysci_offset = (aperture.YDetRef - parent_aperture.YDetRef) * np.cos(np.deg2rad(aperture.DetSciYAngle))


        # shift polynomial coefficients of the parent aperture
        sci2idlx_coefficients_shifted = ShiftCoeffs(sci2idlx_coefficients, xsci_offset, ysci_offset, order=4, verbose=False)
        sci2idly_coefficients_shifted = ShiftCoeffs(sci2idly_coefficients, xsci_offset, ysci_offset, order=4, verbose=False)

        # see calc worksheet in NIRISS SIAFEXCEL
        dx_idl = sci2idlx_coefficients_shifted[0]
        dy_idl = sci2idly_coefficients_shifted[0]

        # remove the zero point offsets from the coefficients
        idl2scix_coefficients_shifted = ShiftCoeffs(idl2scix_coefficients, dx_idl, dy_idl, order=4, verbose=False)
        idl2sciy_coefficients_shifted = ShiftCoeffs(idl2sciy_coefficients, dx_idl, dy_idl, order=4, verbose=False)

        # set 00 coefficient to zero
        sci2idlx_coefficients_shifted[0] = 0
        sci2idly_coefficients_shifted[0] = 0
        idl2scix_coefficients_shifted[0] = 0
        idl2sciy_coefficients_shifted[0] = 0

        # set polynomial coefficients
        k = 0
        for i in range(polynomial_degree + 1):
            for j in np.arange(i + 1):
                setattr(aperture, 'Sci2IdlX{:d}{:d}'.format(i, j), sci2idlx_coefficients_shifted[k])
                setattr(aperture, 'Sci2IdlY{:d}{:d}'.format(i, j), sci2idly_coefficients_shifted[k])
                setattr(aperture, 'Idl2SciX{:d}{:d}'.format(i, j), idl2scix_coefficients_shifted[k])
                setattr(aperture, 'Idl2SciY{:d}{:d}'.format(i, j), idl2sciy_coefficients_shifted[k])
                k += 1

        # set V2ref and V3Ref (this is a standard process for child apertures and should be generalized in a function)
        aperture.V2Ref = parent_aperture.V2Ref + aperture.VIdlParity * dx_idl * np.cos(
            np.deg2rad(aperture.V3IdlYAngle)) + dy_idl * np.sin(np.deg2rad(aperture.V3IdlYAngle))
        aperture.V3Ref = parent_aperture.V3Ref - aperture.VIdlParity * dx_idl * np.sin(
            np.deg2rad(aperture.V3IdlYAngle)) + dy_idl * np.cos(np.deg2rad(aperture.V3IdlYAngle))

    elif instrument == 'NIRCam':
        # do the inverse of what nircam_get_polynomial_both does for the master aperture, then forward for the child aperture

        A, B, C, D = sci2idlx_coefficients, sci2idly_coefficients, idl2scix_coefficients, idl2sciy_coefficients
        AF, BF, CF, DF = convert_polynomial_coefficients(A, B, C, D, inverse=True, parent_aperture=parent_aperture)


        # now shift to child aperture reference point
        AFS_child = ShiftCoeffs(AF, aperture.XDetRef, aperture.YDetRef, polynomial_degree)
        BFS_child = ShiftCoeffs(BF, aperture.XDetRef, aperture.YDetRef, polynomial_degree)
        CFS_child = CF
        DFS_child = DF
        CFS_child[0] -= aperture.XDetRef
        DFS_child[0] -= aperture.YDetRef

        if aperture.AperType == 'OSS':
            oss = True
        else:
            oss = False

        AR, BR, CR, DR, V3SciXAngle, V3SciYAngle, V2Ref, V3Ref = convert_polynomial_coefficients(AFS_child, BFS_child, CFS_child, DFS_child, oss=oss)

        aperture.V2Ref = V2Ref
        aperture.V3Ref = V3Ref
        aperture.V3SciXAngle = V3SciXAngle
        aperture.V3SciYAngle = V3SciYAngle

        # set polynomial coefficients
        k = 0
        for i in range(polynomial_degree + 1):
            for j in np.arange(i + 1):
                setattr(aperture, 'Sci2IdlX{:d}{:d}'.format(i, j), AR[k])
                setattr(aperture, 'Sci2IdlY{:d}{:d}'.format(i, j), BR[k])
                setattr(aperture, 'Idl2SciX{:d}{:d}'.format(i, j), CR[k])
                setattr(aperture, 'Idl2SciY{:d}{:d}'.format(i, j), DR[k])
                k += 1

        if np.abs(aperture.V3SciYAngle) < 90.:
            aperture.V3IdlYAngle = aperture.V3SciYAngle
        else:
            aperture.V3IdlYAngle = aperture.V3SciYAngle - np.sign(aperture.V3SciYAngle)*180.

    return aperture


def v3sciyangle_to_v3idlyangle(v3sciyangle):
    """Convert V3SciYAngle to V3IdlYAngle.

    :param v3sciyangle: angle in degree
    :return: v3idlyangle: angle in deg
    """

    if np.abs(v3sciyangle) < 90.:
        v3idlyangle = v3sciyangle
    else:
        v3idlyangle = v3sciyangle - np.sign(v3sciyangle) * 180.

    return v3sciyangle


<<<<<<< HEAD
def compare_apertures(reference_aperture, comparison_aperture, absolute_tolerance=None, attribute_list=None, print_file=sys.stdout, fractional_tolerance=1e-6, verbose=False):
    """Compare the attributes of two apertures.

    Parameters
    ----------
    reference_aperture
    comparison_aperture
    absolute_tolerance
    attribute_list
    print_file
    fractional_tolerance
    verbose

    Returns
    -------

    """
    if attribute_list is None:
        attribute_list = PRD_REQUIRED_ATTRIBUTES_ORDERED

    comparison_table = Table(names=('aperture', 'attribute', 'reference', 'comparison', 'difference', 'percent'), dtype=['S50']*6)

    add_blank_line = False
    for attribute in attribute_list:
        show = False
        reference_attr = getattr(reference_aperture, attribute)
        comparison_attr = getattr(comparison_aperture, attribute)
        if verbose:
            print('Comparing {} {}: {}{} {}{}'.format(reference_aperture, attribute, type(reference_attr), reference_attr, type(comparison_attr), comparison_attr))
        if reference_attr != comparison_attr:
            show = True
            # if isinstance(reference_attr, float) and isinstance(comparison_attr, float):
            if (type(reference_attr) in [int, float, np.float64]) and (type(comparison_attr) in [int, float, np.float64]):
                difference = np.abs(comparison_attr - reference_attr)
                fractional_difference = difference / np.max(
                    [np.abs(reference_attr), np.abs(comparison_attr)])
                if verbose:
                    print('difference={}, fractional_difference={}'.format(difference, fractional_difference))
                if (absolute_tolerance is not None) and math.isclose(reference_attr, comparison_attr, abs_tol=absolute_tolerance):
                    show = False
                elif fractional_difference <= fractional_tolerance:
                    show = False
                else:
                    fractional_difference_percent_string = '{:.4f}'.format(fractional_difference*100.)
                    difference_string = '{:.6f}'.format(difference)
            else:
                difference_string = 'N/A'
                fractional_difference_percent_string = 'N/A'

        if show:
            add_blank_line = True
            print('{:25} {:>15} {:>21} {:>21} {:>15} {:>10}'.format(reference_aperture.AperName, attribute, str(reference_attr), str(comparison_attr), difference_string, fractional_difference_percent_string), file=print_file)
            # add comparison data to table
            comparison_table.add_row([reference_aperture.AperName, attribute, str(reference_attr), str(comparison_attr), difference_string, fractional_difference_percent_string])

    if add_blank_line:
        print('', file=print_file)

    return comparison_table


def match_v2v3(aperture_1, aperture_2, verbose=False):
    """ Use the V2V3 from aperture_1 in aperture_2  modifying XDetRef, YDetReƒ,
    XSciRef YSciRef to match
    Also shift the polynomial coefficients to reflect the new reference point origin
    and for NIRCam recalculate angles. """

    instrument = aperture_1.InstrName
    assert (aperture_2.AperType in ['FULLSCA', 'SUBARRAY', 'ROI']), "2nd aperture must be pixel-based"
    order = aperture_1.Sci2IdlDeg
    V2Ref1 = aperture_1.V2Ref
    V3Ref1 = aperture_1.V3Ref
    newV2Ref = V2Ref1 # in all cases
    newV3Ref = V3Ref1
    print('Current Vref', aperture_2.V2Ref, aperture_2.V3Ref)
    print('Shift to    ', V2Ref1, V3Ref1)

    # Detector and Science axes may go in opposite directions
    ySign = cos(radians(aperture_2.DetSciYAngle))
    xSign = aperture_2.DetSciParity * ySign

    # Need to work in aperture 2  coordinate systems
    print('Detector 1', aperture_1.AperName[:5], '  Detector 2', aperture_2.AperName[:5])
    V2Ref2 = aperture_2.V2Ref
    V3Ref2 = aperture_2.V3Ref
    theta0 = aperture_2.V3IdlYAngle
    print('Initial VRef', V2Ref2, V3Ref2)
    print('Initial theta', theta0)
    theta = radians(theta0)

    coefficients = aperture_2.get_polynomial_coefficients()
    A = coefficients['Sci2IdlX']
    B = coefficients['Sci2IdlY']
    C = coefficients['Idl2SciX']
    D = coefficients['Idl2SciY']

    if verbose:
        print('\nA')
        triangle(A, order)
        print('B')
        triangle(B, order)
        print('C')
        triangle(C, order)
        print('D')
        triangle(D, order)

    (stat, xmean, ymean, xstd, ystd) = compute_roundtrip_error(A, B, C, D,
                                                                     verbose=True, instrument = instrument)
    print('Round trip     X       Y')
    print('     Means%8.4F %8.4f' %(xmean, ymean))
    print('      STDs%8.4f %8.4f' %(xstd, ystd))

    # Use convert
    print('\nUsing Convert')
    print('VRef1', V2Ref1, V3Ref1)
    (newXSci, newYSci) = aperture_2.convert(V2Ref1, V3Ref1, 'tel', 'sci')
    (newXDet, newYDet) = aperture_2.convert(V2Ref1, V3Ref1, 'tel', 'det')
    print('Sci', newXSci, newYSci)
    print('Det', newXDet, newYDet)
    (newXIdl, newYIdl) = aperture_2.convert(V2Ref1, V3Ref1, 'tel', 'idl')
    print('Idl', newXIdl, newYIdl)

    # Convert back
    (v2c, v3c) = aperture_2.convert(newXSci, newYSci, 'sci', 'tel')
    print('Regained values     ', v2c, v3c)

    dXSciRef = newXSci - aperture_2.XSciRef
    dYSciRef = newYSci - aperture_2.YSciRef
    print('Shift pixel origin by', dXSciRef, dYSciRef)
    AS = ShiftCoeffs(A, dXSciRef, dYSciRef, order)
    BS = ShiftCoeffs(B, dXSciRef, dYSciRef, order)
    print('New Ideal origin', newXIdl, newYIdl)
    CS = ShiftCoeffs(C, AS[0], BS[0], order)
    DS = ShiftCoeffs(D, AS[0], BS[0], order)
    AS[0] = 0.0
    BS[0] = 0.0
    CS[0] = 0.0
    DS[0] = 0.0
    if verbose:
        print('\nShifted Polynomials')
        print('AS')
        triangle(AS, order)
        print('BS')
        triangle(BS, order)
        print('CS')
        triangle(CS, order)
        print('DS')
        triangle(DS, order)
        print('\nABCDS')

    (stat, xmean, ymean, xstd, ystd) = compute_roundtrip_error(AS, BS, CS, DS,
                                                                     verbose=True, instrument=instrument)
    if verbose:
        print('Round trip     X       Y')
        print('     Means%8.4F %8.4f' %(xmean, ymean))
        print('      STDs%8.4f %8.4f' %(xstd, ystd))

    # For NIRCam only, adjust angles
    if instrument == 'NIRCAM':
        newV3IdlYAngle = degrees(atan2(-AS[2], BS[2])) # Everything rotates by this amount
        if abs(newV3IdlYAngle) > 90.0: newV3IdlYAngle = newV3IdlYAngle - copysign(180, newV3IdlYAngle)
        print('New angle', newV3IdlYAngle)
        newA = AS*cos(radians(newV3IdlYAngle)) + BS*sin(radians(newV3IdlYAngle))
        newB = -AS*sin(radians(newV3IdlYAngle)) + BS*cos(radians(newV3IdlYAngle))
        if verbose:
            print('\nnewA')
            triangle(newA, order)
            print('newB')
            triangle(newB, order)

        newC = RotateCoeffs(CS, -newV3IdlYAngle, order)
        newD = RotateCoeffs(DS, -newV3IdlYAngle, order)
        print('Rotate Coeffs by newV3IdlYAngle')
        if verbose:
            print('newC')
            triangle(newC, order)
            print('newD')
            triangle(newD, order)

        (stat, xmean, ymean, xstd, ystd) = compute_roundtrip_error(AS, BS, CS, DS,
                                                                         verbose=True, instrument=instrument)
        print('Final coefficients')
        print('Round trip     X       Y')
        print('     Means%8.4F %8.4f' % (xmean, ymean))
        print('      STDs%8.4f %8.4f' % (xstd, ystd))

        newV3IdlXAngle = aperture_2.V3SciXAngle + newV3IdlYAngle
        newV3SciYAngle = aperture_2.V3SciYAngle + newV3IdlYAngle
        newV3IdlYAngle = aperture_2.V3IdlYAngle + newV3IdlYAngle

    return aperture_2
=======
>>>>>>> bf40d17a6056c2a50d0671625ff19bcdbaff08b3
