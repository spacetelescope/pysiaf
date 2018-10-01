"""A collection of helper functions to support pysiaf calculations and functionalities.

Authors
-------
    - Johannes Sahlmann

"""
from __future__ import absolute_import, print_function, division
import copy
import math
from math import sin, cos, atan2, degrees, radians
import numpy as np

from ..constants import V3_TO_YAN_OFFSET_DEG
from ..iando import read
from .polynomial import shift_coefficients, flip_y, flip_x, add_rotation, \
    prepend_rotation_to_polynomial, poly, print_triangle


def an_to_tel(xan_arcsec, yan_arcsec):
    """Convert from XAN, YAN to V2, V3."""
    v2_arcsec = xan_arcsec
    v3_arcsec = -1*yan_arcsec - V3_TO_YAN_OFFSET_DEG * 3600.
    return v2_arcsec, v3_arcsec


def tel_to_an(v2_arcsec, v3_arcsec):
    """Convert from V2, V3 to XAN, YAN."""
    xan_arcsec = v2_arcsec
    yan_arcsec = -1*v3_arcsec - V3_TO_YAN_OFFSET_DEG * 3600.
    return xan_arcsec, yan_arcsec


def compute_roundtrip_error(A, B, C, D, offset_x=0., offset_y=0., verbose=False, instrument=''):
    """Return the roundtrip error of the distortion transformations specified by A,B,C,D.

    Test whether the forward and inverse idl-sci transformations are consistent.

    Parameters
    ----------
    A : numpy array
        polynomial coefficients
    B : numpy array
        polynomial coefficients
    C : numpy array
        polynomial coefficients
    D : numpy array
        polynomial coefficients
    offset_x : float
        Offset subtracted from input coordinate
    offset_y : float
        Offset subtracted from input coordinate
    verbose : bool
        verbosity
    instrument : str
        Instrument name

    Returns
    -------
    error_estimation_metric, dx.mean(), dy.mean(), dx.std(), dy.std(), data : tuple
        mean and std of errors, data used in computations

    """
    number_of_coefficients = len(A)
    polynomial_degree = np.int((np.sqrt(8 * number_of_coefficients + 1) - 3) / 2)
    order = polynomial_degree

    # regular grid of points (in science pixel coordinates) in the full frame science frame
    grid_amplitude = 2048
    if instrument.lower() == 'miri':
        grid_amplitude = 1024
    x, y = get_grid_coordinates(10, (grid_amplitude/2+1, grid_amplitude/2+1), grid_amplitude)

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

    if verbose:
        print('\nInverse Check')
        for p in range(len(x)):
            print(8*'%10.3f' % (x[p], y[p], u[p], v[p], x2[p], y2[p], x2[p] - x[p], y2[p] - y[p]))

    data = {}
    data['x'] = x
    data['y'] = y
    data['x2'] = x2
    data['y2'] = y2

    # absolute coordinate differences
    dx = np.abs(x2-x)
    dy = np.abs(y2-y)

    if verbose:
        print(4*'%12.3e' % (dx.mean(), dy.mean(), dx.std(), dy.std()))
        # print ('RMS deviation %5.3f' %rms_deviation)

    # compute one number that indicates if something may be wrong
    error_estimation_metric = np.abs(dx.mean()/dx.std()) + np.abs(dx.mean()/dx.std())
    return error_estimation_metric, dx.mean(), dy.mean(), dx.std(), dy.std(), data


def convert_polynomial_coefficients(A_in, B_in, C_in, D_in, oss=False, inverse=False,
                                    parent_aperture=None):
    """Emulate some transformation made in nircam_get_polynomial_both.

    Written by Johannes Sahlmann 2018-02-18, structure largely based on nircamtrans.py code
    by Colin Cox.

    Parameters
    ----------
    A_in : numpy array
        polynomial coefficients
    B_in : numpy array
        polynomial coefficients
    C_in : numpy array
        polynomial coefficients
    D_in : numpy array
        polynomial coefficients
    oss : bool
        Whether this is an OSS aperture or not
    inverse : bool
        Whether this is forward or backward/inverse transformation
    parent_aperture : str
        Name of parent aperture

    Returns
    -------
    AR, BR, CR, DR, V3SciXAngle, V3SciYAngle, V2Ref, V3Ref : tuple of arrays and floats
        Converted polynomial coefficients

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

        # AR, BR = rotate_coefficients(A_in, B_in, V3Angle)
        AR, BR = add_rotation(A_in, B_in, -1*V3Angle)

        CS = shift_coefficients(C_in, V2Ref, V3Ref)
        DS = shift_coefficients(D_in, V2Ref, V3Ref)

        CR = prepend_rotation_to_polynomial(CS, V3Angle)
        DR = prepend_rotation_to_polynomial(DS, V3Angle)

        if oss:
            # OSS apertures
            V3Angle = copy.deepcopy(V3SciYAngle)
        else:
            # non-OSS apertures
            if abs(V3SciYAngle) > 90.0:  # e.g. NRCA2_FULL
                # print 'Reverse Y axis direction'
                AR = -flip_y(AR)
                BR = flip_y(BR)
                CR = flip_x(CR)
                DR = -flip_x(DR)

            else:  # e.g NRCA1_FULL
                # print 'Reverse X axis direction'
                AR = -flip_x(AR)
                BR = flip_x(BR)
                CR = -flip_x(CR)
                DR = flip_x(DR)
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
            AR = -flip_y(A_in)
            BR = flip_y(B_in)
            CR = flip_x(C_in)
            DR = -flip_x(D_in)

        else:  # e.g NRCA1_FULL
            # print 'Reverse X axis direction'
            AR = -flip_x(A_in)
            BR = flip_x(B_in)
            CR = -flip_x(C_in)
            DR = flip_x(D_in)
            V3SciXAngle = revert_correct_V3SciXAngle(V3SciXAngle)

        # rotate the other way
        # A, B = rotate_coefficients(AR, BR, -V3SciYAngle)
        A, B = add_rotation(AR, BR, +1*V3SciYAngle)

        A[0] = parent_aperture.V2Ref
        B[0] = parent_aperture.V3Ref

        # now invert the last part of nircam_get_polynomial_forward
        AFS = A
        BFS = B

        # shift by parent aperture reference point
        AF = shift_coefficients(AFS, -parent_aperture.XDetRef, -parent_aperture.YDetRef)
        BF = shift_coefficients(BFS, -parent_aperture.XDetRef, -parent_aperture.YDetRef)

        CS = prepend_rotation_to_polynomial(CR, -V3SciYAngle)
        DS = prepend_rotation_to_polynomial(DR, -V3SciYAngle)

        C = shift_coefficients(CS, -parent_aperture.V2Ref, -parent_aperture.V3Ref)
        D = shift_coefficients(DS, -parent_aperture.V2Ref, -parent_aperture.V3Ref)

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
    """Return corrected V3SciYAngle.

    Only correct if the original V3SciYAngle in [0,180) deg

    Parameters
    ----------
    V3SciYAngle_deg : float
        angle in deg

    Returns
    -------
    V3SciYAngle_deg : float
        Angle in deg

    """
    if V3SciYAngle_deg < 0.:
        V3SciYAngle_deg += 180.
    return V3SciYAngle_deg


def revert_correct_V3SciXAngle(V3SciXAngle_deg):
    """Return corrected V3SciXAngle.

    Parameters
    ----------
    V3SciXAngle_deg : float
        Angle in deg

    Returns
    -------
    V3SciXAngle_deg : float
        Angle in deg

    """
    V3SciXAngle_deg += 180.
    return V3SciXAngle_deg


def set_reference_point_and_distortion(instrument, aperture, parent_aperture):
    """Set V2Ref and V3ref and distortion coefficients for an aperture with a parent_aperture.

    Parameters
    ----------
    instrument : str
        Instrument name
    aperture : `pysiaf.Aperture` object
        Aperture
    parent_aperture : `pysiaf.Aperture` object
        Parent aperture

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
        xsci_offset = (aperture.XDetRef - parent_aperture.XDetRef) * \
                      np.cos(np.deg2rad(aperture.DetSciYAngle))
        ysci_offset = (aperture.YDetRef - parent_aperture.YDetRef) * \
                      np.cos(np.deg2rad(aperture.DetSciYAngle))

        # shift polynomial coefficients of the parent aperture
        sci2idlx_coefficients_shifted = shift_coefficients(sci2idlx_coefficients, xsci_offset,
                                                           ysci_offset, verbose=False)
        sci2idly_coefficients_shifted = shift_coefficients(sci2idly_coefficients, xsci_offset,
                                                           ysci_offset, verbose=False)

        # see calc worksheet in NIRISS SIAFEXCEL
        dx_idl = sci2idlx_coefficients_shifted[0]
        dy_idl = sci2idly_coefficients_shifted[0]

        # remove the zero point offsets from the coefficients
        idl2scix_coefficients_shifted = shift_coefficients(idl2scix_coefficients, dx_idl, dy_idl,
                                                           verbose=False)
        idl2sciy_coefficients_shifted = shift_coefficients(idl2sciy_coefficients, dx_idl, dy_idl,
                                                           verbose=False)

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

        # set V2ref and V3Ref (this is a standard process for child apertures and should be
        # generalized in a function)
        aperture.V2Ref = parent_aperture.V2Ref + aperture.VIdlParity * dx_idl * np.cos(
            np.deg2rad(aperture.V3IdlYAngle)) + dy_idl * np.sin(np.deg2rad(aperture.V3IdlYAngle))
        aperture.V3Ref = parent_aperture.V3Ref - aperture.VIdlParity * dx_idl * np.sin(
            np.deg2rad(aperture.V3IdlYAngle)) + dy_idl * np.cos(np.deg2rad(aperture.V3IdlYAngle))

    elif instrument == 'NIRCam':
        # do the inverse of what nircam_get_polynomial_both does for the master aperture, then
        # forward for the child aperture

        A, B, C, D = sci2idlx_coefficients, sci2idly_coefficients, idl2scix_coefficients, \
                     idl2sciy_coefficients
        AF, BF, CF, DF = convert_polynomial_coefficients(A, B, C, D, inverse=True,
                                                         parent_aperture=parent_aperture)

        # now shift to child aperture reference point
        AFS_child = shift_coefficients(AF, aperture.XDetRef, aperture.YDetRef)
        BFS_child = shift_coefficients(BF, aperture.XDetRef, aperture.YDetRef)
        CFS_child = CF
        DFS_child = DF
        CFS_child[0] -= aperture.XDetRef
        DFS_child[0] -= aperture.YDetRef

        if aperture.AperType == 'OSS':
            oss = True
        else:
            oss = False

        AR, BR, CR, DR, V3SciXAngle, V3SciYAngle, V2Ref, V3Ref = \
            convert_polynomial_coefficients(AFS_child, BFS_child, CFS_child, DFS_child, oss=oss)

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

    Parameters
    ----------
    v3sciyangle : float
        angle

    Returns
    -------
    v3sciyangle : float
        angle

    """
    if np.abs(v3sciyangle) < 90.:
        v3idlyangle = v3sciyangle
    else:
        v3idlyangle = v3sciyangle - np.sign(v3sciyangle) * 180.

    return v3sciyangle


def match_v2v3(aperture_1, aperture_2, verbose=False):
    """Use the V2V3 from aperture_1 in aperture_2 modifying X[Y]DetRef,X[Y]SciRef to match.

    Also shift the polynomial coefficients to reflect the new reference point origin
    and for NIRCam recalculate angles.

    Parameters
    ----------
    aperture_1 : `pysiaf.Aperture object`
        Aperture whose V2,V3 reference position is to be used
    aperture_2 : `pysiaf.Aperture object`
        The V2,V3 reference position is to be altered to match that of aperture_1
    verbose : bool
        verbosity

    Returns
    -------
    new_aperture_2: `pysiaf.Aperture object`
        An aperture object derived from aperture_2 but with some parameters changed to match
        altered V2V3.

    """
    instrument = aperture_1.InstrName
    assert instrument != 'NIRSPEC', 'Program not working for NIRSpec'
    assert (aperture_2.AperType in ['FULLSCA', 'SUBARRAY', 'ROI']), \
        "2nd aperture must be pixel-based"
    order = aperture_1.Sci2IdlDeg
    V2Ref1 = aperture_1.V2Ref
    V3Ref1 = aperture_1.V3Ref
    newV2Ref = V2Ref1
    newV3Ref = V3Ref1
    if verbose:
        print('Current Vref', aperture_2.V2Ref, aperture_2.V3Ref)
        print('Shift to    ', V2Ref1, V3Ref1)

    # Need to work in aperture 2  coordinate systems
    aperName_1 = aperture_1.AperName
    aperName_2 = aperture_2.AperName
    detector_1 = aperName_1.split('_')[0]
    detector_2 = aperName_2.split('_')[0]
    if verbose:
        print('Detector 1', detector_1, '  Detector 2', detector_2)
    V2Ref2 = aperture_2.V2Ref
    V3Ref2 = aperture_2.V3Ref
    theta0 = aperture_2.V3IdlYAngle
    if verbose:
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
        print_triangle(A)
        print('B')
        print_triangle(B)
        print('C')
        print_triangle(C)
        print('D')
        print_triangle(D)

        (stat, xmean, ymean, xstd, ystd, data) = compute_roundtrip_error(A, B, C, D,
                                                                         verbose=verbose,
                                                                         instrument=instrument)
        print('Round trip     X       Y')
        print('     Means%8.4F %8.4f' % (xmean, ymean))
        print('      STDs%8.4f %8.4f' % (xstd, ystd))

    # Use convert
    (newXSci, newYSci) = aperture_2.convert(V2Ref1, V3Ref1, 'tel', 'sci')
    (newXDet, newYDet) = aperture_2.convert(V2Ref1, V3Ref1, 'tel', 'det')
    (newXIdl, newYIdl) = aperture_2.convert(V2Ref1, V3Ref1, 'tel', 'idl')

    dXSciRef = newXSci - aperture_2.XSciRef
    dYSciRef = newYSci - aperture_2.YSciRef
    AS = shift_coefficients(A, dXSciRef, dYSciRef)
    BS = shift_coefficients(B, dXSciRef, dYSciRef)
    if verbose:
        print('VRef1', V2Ref1, V3Ref1)
        print('Idl', newXIdl, newYIdl)
        print('Shift pixel origin by', dXSciRef, dYSciRef)
        print('New Ideal origin', newXIdl, newYIdl)

    CS = shift_coefficients(C, AS[0], BS[0])
    DS = shift_coefficients(D, AS[0], BS[0])
    AS[0] = 0.0
    BS[0] = 0.0
    CS[0] = 0.0
    DS[0] = 0.0
    if verbose:
        print('\nShifted Polynomials')
        print('AS')
        print_triangle(AS)
        print('BS')
        print_triangle(BS)
        print('CS')
        print_triangle(CS)
        print('DS')
        print_triangle(DS)
        print('\nABCDS')

    (stat, xmean, ymean, xstd, ystd, data) = compute_roundtrip_error(AS, BS, CS, DS,
                                                                     verbose=verbose,
                                                                     instrument=instrument)
    if verbose:
        print('Round trip     X       Y')
        print('     Means%8.4F %8.4f' % (xmean, ymean))
        print('      STDs%8.4f %8.4f' % (xstd, ystd))

    newA = AS
    newB = BS
    newC = CS
    newD = DS

    new_aperture_2 = copy.deepcopy(aperture_2)

    # For NIRCam only, adjust angles
    if instrument == 'NIRCAM':
        newV3IdlYAngle = degrees(atan2(-AS[2], BS[2]))  # Everything rotates by this amount
        if abs(newV3IdlYAngle) > 90.0:
            newV3IdlYAngle = newV3IdlYAngle - copysign(180, newV3IdlYAngle)
        newA = AS*cos(radians(newV3IdlYAngle)) + BS*sin(radians(newV3IdlYAngle))
        newB = -AS*sin(radians(newV3IdlYAngle)) + BS*cos(radians(newV3IdlYAngle))
        if verbose:
            print('New angle', newV3IdlYAngle)
            print('\nnewA')
            print_triangle(newA)
            print('newB')
            print_triangle(newB)

        newC = prepend_rotation_to_polynomial(CS, -newV3IdlYAngle)
        newD = prepend_rotation_to_polynomial(DS, -newV3IdlYAngle)

        if verbose:
            print('newC')
            print_triangle(newC)
            print('newD')
            print_triangle(newD)

            (stat, xmean, ymean, xstd, ystd, data) = compute_roundtrip_error(newA, newB, newC, newD,
                                                                             verbose=verbose,
                                                                             instrument=instrument)
            print('\nFinal coefficients')
            print('Round trip     X       Y')
            print('     Means%8.4F %8.4f' % (xmean, ymean))
            print('      STDs%8.4f %8.4f' % (xstd, ystd))

        newV3SciXAngle = aperture_2.V3SciXAngle + newV3IdlYAngle
        newV3SciYAngle = aperture_2.V3SciXAngle + newV3IdlYAngle
        newV3IdlYAngle = aperture_2.V3IdlYAngle + newV3IdlYAngle
        new_aperture_2.V3SciXAngle = newV3SciXAngle
        new_aperture_2.V3SciYAngle = newV3SciYAngle
        new_aperture_2.V3IdlYAngle = newV3IdlYAngle

    # Set new values in new_aperture_2
    new_aperture_2.V2Ref = newV2Ref
    new_aperture_2.V3Ref = newV3Ref
    new_aperture_2.XDetRef = newXDet
    new_aperture_2.YDetRef = newYDet
    new_aperture_2.XSciRef = newXSci
    new_aperture_2.YSciRef = newYSci
    if verbose:
        print('Initial', aperture_2.V2Ref, aperture_2.V3Ref, aperture_2.XDetRef, aperture_2.YDetRef)
        print('Changes', newV2Ref, newV3Ref, newXDet, newYDet)
        print('Modified', new_aperture_2.V2Ref, new_aperture_2.V3Ref, new_aperture_2.XDetRef,
              new_aperture_2.YDetRef)

    new_aperture_2.set_polynomial_coefficients(newA, newB, newC, newD)
    (xcorners, ycorners) = new_aperture_2.corners('idl', rederive=True)
    for c in range(4):
        suffix = "{}".format(c+1)
        setattr(new_aperture_2, 'XIdlVert' + suffix, xcorners[c])
        setattr(new_aperture_2, 'YIdlVert' + suffix, ycorners[c])

    return new_aperture_2
