#!/usr/bin/env python
"""Generate MIRI SIAF and flight-like SIAF reference files.

Authors
-------
    Johannes Sahlmann

References
----------
    This script was partially adapted from Colin Cox' miriifu.py.

    For a detailed description of the MIRI SIAF, the underlying reference files, and the
    transformations, see Law et al., (latest revision): MIRI SIAF Input (JWST-STScI-004741).

    The term `worksheet` refers to the excel worksheet in the respective SIAF.xlsx, which contained
    some of the SIAF generation logic previously.

"""
from collections import OrderedDict
import copy
import os

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as pl

import pysiaf
from pysiaf import iando
from pysiaf.constants import JWST_SOURCE_DATA_ROOT, JWST_TEMPORARY_DATA_ROOT, \
    JWST_DELIVERY_DATA_ROOT
from pysiaf.tests import test_miri
from pysiaf.utils import compare
from pysiaf.utils import polynomial

import generate_reference_files

#############################
instrument = 'MIRI'

# regenerate SIAF reference files if needed
regerenate_basic_reference_files = False
if regerenate_basic_reference_files:
    generate_reference_files.generate_siaf_detector_layout()
    generate_reference_files.generate_siaf_detector_reference_file(instrument)
    generate_reference_files.generate_siaf_ddc_mapping_reference_file(instrument)

# DDC name mapping
ddc_apername_mapping = iando.read.read_siaf_ddc_mapping_reference_file(instrument)

# MIRI detector parameters, e.g. XDetSize
siaf_detector_parameters = iando.read.read_siaf_detector_reference_file(instrument)

# definition of the master apertures
detector_layout = iando.read.read_siaf_detector_layout()
master_aperture_names = detector_layout['AperName'].data

# directory containing reference files delivered by instrument team(s)
source_data_dir = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, 'delivery')
print('Loading source data files from directory: {}'.format(source_data_dir))

miri_distortion_file = 'MIRI_FM_MIRIMAGE_DISTORTION_07.04.01.fits'

# Fundamental aperture definitions: names, types, reference positions, dependencies
# for MIRI this file is part of the delivered source files and contains more columns
siaf_aperture_definitions = iando.read.read_siaf_aperture_definitions(instrument,
                                                                      directory=source_data_dir)

def untangle(square):
    """Turn a square n x n array into a linear array.

    Parameters
    ----------
    square : n x n array
        Input array

    Returns
    -------
    linear : array
        Linearized array

    """
    n = square.shape[0]
    t = n * (n + 1) // 2
    linear = np.zeros(t)
    for i in range(n):
        for j in range(n - i):
            k = (i + j) * (i + j + 1) // 2 + i
            linear[k] = square[i, j]
    return linear


def invcheck(A, B, C, D, order, low, high):
    """Round trip calculation to test inversion.

    Parameters
    ----------
    A
    B
    C
    D
    order
    low
    high

    """
    x = np.random.random(10)
    x = low + (high - low) * x
    y = np.random.random(10)
    y = low + (high - low) * y
    u = polynomial.poly(A, x, y, order)
    v = polynomial.poly(B, x, y, order)
    x2 = polynomial.poly(C, u, v, order)
    y2 = polynomial.poly(D, u, v, order)
    print('\n INVERSE CHECK')
    for i in range(10):
        print('%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.2e%10.2e' %
              (x[i], y[i], u[i], v[i], x2[i], y2[i], x2[i] - x[i], y2[i] - y[i]))

    print('Round trip errors %10.2e %10.2e' % ((x - x2).std(), (y - y2).std()))
    print('Round trip errors %10.3f %10.3f' % ((x - x2).std(), (y - y2).std()))


def get_mirim_coefficients(distortion_file, verbose=False):
    """Read delivered FITS file for MIRI imager and return data to be ingested in SIAF.

    Parameters
    ----------
    distortion_file : str
        Name of distortion file.
    verbose : bool
        verbosity

    Returns
    -------
    csv_data : dict
        Dictionary containing the data

    """
    miri = fits.open(os.path.join(source_data_dir, distortion_file))

    T = miri['T matrix'].data
    TI = miri['TI matrix'].data

    # CDP7 T matrices transform from/to v2,v3 in arcsec
    # set VtoAN and ANtoV to unit matrix
    VtoAN = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ANtoV = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    TV = np.dot(T, VtoAN)
    VT = np.dot(ANtoV, TI)
    prod = np.dot(VT, TV)
    TT = np.dot(T, TI)

    if verbose:
        print('T\n', T)
        print('TI\n', TI)
        print('VtoAN\n', VtoAN)
        print('\n TV V2V3 to XY Entrance')
        print(TV)
        print(1.0 / TV[1, 1], 'arcsec/mm')
        print('\nANtoV\n', ANtoV)
        print('\n VTXY entrance to V2V3')
        print('VT\n', VT)
        print()
        print('VT comparison\n', prod)
        print('T comparison\n', TT)


    # Get linear coefficient layout
    A = miri['AI matrix'].data
    B = miri['BI matrix'].data
    C = miri['A matrix'].data
    D = miri['B matrix'].data
    AL = untangle(A)
    BL = untangle(B)
    CL = untangle(C)
    DL = untangle(D)
    if verbose:
        print('Initial AL\n', AL)
        print('Initial BL\n', BL)
        print('CL\n', CL)
        print('DL\n', DL)

    # scale factor corresponding to 25 mum pixel size, i.e. 40 pixels/mm
    order = 4
    k = 0
    for i in range(order + 1):
        factor = 0.025 ** i
        for j in range(i + 1):
            AL[k] = AL[k] * factor
            BL[k] = BL[k] * factor
            k += 1


    AF = VT[0, 0] * AL + VT[0, 1] * BL
    AF[0] = AF[0] + VT[0, 2]
    BF = VT[1, 0] * AL + VT[1, 1] * BL
    BF[0] = BF[0] + VT[1, 2]

    if verbose:
        polynomial.print_triangle(AF)
        polynomial.print_triangle(BF)

        print('AL scaled\n', AL)
        print('\n A FINAL')
        print('\n B FINAL')

    ## print('INVERSE TRANSFORMATIONS')
    # Combine TV with polynomial using polynomial.two_step
    # combination of several polynomial coefficients
    a = np.array([TV[0, 2], TV[0, 0], TV[0, 1]])
    b = np.array([TV[1, 2], TV[1, 0], TV[1, 1]])
    (C2, D2) = polynomial.two_step(CL, DL, a, b)
    CF = 40 * C2
    DF = 40 * D2
    if verbose:
        polynomial.print_triangle(CF)
        polynomial.print_triangle(DF)

        print('a', a)
        print('b', b)
        print('\nC Final')
        print('\nD Final')

    # if verbose:

        # Test two_step
        v2 = -280
        v3 = -430

        xin = TV[0, 0] * v2 + TV[0, 1] * v3 + TV[0, 2]
        yin = TV[1, 0] * v2 + TV[1, 1] * v3 + TV[1, 2]

        xmm = polynomial.poly(CL, xin, yin, 4)
        ymm = polynomial.poly(DL, xin, yin, 4)

        xmm2 = polynomial.poly(C2, v2, v3, 4)
        ymm2 = polynomial.poly(D2, v2, v3, 4)

        # Backwards check
        xp = 0
        yp = 0
        v2 = polynomial.poly(AF, xp, yp, 4)
        v3 = polynomial.poly(BF, xp, yp, 4)
        xpix = polynomial.poly(CF, v2, v3, 4)
        ypix = polynomial.poly(DF, v2, v3, 4)

        print('IN', xin, yin)
        print('MM', xmm, ymm)
        print('MM2', xmm2, ymm2)
        print('V', v2, v3)
        print('Original ', xp, yp)
        print('Recovered', xpix, ypix)
        print('Change   ', xpix - xp, ypix - yp)

        invcheck(AF, BF, CF, DF, 4, -512.0, 512.0)

    CS = polynomial.shift_coefficients(CF, AF[0], BF[0])
    DS = polynomial.shift_coefficients(DF, AF[0], BF[0])
    CS[0] = 0.0
    DS[0] = 0.0

    # extract V2,V3 reference position
    V2cen = AF[0]
    V3cen = BF[0]

    # reset zero order coefficients to zero
    AF[0] = 0.0
    BF[0] = 0.0

    if verbose:
        polynomial.print_triangle(CS)
        polynomial.print_triangle(DS)
        invcheck(AF, BF, CS, DS, 4, -512.0, 512.0)

        print('\nCS')
        print('\nDS')
        print('\nDetector Center')

    # if verbose:
        xscalec = np.hypot(AF[1], BF[1])
        yscalec = np.hypot(AF[2], BF[2])

    # compute angles
    xanglec = np.rad2deg(np.arctan2(AF[1], BF[1]))
    yanglec = np.rad2deg(np.arctan2(AF[2], BF[2]))

    if verbose:
        print('Position', V2cen, V3cen)
        print('Scales %10.6f %10.6f' % (xscalec, yscalec))
        print('Angles %10.6f %10.6f' % (xanglec, yanglec))

    # if verbose:
        xcen = 1033 / 2
        ycen = 1025 / 2
        xref = 693.5 - xcen
        yref = 512.5 - ycen
        V2Ref = polynomial.poly(AF, xref, yref, 4) + V2cen
        V3Ref = polynomial.poly(BF, xref, yref, 4) + V3cen
        dV2dx = polynomial.dpdx(AF, xref, yref)
        dV3dx = polynomial.dpdx(BF, xref, yref)
        dV2dy = polynomial.dpdy(AF, xref, yref)
        dV3dy = polynomial.dpdy(BF, xref, yref)
        xangler = np.arctan2(dV2dx, dV3dx)
        yangler = np.arctan2(dV2dy, dV3dy)
    # if verbose:
        print('Axis angles', np.rad2deg(xangler), np.rad2deg(yangler))

    # if verbose:
        # Illum reference position
        xscaler = np.hypot(dV2dx, dV3dx)
        yscaler = np.hypot(dV2dy, dV3dy)
        xangler = np.rad2deg(np.arctan2(dV2dx, dV3dx))
        yangler = np.rad2deg(np.arctan2(dV2dy, dV3dy))

    # if verbose:
        print('\nIllum reference position')
        print('xref=', xref)
        print('Position', V2Ref, V3Ref)
        print('Scales %10.6f %10.6f' % (xscaler, yscaler))
        print('Angles %10.6f %10.6f %10.6f' % (xangler, yangler, yangler - xangler))

    # if verbose:
        # Slit position
        xslit = (326.13)
        yslit = (300.70)
        dxslit = xslit - xcen
        dyslit = yslit - ycen
        V2slit = polynomial.poly(AF, dxslit, dyslit, 4) + V2cen
        V3slit = polynomial.poly(BF, dxslit, dyslit, 4) + V3cen
        dV2dx = polynomial.dpdx(AF, dxslit, yslit)
        dV3dx = polynomial.dpdx(BF, dxslit, dyslit)
        dV2dy = polynomial.dpdy(AF, dxslit, dyslit)
        dV3dy = polynomial.dpdy(BF, dxslit, dyslit)
        xangles = np.arctan2(dV2dx, dV3dx)
        yangles = np.arctan2(dV2dy, dV3dy)

    # if verbose:
        print('\nSlit')
        print('Position', dxslit, dyslit)
        print('V2,V3', V2slit, V3slit)
        print('Slit angles', np.rad2deg(xangles), np.rad2deg(yangles))

    # if verbose:
        # Corners
        xc = np.array([-516.0, 516.0, 516.0, -516.0, -516.0])
        yc = np.array([-512.0, -512.0, 512.0, 512.0, -512.0])
        V2c = polynomial.poly(AF, xc, yc, 4)
        V3c = polynomial.poly(BF, xc, yc, 4)
        V2c = V2c + V2cen
        V3c = V3c + V3cen
    # if verbose:
        print('\nCorners')
        print('V2 %10.4f %10.4f %10.4f %10.4f' % (V2c[0], V2c[1], V2c[2], V2c[3]))
        print('V3 %10.4f %10.4f %10.4f %10.4f' % (V3c[0], V3c[1], V3c[2], V3c[3]))

        # make figure
        pl.figure(1)
        pl.clf()
        pl.title('MIRI Detector')
        pl.plot(V2cen, V3cen, 'r+')
        pl.plot(V2c, V3c, ':')
        pl.grid(True)
        pl.axis('equal')
        pl.plot(V2Ref, V3Ref, 'b+')
        pl.plot(V2slit, V3slit, 'c+')
        pl.gca().invert_xaxis()
        pl.show()

    ## Rotated versions
        print('Angle', yanglec)
        print('Rotated')

    # incorporate rotation in coefficients
    a = np.deg2rad(yanglec)
    AR = AF * np.cos(a) - BF * np.sin(a)
    BR = AF * np.sin(a) + BF * np.cos(a)

    CR = polynomial.prepend_rotation_to_polynomial(CS, yanglec)
    DR = polynomial.prepend_rotation_to_polynomial(DS, yanglec)

    if verbose:
        print('AR')
        polynomial.print_triangle(AR)
        print('BR')
        polynomial.print_triangle(BF)
        print('\n', AR[2], ' near zero')
    # if verbose:
        invcheck(AR, BR, CR, DR, 4, -512.0, 512.0)

    # Check positions using rotated (Ideal) coefficients
    # if verbose:

        xi = polynomial.poly(AR, xc, yc, 4)
        yi = polynomial.poly(BR, xc, yc, 4)
        v2r = xi * np.cos(a) + yi * np.sin(a) + V2cen
        v3r = -xi * np.sin(a) + yi * np.cos(a) + V3cen
    # if verbose:
        print('V2', v2r)
        print('V3', v3r)
        pl.plot(v2r, v3r, '--')

    CRFl = polynomial.flip_x(CR)
    DRFl = polynomial.flip_x(DR)

    # see TR: "polynomial origin being at the detector center with
    # pixel position (516.5, 512.5). "
    detector_center_pixel_x = 516.5
    detector_center_pixel_y = 512.5

    # dictionary holding data written to csv
    csv_data = {}
    csv_data['DET_OSS'] = {}
    csv_data['DET_OSS']['A'] = AR
    csv_data['DET_OSS']['B'] = BR
    csv_data['DET_OSS']['C'] = CR
    csv_data['DET_OSS']['D'] = DR
    csv_data['DET_OSS']['Xref'] = detector_center_pixel_x
    csv_data['DET_OSS']['Yref'] = detector_center_pixel_y
    csv_data['DET_OSS']['Xref_inv'] = V2cen
    csv_data['DET_OSS']['Yref_inv'] = V3cen
    csv_data['DET_OSS']['xAngle'] = xanglec
    csv_data['DET_OSS']['yAngle'] = yanglec
    csv_data['DET_DMF'] = {}
    csv_data['DET_DMF']['A'] = -AR
    csv_data['DET_DMF']['B'] = BR
    csv_data['DET_DMF']['C'] = CRFl
    csv_data['DET_DMF']['D'] = DRFl
    csv_data['DET_DMF']['Xref'] = detector_center_pixel_x
    csv_data['DET_DMF']['Yref'] = detector_center_pixel_y
    csv_data['DET_DMF']['Xref_inv'] = V2cen
    csv_data['DET_DMF']['Yref_inv'] = V3cen
    csv_data['DET_DMF']['xAngle'] = xanglec
    csv_data['DET_DMF']['yAngle'] = yanglec

    return csv_data


def extract_ifu_data(aperture_table):
    """Extract relevant information from IFU slice reference files.

    Return one single table with columns that directly map to SIAF aperture entries.

    Parameters
    ----------
    aperture_table : astropy.table.Table
        Table with aperture information

    Returns
    -------
    table : astropy.table.Table instance
        Table containing data

    """
    column_name_mapping = {}
    column_name_mapping['X1'] = 'v2ll'
    column_name_mapping['Y1'] = 'v3ll'
    column_name_mapping['X2'] = 'v2lr'
    column_name_mapping['Y2'] = 'v3lr'
    column_name_mapping['X3'] = 'v2ur'
    column_name_mapping['Y3'] = 'v3ur'
    column_name_mapping['X4'] = 'v2ul'
    column_name_mapping['Y4'] = 'v3ul'

    ifu_index = np.array([i for i, name in enumerate(aperture_table['AperName']) if 'MIRIFU_' in name])
    table = copy.deepcopy(aperture_table[ifu_index])

    table['V2Ref'] = table['v2ref']
    table['V3Ref'] = table['v3ref']

    # see IFU worksheet
    for axis in ['X', 'Y']:
        for index in [1, 2, 3, 4]:
            if axis == 'X':
                table['{}IdlVert{}'.format(axis, index)] = table['V2Ref'] \
                                                           - table[column_name_mapping['{}{}'.format(axis, index)]]
            elif axis == 'Y':
                table['{}IdlVert{}'.format(axis, index)] = table[ column_name_mapping['{}{}'.format(axis, index)]] \
                                                           - table['V3Ref']
    return table


csv_data = get_mirim_coefficients(miri_distortion_file, verbose=False)

number_of_coefficients = len(csv_data['DET_OSS']['A'])
polynomial_degree = polynomial.polynomial_degree(number_of_coefficients)

# convert to column names in Calc worksheet
for AperName in csv_data.keys():
    csv_data[AperName]['dx'] = csv_data[AperName]['Xref']
    csv_data[AperName]['dy'] = csv_data[AperName]['Yref']
    csv_data[AperName]['dxIdl'] = csv_data[AperName]['Xref_inv']
    csv_data[AperName]['dyIdl'] = csv_data[AperName]['Yref_inv']
    k = 0
    for i in range(polynomial_degree + 1):
        for j in np.arange(i + 1):
            csv_data[AperName]['Sci2IdlX{:d}{:d}'.format(i, j)] = csv_data[AperName]['A'][k]
            csv_data[AperName]['Sci2IdlY{:d}{:d}'.format(i, j)] = csv_data[AperName]['B'][k]
            csv_data[AperName]['Idl2SciX{:d}{:d}'.format(i, j)] = csv_data[AperName]['C'][k]
            csv_data[AperName]['Idl2SciY{:d}{:d}'.format(i, j)] = csv_data[AperName]['D'][k]
            k += 1

# get IFU aperture definitions
slice_table = extract_ifu_data(siaf_aperture_definitions)

idlvert_attributes = ['XIdlVert{}'.format(i) for i in [1, 2, 3, 4]] + [
    'YIdlVert{}'.format(i) for i in [1, 2, 3, 4]]

aperture_dict = OrderedDict()
aperture_name_list = siaf_aperture_definitions['AperName'].tolist()

for aperture_index, AperName in enumerate(aperture_name_list):
    # new aperture to be constructed
    aperture = pysiaf.JwstAperture()
    aperture.AperName = AperName
    aperture.InstrName = siaf_detector_parameters['InstrName'][0].upper()

    # index in the aperture definition table
    aperture_definitions_index = siaf_aperture_definitions['AperName'].tolist().index(AperName)

    aperture.AperShape = siaf_detector_parameters['AperShape'][0]

    # Retrieve basic aperture parameters from definition files
    for attribute in 'XDetRef YDetRef AperType XSciSize YSciSize XSciRef YSciRef'.split():
        value = siaf_aperture_definitions[attribute][aperture_definitions_index]
        if np.ma.is_masked(value):
            value = None
        setattr(aperture, attribute, value)

    if aperture.AperType not in ['COMPOUND', 'SLIT']:
        for attribute in 'XDetSize YDetSize'.split():
            setattr(aperture, attribute, siaf_detector_parameters[attribute][0])

    aperture.DDCName = 'not set'
    aperture.Comment = None
    aperture.UseAfterDate = '2014-01-01'

    master_aperture_name = 'MIRIM_FULL'
    # process master apertures
    if aperture.AperType not in ['COMPOUND', 'SLIT']:

        if aperture.AperType == 'OSS':
            aperture.VIdlParity = 1
            aperture.DetSciYAngle = 0
            aperture.DetSciParity = 1
            csv_aperture_name = 'DET_OSS'
        else:
            detector_layout_index = detector_layout['AperName'].tolist().index(master_aperture_name)
            for attribute in 'DetSciYAngle DetSciParity VIdlParity'.split():
                setattr(aperture, attribute, detector_layout[attribute][detector_layout_index])

            # this is the name given to the pseudo-aperture in the Calc worksheet
            csv_aperture_name = 'DET_DMF'

        aperture.Sci2IdlDeg = polynomial_degree

        dx = aperture.XDetRef - csv_data[csv_aperture_name]['dx']
        dy = aperture.YDetRef - csv_data[csv_aperture_name]['dy']

        csv_data[csv_aperture_name]['A_shifted'] = polynomial.shift_coefficients(
            csv_data[csv_aperture_name]['A'], dx, dy, verbose=False)
        csv_data[csv_aperture_name]['B_shifted'] = polynomial.shift_coefficients(
            csv_data[csv_aperture_name]['B'], dx, dy, verbose=False)

        # apply polynomial to get reference location in ideal plane
        dxIdl = polynomial.poly(csv_data[csv_aperture_name]['A'], dx, dy, order=polynomial_degree)
        dyIdl = polynomial.poly(csv_data[csv_aperture_name]['B'], dx, dy, order=polynomial_degree)

        csv_data[csv_aperture_name]['C_shifted'] = polynomial.shift_coefficients(
            csv_data[csv_aperture_name]['C'], dxIdl, dyIdl, verbose=False)
        csv_data[csv_aperture_name]['D_shifted'] = polynomial.shift_coefficients(
            csv_data[csv_aperture_name]['D'], dxIdl, dyIdl, verbose=False)

        # set 00 coefficients to zero
        for coefficient_name in ['{}_shifted'.format(c) for c in 'A B C D'.split()]:
            csv_data[csv_aperture_name][coefficient_name][0] = 0.

        k = 0
        for i in range(polynomial_degree + 1):
            for j in np.arange(i + 1):
                setattr(aperture, 'Sci2IdlX{:d}{:d}'.format(i, j), csv_data[csv_aperture_name]['A_shifted'][k])
                setattr(aperture, 'Sci2IdlY{:d}{:d}'.format(i, j), csv_data[csv_aperture_name]['B_shifted'][k])
                setattr(aperture, 'Idl2SciX{:d}{:d}'.format(i, j), csv_data[csv_aperture_name]['C_shifted'][k])
                setattr(aperture, 'Idl2SciY{:d}{:d}'.format(i, j), csv_data[csv_aperture_name]['D_shifted'][k])
                k += 1

        aperture.V3SciYAngle = csv_data[csv_aperture_name]['yAngle']
        aperture.V3SciXAngle = csv_data[csv_aperture_name]['xAngle']
        aperture.V3IdlYAngle = aperture.V3SciYAngle
        aperture.V2Ref = csv_data[csv_aperture_name]['Xref_inv'] + aperture.VIdlParity * dxIdl * np.cos(np.deg2rad(aperture.V3IdlYAngle)) + dyIdl * np.sin(np.deg2rad(aperture.V3IdlYAngle))
        aperture.V3Ref = csv_data[csv_aperture_name]['Yref_inv'] - aperture.VIdlParity * dxIdl * np.sin(np.deg2rad(aperture.V3IdlYAngle)) + dyIdl * np.cos(np.deg2rad(aperture.V3IdlYAngle))

        # overwrite V3IdlYAngle if set in definition files
        for attribute in 'V3IdlYAngle'.split():
            value = siaf_aperture_definitions[attribute][aperture_definitions_index]
            if np.ma.is_masked(value) is False:
                setattr(aperture, attribute, value)

        aperture.complement()

    elif AperName in slice_table['AperName']:
        slice_index = slice_table['AperName'].tolist().index(AperName)
        for attribute in 'V2Ref V3Ref V3IdlYAngle'.split() + idlvert_attributes:  #
            setattr(aperture, attribute, slice_table[attribute][slice_index])
        aperture.AperShape = siaf_detector_parameters['AperShape'][0]
        aperture.VIdlParity = -1

    elif AperName == 'MIRIM_SLIT':

        # get MIRIM_SLIT definitions from source_file
        mirim_slit_definitions = copy.deepcopy(siaf_aperture_definitions[aperture_index])
        aperture.V2Ref = mirim_slit_definitions['v2ref']
        aperture.V3Ref = mirim_slit_definitions['v3ref']
        for attribute_name in 'VIdlParity V3IdlYAngle'.split():
            setattr(aperture, attribute_name, mirim_slit_definitions[attribute_name])
        # the mapping is different from above because now we are treating this as 'true' v2v3 and transform to idl
        column_name_mapping = {}
        column_name_mapping['X1'] = 'v2ll'
        column_name_mapping['Y1'] = 'v3ll'
        column_name_mapping['X4'] = 'v2lr'
        column_name_mapping['Y4'] = 'v3lr'
        column_name_mapping['X3'] = 'v2ur'
        column_name_mapping['Y3'] = 'v3ur'
        column_name_mapping['X2'] = 'v2ul'
        column_name_mapping['Y2'] = 'v3ul'
        for index in [1, 2, 3, 4]:
            x_idl, y_idl = aperture.tel_to_idl(mirim_slit_definitions[column_name_mapping['{}{}'.format('X', index)]],
                                               mirim_slit_definitions[column_name_mapping['{}{}'.format('Y', index)]])
            setattr(aperture, '{}IdlVert{}'.format('X', index), x_idl)
            setattr(aperture, '{}IdlVert{}'.format('Y', index), y_idl)
    aperture_dict[AperName] = aperture

aperture_dict = OrderedDict(sorted(aperture_dict.items(), key=lambda t: aperture_name_list.index(t[0])))

# third pass to set DDCNames apertures, which depend on other apertures
ddc_siaf_aperture_names = np.array([key for key in ddc_apername_mapping.keys()])
ddc_v2 = np.array(
    [aperture_dict[aperture_name].V2Ref for aperture_name in ddc_siaf_aperture_names])
ddc_v3 = np.array(
    [aperture_dict[aperture_name].V3Ref for aperture_name in ddc_siaf_aperture_names])
for AperName in aperture_name_list:
    separation_tel_from_ddc_aperture = np.sqrt(
        (aperture_dict[AperName].V2Ref - ddc_v2) ** 2 + (
        aperture_dict[AperName].V3Ref - ddc_v3) ** 2)
    aperture_dict[AperName].DDCName = ddc_apername_mapping[
        ddc_siaf_aperture_names[np.argmin(separation_tel_from_ddc_aperture)]]


######################################
# SIAF content generation finished
######################################

aperture_collection = pysiaf.ApertureCollection(aperture_dict)

emulate_delivery = True

if emulate_delivery:
    pre_delivery_dir = os.path.join(JWST_DELIVERY_DATA_ROOT, instrument)
    if not os.path.isdir(pre_delivery_dir):
        os.makedirs(pre_delivery_dir)

    # write the SIAF files to disk
    filenames = pysiaf.iando.write.write_jwst_siaf(aperture_collection, basepath=pre_delivery_dir, file_format=['xml', 'xlsx'])

    pre_delivery_siaf = pysiaf.Siaf(instrument, basepath=pre_delivery_dir)

    compare_against_prd = True
    compare_against_cdp7b = True

    print('\nRunning regression test of pre_delivery_siaf against test_data:')
    test_miri.test_against_test_data(siaf=pre_delivery_siaf, verbose=True)

    for compare_to in [pysiaf.JWST_PRD_VERSION]:
        if compare_to == 'cdp7b':
            ref_siaf = pysiaf.Siaf(instrument,
                                   filename=os.path.join(pre_delivery_dir, 'MIRI_SIAF_cdp7b.xml'))
        else:
            # compare new SIAF with PRD version
            ref_siaf = pysiaf.Siaf(instrument)

        tags = {'reference': compare_to, 'comparison': 'pre_delivery'}

        compare.compare_siaf(pre_delivery_siaf, reference_siaf_input=ref_siaf,
                             fractional_tolerance=1e-6, report_dir=pre_delivery_dir, tags=tags)

        compare.compare_transformation_roundtrip(pre_delivery_siaf,
                                                 reference_siaf_input=ref_siaf, tags=tags,
                                                 report_dir=pre_delivery_dir)

        compare.compare_inspection_figures(pre_delivery_siaf, reference_siaf_input=ref_siaf,
                                           report_dir=pre_delivery_dir, tags=tags,
                                           xlimits=(-360, -520), ylimits=(-440, -300))

    # run some tests on the new SIAF
    from pysiaf.tests import test_aperture
    print('\nRunning aperture_transforms test for pre_delivery_siaf')
    test_aperture.test_jwst_aperture_transforms([pre_delivery_siaf], verbose=False, threshold=0.04)
    print('\nRunning aperture_vertices test for pre_delivery_siaf')
    test_aperture.test_jwst_aperture_vertices([pre_delivery_siaf])

else:

    test_dir = os.path.join(JWST_TEMPORARY_DATA_ROOT, instrument, 'generate_test')
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    # write the SIAFXML to disk
    [filename] = pysiaf.iando.write.write_jwst_siaf(aperture_collection, basepath=test_dir,
                                                    file_format=['xml'])
    print('SIAFXML written in {}'.format(filename))
