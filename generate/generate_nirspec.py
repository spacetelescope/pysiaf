#!/usr/bin/env python
"""Script to generate NIRSpec SIAF content and files using pysiaf and flight-like SIAF reference files

The distortion polynomials used in NIRSpec FULLSCA, OSS, and TRANSFORM entries are derived from the
NIRSpec IDT’s Parametric Instrument Mode
As discussed in JWST-STScI-005921, the NIRSpec use and implementation of the FULLSCA rows differs
significantly from the normal SIAF conventions.

Authors
-------

    Johannes Sahlmann

References
----------

    This code was partially adapted from Colin Cox' nirspecpcf.py, msa.py, and slits.ipynb.

    For a detailed description of the NIRSpec SIAF, the underlying reference files, and the
    transformations, see Proffitt et al., 2017: The Pre-Flight SI Aperture File, Part 4: NIRSpec
    (JWST-STScI-005921).


"""


from collections import OrderedDict
import os

from astropy.table import Table
import numpy as np

import pysiaf
from pysiaf import iando
from pysiaf.aperture import DISTORTION_ATTRIBUTES
from pysiaf.utils import compare, polynomial
from pysiaf.constants import JWST_SOURCE_DATA_ROOT, JWST_TEMPORARY_DATA_ROOT, REPORTS_ROOT, \
    V3_TO_YAN_OFFSET_DEG, JWST_DELIVERY_DATA_ROOT, JWST_PRD_DATA_ROOT
import generate_reference_files


def process_nirspec_aperture(aperture, verbose=False):
    """Set aperture parameters for master apertures and FULLSCA and OSS apertures.

    Parameters
    ----------
    aperture
    verbose

    Returns
    -------

    """

    AperName = aperture.AperName

    index = siaf_aperture_definitions['AperName'].tolist().index(AperName)

    parent_aperture_name = None
    if (siaf_aperture_definitions['parent_apertures'][index] is not None) and (
        siaf_aperture_definitions['dependency_type'][index] == 'default'):
        aperture._parent_apertures = siaf_aperture_definitions['parent_apertures'][index]
        parent_aperture = aperture_dict[aperture._parent_apertures]
        parent_aperture_name = parent_aperture.AperName
        for attribute in 'DetSciYAngle Sci2IdlDeg DetSciParity VIdlParity'.split():
            setattr(aperture, attribute, getattr(parent_aperture, attribute))

    polynomial_degree = 5
    aperture.Sci2IdlDeg = polynomial_degree

    if (AperName in ['NRS1_FULL', 'NRS1_FULL_OSS']) or (parent_aperture_name == 'NRS1_FULL'):
        pcf_name = '491_GWA'
    elif (AperName in ['NRS2_FULL', 'NRS2_FULL_OSS']) or (parent_aperture_name == 'NRS2_FULL'):
        pcf_name = '492_GWA'

    if parent_aperture_name is None:
        Xref = aperture.XDetRef
        Yref = aperture.YDetRef
    else:
        Xref = parent_aperture.XDetRef
        Yref = parent_aperture.YDetRef

    for axis in ['A', 'B']:
        # modified is _shifted or _XYflipped, see Calc worksheet Rows 8,9,10
        pcf_data[pcf_name]['{}_modified'.format(axis)] = polynomial.shift_coefficients(
            pcf_data[pcf_name]['{}'.format(axis)], Xref, Yref, verbose=False)
        if (AperName in ['NRS2_FULL']) or (parent_aperture_name == 'NRS2_FULL'):
            # Add an XY flip (The definition of the SCI frame differs from that of the DET frame,
            # therefore the polynomial coefficients are redefined so the net transformation from
            # the DET to GWA plane is the same as is obtained when the NRS2_FULL_OSS row is used.
            # see JWST-STScI-005921.)
            pcf_data[pcf_name]['{}_modified'.format(axis)] = polynomial.flip_xy(
                pcf_data[pcf_name]['{}_modified'.format(axis)])

    if 'MIMF' not in AperName:
        Xoffset = 0
        Yoffset = 0
    else:
        Xoffset = aperture.XSciRef - parent_aperture.XSciRef
        Yoffset = aperture.YSciRef - parent_aperture.YSciRef

    sci2idlx_coefficients = polynomial.shift_coefficients(pcf_data[pcf_name]['{}_modified'.format('A')],
                                                          Xoffset, Yoffset, verbose=False)
    sci2idly_coefficients = polynomial.shift_coefficients(pcf_data[pcf_name]['{}_modified'.format('B')],
                                                          Xoffset, Yoffset, verbose=False)

    # set polynomial coefficients for transformation that goes directly to the GWA pupil plane
    idl2sci_factor = +1
    if (AperName in ['NRS2_FULL']) or ('NRS2_FP' in AperName):
        idl2sci_factor = -1
    k = 0
    for i in range(polynomial_degree + 1):
        for j in np.arange(i + 1):
            setattr(aperture, 'Sci2IdlX{:d}{:d}'.format(i, j), sci2idlx_coefficients[k])
            setattr(aperture, 'Sci2IdlY{:d}{:d}'.format(i, j), sci2idly_coefficients[k])
            setattr(aperture, 'Idl2SciX{:d}{:d}'.format(i, j),
                    idl2sci_factor * pcf_data[pcf_name]['C'][k])
            setattr(aperture, 'Idl2SciY{:d}{:d}'.format(i, j),
                    idl2sci_factor * pcf_data[pcf_name]['D'][k])
            k += 1

    aperture.Idl2SciX00 = aperture.Idl2SciX00 - idl2sci_factor * aperture.XDetRef
    aperture.Idl2SciY00 = aperture.Idl2SciY00 - idl2sci_factor * aperture.YDetRef

    # get offsets from first coefficients
    Xgwa = aperture.Sci2IdlX00
    Ygwa = aperture.Sci2IdlY00

    # see Calc worksheet row 30
    Xgwa_mod = -Xgwa
    Ygwa_mod = -Ygwa

    # apply polynomial transform to XAN,YAN
    XAN = polynomial.poly(pcf_data['CLEAR_GWA_OTE']['A'], Xgwa_mod, Ygwa_mod,
                          order=polynomial_degree)
    YAN = polynomial.poly(pcf_data['CLEAR_GWA_OTE']['B'], Xgwa_mod, Ygwa_mod,
                          order=polynomial_degree)

    # convert from XAN,YAN to V2,V3 and from degree to arcsecond (e.g. Cell F32)
    aperture.V2Ref = +1 * 3600. * XAN
    aperture.V3Ref = -1 * 3600. * (YAN + V3_TO_YAN_OFFSET_DEG)

    if verbose:
        print('Xgwa, Ygwa:', Xgwa, Ygwa)
        print('Xgwa_mod, Ygwa_mod:', Xgwa_mod, Ygwa_mod)
        print('XAN, YAN:', XAN, YAN)
        print('aperture.V2Ref, aperture.V3Ref:', aperture.V2Ref, aperture.V3Ref)

    # derivatives
    dXAN_dXgwa = polynomial.shift_coefficients(pcf_data['CLEAR_GWA_OTE']['A'], Xgwa_mod, Ygwa_mod, verbose=False)[1]
    dXAN_dYgwa = polynomial.shift_coefficients(pcf_data['CLEAR_GWA_OTE']['A'], Xgwa_mod, Ygwa_mod, verbose=False)[2]
    dYAN_dXgwa = polynomial.shift_coefficients(pcf_data['CLEAR_GWA_OTE']['B'], Xgwa_mod, Ygwa_mod, verbose=False)[1]
    dYAN_dYgwa = polynomial.shift_coefficients(pcf_data['CLEAR_GWA_OTE']['B'], Xgwa_mod, Ygwa_mod, verbose=False)[2]

    if verbose:
        print('dXAN_dXgwa, dXAN_dYgwa:', dXAN_dXgwa, dXAN_dYgwa)
        print('dYAN_dXgwa, dYAN_dYgwa:', dYAN_dXgwa, dYAN_dYgwa)

    if parent_aperture_name is None:
        dV2_dXSci = -3600. * (dXAN_dXgwa * aperture.Sci2IdlX10 + dXAN_dYgwa * aperture.Sci2IdlY10)
        dV2_dYSci = -3600. * (dXAN_dXgwa * aperture.Sci2IdlX11 + dXAN_dYgwa * aperture.Sci2IdlY11)
        dV3_dXSci =  3600. * (dYAN_dXgwa * aperture.Sci2IdlX10 + dYAN_dYgwa * aperture.Sci2IdlY10)
        dV3_dYSci =  3600. * (dYAN_dXgwa * aperture.Sci2IdlX11 + dYAN_dYgwa * aperture.Sci2IdlY11)
    else:
        dV2_dXSci = -3600. * (dXAN_dXgwa * parent_aperture.Sci2IdlX10 + dXAN_dYgwa * parent_aperture.Sci2IdlY10)
        dV2_dYSci = -3600. * (dXAN_dXgwa * parent_aperture.Sci2IdlX11 + dXAN_dYgwa * parent_aperture.Sci2IdlY11)
        dV3_dXSci =  3600. * (dYAN_dXgwa * parent_aperture.Sci2IdlX10 + dYAN_dYgwa * parent_aperture.Sci2IdlY10)
        dV3_dYSci =  3600. * (dYAN_dXgwa * parent_aperture.Sci2IdlX11 + dYAN_dYgwa * parent_aperture.Sci2IdlY11)

    # approximate scale terms
    aperture.XSciScale = np.sqrt(dV2_dXSci ** 2 + dV3_dXSci ** 2)
    aperture.YSciScale = np.sqrt(dV2_dYSci ** 2 + dV3_dYSci ** 2)

    # compute the approximate angles
    betaY = np.rad2deg(np.arctan2(dV2_dYSci, dV3_dYSci))
    betaX = np.rad2deg(np.arctan2(dV2_dXSci, dV3_dXSci))
    if verbose:
        print('dV2_dXSci, dV2_dYSci, dV3_dXSci, dV3_dYSci:', dV2_dXSci, dV2_dYSci, dV3_dXSci, dV3_dYSci)
        print('betaY:', betaY)

    # set the aperture attributes
    aperture.V3SciXAngle = betaX
    aperture.V3SciYAngle = betaY
    aperture.V3IdlYAngle = aperture.V3SciYAngle

    # The usual SIAF ideal plane is completely bypassed in the target acquisition calculations.
    # In the OSS and FULLSCA rows, an ideal plane is nevertheless defined by choosing a reference
    # point near the center of each detector and using the combined TA transformations to project
    #  the detector reference points and corners onto the sky.

    # Compute aperture corners in different frames: Calc worksheep row 43
    sci_corners_x, sci_corners_y = aperture.corners('sci', rederive=True)

    # offset from reference location
    sci_corners_x -= aperture.XSciRef
    sci_corners_y -= aperture.YSciRef

    # compute GWA plane corners
    # These coefficients are name overloaded and implement the transformation to GWA plane
    gwa_coefficients_x = np.array(
        [getattr(aperture, s) for s in DISTORTION_ATTRIBUTES if 'Sci2IdlX' in s])
    gwa_coefficients_y = np.array(
        [getattr(aperture, s) for s in DISTORTION_ATTRIBUTES if 'Sci2IdlY' in s])

    gwa_corners_x = np.zeros(len(sci_corners_x))
    gwa_corners_y = np.zeros(len(sci_corners_y))

    # apply transformation to GWA plane
    for j in range(len(gwa_corners_x)):
        gwa_corners_x[j] = polynomial.poly(gwa_coefficients_x, sci_corners_x[j], sci_corners_y[j],
                                           order=aperture.Sci2IdlDeg)
        gwa_corners_y[j] = polynomial.poly(gwa_coefficients_y, sci_corners_x[j], sci_corners_y[j],
                                           order=aperture.Sci2IdlDeg)

    # compute corners in V2V3/Tel
    gwa_to_ote_coefficients_x = pcf_data['CLEAR_GWA_OTE']['A']
    gwa_to_ote_coefficients_y = pcf_data['CLEAR_GWA_OTE']['B']

    tel_corners_x = np.zeros(len(sci_corners_x))
    tel_corners_y = np.zeros(len(sci_corners_y))
    for j in range(len(gwa_corners_x)):
        tel_corners_x[j] = +3600 * polynomial.poly(gwa_to_ote_coefficients_x, -gwa_corners_x[j],
                                                   -gwa_corners_y[j], order=aperture.Sci2IdlDeg)
        tel_corners_y[j] = -3600 * (
            polynomial.poly(gwa_to_ote_coefficients_y, -gwa_corners_x[j], -gwa_corners_y[j],
                            order=aperture.Sci2IdlDeg) + V3_TO_YAN_OFFSET_DEG)

    # Ideal corners
    idl_corners_x = np.zeros(len(sci_corners_x))
    idl_corners_y = np.zeros(len(sci_corners_y))
    for j in range(len(gwa_corners_x)):
        idl_corners_x[j] = aperture.VIdlParity * (tel_corners_x[j] - aperture.V2Ref) * np.cos(
            np.deg2rad(aperture.V3IdlYAngle)) - aperture.VIdlParity * (
            tel_corners_y[j] - aperture.V3Ref) * np.sin(np.deg2rad(aperture.V3IdlYAngle))
        idl_corners_y[j] = (tel_corners_x[j] - aperture.V2Ref) * np.sin(
            np.deg2rad(aperture.V3IdlYAngle)) + (tel_corners_y[j] - aperture.V3Ref) * np.cos(
            np.deg2rad(aperture.V3IdlYAngle))
        setattr(aperture, 'XIdlVert{}'.format(j + 1), idl_corners_x[j])
        setattr(aperture, 'YIdlVert{}'.format(j + 1), idl_corners_y[j])

    if verbose:
        print('sci_corners_x, sci_corners_y:', sci_corners_x, sci_corners_y)
        print(gwa_corners_x, gwa_corners_y)
        print(tel_corners_x, tel_corners_y)
        print(idl_corners_x, idl_corners_y)

    return aperture


def read_pcf_gtp(file_name):
    """Generic function that reads in NIRSpec gtp or pcf file and returns dictionary.

    Parameters
    ----------
    file_name

    Returns
    -------

    Authors
    -------

    - Johannes Sahlmann


    """

    data = OrderedDict()
    data['Filename'] = file_name
    with open(file_name) as f:
        EOF = False  # end of file
        while EOF is False:
            line = f.readline()
            if line.strip() == '':
                EOF = True
            elif line[0] == '#':
                # skip comments
                line = f.readline()
            elif line[0] == '*':
                EOS = False # end of section
                key = line[1:].strip().split()[0]
                data[key] = []
                empty_line_counter = 0
                while EOS is False:
                    last_pos = f.tell()
                    line = f.readline()
                    if line.strip() == '':
                        empty_line_counter += 1
                    elif line[0] == '*':
                        f.seek(last_pos)
                        EOS = True
                    else:
                        data[key].append(line.strip())
                    if empty_line_counter > 5:
                        EOS = True

    # transform to numpy array when possible
    for key in data.keys():
        try:
            data[key] = np.array(data[key][0].split()).astype(np.float)
        except:
            pass

    return data


def rearrange(X):
    """ See section 5.4 of JWST-STScI-005921 ?

    Parameters
    ----------
    X

    Returns
    -------

    """
    order = 5
    terms = (order + 1) * (order + 2) // 2
    square = np.zeros((order + 1, order + 1))
    XL0 = np.zeros(terms)
    XL1 = np.zeros(terms)

    k = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            square[i, j] = X[k]
            # print ('%15.6e' % X[k], end = '')
            k += 1
    #     print ()
    # print ()
    # Now put in conventiomal layout

    k1 = 0
    for i in range(order + 1):
        for j in range(i + 1):
            XL0[k1] = square[i - j, j]
            k1 += 1

    # Now do L1 set - continue k index, but reset k1
    for i in range(order + 1):
        for j in range(order + 1 - i):
            square[i, j] = X[k]
            # print ('%15.6e' %X[k], end = '')
            k += 1
    #     print ()
    # print ()

    k1 = 0
    for i in range(order + 1):
        for j in range(i + 1):
            XL1[k1] = square[i - j, j]
            k1 += 1

    return XL0, XL1


def reorder(pcfName, verbose=False):
    """Use pcf files"""

    print('\n  =============================================%\n')
    print(pcfName)
    xForward = []
    yForward = []
    xBackward = []
    yBackward = []
    pcf = open(pcfName)  # One of the pcf files
    text = pcf.readline()
    while text != '':
        text = pcf.readline()
        if '*DATE' in text:
            print (pcf.readline())

        if '*FitOrder' in text:
            order = int(pcf.readline())
            terms = (order + 1) * (order + 2) // 2
            print ('Order', order, terms, ' terms')

        if '*xForward' in text:
            # text = pcf.readline()
            # f = text.split()  # Array of terms text strings
            for k in range(terms):
                text = pcf.readline()
                xForward.append(float(text))  # f[k]
            xForward = np.array(xForward)

        if '*xBackward' in text:
            # text = pcf.readline()
            # f = text.split()  # Array of terms text strings
            for k in range(terms):
                text = pcf.readline()
                xBackward.append(float(text))  # f[k]
            xBackward = np.array(xBackward)

        if '*yForward' in text:
            # text = pcf.readline()
            # f = text.split()  # Array of terms text strings
            for k in range(terms):
                text = pcf.readline()
                yForward.append(float(text))  # f[k]
            yForward = np.array(yForward)

        if '*yBackward' in text:
            # text = pcf.readline()
            # f = text.split()  # Array of terms text strings
            for k in range(terms):
                text = pcf.readline()
                yBackward.append(float(text))  # f[k]
            yBackward = np.array(yBackward)
    pcf.close()

    # Now reorder coefficients
    Aarray = np.zeros((order + 1, order + 1))
    Barray = np.zeros((order + 1, order + 1))
    Carray = np.zeros((order + 1, order + 1))
    Darray = np.zeros((order + 1, order + 1))
    terms = (order + 1) * (order + 2) // 2
    A2 = np.zeros(terms)
    B2 = np.zeros(terms)
    C2 = np.zeros(terms)
    D2 = np.zeros(terms)
    k1 = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            Aarray[j, i] = xForward[k1]
            Barray[j, i] = yForward[k1]
            Carray[j, i] = xBackward[k1]
            Darray[j, i] = yBackward[k1]
            k1 += 1
    k2 = 0
    for i in range(order + 1):
        for j in range(i + 1):
            A2[k2] = Aarray[j, i - j]
            B2[k2] = Barray[j, i - j]
            C2[k2] = Carray[j, i - j]
            D2[k2] = Darray[j, i - j]
            k2 += 1

    if verbose:
        print('\n', pcfName)
        print('A')
        polynomial.print_triangle(A2)
        print('\nB')
        polynomial.print_triangle(B2)
        print('\nC')
        polynomial.print_triangle(C2)
        print('\nD')
        polynomial.print_triangle(D2)

    # Convert V2V3 output polynomials to XAN,YAN type
    # print (year, pcfName)
    # print (pcfName)
    year = '2017'
    if year == '2016' and 'GWA2OTE' in pcfName:

        B2 = -B2
        B2[0] = B2[0] - 0.13
        (C2, D2) = polynomial.TwoStep(C2, D2, [0.0, 1.0, 0.0], [-0.13, 0.0, -1.0], 5)
        print('\nAdjusted Polynomials')
        print('A')
        polynomial.print_triangle(A2)
        print('\nB')
        polynomial.print_triangle(B2)
        print('\nC')
        polynomial.print_triangle(C2)
        print('\nD')
        polynomial.print_triangle(D2)

    return A2, B2, C2, D2


def rows(pcfName, new_pcf_format=False):
    print('=============================================')
    xForward = []
    yForward = []
    xBackward = []
    yBackward = []

    pcf = open(pcfName)
    text = pcf.readline()
    print('First Line\n', text)
    while text != '':
        text = pcf.readline()
        if 'Factor' in text:
            text = pcf.readline()
            [xfactor, yfactor] = text.split()
            xfactor = float(xfactor)
            yfactor = float(yfactor)
            print('xfactor', xfactor)
            print('yfactor', yfactor)

        if '*FitOrder' in text:
            text = pcf.readline()
            order = int(text.split()[0])
            print('order', order)

        if '*Rotation' in text:
            rotation = float(pcf.readline())
            print('rotation', rotation)

        if '*InputRotation' in text:
            text = pcf.readline()
            [xCenterIn, yCenterIn] = text.split()
            xCenterIn = float(xCenterIn)
            yCenterIn = float(yCenterIn)
            print('CenterIn', xCenterIn, yCenterIn)

        if '*OutputRotation' in text:
            text = pcf.readline()
            [xCenterOut, yCenterOut] = text.split()
            xCenterOut = float(xCenterOut)
            yCenterOut = float(yCenterOut)
            print('CenterOut', xCenterOut, yCenterOut)

        # if ('OTE' in pcfName) or ('Fore_' in pcfName) or (new_pcf_format is True):  # Different layout
        if ('OTE' in pcfName) or (new_pcf_format is True):  # Different layout
            if '*xForward' in text:
                text = pcf.readline()
                cfList = text.split()
                for cf in cfList:
                    cff = float(cf)
                    xForward.append(cff)  # L0 set
                text = pcf.readline()
                cfList = text.split()
                for cf in cfList:
                    cff = float(cf)
                    xForward.append(cff)  # L1 set
                xForward = np.array(xForward)

            if '*xBackward' in text:
                text = pcf.readline()
                cfList = text.split()
                for cf in cfList:
                    cff = float(cf)
                    xBackward.append(cff)  # L0 set
                text = pcf.readline()
                cfList = text.split()
                for cf in cfList:
                    cff = float(cf)
                    xBackward.append(cff)  # L1 set
                xBackward = np.array(xBackward)

            if '*yForward' in text:
                text = pcf.readline()
                cfList = text.split()
                for cf in cfList:
                    cff = float(cf)
                    yForward.append(cff)  # L0 set
                text = pcf.readline()
                cfList = text.split()
                for cf in cfList:
                    cff = float(cf)
                    yForward.append(cff)  # L1 set
                yForward = np.array(yForward)

            if '*yBackward' in text:
                text = pcf.readline()
                cfList = text.split()
                for cf in cfList:
                    cff = float(cf)
                    yBackward.append(cff)  # L0 set
                text = pcf.readline()
                cfList = text.split()
                for cf in cfList:
                    cff = float(cf)
                    yBackward.append(cff)  # L1 set
                yBackward = np.array(yBackward)

        else:  # Other filter files
            if '*xForward' in text:
                for k in range(42):
                    text = pcf.readline()
                    f = float(text.split()[0])
                    xForward.append(f)
                xForward = np.array(xForward)

                # print ('xForward')
                # print (xForward)

            if '*xBackward' in text:
                for k in range(42):
                    text = pcf.readline()
                    f = float(text.split()[0])
                    xBackward.append(f)
                xBackward = np.array(xBackward)

            if '*yForward' in text:
                for k in range(42):
                    text = pcf.readline()
                    f = float(text.split()[0])
                    yForward.append(f)
                yForward = np.array(yForward)

            if '*yBackward' in text:
                for k in range(42):
                    text = pcf.readline()
                    f = float(text.split()[0])
                    yBackward.append(f)
                yBackward = np.array(yBackward)

    print('Finished reading PCF file')

    # Now generate two SIAF rows First half of each coefficient set gets reorderd and goes in first L0 row
    # Second half goes in second L1 row
    # Design AperName using name of pcf file
    print('pcfName', pcfName)

    if 'OTE' in pcfName:
        ApName1 = 'NRS_SKY_OTEIP'
        # ApName2 = 'DISCARD'
    else:
        i1 = pcfName.find('_')
        i2 = pcfName.find('.')
        filter = (pcfName[i1 + 1:i2])
        ApName1 = 'NRS_' + filter + '_OTEIP_MSA_L0'
        ApName2 = 'NRS_' + filter + '_OTEIP_MSA_L1'

    # print ('\nAL0, AL1')
    (AL0, AL1) = rearrange(xForward)
    # polynomial.print_triangle(AL0)
    # print ()
    # polynomial.print_triangle(AL1)

    # print ('\nBL0, BL1')
    (BL0, BL1) = rearrange(yForward)
    # polynomial.print_triangle(BL0)
    # print ()
    # polynomial.print_triangle(BL1)

    # print ('\nCL0, CL1')
    (CL0, CL1) = rearrange(xBackward)
    # polynomial.print_triangle(CL0)
    # print ()
    # polynomial.print_triangle(CL1)

    # print ('\nDL0, DL1')
    (DL0, DL1) = rearrange(yBackward)
    # polynomial.print_triangle(DL0)
    # print ()
    # polynomial.print_triangle(DL1)

    data = {}
    data['L0'] = {}
    data['L1'] = {}
    data['L0']['A'] = AL0
    data['L1']['A'] = AL1
    data['L0']['B'] = BL0
    data['L1']['B'] = BL1
    data['L0']['C'] = CL0
    data['L1']['C'] = CL1
    data['L0']['D'] = DL0
    data['L1']['D'] = DL1

    return data


#############################
instrument = 'NIRSpec'
test_dir = os.path.join(JWST_TEMPORARY_DATA_ROOT, instrument, 'generate_test')
if not os.path.isdir(test_dir):
    os.makedirs(test_dir)

# regenerate SIAF reference files if needed
if 0:
    generate_reference_files.generate_siaf_detector_layout()
    generate_reference_files.generate_initial_siaf_aperture_definitions(instrument)
    generate_reference_files.generate_siaf_detector_reference_file(instrument)
    generate_reference_files.generate_siaf_ddc_mapping_reference_file(instrument)

# DDC name mapping
ddc_apername_mapping = iando.read.read_siaf_ddc_mapping_reference_file(instrument)

# NIRSpec detected parameters, e.g. XDetSize
siaf_detector_parameters = iando.read.read_siaf_detector_reference_file(instrument)

# Fundamental aperture definitions: names, types, reference positions, dependencies
siaf_aperture_definitions = iando.read.read_siaf_aperture_definitions(instrument)

# definition of the master apertures, the 16 SCAs
detector_layout = iando.read.read_siaf_detector_layout()
master_aperture_names = detector_layout['AperName'].data

# directory containing reference files delivered by IDT
source_data_dir = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, 'delivery')
print ('Loading reference files from directory: {}'.format(source_data_dir))

# XSciRef etc. data for some of the transform apertures, see Section 4.7.1 and Table 1 of JWST-STScI-005921
tiltx_gtp_file = os.path.join(source_data_dir, 'disperser_MIRROR_TiltX_TA.gtp')
tilty_gtp_file = os.path.join(source_data_dir, 'disperser_MIRROR_TiltY_TA.gtp')
disperser_mirror_tiltx = read_pcf_gtp(tiltx_gtp_file)
disperser_mirror_tilty = read_pcf_gtp(tilty_gtp_file)

# TA transforms: mapping of row names in the Calc worksheet to reference files
pcf_file_mapping = {}
pcf_file_mapping['491_GWA'] = 'delivery_SCA491toGWA.pcf'
pcf_file_mapping['492_GWA'] = 'delivery_SCA492toGWA.pcf'
pcf_file_mapping['CLEAR_GWA_OTE'] = 'delivery_CLEAR_GWA2XanYan.pcf'
pcf_file_mapping['F110W_GWA_OTE'] = 'delivery_F110W_GWA2XanYan.pcf'
pcf_file_mapping['F140X_GWA_OTE'] = 'delivery_F140X_GWA2XanYan.pcf'
pcf_data = {}
for field in pcf_file_mapping.keys():
    pcf_data[field] = {}
    pcf_data[field]['A'], pcf_data[field]['B'], pcf_data[field]['C'], pcf_data[field]['D'] = reorder(os.path.join(source_data_dir, pcf_file_mapping[field]), verbose=True)

# reference file delivered by IDT
nirspec_slit_apertures_file = os.path.join(source_data_dir, 'positionsSIAFApertures.fits')
nirspec_slit_apertures_data = Table.read(nirspec_slit_apertures_file)
nirspec_slit_aperture_names = nirspec_slit_apertures_data['SIAF_NAME'].data.astype(str).tolist()

# dictionary that maps NIRSpec nomenclature to SIAF nomenclature
nirspec_slit_apertures_data_mapping = {}
nirspec_slit_apertures_data_mapping['V2Ref'] = 'RefXPOSKY'
nirspec_slit_apertures_data_mapping['V3Ref'] = 'RefYPOSKY'
nirspec_slit_apertures_data_mapping['V3IdlYAngle'] = 'AngleV3'

# compute 'Idl Vertices' from the V2V3 vertices given in nirspec_slit_apertures_data,
# see Section 5.2 of TR and see Calc worksheet in NIRSpec_SIAF.xlsx
for index in [1,2,3,4]:
    nirspec_slit_apertures_data['XIdlVert{}'.format(index)] = -1 * ((nirspec_slit_apertures_data['C{}_XPOSSKY'.format(index)] - nirspec_slit_apertures_data['RefXPOSKY']) * np.cos(np.deg2rad(nirspec_slit_apertures_data['AngleV3'])) - (nirspec_slit_apertures_data['C{}_YPOSSKY'.format(index)] - nirspec_slit_apertures_data['RefYPOSKY']) * np.sin(np.deg2rad(nirspec_slit_apertures_data['AngleV3'])))

    nirspec_slit_apertures_data['YIdlVert{}'.format(index)] = +1 * ((nirspec_slit_apertures_data['C{}_XPOSSKY'.format(index)] - nirspec_slit_apertures_data['RefXPOSKY']) * np.sin(np.deg2rad(nirspec_slit_apertures_data['AngleV3'])) + (nirspec_slit_apertures_data['C{}_YPOSSKY'.format(index)] - nirspec_slit_apertures_data['RefYPOSKY']) * np.cos(np.deg2rad(nirspec_slit_apertures_data['AngleV3'])))

# map aperture names to Fore_*.pcf file names
fore_pcf_file_mapping = {}
fore_pcf_file_mapping['NRS_SKY_OTEIP'] = 'OTE.pcf'
fore_pcf_file_mapping['NRS_CLEAR_OTEIP_MSA_L0'] = 'Fore_CLEAR.pcf'
fore_pcf_file_mapping['NRS_CLEAR_OTEIP_MSA_L1'] = 'Fore_CLEAR.pcf'
fore_pcf_file_mapping['NRS_F070LP_OTEIP_MSA_L0'] = 'Fore_F070LP.pcf'
fore_pcf_file_mapping['NRS_F070LP_OTEIP_MSA_L1'] = 'Fore_F070LP.pcf'
fore_pcf_file_mapping['NRS_F100LP_OTEIP_MSA_L0'] = 'Fore_F100LP.pcf'
fore_pcf_file_mapping['NRS_F100LP_OTEIP_MSA_L1'] = 'Fore_F100LP.pcf'
fore_pcf_file_mapping['NRS_F170LP_OTEIP_MSA_L0'] = 'Fore_F170LP.pcf'
fore_pcf_file_mapping['NRS_F170LP_OTEIP_MSA_L1'] = 'Fore_F170LP.pcf'
fore_pcf_file_mapping['NRS_F290LP_OTEIP_MSA_L0'] = 'Fore_F290LP.pcf'
fore_pcf_file_mapping['NRS_F290LP_OTEIP_MSA_L1'] = 'Fore_F290LP.pcf'
fore_pcf_file_mapping['NRS_F110W_OTEIP_MSA_L0'] = 'Fore_F110W.pcf'
fore_pcf_file_mapping['NRS_F110W_OTEIP_MSA_L1'] = 'Fore_F110W.pcf'
fore_pcf_file_mapping['NRS_F140X_OTEIP_MSA_L0'] = 'Fore_F140X.pcf'
fore_pcf_file_mapping['NRS_F140X_OTEIP_MSA_L1'] = 'Fore_F140X.pcf'

aperture_dict = OrderedDict()
aperture_name_list = siaf_aperture_definitions['AperName'].tolist()

for AperName in aperture_name_list:
    # new aperture to be constructed
    aperture = pysiaf.JwstAperture()
    aperture.AperName = AperName
    aperture.InstrName = siaf_detector_parameters['InstrName'][0].upper()

    # index in the aperture definition table
    aperture_definitions_index = siaf_aperture_definitions['AperName'].tolist().index(AperName)

    # Retrieve basic aperture parameters from definition files
    for attribute in 'XDetRef YDetRef AperType XSciSize YSciSize XSciRef YSciRef'.split():
        # setattr(aperture, attribute, getattr(parent_aperture, attribute))
        value = siaf_aperture_definitions[attribute][aperture_definitions_index]
        if np.ma.is_masked(value):
            value = None
        setattr(aperture, attribute, value)

    aperture.DDCName = 'not set'
    aperture.Comment = None
    aperture.UseAfterDate = '2014-01-01'

    if aperture.AperType not in ['TRANSFORM']:
        aperture.AperShape = siaf_detector_parameters['AperShape'][0]

    if aperture.AperType == 'OSS':
        aperture.VIdlParity = 1
        aperture.DetSciParity = 1
        aperture.DetSciYAngle = 0

    if AperName in ['NRS_FULL_MSA', 'NRS_VIGNETTED_MSA']:
        aperture.VIdlParity = -1

    if aperture.AperType not in ['SLIT', 'TRANSFORM']:
        aperture.XDetSize = siaf_detector_parameters['XDetSize'][0]
        aperture.YDetSize = siaf_detector_parameters['YDetSize'][0]

    # process master apertures
    if AperName in master_aperture_names:
        detector_layout_index = detector_layout['AperName'].tolist().index(AperName)
        for attribute in 'DetSciYAngle DetSciParity VIdlParity'.split():
            setattr(aperture, attribute, detector_layout[attribute][detector_layout_index])

        aperture = process_nirspec_aperture(aperture)

    # SLIT apertures, correspond to physical apertures and other locations in the MSA and SLICER planes
    # the information on the physical location of each aperture in the MSA plane is not recorded in the SIAF
    elif AperName in nirspec_slit_aperture_names:
        index = nirspec_slit_aperture_names.index(AperName)

        # copy parameters from IDT files
        for attribute in nirspec_slit_apertures_data_mapping.keys():
            setattr(aperture, attribute, nirspec_slit_apertures_data[nirspec_slit_apertures_data_mapping[attribute]][index])

        # set ideal vertices
        for attribute in [name for name in nirspec_slit_apertures_data.colnames if 'IdlVert' in name]:
            setattr(aperture, attribute, nirspec_slit_apertures_data[attribute][index])

    # Target Acquisition Transforms: Transformation coefficients from .pcf files ['491_GWA', '492_GWA',
    # 'F140X_GWA_OTE', 'F110W_GWA_OTE', 'CLEAR_GWA_OTE']
    elif AperName in pcf_file_mapping.keys():
        number_of_coefficients = len(pcf_data[AperName]['A'])
        polynomial_degree = np.int((np.sqrt(8 * number_of_coefficients + 1) - 3) / 2)
        aperture.Sci2IdlDeg = polynomial_degree
        k = 0
        # polynomial coefficients for transformation that goes directly from the GWA pupil plane to the sky
        for i in range(polynomial_degree + 1):
            for j in np.arange(i + 1):
                setattr(aperture, 'Sci2IdlX{:d}{:d}'.format(i, j), pcf_data[AperName]['A'][k])
                setattr(aperture, 'Sci2IdlY{:d}{:d}'.format(i, j), pcf_data[AperName]['B'][k])
                setattr(aperture, 'Idl2SciX{:d}{:d}'.format(i, j), pcf_data[AperName]['C'][k])
                setattr(aperture, 'Idl2SciY{:d}{:d}'.format(i, j), pcf_data[AperName]['D'][k])
                k += 1

        # coefficients to apply the reflection in the MIRROR taking into account the correction
        # to the GWA position as derived from the sensor readings and their calibration relation
        aperture.XSciScale = np.float(disperser_mirror_tiltx['CoeffsTemperature00'][0])
        aperture.YSciScale = np.float(disperser_mirror_tilty['CoeffsTemperature00'][0])
        aperture.XSciRef = np.float(disperser_mirror_tiltx['Zeroreadings'][0])
        aperture.YSciRef = np.float(disperser_mirror_tilty['Zeroreadings'][0])
        aperture.DDCName = 'None'

    # TRANSFORM apertures for the conversion between the OTE image plane and the MSA plane
    elif aperture.AperType in ['TRANSFORM']:

        # get the name of the applicable Fore_*.pcf file
        pcf_file = fore_pcf_file_mapping[AperName]

        sequence_string = AperName[-2:]
        if sequence_string not in ['L0', 'L1']:
            sequence_string = 'L0'

        # read .pcf file into dictionary
        fore_pcf = os.path.join(source_data_dir, pcf_file)
        # print('reading {}'.format(fore_pcf))
        fore_pcf_data = read_pcf_gtp(fore_pcf)

        # deal with different formats of the .pcf files
        fore_year = np.int(fore_pcf_data['DATE'][0][0:4])
        if fore_year > 2016:
            new_pcf_format = True
        else:
            new_pcf_format = False

        aperture.XSciRef = fore_pcf_data['InputRotationCentre'][0]
        aperture.YSciRef = fore_pcf_data['InputRotationCentre'][1]
        aperture.V2Ref = fore_pcf_data['OutputRotationCentre'][0]
        aperture.V3Ref = fore_pcf_data['OutputRotationCentre'][1]
        aperture.XSciScale = fore_pcf_data['Factor'][0]
        aperture.YSciScale = fore_pcf_data['Factor'][1]
        aperture.V3IdlYAngle = fore_pcf_data['Rotation'][0]

        data = rows(fore_pcf, new_pcf_format=new_pcf_format)

        polynomial_degree = 5
        aperture.Sci2IdlDeg = polynomial_degree
        k = 0
        #  IDT parametric model convention “forward” direction maps to Sci2IdlX
        for i in range(polynomial_degree + 1):
            for j in np.arange(i + 1):
                setattr(aperture, 'Sci2IdlX{:d}{:d}'.format(i, j), data[sequence_string]['A'][k])  # *xForward
                setattr(aperture, 'Sci2IdlY{:d}{:d}'.format(i, j), data[sequence_string]['B'][k])  # *yForward
                setattr(aperture, 'Idl2SciX{:d}{:d}'.format(i, j), data[sequence_string]['C'][k])  # *xBackward
                setattr(aperture, 'Idl2SciY{:d}{:d}'.format(i, j), data[sequence_string]['D'][k])  # *yBackward
                k += 1

        aperture.DDCName = 'None'

    aperture_dict[AperName] = aperture

# second pass to set parameters for apertures that depend on other apertures
for AperName in aperture_name_list:
    index = siaf_aperture_definitions['AperName'].tolist().index(AperName)
    aperture = aperture_dict[AperName]

    parent_aperture_name = siaf_aperture_definitions['parent_apertures'][index]
    if (parent_aperture_name is not None) and (not np.ma.is_masked(parent_aperture_name)):
        aperture._parent_apertures = parent_aperture_name
        parent_aperture = aperture_dict[aperture._parent_apertures]

        if aperture.AperType in ['FULLSCA', 'OSS']:
            aperture = process_nirspec_aperture(aperture, verbose=False)

        if siaf_aperture_definitions['dependency_type'][index] == 'default':
            aperture.VIdlParity = parent_aperture.VIdlParity

        # first MIMF field point inherits properties from parent SLIT aperture
        elif siaf_aperture_definitions['dependency_type'][index] == 'FP1MIMF':
            idlvert_attributes = ['XIdlVert{}'.format(i) for i in [1,2,3,4]] + ['YIdlVert{}'.format(i) for i in [1,2,3,4]]
            for attribute in 'V2Ref V3Ref V3IdlYAngle VIdlParity'.split() + idlvert_attributes:
                setattr(aperture, attribute, getattr(parent_aperture, attribute))

    aperture_dict[AperName] = aperture


# sort SIAF entries in the order of the aperture definition file
aperture_dict = OrderedDict(sorted(aperture_dict.items(), key=lambda t: aperture_name_list.index(t[0])))

# third pass to set DDCNames apertures, which depend on other apertures
ddc_siaf_aperture_names = np.array([key for key in ddc_apername_mapping.keys()])
ddc_v2 = np.array([aperture_dict[aperture_name].V2Ref for aperture_name in ddc_siaf_aperture_names])
ddc_v3 = np.array([aperture_dict[aperture_name].V3Ref for aperture_name in ddc_siaf_aperture_names])
for AperName in aperture_name_list:
    if aperture_dict[AperName].AperType not in ['TRANSFORM']:
        separation_tel_from_ddc_aperture = np.sqrt((aperture_dict[AperName].V2Ref - ddc_v2)**2 + (aperture_dict[AperName].V3Ref - ddc_v3)**2)
        aperture_dict[AperName].DDCName = ddc_apername_mapping[ddc_siaf_aperture_names[np.argmin(separation_tel_from_ddc_aperture)]]

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
    filenames = pysiaf.iando.write.write_jwst_siaf(aperture_collection, basepath=pre_delivery_dir,
                                                   file_format=['xml', 'xlsx'])

    pre_delivery_siaf = pysiaf.Siaf(instrument, basepath=pre_delivery_dir)

    # compare new SIAF with PRD version
    for compare_to in [pysiaf.JWST_PRD_VERSION]:
        if compare_to == 'NIRSpec_SIAF_fullsca':
            ref_siaf = pysiaf.Siaf(instrument, filename=os.path.join(pre_delivery_dir,
                                                                     'NIRSpec_SIAF_fullsca.xml'))
        elif compare_to == 'NIRSpec_SIAF_bugfix-only':
            ref_siaf = pysiaf.Siaf(instrument, filename=os.path.join(pre_delivery_dir,
                                                                     'NIRSpec_SIAF_bugfix-only.xml'))
        elif compare_to == 'PRDOPSSOC-027':
            ref_siaf = pysiaf.Siaf(instrument, filename=os.path.join(pre_delivery_dir, 'NIRSpec_SIAF-027.xml'))
        elif compare_to == 'PRDOPSSOC-M-026':
            ref_siaf = pysiaf.Siaf(instrument, filename=os.path.join(JWST_PRD_DATA_ROOT.replace(
                pysiaf.JWST_PRD_VERSION, compare_to), 'NIRSpec_SIAF.xml'))
        else:
            # compare new SIAF with PRD version
            ref_siaf = pysiaf.Siaf(instrument)

        tags = {'reference': compare_to, 'comparison': 'pre_delivery'}

        compare.compare_siaf(pre_delivery_siaf, reference_siaf_input=ref_siaf,
                             fractional_tolerance=1e-6, report_dir=pre_delivery_dir, tags=tags)

        compare.compare_inspection_figures(pre_delivery_siaf, reference_siaf_input=ref_siaf,
                                           report_dir=pre_delivery_dir, tags=tags,
                                           skipped_aperture_type=['TRANSFORM'],
                                           selected_aperture_name=['NRS1_FP1MIMF', 'NRS1_FP2MIMF', 'NRS1_FP3MIMF', 'NRS2_FP4MIMF', 'NRS2_FP5MIMF'],
                                           mark_ref=True, xlimits=(100, 700), ylimits=(-700, -100),
                                           filename_appendix='MIMF_apertures')

        compare.compare_transformation_roundtrip(pre_delivery_siaf,
                                                 reference_siaf_input=ref_siaf, tags=tags,
                                                 report_dir=pre_delivery_dir,
                                                 skipped_aperture_type=['TRANSFORM', 'SLIT'],
                                                 selected_aperture_name=['NRS1_FULL', 'NRS2_FULL', 'NRS1_FULL_OSS', 'NRS2_FULL_OSS'])


        compare.compare_inspection_figures(pre_delivery_siaf, reference_siaf_input=ref_siaf,
                                           report_dir=pre_delivery_dir, tags=tags,
                                           skipped_aperture_type=['TRANSFORM'])

    # run some tests on the new SIAF
    from pysiaf.tests import test_nirspec

    print('\nRunning regression test of pre_delivery_siaf against IDT test_data:')
    test_nirspec.test_against_test_data(siaf=pre_delivery_siaf)

    print('\nRunning nirspec_aperture_transforms test for pre_delivery_siaf')
    test_nirspec.test_nirspec_aperture_transforms(siaf=pre_delivery_siaf, verbose=False)

    print('\nRunning nirspec_slit_transforms test for pre_delivery_siaf')
    test_nirspec.test_nirspec_slit_transformations(siaf=pre_delivery_siaf, verbose=False)

    new_siaf = pre_delivery_siaf

else:
    # filename = pysiaf.iando.write.write_jwst_siaf(aperture_collection, basepath=test_dir, label='pysiaf')
    [filename] = pysiaf.iando.write.write_jwst_siaf(aperture_collection, basepath=test_dir, file_format=['xml'])
    print('SIAFXML written in {}'.format(filename))

    # compare to SIAFXML produced the old way
    # ref_siaf = pysiaf.Siaf(instrument, os.path.join(test_dir , '{}'.format('NIRISS_SIAF_2017-10-18.xml')))
    ref_siaf = pysiaf.Siaf(instrument)
    # ref_siaf = pysiaf.Siaf(instrument, os.path.join(test_dir, 'NIRSpec_SIAF_2018-04-13.xml'))
    # new_siaf = pysiaf.Siaf(instrument, os.path.join(test_dir, 'NIRSpec_SIAF_2018-04-13.xml'))
    new_siaf = pysiaf.Siaf(instrument, filename)

    report_dir = os.path.join(REPORTS_ROOT, instrument)
    comparison_aperture_names = [AperName for AperName in aperture_name_list if 'MIMF' in AperName]
    # comparison_aperture_names = [AperName for AperName, aperture in aperture_dict.items() if
    #                              aperture.AperType == 'SLIT']
    # comparison_aperture_names = [AperName for AperName, aperture in aperture_dict.items() if
    #                              aperture.AperType in ['FULLSCA', 'OSS']]
    # comparison_aperture_names = pcf_file_mapping.keys()

    # comparison_aperture_names = ['NRS_SKY_OTEIP']
    # comparison_aperture_names = ['NRS_SKY_OTEIP']
    comparison_aperture_names = ['NRS1_FULL', 'NRS2_FULL', 'NRS1_FULL_OSS', 'NRS2_FULL_OSS']
    # compare.compare_siaf(new_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-6,
    #                      selected_aperture_name=['NRS1_FULL', 'NRS2_FULL', 'NRS1_FULL_OSS', 'NRS2_FULL_OSS'])
    # compare.compare_siaf(new_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-6,
    #                      selected_aperture_name=['NRS_SKY_OTEIP'])
    # compare.compare_siaf(new_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-6,
    #                      selected_aperture_name=comparison_aperture_names, report_dir=report_dir)
    compare.compare_siaf(new_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-6)
    # tools.compare_siaf_xml(ref_siaf, new_siaf)

# selected_aperture_name = [AperName for AperName in aperture_name_list if ('GWA' not in AperName) and \
#                           ('MSA' not in AperName) and ('SKY' not in AperName)]
# # run roundtrip test on all apertures
# compare.compare_transformation_roundtrip(new_siaf, reference_siaf_input=ref_siaf,
#                                          selected_aperture_name=selected_aperture_name)
