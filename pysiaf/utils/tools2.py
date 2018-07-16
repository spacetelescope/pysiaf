"""A collection of helper functions to support pysiaf calculations and functionalities

Authors
-------

    - Colin Cox

References
----------



"""

import sys
import numpy as np
from math import *
import copy
import pylab as pl

from astropy.table import Table

from ..aperture import PRD_REQUIRED_ATTRIBUTES_ORDERED
from .polynomial import poly, ShiftCoeffs, Rotate, FlipY, FlipX, rotate_coefficients, RotateCoeffs, triangle, invert
from ..siaf import Siaf
from pysiaf.aperture import DISTORTION_ATTRIBUTES
from pysiaf.utils import tools


def match_v2v3(aperture_1, aperture_2, verbose=False):
    """ Use the V2V3 from aperture_1 in aperture_2  modifying XDetRef, YDetReƒf,
    XSciRef YSciRef to match
    Also  copy or shift the polynomial coefficients to reflect the new reference point origin
    and for NIRCam recalculate angles. """

    instrument_codes = {'NRC':'NIRCam', 'MIR':'MIRI', 'FGS':'FGS', 'NRS':'NIRSpec', 'NIS':'NIRISS'}
    code = aperture_1.AperName[:3]
    instrument = instrument_codes[code]
    print (instrument)
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


    if aperture_1.AperName[:5] == aperture_2.AperName[:5]: # Common detector - most things copied
        print('Detector ', apName1[:5])
        newXDetRef = aperture_1.XDetRef
        newYDetRef = aperture_1.YDetRef
        newXSciRef = aperture_2.XSciRef + xSign*(newXDetRef-aperture_2.XDetRef) 
        newYSciRef = aperture_2.YSciRef + ySign*(newYDetRef-aperture_2.YDetRef)
        newV3SciXAngle = aperture_1.V3SciXAngle
        newV3SciYAngle = aperture_1.V3SciYAngle
        newV3IdlYAngle = aperture_1.V3IdlYAngle
        coefficients = aperture_1.get_polynomial_coefficients()
        newA = coefficients['Sci2IdlX']
        newB = coefficients['Sci2IdlY']
        newC = coefficients['Idl2SciX']
        newD = coefficients['Idl2SciY']
        print('NewA')
        triangle(newA, order)
    else:
        # Need to work in aperture 2 Ideal coordinate system
        print('Detector 1', aperture_1.AperName[:5], '  Detector 2', aperture_2.AperName[:5])
        V2Ref2 = aperture_2.V2Ref
        V3Ref2 = aperture_2.V3Ref
        theta0 = aperture_2.V3IdlYAngle
        print('Initial VRef', V2Ref2, V3Ref2)
        print('Initial theta', theta0)
        theta = radians(theta0)

        coefficients = aperture_1.get_polynomial_coefficients()
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

        (stat, xmean, ymean, xstd, ystd) = tools.compute_roundtrip_error(A, B, C, D,
                                           verbose=True, instrument = instrument)
        print('Round trip', stat, xmean, ymean, xstd, ystd)

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
            triangle(AS,order)
            print('BS')
            triangle(BS, order)
            print('CS')
            triangle(CS,order)
            print('DS')
            triangle(DS, order)
            print('\nABCDS')
            
        (stat, xmean, ymean, xstd, ystd) = tools.compute_roundtrip_error(AS,BS,CS,DS,
                                           verbose=True, instrument=instrument )
        print('Round trip', stat, xmean, ymean, xstd, ystd)

        # For NIRCam only, adjust angles
        if instrument == 'NIRCam':
            newV3IdlYAngle = degrees(atan2(-AS[2], BS[2])) # Everything rotates by this amount
            if abs(newV3IdlYAngle) > 90.0: newV3IdlYAngle = newV3IdlYAngle - copysign(180, newV3IdlYAngle)
            print('New angle', newV3IdlYAngle)
            newA = AS*cos(radians(newV3IdlYAngle)) + BS*sin(radians(newV3IdlYAngle))
            newB = -AS*sin(radians(newV3IdlYAngle)) + BS*cos(radians(newV3IdlYAngle))
            print('\nnewA')
            triangle(newA,5)
            print('newB')
            triangle(newB,5)

            newC = RotateCoeffs(CS, -newV3IdlYAngle, order)
            newD = RotateCoeffs(DS, -newV3IdlYAngle, order)
            print('Rotate Coeffs by newV3IdlYAngle')
            print('newC')
            triangle(newC, order)
            print('newD')
            triangle(newD, order)
            print('\nTest final coefficients')
            (stat, xmean, ymean, xstd, ystd) = tools.compute_roundtrip_error(AS, BS, CS, DS,
                                               verbose=True, instrument=instrument)
            print('Round trip', stat, xmean, ymean, xstd, ystd)

            newV3IdlXAngle = aperture_2.V3SciXAngle + newV3IdlYAngle
            newV3SciYAngle = aperture_2.V3SciYAngle + newV3IdlYAngle
            newV3IdlYAngle = aperture_2.V3IdlYAngle + newV3IdlYAngle

        print('\nParameters')
        print('  XDetRef   YDetRef    XSciRef   YSciRef   V3SciXAngle V3SciYAngle V3IdlYAngle')
        print('{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(aperture_2.XDetRef, aperture_2.YDetRef,
              aperture_2.XSciRef, aperture_2.YSciRef, aperture_2.V3SciXAngle, aperture_2.V3SciYAngle, aperture_2.V3IdlYAngle))
        #print('{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} '.format(newXDetRef, newYDetRef,
        #                                                                               newXSciRef, newYSciRef, newV3SciXAngle, newV3SciYAngle, newV3IdlYAngle))


    return aperture_2