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

def checkinv(A, B, C, D, order):
    '''Check that forward and inverse transformations are consistent'''
    [x,y] = np.mgrid[-200:201:200, -200:201:200]
    v2 = poly(A, x, y, order)
    v3 = poly(B, x, y, order)
    x2 = poly(C, v2, v3, order)
    y2 = poly(D, v2, v3, order)

    print ('     x         y          V2        V3        xp        yp       dx        dy')
    ######   -200.0000 -200.0000  -12.5849  -12.5939 -199.7371 -200.2741    0.2629   -0.2741

    for i in range(3):
        for j in range(3):
            print(8*'%10.4f' %(x[i,j],y[i,j],v2[i,j],v3[i,j],x2[i,j],y2[i,j], x2[i,j]-x[i,j], y2[i,j]-y[i,j]))

    # Summary
    print ('\nShifts', (x2-x).mean(), (y2-y).mean())
    print ('RMS {:10.2e} {:10.2e}'.format( (x2-x).std(), (y2-y).std() ))
    print()

    return

def get_polynomial_coefficients(aperture, group):
    '''Extract one of the four polynomial groups for an aperture
    The group will be one of 'Sci2IdlX', 'Sci2IdlY', 'Idl2SciX', 'Idl2SciY' '''
    coeffList = [getattr(aperture, s) for s in DISTORTION_ATTRIBUTES if group in s]
    coefficients = np.array(coeffList, dtype=float)
    return coefficients

def matchV2V3(apName1, apName2):
    """ Use the V2V3 from aperture named apName1 in aperture apName2, modifying XDetRef, YDetRef,
    XSciRef YSciRef to match
    Also  copy or shift the polynomial coefficients to reflect the new reference point origin
    and for NIRCam recalculate angles. """

    instruments = {'NRC':'NIRCam', 'NRS':'NIRSpec', 'FGS':'FGS', 'MIR':'MIRI', 'NIS':'NIRISS'}
    insCode = apName1[:3]
    ins = instruments[insCode]
    print ('\n******************************************************')
    print ('Instrument', ins)
    siaf = Siaf(ins)
    ap1 = siaf.apertures[apName1]
    ap2 = siaf.apertures[apName2]
    assert (ap2.AperType in ['FULLSCA', 'SUBARRAY', 'ROI']), "2nd aperture must be pixel-based"
    order = ap1.Sci2IdlDeg
    V2Ref1 = ap1.V2Ref
    V3Ref1 = ap1.V3Ref
    newV2Ref = V2Ref1 # in all cases
    newV3Ref = V3Ref1
    print ('Current Vref', ap2.V2Ref, ap2.V3Ref)
    print ('Shift to    ', V2Ref1, V3Ref1)

    # Detector and Science axes may go in opposite directions
    ySign = cos(radians(ap2.DetSciYAngle))
    xSign = ap2.DetSciParity * ySign


    if apName1[:5] == apName2[:5]: # Common detector - most things copied
        print ('Detector ', apName1[:5])
        newXDetRef = ap1.XDetRef
        newYDetRef = ap1.YDetRef
        newXSciRef = ap2.XSciRef + xSign*(newXDetRef-ap2.XDetRef) 
        newYSciRef = ap2.YSciRef + ySign*(newYDetRef-ap2.YDetRef)
        newV3SciXAngle = ap1.V3SciXAngle
        newV3SciYAngle = ap1.V3SciYAngle
        newV3IdlYAngle = ap1.V3IdlYAngle
        newA = get_polynomial_coefficients(ap1, 'Sci2IdlX')
        newB = get_polynomial_coefficients(ap1, 'Sci2IdlY')
        newC = get_polynomial_coefficients(ap1, 'Idl2SciX')
        newD = get_polynomial_coefficients(ap1, 'Idl2SciY')
        print ('{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(ap2.V2Ref, ap2.V3Ref, ap2.XDetRef, ap2.YDetRef,
                                                                                       ap2.XSciRef,ap2.YSciRef, ap2.V3SciXAngle, ap2.V3SciYAngle, ap2.V3IdlYAngle))
        print ('{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} '.format(newV2Ref, newV3Ref, newXDetRef, newYDetRef,
                                                                                        newXSciRef,newYSciRef, newV3SciXAngle, newV3SciYAngle, newV3IdlYAngle))
    else:
        # Need to work in aperture 2 Ideal coordinate system
        print ('Detector 1', ap1.AperName[:5], '  Detector 2', ap2.AperName[:5])
        V2Ref2 = ap2.V2Ref
        V3Ref2 = ap2.V3Ref
        theta0 = ap2.V3IdlYAngle
        print ('Initial VRef', V2Ref2, V3Ref2)
        print ('Initial theta', theta0)
        theta = radians(theta0)
        # Express reference point 1 V2V3  in Ideal2 coordinates
        XIdl2 = ap2.VIdlParity*((V2Ref1-V2Ref2)*cos(theta) - (V3Ref1-V3Ref2)*sin(theta))
        YIdl2 = (V2Ref1-V2Ref2)*sin(theta) + (V3Ref1-V3Ref2)*cos(theta)
        print ('Idl2 %10.4f %10.4f' %(XIdl2, YIdl2) )

        A = get_polynomial_coefficients(ap2, 'Sci2IdlX')
        B = get_polynomial_coefficients(ap2, 'Sci2IdlY')
        C = get_polynomial_coefficients(ap2, 'Idl2SciX')
        D = get_polynomial_coefficients(ap2, 'Idl2SciY')
        print ('\nA')
        triangle(A,5)
        print ('B')
        triangle(B,5)
        print('C')
        triangle(C,5)
        print('D')
        triangle(D,5)

        xPixel = poly(C, XIdl2, YIdl2, order)
        yPixel = poly(D, XIdl2, YIdl2, order)
        print ('New pixel position', xPixel, yPixel)

        print ('Current Sci position', ap2.XSciRef, ap2.YSciRef)
        #(dXSciRef, dYSciRef, err, steps) = invert(A, B, XIdl2, YIdl2, order)
        #print ('invert {:10.3e} error after {:2d} steps'.format(err,steps))
        #print ('Sci shift', dXSciRef, dYSciRef)
        #print ('New Sci position {:8.2f} {:8.2f}'.format(newXSciRef, newYSciRef))
        newXDetRef = ap2.XDetRef + xSign*dXSciRef
        newYDetRef = ap2.YDetRef + ySign*dYSciRef
        #print ('V shift', V2Ref2-V2Ref1, V3Ref2-V3Ref1)
        #print('New DetRef', newXDetRef, newYDetRef)

        # Use convert
        print ('VRef1', V2Ref1, V3Ref1)
        (xsc, ysc) = ap2.convert(V2Ref1, V3Ref1, 'tel', 'sci')
        print ('Convert method', xsc, ysc)
        # Convert back
        (v2c, v3c) = ap2.convert(xsc, ysc, 'sci', 'tel')
        print ('Regained values     ', v2c, v3c)
        dXSciRef = xsc - xPixel
        dYSciRef = ysc - yPixel
        newXSciRef = ap2.XSciRef + dXSciRef
        newYSciRef = ap2.YSciRef + dYSciRef
        AS = ShiftCoeffs(A, dXSciRef, dYSciRef, order)
        BS = ShiftCoeffs(B, dXSciRef, dYSciRef, order)
        print ('New origin', AS[0], BS[0])
        CS = C
        DS = D
        CS[0] = CS[0] - xPixel
        DS[0] = DS[0] - yPixel

        print ('AS')
        triangle(AS,order)
        print ('BS')
        triangle(BS, order)
        print ('CS')
        triangle(CS,order)
        print('DS')
        triangle(DS, order)
        print('\nABCDS')
        checkinv(AS,BS,CS,DS, order)

        # For NIRCam only, adjust angles
        if ins == 'NIRCam':
            newV3IdlYAngle = degrees(atan2(-AS[2], BS[2])) # Everything rotates by this amount
            if abs(newV3IdlYAngle) > 90.0: newV3IdlYAngle = newV3IdlYAngle - copysign(180, newV3IdlYAngle)
            print ('New angle', newV3IdlYAngle)
            as0 = AS[0]
            bs0 = BS[0]
            AS[0] = 0.0
            BS[0] = 0.0
            newA = AS*cos(radians(newV3IdlYAngle)) + BS*sin(radians(newV3IdlYAngle))
            newB = -AS*sin(radians(newV3IdlYAngle)) + BS*cos(radians(newV3IdlYAngle))
            print ('\nnewA')
            triangle(newA,5)
            print('newB')
            triangle(newB,5)
            c0 = CS[0]
            d0 = DS[0]
            print ('Inverse offsets', c0, d0)
            CS[0] = 0.0
            DS[0] = 0.0
            print ('Zero centered coeffs ')
            checkinv(AS, BS, CS, DS, order)

            newC = RotateCoeffs(CS, -newV3IdlYAngle, order)
            newD = RotateCoeffs(DS, -newV3IdlYAngle, order)
            print ('Rotate Coeffs by newV3IdlYAngle')
            print('newC')
            triangle(newC, order)
            print ('newD')
            triangle(newD, order)
            print ('\nTest final coefficients')
            checkinv(newA, newB, newC, newD, order)
            newV3SciXAngle = ap2.V3SciXAngle + newV3IdlYAngle
            newV3SciYAngle = ap2.V3SciYAngle + newV3IdlYAngle
            newV3IdlYAngle = ap2.V3IdlYAngle + newV3IdlYAngle

        print ('\nParameters')
        print ('  XDetRef   YDetRef    XSciRef   YSciRef   V3SciXAngle V3SciYAngle V3IdlYAngle')
        print ('{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(ap2.XDetRef, ap2.YDetRef,
                                                                                       ap2.XSciRef, ap2.YSciRef, ap2.V3SciXAngle, ap2.V3SciYAngle, ap2.V3IdlYAngle))
        print ('{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} '.format(newXDetRef, newYDetRef,
                                                                                        newXSciRef, newYSciRef, newV3SciXAngle, newV3SciYAngle, newV3IdlYAngle))


    return (newV2Ref, newV3Ref, newXDetRef, newYDetRef, newXSciRef, newYSciRef)