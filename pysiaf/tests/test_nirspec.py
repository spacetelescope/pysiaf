#!/usr/bin/env python
"""Test NIRSpec transformations internally and against the test dataset delivered by IDT.

Authors
-------

    Johannes Sahlmann

"""

import os

from astropy.io import fits
from astropy.table import Table
import copy
import numpy as np
import pylab as pl
# import pytest



from ..constants import JWST_TEMPORARY_DATA_ROOT, TEST_DATA_ROOT, JWST_SOURCE_DATA_ROOT
from ..siaf import Siaf


instrument = 'NIRSpec'

def test_against_test_data(siaf=None):
    """NIRSpec test data comparison.

    Mean and RMS difference between the IDT computations and the pysiaf computations are
    computed and compared against acceptable thresholds.

    """
    if siaf is None:
        siaf = Siaf(instrument)
    else:
        # safeguard against side-effects when running several tests on a provided siaf, e.g.
        # setting tilt to non-zero value
        siaf = copy.deepcopy(siaf)
    # directory that holds SIAF XML file
    # test_dir = os.path.join(JWST_TEMPORARY_DATA_ROOT, instrument, 'generate_test')
    # siaf_xml_file = os.path.join(test_dir, '{}_SIAF.xml'.format(instrument))
    # siaf = Siaf(instrument, filename=siaf_xml_file)

    # test_data_dir = os.path.join(TEST_DATA_ROOT, instrument)
    test_data_dir = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, 'delivery', 'test_data')

    print(' ')
    for include_tilt in [False, True]:

        if include_tilt is False:
            ta_transform_data_dir = os.path.join(test_data_dir, 'TA_testDataNoTilt')
        else:
            ta_transform_data_dir = os.path.join(test_data_dir, 'TA_testDataWithGWATilt')

        filter_list = 'CLEAR F110W F140X'.split()
        sca_list = ['SCA491', 'SCA492']

        difference_metrics = {}
        index = 0
        for sca_name in sca_list:
            for filter_name in filter_list:

                test_data_file = os.path.join(ta_transform_data_dir, 'testDataTA_{}{}.fits'.format(sca_name, filter_name))
                test_data = Table(fits.getdata(test_data_file))
                if include_tilt is False:
                    tilt = None
                else:
                    test_header = fits.getheader(test_data_file)
                    tilt = (np.float(test_header['GWA_XTIL']), np.float(test_header['GWA_YTIL']))

                if sca_name == 'SCA491':
                    AperName = 'NRS1_FULL_OSS'
                elif sca_name == 'SCA492':
                    AperName = 'NRS2_FULL_OSS'

                aperture = siaf[AperName]
                aperture.filter_name = filter_name
                aperture.tilt = tilt

                if 0:
                    pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k'); pl.clf()
                    aperture.plot(name_label=True)
                    siaf['NRS2_FULL_OSS'].plot(name_label=True)
                    pl.plot(test_data['XAN']*u.deg.to(u.arcsecond), test_data['YAN']*u.deg.to(u.arcsecond), 'b.')
                    pl.show()

                    1/0


                # SCI to GWA detector side (Step 1. in Sections 2.3.3, 5.5.2 of JWST-STScI-005921 , see also Table 4.7.1)
                test_data['pysiaf_GWAout_X'], test_data['pysiaf_GWAout_Y'] = aperture.sci_to_gwa(test_data['SCA_X'], test_data['SCA_Y'])

                # effect of mirror, transform from GWA detector side to GWA skyward side
                test_data['pysiaf_GWAin_X'], test_data['pysiaf_GWAin_Y'] = aperture.gwaout_to_gwain(test_data['pysiaf_GWAout_X'] , test_data['pysiaf_GWAout_Y'])

                # transform to OTE frame (XAN, YAN)
                test_data['pysiaf_XAN'], test_data['pysiaf_YAN'] = aperture.gwa_to_ote(
                    test_data['pysiaf_GWAin_X'], test_data['pysiaf_GWAin_Y'])

                for axis_name in ['X', 'Y']:
                    for parameter_name in ['{}AN'.format(axis_name)]:

                        # compute differences between SIAF implementation and IDT test dataset
                        test_data['difference_{}'.format(parameter_name)] = test_data['pysiaf_{}'.format(parameter_name)] - test_data['{}'.format(parameter_name)]

                        for key_seed in ['mean', 'rms']:
                            key_name = 'diff_{}_{}'.format(parameter_name, key_seed)
                            if key_name not in difference_metrics.keys():
                                difference_metrics[key_name] = []
                            if key_seed == 'mean':
                                difference_metrics[key_name].append(np.mean(test_data['difference_{}'.format(parameter_name)]))
                            elif key_seed == 'rms':
                                difference_metrics[key_name].append(np.std(test_data['difference_{}'.format(parameter_name)]))



                        print('{} {} SCA_to_OTE transform comparison to {:>10}  tilt={} {:>10} MEAN={:+1.3e} RMS={:1.3e}'.format(sca_name, filter_name, AperName, include_tilt, parameter_name, difference_metrics['diff_{}_{}'.format(parameter_name, 'mean')][index], difference_metrics['diff_{}_{}'.format(parameter_name, 'rms')][index]))

                        assert difference_metrics['diff_{}_{}'.format(parameter_name, 'mean')][index] < 1e-9
                        assert difference_metrics['diff_{}_{}'.format(parameter_name, 'rms')][index] < 5e-9

                        if 0:
                            threshold = 1e-6
                            if (difference_metrics['diff_{}_{}'.format(parameter_name, 'rms')][index] > threshold):
                                pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k'); pl.clf()
                                pl.quiver(test_data['SCA_X'], test_data['SCA_Y'], test_data['difference_XAN'], test_data['difference_YAN'], angles='xy')
                                pl.title('Difference IDT and pysiaf')
                                pl.show()

                index += 1


def test_nirspec_aperture_transforms(verbose=False, siaf=None):
    """Test transformations between frames.

    Transform back and forth between frames and verify that input==output.

    Parameters
    ----------
    verbose

    """
    if siaf is None:
        siaf = Siaf(instrument)
    else:
        siaf = copy.deepcopy(siaf)

    labels = ['X', 'Y']
    threshold = 0.2

    from_frame = 'sci'
    to_frames = 'det gwa idl tel'.split()

    x_sci = np.linspace(-10, 10, 3)
    y_sci = np.linspace(10, -10, 3)

    for include_tilt in [False, True]:

        for aper_name in siaf.apertures.keys():
            skip = False

            # aperture
            aperture = siaf[aper_name]
            # offset slightly from default tilt values

            if (aperture.AperType in ['COMPOUND', 'TRANSFORM', 'SLIT']) or ('_FULL' not in aper_name):
                skip = True

            if skip is False:
                if(include_tilt is True):
                   # set tilt to a representative off nominal value
                    gwa_aperture = getattr(aperture, '_CLEAR_GWA_OTE')
                    rx0 = getattr(gwa_aperture, 'XSciRef')
                    ry0 = getattr(gwa_aperture, 'YSciRef')
                    aperture.tilt = (ry0 - 0.002, rx0 - 0.01)
                   
                # test transformations
                if verbose:
                    print('testing {} {} Tilt={}'.format(siaf.instrument, aper_name, aperture.tilt))

                for to_frame in to_frames:
                    forward_transform = getattr(aperture, '{}_to_{}'.format(from_frame, to_frame))
                    backward_transform = getattr(aperture, '{}_to_{}'.format(to_frame, from_frame))

                    x_out, y_out = backward_transform(*forward_transform(x_sci, y_sci))
                    x_mean_error = np.mean(np.abs(x_sci - x_out))
                    y_mean_error = np.mean(np.abs(y_sci - y_out))
                    for i, error in enumerate([x_mean_error, y_mean_error]):
                        if verbose:
                            print('{} {}: Error in {}<->{} {}-transform is {:02.6f})'.format(
                                siaf.instrument, aper_name, from_frame, to_frame, labels[i], error))
                        assert error < threshold


def test_nirspec_slit_transformations(verbose=False, siaf=None):
    """Test that slit to detector transforms give the same answer as the equivalent SCA transform.

    Check that reference_point and corners work for slit.

    Parameters
    ----------
    verbose : bool

    Authors
    -------
    Charles Proffitt
    Johannes Sahlmann

    """
    if siaf is None:
        siaf = Siaf(instrument)
    else:
        siaf = copy.deepcopy(siaf)

    threshold = 0.010  # arc-seconds
    pixel_threshold = 10 * threshold

    labels = ['X', 'Y']
    from_frame = 'sci'
    to_frames = 'det tel'.split()
    x_sci = np.linspace(-10, 10, 3)
    y_sci = np.linspace(10, -10, 3)


    # for aper_name in 'NRS_S1600A1_SLIT NRS_S200B1_SLIT NRS_FIELD1_MSA4 NRS1_FULL'.split():
    for aper_name in siaf.apertures.keys():
        skip = False
        aperture = siaf[aper_name]

        if (aperture.AperType not in ['SLIT']) or ('MIMF' in aper_name) or (
        not hasattr(aperture, '_parent_aperture')):
            skip = True

        if skip is False:
            parent_aperture = siaf[aperture._parent_aperture.AperName]
            if verbose:
                print(
                'testing {} {} parent {}'.format(siaf.instrument, aper_name, parent_aperture.AperName))

            # verify that correct reference point can be retrieved
            v2ref, v3ref = aperture.reference_point('tel')
            assert np.abs(v2ref - aperture.V2Ref) < threshold
            assert np.abs(v3ref - aperture.V3Ref) < threshold

            # verify that we get the same tel to sci transform whether using slit or parent
            # aperture name
            xsciref, ysciref = aperture.reference_point('sci')
            xscidref, yscidref = parent_aperture.tel_to_sci(v2ref, v3ref)
            xsciaref, ysciaref = aperture.tel_to_sci(v2ref, v3ref)
            error = np.sqrt((xsciref - xscidref) ** 2 + (ysciref - yscidref) ** 2)
            if verbose:
                print(
                '{} {}: Error in reference point {:02.6f} pixels. (parent aperture is {})'.format(siaf.instrument, aper_name,
                                                                    error, parent_aperture.AperName))
            assert error < pixel_threshold

            # verify that corners can be retrieved and check 1st vertice
            ixc, iyc = aperture.corners('idl')
            assert np.abs(ixc[0] - aperture.XIdlVert1) < pixel_threshold
            assert np.abs(iyc[0] - aperture.YIdlVert1) < pixel_threshold

            # verify that we get the same tel to det transform whether using slit or parent
            # aperture name
            v2c, v3c = aperture.corners('tel')
            xc, yc = aperture.corners('det')
            xdc, ydc = parent_aperture.tel_to_det(v2c, v3c)
            xac, yac = aperture.tel_to_det(v2c, v3c)
            xic, yic = aperture.idl_to_det(ixc, iyc)
            error = np.max(np.abs(
                np.concatenate((xc - xdc, yc - ydc, xc - xac, yc - yac, xc - xic, yc - yic))))
            if verbose:
                print(
                '{} {}: Max error in corners {:02.6f} pixels.'.format(siaf.instrument, aper_name,
                                                                      error))
            assert error < pixel_threshold

            #testing roundtrip error
            for to_frame in to_frames:
                forward_transform = getattr(aperture, '{}_to_{}'.format(from_frame, to_frame))
                backward_transform = getattr(aperture, '{}_to_{}'.format(to_frame, from_frame))

                x_out, y_out = backward_transform(*forward_transform(x_sci, y_sci))
                x_mean_error = np.mean(np.abs(x_sci - x_out))
                y_mean_error = np.mean(np.abs(y_sci - y_out))
                for i, error in enumerate([x_mean_error, y_mean_error]):
                    if verbose:
                        print('{} {}: Error in {}<->{} {}-transform is {:02.6f})'.format(
                            siaf.instrument, aper_name, from_frame, to_frame, labels[i], error))
                    assert error < pixel_threshold

