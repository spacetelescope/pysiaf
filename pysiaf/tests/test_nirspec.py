#!/usr/bin/env python
"""Test NIRSpec transformations internally and against the test dataset delivered by IDT.

Authors
-------

    Johannes Sahlmann

"""

import os

from astropy.io import fits
from astropy.table import Table
import numpy as np
import pylab as pl
# import pytest



from ..constants import JWST_TEMPORARY_DATA_ROOT, TEST_DATA_ROOT
from ..siaf import Siaf


instrument = 'NIRSpec'

def test_against_test_data():
    """NIRSpec test data comparison.

    Mean and RMS difference between the IDT computations and the pysiaf computations are
    computed and compared against acceptable thresholds.

    """
    siaf = Siaf(instrument)
    # directory that holds SIAF XML file
    # test_dir = os.path.join(JWST_TEMPORARY_DATA_ROOT, instrument, 'generate_test')
    # siaf_xml_file = os.path.join(test_dir, '{}_SIAF.xml'.format(instrument))
    # siaf = Siaf(instrument, filename=siaf_xml_file)

    test_data_dir = os.path.join(TEST_DATA_ROOT, instrument)


    include_tilt = False

    if include_tilt is False:
        ta_transform_data_dir = os.path.join(test_data_dir, 'testDataSet_TA', 'testDataNoTilt')

    filter_list = 'CLEAR F110W F140X'.split()
    sca_list = ['SCA491', 'SCA492']
    # filter_list = 'CLEAR'.split()
    # sca_list = ['SCA491']
    # sca_list = ['SCA492']

    difference_metrics = {}
    index = 0
    for sca_name in sca_list:
        for filter_name in filter_list:

            test_data_file = os.path.join(ta_transform_data_dir, 'testDataTA_{}{}.fits'.format(sca_name, filter_name))
            test_data = Table(fits.getdata(test_data_file))

            if sca_name == 'SCA491':
                AperName = 'NRS1_FULL_OSS'
            elif sca_name == 'SCA492':
                AperName = 'NRS2_FULL_OSS'

            aperture = siaf[AperName]

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
            if include_tilt is False:
                # last equation in Secion 5.5.2
                test_data['pysiaf_GWAin_X'] = -1 * test_data['pysiaf_GWAout_X']
                test_data['pysiaf_GWAin_Y'] = -1 * test_data['pysiaf_GWAout_Y']

            # transform to OTE frame (XAN, YAN)
            test_data['pysiaf_XAN'], test_data['pysiaf_YAN'] = aperture.gwa_to_ote(
                test_data['pysiaf_GWAin_X'], test_data['pysiaf_GWAin_Y'], filter_name)

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



                    print('{} {} SCA_to_OTE transform comparison to {:>10}  {:>10} MEAN={:+1.3e} RMS={:1.3e}'.format(sca_name, filter_name, AperName, parameter_name, difference_metrics['diff_{}_{}'.format(parameter_name, 'mean')][index], difference_metrics['diff_{}_{}'.format(parameter_name, 'rms')][index]))

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
