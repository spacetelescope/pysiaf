#!/usr/bin/env python
"""Tests for the pysiaf aperture classes.

Authors
-------

    Johannes Sahlmann

"""

import numpy as np
import pytest

from ..aperture import HstAperture
from ..iando import read
from ..siaf import Siaf, get_jwst_apertures
from ..utils.tools import get_grid_coordinates

@pytest.fixture(scope='module')
def siaf_objects():
    """Return list of Siaf objects.

    :return:
    """
    # for instrument in 'NIRISS NIRCam MIRI FGS NIRSpec'.split():
    siafs = []
    for instrument in 'NIRISS FGS'.split():
        siaf = Siaf(instrument)
        siafs.append(siaf)
    return siafs


def test_hst_aperture_init():
    """Test the initialization of an HstAperture object."""
    hst_aperture = HstAperture()
    hst_aperture.a_v2_ref = -100.
    assert hst_aperture.a_v2_ref == hst_aperture.V2Ref #, 'HST aperture initialisation failed')

def test_jwst_aperture_transforms(siaf_objects, verbose=False):
    """Test transformations between frames.

    Transform back and forth between frames and verify that input==output.

    Parameters
    ----------
    siaf_objects
    verbose

    """
    labels = ['X', 'Y']
    threshold = 0.1

    from_frame = 'sci'
    to_frames = 'det idl tel'.split()

    x_sci = np.linspace(-10, 10, 3)
    y_sci = np.linspace(10, -10, 3)

    for siaf in siaf_objects:
        for aper_name in siaf.apertures.keys():
            skip = False

            # aperture
            aperture = siaf[aper_name]

            if (aperture.AperType in ['COMPOUND', 'TRANSFORM']) or (
                    siaf.instrument in ['NIRCam', 'MIRI', 'NIRSpec'] and
                    aperture.AperType == 'SLIT'):
                skip = True

            if skip is False:
                # test transformations
                if verbose:
                    print('testing {} {}'.format(siaf.instrument, aper_name))

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

def test_jwst_aperture_vertices(siaf_objects):
    """Test the JwstAperture vertices by rederiving them and comparing to SIAF.

    Rederive Idl vertices and compare with content of SIAFXML

    """
    verbose = False

    threshold = 0.2
    labels = ['X', 'Y']

    for siaf in siaf_objects:
        for aper_name in siaf.apertures.keys():
            skip = False

            #aperture
            aperture = siaf[aper_name]

            if (aperture.AperType in ['COMPOUND', 'TRANSFORM']) or \
                    (siaf.instrument in ['NIRCam', 'MIRI', 'NIRSpec']
                     and aperture.AperType == 'SLIT'):
                skip = True

            if skip is False:
                if verbose:
                    print('testing {} {}'.format(siaf.instrument, aper_name))

                # Idl corners from Sci attributes (XSciRef, XSciSize etc.)
                x_idl_vertices_rederived, y_idl_vertices_rederived = aperture.corners('idl',
                                                                                      rederive=True)

                # Idl corners from SIAFXML
                x_idl_vertices = np.array([getattr(aperture, 'XIdlVert{:d}'.format(j)) for j in [1, 2, 3, 4]])
                y_idl_vertices = np.array([getattr(aperture, 'YIdlVert{:d}'.format(j)) for j in [1, 2, 3, 4]])

                if verbose:
                    print(x_idl_vertices, x_idl_vertices_rederived)
                    print(y_idl_vertices, y_idl_vertices_rederived)

                x_mean_error = np.abs(np.mean(x_idl_vertices) - np.mean(x_idl_vertices_rederived))
                y_mean_error = np.abs(np.mean(y_idl_vertices) - np.mean(y_idl_vertices_rederived))

                if verbose:
                    for i, error in enumerate([x_mean_error, y_mean_error]):
                        print('{} {}: Error in {}Idl_vertices is {:02.6f})'.format(siaf.instrument, aper_name, labels[i], error))

                assert x_mean_error < threshold
                assert y_mean_error < threshold

def test_raw_transformations(verbose=False):
    """Test raw_to_sci and sci_to_raw transformations"""
    siaf_detector_layout = read.read_siaf_detector_layout()
    master_aperture_names = siaf_detector_layout['AperName'].data
    apertures_dict = {'instrument': siaf_detector_layout['InstrName'].data}
    apertures_dict['pattern'] = master_aperture_names
    apertures = get_jwst_apertures(apertures_dict, exact_pattern_match=True)

    grid_amplitude = 2048
    x_raw, y_raw = get_grid_coordinates(10, (grid_amplitude/2, grid_amplitude/2), grid_amplitude)

    labels = ['X', 'Y']
    threshold = 0.1
    from_frame = 'raw'
    to_frame = 'sci'

    # compute roundtrip error
    for aper_name, aperture in apertures.apertures.items():
        forward_transform = getattr(aperture, '{}_to_{}'.format(from_frame, to_frame))
        backward_transform = getattr(aperture, '{}_to_{}'.format(to_frame, from_frame))

        x_out, y_out = backward_transform(*forward_transform(x_raw, y_raw))
        x_mean_error = np.mean(np.abs(x_raw - x_out))
        y_mean_error = np.mean(np.abs(y_raw - y_out))
        for i, error in enumerate([x_mean_error, y_mean_error]):
            if verbose:
                print('{} {}: Error in {}<->{} {}-transform is {:02.6f})'.format(
                    aperture.InstrName, aper_name, from_frame, to_frame, labels[i], error))
            assert error < threshold


