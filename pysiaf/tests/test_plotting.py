#!/usr/bin/env python
"""Test plotting methods of Aperture and Siaf class.

Authors
-------

    Johannes Sahlmann

"""

import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl
import pytest

from ..constants import JWST_TEMPORARY_DATA_ROOT
from ..siaf import Siaf, plot_master_apertures

# @pytest.mark.skip(reason="Need to figure out how to set backend")
def test_aperture_plotting():
    """Generate aperture plots and save to png.

    """
    save_plot = True
    plot_dir = os.path.join(JWST_TEMPORARY_DATA_ROOT)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    instrument = 'NIRISS'
    siaf = Siaf(instrument)

    # plot all apertures in SIAF
    pl.figure(figsize=(4, 4), facecolor='w', edgecolor='k')
    for aperture_name, aperture in siaf.apertures.items():
        aperture.plot(color='b')
    pl.title('{} apertures'.format(instrument))
    if save_plot:
        fig_name = os.path.join(plot_dir, '{}_apertures.png'.format(instrument))
        pl.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

    assert os.path.isfile(fig_name)

    # plot 'master' apertures
    pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
    plot_master_apertures(mark_ref=True, color='b')
    pl.title('JWST master apertures')
    if save_plot:
        fig_name = os.path.join(plot_dir, 'JWST_master_apertures.png')
        pl.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

    assert os.path.isfile(fig_name)

    # plot HST apertures
    siaf = Siaf('HST')
    aperture_names = ['FGS1', 'FGS2', 'FGS3', 'IUVIS1FIX', 'IUVIS2FIX', 'JWFC1FIX', 'JWFC2FIX']

    pl.figure(figsize=(4, 4), facecolor='w', edgecolor='k')
    for aperture_name in aperture_names:
        siaf[aperture_name].plot(color='r', fill_color='darksalmon', mark_ref=True)
    ax = pl.gca()
    ax.set_aspect('equal')
    ax.invert_yaxis()
    pl.title('Some HST apertures')
    if save_plot:
        fig_name = os.path.join(plot_dir, 'HST_apertures.png')
        pl.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

    assert os.path.isfile(fig_name)
