#! /usr/bin/env python
"""Module to handle Science Instrument Aperture Files (SIAF)

The siaf module defined classes and functions to support working
with SIAF files. The main class is ApertureCollection, and the Siaf
class inherits from it. The class methods support basic operations and
plotting.

ApertureCollection is essentially a container for a set of pysiaf
 aperture objects.

Authors
-------

    - Johannes Sahlmann

References
----------

    Many of the Siaf class methods were adapted from the jwxml package
    written by Marshall Perrin and Joseph Long
    (https://github.com/mperrin/jwxml).

"""
from collections import OrderedDict
import re

from astropy.table import Table
import numpy as np
import pylab as pl

from .iando import read



class ApertureCollection(object):
    """Structure class for an aperture collection, e.g. read from a SIAF file."""

    def __init__(self, aperture_dict=None):
        if aperture_dict is not None:
            if type(aperture_dict) not in [dict, OrderedDict]:
                raise RuntimeError('Argument has to be of type `dict`')
            self.apertures = aperture_dict

            # table of content
            self.generate_toc()

    def generate_toc(self):
        """Generate a table of contents."""

        toc = Table()
        for attribute in 'InstrName AperName AperShape AperType'.split():
            toc[attribute] = [getattr(a, attribute) for key, a in self.apertures.items()]
        self.toc = toc

    def __getitem__(self, key):
        return self.apertures[key]

    def __str__(self):
        return '{} ApertureCollection containing {} apertures'.format(self.observatory, len(self.apertures))

    def list_apertures(self, instrument=None, shape=None):
        """Print a list of apertures to screen."""
        idx1 = range(len(self.toc))
        idx2 = range(len(self.toc))
        if instrument is not None:
            idx1 = np.where(self.toc['InstrName'] == instrument)[0]
        if shape is not None:
            idx2 = np.where(self.toc['AperShape'] == shape)[0]
        idx = np.intersect1d(idx1, idx2)
        self.toc[idx].pprint()

    def __len__(self):
        return len(self.apertures)

def get_jwst_apertures(apertures_dict, verbose=0, include_oss_apertures=False, exact_pattern_match=False):
    """Return ApertureCollection that corresponds to constraints specified in apertures_dict

    :param apertures_dict:
    :param verbose:
    :return:

    Example:
    apertures_dict = {'instrument':['FGS']}
    apertures_dict['pattern'] = ['FULL']*len(apertures_dict['instrument'])
    fgs_apertures_all = get_jwst_apertures(apertures_dict)



    """

    # tolerate inconsistent capitalization
    if 'NIRCAM' in apertures_dict['instrument']:
        instrument_names = np.array([s.replace('NIRCAM', 'NIRCam') for s in apertures_dict['instrument']])
        apertures_dict['instrument'] = instrument_names
    if 'NIRSPEC' in apertures_dict['instrument']:
        instrument_names = np.array([s.replace('NIRSPEC', 'NIRSpec') for s in apertures_dict['instrument']])
        apertures_dict['instrument'] = instrument_names

    all_aps = {}
    for j, instrument in enumerate(apertures_dict['instrument']):
        siaf = Siaf(instrument)
        for AperName, aperture in siaf.apertures.items():
            if exact_pattern_match:
                matched = AperName == apertures_dict['pattern'][j]
            else:
                pattern = re.compile(apertures_dict['pattern'][j])
                matched = pattern.search(AperName)
            if matched:
                if (include_oss_apertures is False) and ('_OSS' in AperName):
                    continue
                all_aps[AperName] = aperture

    return ApertureCollection(aperture_dict=all_aps)

def plot_all_apertures(subarrays=True, showorigin=True, showchannels=True, **kwargs):
    """ Plot All instrument """

    for instr in ['NIRCam', 'NIRISS', 'NIRSpec', 'FGS', 'MIRI']:
        aps = Siaf(instr, **kwargs)
        print("{0} has {1} apertures".format(aps.instrument, len(aps)))

        aps.plot(clear=False, subarrays=subarrays, **kwargs)
        if showorigin: aps.plot_detector_origin()
        if showchannels: aps.plot_detector_channels()

def plot_main_apertures(label=False, darkbg=False, detector_channels=False, **kwargs):
    if darkbg:
        col_imaging = 'aqua'
        col_coron = 'lime'
        col_msa = 'violet'
    else:
        col_imaging = 'blue'
        col_coron = 'green'
        col_msa = 'magenta'

    nircam = Siaf('NIRCam')
    niriss = Siaf('NIRISS')
    fgs = Siaf('FGS')
    nirspec = Siaf('NIRSpec')
    miri = Siaf('MIRI')

    im_aps = [
        nircam['NRCA5_FULL'],
        nircam['NRCB5_FULL'],
        niriss['NIS_CEN'],
        miri['MIRIM_ILLUM'],
        fgs['FGS1_FULL'],
        fgs['FGS2_FULL']
    ]

    for letter in ['A', 'B']:
        for num in range(5):
            im_aps.append(nircam['NRC{}{}_FULL'.format(letter, num + 1)])

    coron_aps = [
        nircam['NRCA2_MASK210R'],
        nircam['NRCA4_MASKSWB'],
        nircam['NRCA5_MASK335R'],
        nircam['NRCA5_MASK430R'],
        nircam['NRCA5_MASKLWB'],
        nircam['NRCB3_MASKSWB'],
        nircam['NRCB1_MASK210R'],
        nircam['NRCB5_MASK335R'],
        nircam['NRCB5_MASK430R'],
        nircam['NRCB5_MASKLWB'],
        miri['MIRIM_MASK1065'],
        miri['MIRIM_MASK1140'],
        miri['MIRIM_MASK1550'],
        miri['MIRIM_MASKLYOT']
    ]
    msa_aps = [nirspec['NRS_FULL_MSA' + str(n + 1)] for n in range(4)]
    msa_aps.append(nirspec['NRS_S1600A1_SLIT'])  # square aperture

    for aplist, col in zip([im_aps, coron_aps, msa_aps], [col_imaging, col_coron, col_msa]):
        for ap in aplist:
            ap.plot(color=col, frame='Tel', name_label=label, **kwargs)
            if detector_channels:
                try:
                    ap.plot_detector_channels('Tel')
                except TypeError:
                    pass


    # pl.xlim((-800,800))
    # pl.ylim((-1300, 300))

    # ensure V2 increases to the left
    ax = pl.gca()
    xlim = ax.get_xlim()

    if xlim[0] < xlim[1]:
        ax.set_xlim(xlim[::-1])

def plot_master_apertures(**kwargs):
    """Plot only master apertures contours

    :return:
    """
    siaf_detector_layout = read.read_siaf_detector_layout()
    master_aperture_names = siaf_detector_layout['AperName'].data
    apertures_dict = {'instrument': siaf_detector_layout['InstrName'].data}
    apertures_dict['pattern'] = master_aperture_names
    apertures = get_jwst_apertures(apertures_dict, exact_pattern_match=True)
    # print('Plotting {} master apertures'.format(len(apertures.apertures)))
    for AperName, aperture in apertures.apertures.items():
        aperture.plot(**kwargs)

    # ensure V2 increases to the left
    ax = pl.gca()
    ax.set_aspect('equal')
    xlim = ax.get_xlim()
    if xlim[0] < xlim[1]:
        ax.set_xlim(xlim[::-1])

class Siaf(ApertureCollection):
    """Science Instrument Aperture File class.

    This is a class interface to SIAF information, e.g. stored in an XML file in the PRD.
    It enables apertures retrieval by name, plotting, and other functionality.
    See the Aperture class for the detailed implementation of the transformations.

    Adapted from https://github.com/mperrin/jwxml

    The HST case is treated here as an instrument, because it's single SIAF contains all apertures of all HST-instruments

    Examples
    ---------

    fgs_siaf = SIAF('FGS')
    fgs_siaf.apernames                # returns a list of aperture names
    ap = fgs_siaf['FGS1_FULL']        # returns an aperture object
    ap.plot(frame='Tel')              # plot one aperture
    fgs_siaf.plot()                   # plot all apertures in this file

    """

    def __init__(self, instrument, filename=None, basepath=None, AperNames=None):
        """
        Read a SIAF from disk

        Parameters
        -----------
        instrument : string
            one of 'NIRCam', 'NIRSpec', 'NIRISS', 'MIRI', 'FGS'; case sensitive.
        basepath : string
            Directory to look in for SIAF files
        filename : string, optional
            Alternative method to specify a specific SIAF XML file.

        """
        super().__init__()

        if instrument is None:
            return

        elif instrument not in ['NIRCam', 'NIRSpec', 'NIRISS', 'MIRI', 'FGS', 'HST']:
            raise ValueError(
                'Invalid instrument name: {0}. Note that this is case sensitive.'.format(instrument))

        self.instrument = instrument

        if instrument == 'HST':
            self.apertures = read.read_hst_siaf()
            self.observatory = 'HST'
        else:
            self.apertures = read.read_jwst_siaf(instrument, filename=filename, basepath=basepath)
            self.observatory = 'JWST'

    def __repr__(self):
        return "<pysiaf.Siaf object Instrument={} >".format(self.instrument)

    def __str__(self):
        return '{} {} Siaf with {} apertures'.format(self.observatory, self.instrument, len(self))

    def _getFullApertures(self):
        """ Return whichever subset of apertures correspond to the entire detectors. This is a
        helper function for the various plotting routines following"""

        fullaps = []
        if self.instrument == 'NIRCam':
            fullaps.append(self.apertures['NRCA5_FULL'])
            fullaps.append(self.apertures['NRCB5_FULL'])
        elif self.instrument == 'NIRSpec':
            fullaps.append(self.apertures['NRS_FULL_MSA1'])
            fullaps.append(self.apertures['NRS_FULL_MSA2'])
            fullaps.append(self.apertures['NRS_FULL_MSA3'])
            fullaps.append(self.apertures['NRS_FULL_MSA4'])
        elif self.instrument == 'NIRISS':
            fullaps.append(self.apertures['NIS_CEN'])
        elif self.instrument == 'MIRI':
            fullaps.append(self.apertures['MIRIM_FULL'])
        elif self.instrument == 'FGS':
            fullaps.append(self.apertures['FGS1_FULL'])
            fullaps.append(self.apertures['FGS2_FULL'])
        return fullaps

    def delete_aperture(self, aperture_name):
        """Remove an aperture from the Siaf.

        :param aperture_name: str or list
        :return:
        """
        for aper_name in list(aperture_name):
            del self.apertures[aper_name]

    @property
    def apernames(self):
        """ List of aperture names defined in this SIAF"""
        return self.apertures.keys()

    def plot(self, frame='tel', names=None, label=True, units=None, clear=True, annotate=False,
             mark_ref=False, subarrays=True, **kwargs):
        """ Plot all apertures in this SIAF

        Parameters
        -----------
        names : list of strings
            A subset of aperture names, if you wish to plot only a subset
        subarrays : bool
            Plot all the minor subarrays if True, else just plot the "main" apertures
        label : bool
            Add text labels stating aperture names
        units : str
            one of 'arcsec', 'arcmin', 'deg'
        clear : bool
            Clear plot before plotting (set to false to overplot)
        annotate : bool
            Add annotations for detector (0,0) pixels
        mark_ref : bool
            Add markers for the reference (V2Ref, V3Ref) point in each apertyre
        frame : str
            Which coordinate system to plot in: 'Tel', 'Idl', 'Sci', 'Det'

        Other matplotlib standard parameters may be passed in via **kwargs
        to adjust the style of the displayed lines.

        """
        if clear:
            pl.clf()
        ax = pl.subplot(111)
        ax.set_aspect('equal')

        # which list of apertures to iterate over?
        if subarrays:
            iterable = self.apertures.values
        else:
            iterable = self._getFullApertures

        for ap in iterable():
            if names is not None:
                if ap.AperName not in names: continue

            ap.plot(frame=frame, name_label=label, ax=ax, units=None, mark_ref=mark_ref, **kwargs)
            if annotate:
                ap.plot_detector_origin(frame=frame)
        ax.set_xlabel('V2 (arcsec)')
        ax.set_ylabel('V3 (arcsec)')

        if frame == 'Tel' or frame == 'Idl':
            # enforce V2 increasing toward the left
            ax.autoscale_view(True, True, True)
            xlim = ax.get_xlim()
            if xlim[1] > xlim[0]: ax.set_xlim(xlim[::-1])
            ax.set_autoscalex_on(True)

        self._last_plot_frame = frame

    def plot_detector_origin(self, which='both', frame=None):
        """ Mark on the plot the detector's origin in Det and Sci coordinates

        Parameters
        -----------
        which : str
            Which detector origin to plot: 'both', 'Det', 'Sci'
        frame : str
            Which coordinate system to plot in: 'Tel', 'Idl', 'Sci', 'Det'
            Optional if you have already called plot() to specify a
            coordinate frame.

        """
        if frame is None: frame = self._last_plot_frame
        for ap in self._getFullApertures():
            ap.plot_detector_origin(frame=frame, which=which)

    def plot_detector_channels(self, frame=None):
        """ Mark on the plot the various detector readout channels

        These are depicted as alternating light/dark bars to show the
        regions read out by each of the output amps.

        Parameters
        ----------
        frame : str
            Which coordinate system to plot in: 'Tel', 'Idl', 'Sci', 'Det'
            Optional if you have already called plot() to specify a
            coordinate frame.

        """
        raise NotImplementedError

        if frame is None: frame = self._last_plot_frame
        for ap in self._getFullApertures():
            ap.plot_detector_channels(frame=frame)