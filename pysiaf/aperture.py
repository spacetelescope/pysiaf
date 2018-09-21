"""Module to handle SIAF apertures.

The aperture module defined classes and functions to support working
with SIAF apertures. The main class is Aperture, JwstAperture and
HstAperture inherit from it. The class methods support transformations
between any of the 4 SIAF frames (detector, science/DMS, ideal,
V-frame/telescope). Aperture attributes correspond to entries in the
SIAF xml files in the Project Reference Database (PRD), and are
checked for format compliance. Methods for aperture validation and
plotting are provided. The nomenclature is adapted from the JWST
SIAF document Cox & Lallo: JWST-STScI-001550.

Authors
-------
    - Johannes Sahlmann

References
----------
    Numerous contributions and code snippets by Colin Cox were
    incorporated.

    Some methods were adapted from the jwxml package written by
    Marshall Perrin and Joseph Long (https://github.com/mperrin/jwxml).

    Some of the polynomial transformation code was adapted from
    Bryan Hilbert's ramp_simulator code (e.g. https://github.com/
    spacetelescope/ramp_simulator/blob/master/read_siaf_table.py)


Use
---

Dependencies
------------

Notes
-----


TODO
----

    check for sanity of entries, sciref vs. detref, number of valid distortion parameters versus
    degree
    a.verify()

"""


from __future__ import absolute_import, print_function, division

import copy
import os
import numpy as np
import pylab as pl
import subprocess
import sys

from astropy.modeling import models
from astropy.table import Table
import astropy.units as u
import matplotlib

from .utils import rotations, projection
from .utils.tools import an_to_tel, tel_to_an
from .iando import read

# global variables

# shorthands for supported coordinate systems
FRAMES = ('det', 'sci', 'idl', 'tel')

# list of attributes for the distortion coefficients up to degree 5
POLYNOMIAL_COEFFICIENT_NAMES = 'Sci2IdlX Sci2IdlY Idl2SciX Idl2SciY'.split()
DISTORTION_ATTRIBUTES = []
for i in range(5 + 1):
    for j in np.arange(i + 1):
        for name in POLYNOMIAL_COEFFICIENT_NAMES:
            DISTORTION_ATTRIBUTES.append('{}{:d}{:d}'.format(name, i, j))

# list of attributes that have to be defined for a new aperture
VALIDATION_ATTRIBUTES = ('InstrName AperName AperType AperShape '
                         'XDetSize YDetSize XDetRef YDetRef '
                         'XSciSize YSciSize XSciRef YSciRef '
                         'V2Ref V3Ref V3IdlYAngle VIdlParity '
                         'DetSciYAngle DetSciParity '
                         'V3SciXAngle V3SciYAngle '
                         'Sci2IdlDeg'.split()) + DISTORTION_ATTRIBUTES

# list of attributes written to the JWST SIAFXML required by the JWST PRD
# the order of the XML tags in the SIAFXML is relevant, therefore define IRCD order here
# see JWST PRDS IRCD Volume III: S&OC Subsystems (JWST-STScI-000949) Table 4-3
# SIAF_XML_FIELD_FORMAT = read.read_siaf_xml_field_format_reference_file(
#     'NIRCam')  # use NIRCam as model temporarily
SIAF_XML_FIELD_FORMAT = read.read_siaf_xml_field_format_reference_file()  # generic format for all instruments
PRD_REQUIRED_ATTRIBUTES_ORDERED = list(SIAF_XML_FIELD_FORMAT['field_name'])

# As per JWST PRDS IRCD Volume III: S&OC Subsystems (JWST-STScI-000949) Table 4-3,
# these attributes have to be integers, DetSciYAngle to be discussed (because it differs from SI
# to SI)
INTEGER_ATTRIBUTES = list(
    SIAF_XML_FIELD_FORMAT['field_name'][SIAF_XML_FIELD_FORMAT['format'] == 'integer'])

# these attributes have to be strings
STRING_ATTRIBUTES = list(
    SIAF_XML_FIELD_FORMAT['field_name'][SIAF_XML_FIELD_FORMAT['format'] == 'string'])

# these attributes have to be doubles/floats
FLOAT_ATTRIBUTES = np.setdiff1d(PRD_REQUIRED_ATTRIBUTES_ORDERED,
                                INTEGER_ATTRIBUTES + STRING_ATTRIBUTES)

# according to SIAFXML in the PRD (January 2018) the following attributes can be None
ATTRIBUTES_THAT_CAN_BE_NONE = 'XDetSize YDetSize XDetRef YDetRef XSciSize YSciSize XSciRef ' \
                              'YSciRef XSciScale YSciScale DetSciYAngle DetSciParity V3SciXAngle ' \
                              'V3SciYAngle Comment Sci2IdlDeg'.split() + DISTORTION_ATTRIBUTES

# to account for NIRSpec:
ATTRIBUTES_THAT_CAN_BE_NONE += 'AperShape V2Ref V3Ref V3IdlYAngle VIdlParity XIdlVert1 XIdlVert2 ' \
                               'XIdlVert3 XIdlVert4 YIdlVert1 YIdlVert2 YIdlVert3 YIdlVert4'.split()

# NIRSpec target acquisition filters
NIRSPEC_TA_FILTER_NAMES = 'CLEAR F110W F140X'.split()

# private functions
def _telescope_transform_model(from_sys, to_sys, par, angle):
    """
    Creates an astropy.modeling.Model object
    for the undistorted ("ideal") to V2V3 coordinate translation

    angle has to be in radians

    sics_to_v2v3 (HST)
    x_v2v3 = v2_origin + parity * x_sics * np.cos(np.deg2rad(theta_deg)) + y_sics * np.sin(
    np.deg2rad(theta_deg))
    y_v2v3 = v3_origin - parity * x_sics * np.sin(np.deg2rad(theta_deg)) + y_sics * np.cos(
    np.deg2rad(theta_deg))

    adapted from https://github.com/spacetelescope/ramp_simulator/blob/master/read_siaf_table.py
    """
    if from_sys != 'tel' and to_sys != 'tel':
        raise ValueError(
            'This function is designed to generate the transformation either to or from V2V3.')

    # cast the transform functions as 1st order polynomials
    xc = {}
    yc = {}
    if to_sys == 'tel':
        xc['c1_0'] = par * np.cos(angle)
        xc['c0_1'] = np.sin(angle)
        yc['c1_0'] = (0. - par) * np.sin(angle)
        yc['c0_1'] = np.cos(angle)

    if from_sys == 'tel':
        xc['c1_0'] = par * np.cos(angle)
        xc['c0_1'] = par * (0. - np.sin(angle))
        yc['c1_0'] = np.sin(angle)
        yc['c0_1'] = np.cos(angle)

    # 0,0 coeff set to zero, because offsets/shifts are handled external to this function
    xc['c0_0'] = 0
    yc['c0_0'] = 0

    xmodel = models.Polynomial2D(1, **xc)
    ymodel = models.Polynomial2D(1, **yc)

    return xmodel, ymodel

def _x_from_polar(x0, radius, phi_rad):
    """ Convert polar to rectangular x coordinate"""
    return x0 + radius * np.sin(phi_rad)

def _y_from_polar(y0, radius, phi_rad):
    """ Convert polar to rectangular x coordinate"""
    return y0 + radius * np.cos(phi_rad)


class Aperture(object):
    """A class for aperture definitions of astronomical telescopes.

    HST and JWST SIAF entries are supported via class inheritance.
    Frames, transformations, conventions and property/attribute names are as defined for JWST in
    JWST-STScI-001550.

    4 Coordinate systems are supported:
        * Detector:  pixels, in SIAF detector read out axes orientation as defined in SIAF ("det").
                     This system differs from the DMS detector frame definition.
        * Science:   pixels, in conventional DMS axes orientation ("sci")
        * Ideal:     arcsecs, tangent plane projection relative to aperture reference location. (
        "idl")
        * Telescope: arcsecs, spherical V2,V3 ("tel")

    Example
    ========

    ap = some_siaf['desired_aperture_name']     # extract one aperture from a Siaf object

    ap.det_to_tel(1024, 512)                    # convert pixel coordinates to V2V3 / tel coords.
                                                # takes pixel coords, returns arcsec

    ap.idl_to_sci( 10, 3)                       # convert idl coords to sci pixels
                                                # takes arcsec, returns pixels

    There exist methods for all of the possible
    {tel,idl,sci,det}_to_{tel,idl,sci,det} combinations.

    Frames can also be defined by strings:
    ap.convert(1024, 512, from_frame='det', to_frame='tel')  # same as first example above

    ap.corners('tel')                           # Get aperture vertices/corners in tel frame
    ap.reference_point('tel')                   # Get the reference point defined in the SIAF

    ap.plot('tel')                              # plot coordinates in idl frame
    """

    def __init__(self):
        self._observatory = None
        for key in PRD_REQUIRED_ATTRIBUTES_ORDERED:
            self.__dict__[key] = None

        # attribute that indicates whether the aperture has been validated for formal correctness
        self.__dict__['_initial_attributes_validated'] = False

        # attribute that controls whether DVA is corrected for when transforming from Ideal to
        # tel/V2V3 frame
        self.__dict__['_correct_dva'] = False
        self.__dict__['_dva_parameters'] = None

        # parent apertures, if any
        self.__dict__['_parent_apertures'] = None

    def __setattr__(self, key, value):
        """Set an aperture attribute and verify that is has the correct format.

        Parameters
        ----------
        key
        value

        """

        if (key == 'AperType') and (value not in self._accepted_aperture_types):
            raise AttributeError(
                '{} attributes has to be one of {}'.format(key, self._accepted_aperture_types))

        if (value is None) and (key in ATTRIBUTES_THAT_CAN_BE_NONE):
            pass
        elif (value is None) and (key in ['DDCName']) and (
            self.AperType in ['TRANSFORM', None]):  # NIRSpec case
            pass
        elif (key in INTEGER_ATTRIBUTES) and (type(value) not in [int, np.int64]):
            raise AttributeError('pysiaf Aperture attribute `{}` has to be an integer.'.format(key))
        elif (key in STRING_ATTRIBUTES) and (type(value) not in [str, np.str_]):
            raise AttributeError(
                'pysiaf Aperture attribute `{}` has to be a string (tried to assign it type {'
                '}).'.format(
                    key, type(value)))
        elif (key in FLOAT_ATTRIBUTES) and (type(value) not in [float, np.float32, np.float64]):
            raise AttributeError('pysiaf Aperture attribute `{}` has to be a float.'.format(key))

        self.__dict__[key] = value

    def __str__(self):
        """Return string describing the instance."""
        return '{} {} aperture named {}'.format(self.observatory, self.InstrName, self.AperName)

    def __repr__(self):
        return "<pysiaf.Aperture object AperName={0} >".format(self.AperName)

    def closed_polygon_points(self, to_frame, rederive=True):
        """
        Compute closed polygon points of aperture outline. Used for plotting and path generation.
        :param to_frame:
        :return:
        """
        points_x, points_y = self.corners(to_frame, rederive=rederive)
        return points_x[np.append(np.arange(len(points_x)), 0)], points_y[
            np.append(np.arange(len(points_y)), 0)]

    def complement(self):
        """
        'XIdlVert1 XIdlVert2 XIdlVert3 XIdlVert4 YIdlVert1 YIdlVert2 YIdlVert3 YIdlVert4 '
        XSciScale
        YSciScale

        :return:

        TODO:
            Implement exact scale computation
        """
        if not self._initial_attributes_validated:
            self.validate()

        if self.InstrName.lower() == 'nirspec':
            if self.AperName not in ['NRS1_FULL']:
                raise NotImplementedError(
                    'NIRSpec aperture {} is not implemented.'.format(self.AperName))

        # approximate scale at reference point positions
        self.XSciScale = np.sqrt(self.Sci2IdlX10 ** 2 + self.Sci2IdlY10 ** 2)
        self.YSciScale = np.sqrt(self.Sci2IdlX11 ** 2 + self.Sci2IdlY11 ** 2)

        corners_Idl_x, corners_Idl_y = self.corners('idl', rederive=True)
        for j in [1, 2, 3, 4]:
            setattr(self, 'XIdlVert{:d}'.format(j), corners_Idl_x[j - 1])
            setattr(self, 'YIdlVert{:d}'.format(j), corners_Idl_y[j - 1])

    def convert(self, X, Y, from_frame, to_frame):
        """
        Generic conversion routine, that calls one of the
        specific conversion routines based on the provided frame names as strings.
        Adapted from jwxml package.
        """

        #if self.InstrName.lower() == 'nirspec':
            #print('WARNING: {} transformations may be unreliable'.format(self.InstrName))
            # raise NotImplementedError('NIRSpec were transformations not yet implemented.')

        if from_frame not in FRAMES or to_frame not in FRAMES:
            raise ValueError("from_frame value must be one of: [{}]".format(', '.join(FRAMES)))

        if from_frame == to_frame:
            return X, Y  # null transformation

        else:

            # With valid from_frame and to_frame, this method must exist:
            # print('calling {}_to_{}'.format(from_frame.lower(), to_frame.lower()))
            conversion_method = getattr(self,
                                        '{}_to_{}'.format(from_frame.lower(), to_frame.lower()))

            return conversion_method(X, Y)

    def correct_for_dva(self, v2_arcsec, v3_arcsec, verbose=False):
        """Apply differential velocity aberration correction to input arrays of V2/V3 coordinates

        :param v2_arcsec:
        :param v3_arcsec:
        :return:
        """

        if self._dva_parameters is None:
            raise RuntimeError('DVA parameters not specified.')

        data = Table([v2_arcsec, v3_arcsec])
        tmp_file_in = os.path.join(os.environ['HOME'], 'hst_dva_temporary_file.txt')
        tmp_file_out = os.path.join(os.environ['HOME'], 'hst_dva_temporary_file_out.txt')
        # data.write(tmp_file_in, format='ascii.fixed_width_no_header', delimiter=' ',
        # bookend=False, overwrite=True)
        data.write(tmp_file_in, format='ascii.fixed_width_no_header', delimiter=' ', bookend=False)

        dva_source_dir = self._dva_parameters['dva_source_dir']
        parameter_file = self._dva_parameters['parameter_file']

        system_command = '{} {} {} {}'.format(os.path.join(dva_source_dir, 'compute-DVA.e'),
                                              parameter_file,
                                              tmp_file_in, tmp_file_out)
        if verbose:
            print('Running system command \n{}'.format(system_command))
        subprocess.call(system_command, shell=True)

        data = Table.read(tmp_file_out, format='ascii.no_header',
                          names=('v2_original', 'v3_original', 'v2_corrected', 'v3_corrected'))

        # clean up
        for filename in [tmp_file_in, tmp_file_out]:
            if os.path.isfile(filename):
                os.remove(filename)

        return np.array(data['v2_corrected']), np.array(data['v3_corrected'])


    def corners(self, to_frame, rederive=True):
        """
        Return coordinates of the aperture outline in the specified frame.
        """

        if rederive or not hasattr(self, 'XIdlVert1'):
            # see Colin's Calc worksheet
            x1 = -1 * getattr(self, 'XSciRef') + 0.5
            x2 = -1 * getattr(self, 'XSciRef') + 0.5 + getattr(self, 'XSciSize')
            y1 = -1 * getattr(self, 'YSciRef') + 0.5
            y3 = -1 * getattr(self, 'YSciRef') + 0.5 + getattr(self, 'YSciSize')
            x_sci = np.array([x1, x2, x2, x1]) + getattr(self, 'XSciRef')
            y_sci = np.array([y1, y1, y3, y3]) + getattr(self, 'YSciRef')
            corners = SiafCoordinates(x_sci, y_sci, 'sci')
        elif hasattr(self, 'XIdlVert1'):
            x_Idl = np.array([getattr(self, 'XIdlVert{:d}'.format(j)) for j in [1, 2, 3, 4]])
            y_Idl = np.array([getattr(self, 'YIdlVert{:d}'.format(j)) for j in [1, 2, 3, 4]])
            corners = SiafCoordinates(x_Idl, y_Idl, 'idl')
        else:
            raise NotImplementedError()

        return self.convert(corners.x, corners.y, corners.frame, to_frame)

    def get_polynomial_coefficients(self):
        """Return a dictionary of arrays holding the significant idl/sci coefficients.

        Returns
        -------
        dict : dictionary

        """
        if self.Sci2IdlDeg is None:
            return None
        else:
            number_of_coefficients = np.int((self.Sci2IdlDeg + 1) * (self.Sci2IdlDeg + 2) / 2)
            dict = {}
            for seed in 'Sci2IdlX Sci2IdlY Idl2SciX Idl2SciY'.split():
                dict[seed] = np.array(
                    [getattr(self, s) for s in DISTORTION_ATTRIBUTES if seed in s])[
                             0:number_of_coefficients]
            return dict
    def set_polynomial_coefficients(self, sci2idlx, sci2idly, idl2scix, idl2sciy):
        """Place a full set of coefficients into aperture
        Input
        : param sci2idlx - array of Sci2IdlX coefficients in JWST order
        : param sci2idly - array of Sci2IdlY coefficients
        : param idl2scix - array of Idl2SciX coefficients
        : param idl2sciy - array of Idl2SciY coefficients
        """

        if self.Sci2IdlDeg is None:
            print('This aperture has no polynomial coefficients')
            return None
        else:
            number_of_coefficients = (self.Sci2IdlDeg + 1) * (self.Sci2IdlDeg + 2) // 2
            for  i in range (number_of_coefficients):
                    setattr(self, DISTORTION_ATTRIBUTES[4*i], sci2idlx[i])
                    setattr(self, DISTORTION_ATTRIBUTES[4*i+1], sci2idly[i])
                    setattr(self, DISTORTION_ATTRIBUTES[4*i+2], idl2scix[i])
                    setattr(self, DISTORTION_ATTRIBUTES[4*i+3], idl2sciy[i])
            return None




    def path(self, to_frame):
        """
        Generate path from aperture vertices
        :return:
        """
        # self.path = \
        return matplotlib.path.Path(np.array(self.closed_polygon_points(to_frame)).T)


    def plot(self, frame='tel', name_label=False, ax=None, title=False, units='arcsec',
             annotate=False, mark_ref=False, fill=True, fill_color='cyan', **kwargs):
        """Plot this aperture.

        Partially adapted from https://github.com/mperrin/jwxml

        Parameters
        -----------
        frame : str
            Which coordinate system to plot in: 'tel', 'idl', 'sci', 'det'
        name_label : bool
            Add text label stating aperture name
        ax : matplotlib.Axes
            Desired destination axes to plot into (If None, current
            axes are inferred from pyplot.)
        units : str
            one of 'arcsec', 'arcmin', 'deg'
        annotate : bool
            Add annotations for detector (0,0) pixels
        mark_ref : bool
            Add marker for the (V2Ref, V3Ref) reference point defining this aperture.
        title : str
            If set, add a label to the plot indicating which frame was plotted.
        color : matplotlib-compatible color
            Color specification for this aperture's outline,
            passed through to `matplotlib.Axes.plot`

        Other matplotlib standard parameters may be passed in via **kwargs
        to adjust the style of the displayed lines.

        TODO:
        plotting in Sky frame, requires attitude
        #     elif system == 'radec':
        #         if attitude_ref is not None:
        #             vertices = np.array(
        #                 siaf_rotations.pointing(attitude_ref, self.points_closed.T[0],
        self.points_closed.T[1])).T
        #             self.path_radec = matplotlib.path.Path(vertices)
        #         else:
        #             error('Please provide attitude_ref')

        """
        if self.AperType == "TRANSFORM":
            raise TypeError("Cannot plot aperture of type: TRANSFORM")

        if units is None:
            units = 'arcsec'

        if ax is None:
            ax = pl.gca()
            ax.set_aspect('equal')
            if frame == 'tel':
                ax.set_xlabel('V2 ({0})'.format(units))
                ax.set_ylabel('V3 ({0})'.format(units))
            elif frame == 'idl':
                ax.set_xlabel('Ideal X ({0})'.format(units))
                ax.set_ylabel('Ideal Y ({0})'.format(units))
            elif frame == 'sci' or frame == 'det':
                ax.set_xlabel('X pixels ({0})'.format(frame))
                ax.set_ylabel('Y pixels ({0})'.format(frame))

        x, y = self.corners(frame, rederive=False)
        x2, y2 = self.closed_polygon_points(frame, rederive=False)

        if units.lower() == 'arcsec':
            scale = 1
        elif units.lower() == 'arcmin':
            scale = 1. / 60
        elif units.lower() == 'deg':
            scale = 1. / 60 / 60
        else:
            raise ValueError("Unknown units: " + units)

        # convert arcsec to arcmin and plot
        # if color is not None:
        #     ax.plot(x2 * scale, y2 * scale, color=color, ls=line_style, label=line_label)
        # else:
        # ax.plot(x2 * scale, y2 * scale, ls=line_style, label=line_label)
        ax.plot(x2 * scale, y2 * scale, **kwargs)

        if name_label:
            # partially mitigate overlapping NIRCam labels
            rotation = 30 if self.AperName.startswith('NRC') else 0
            ax.text(
                x.mean() * scale, y.mean() * scale, self.AperName,
                verticalalignment='center',
                horizontalalignment='center',
                rotation=rotation,
                color=ax.lines[-1].get_color())
        if fill:
            pl.fill(x2, y2, color=fill_color, zorder=-40)
        if title:
            ax.set_title("{0} frame".format(frame))
        if annotate:
            self.plot_detector_origin(frame)
        if mark_ref:
            x_ref, y_ref = self.reference_point(frame)
            ax.plot([x_ref], [y_ref], marker='+', color=ax.lines[-1].get_color())

        if (frame == 'tel') and (self.observatory == 'JWST'):
            # ensure V2 increases to the left
            ax = pl.gca()
            xlim = ax.get_xlim()
            if xlim[0] < xlim[1]:
                ax.invert_xaxis()


    def plot_detector_channels(self, frame, color='0.5', alpha=0.3, evenoddratio=0.5):
        """ Mark on the plot the various detector readout channels

        These are depicted as alternating light/dark bars to show the
        regions read out by each of the output amps.

        Parameters
        ----------
        frame : str
            Which coordinate system to plot in: 'tel', 'idl', 'sci', 'det'
        color : matplotlib-compatible color
            Color specification for the amplifier shaded region,
            passed through to `matplotlib.patches.Polygon` as `facecolor`
        alpha : float
            Opacity of odd-numbered amplifier region overlays
            (for even, see `evenoddratio`)
        evenoddratio : float
            Ratio of opacity between even and odd amplifier region
            overlays
       """

        # if self.InstrName in ['NIRISS','FGS']: # channels have wrong orientation
        #     raise NotImplementedError

        npixels = self.XDetSize
        # if self.InstrName == 'MIRI':
        #     npixels = 1024
        # else:
        #     npixels = 2048
        ch = npixels / 4

        ax = pl.gca()
        if self.InstrName in ['NIRISS', 'FGS', 'NIRSPEC']:
            pts = ((0, 0), (0, ch), (npixels, ch), (npixels, 0))
        else:
            pts = ((0, 0), (ch, 0), (ch, npixels), (0, npixels))
        for chan in range(4):
            plotpoints = np.zeros((4, 2))
            for i, xy in enumerate(pts):
                if self.InstrName in ['NIRISS', 'FGS', 'NIRSPEC']:
                    # plotpoints[i] = self.convert(xy[0] + chan * ch, xy[1], 'det', frame)
                    plotpoints[i] = self.convert(xy[0], xy[1] + chan * ch, 'det', frame)
                else:
                    plotpoints[i] = self.convert(xy[0] + chan * ch, xy[1], 'det', frame)
            chan_alpha = alpha if chan % 2 == 1 else alpha * evenoddratio
            rect = matplotlib.patches.Polygon(
                plotpoints,
                closed=True,
                alpha=chan_alpha,
                facecolor=color,
                edgecolor='none',
                lw=0
            )
            ax.add_patch(rect)


    def plot_detector_origin(self, frame, which='both'):
        """ Draw red and blue squares to indicate the raw detector
        readout and science frame readout, respectively

        Parameters
        -----------
        which : str
            Which detector origin to plot: 'both', 'det', 'sci'
        frame : str
            Which coordinate system to plot in: 'tel', 'idl', 'sci', 'det'
        """

        # raw detector frame
        if which.lower() == 'det' or which.lower() == 'both':
            c1, c2 = self.convert(0, 0, 'det', frame)
            pl.plot(c1, c2, color='red', marker='s', markersize=9)

        # science frame
        if which.lower() == 'sci' or which.lower() == 'both':
            c1, c2 = self.convert(0, 0, 'sci', frame)
            pl.plot(c1, c2, color='blue', marker='s')


    def reference_point(self, to_frame):
        """ Return the defining reference point of the aperture."""
        return self.convert(self.V2Ref, self.V3Ref, 'tel', to_frame)


# Definition of frame transforms
    def detector_transform(self, from_system, to_system, angle_deg=None, parity=None):
        """
        Generate transformation model to transform between det <-> sci

        DetSciYAngle_deg, DetSciParity can be defined externally to override value saved in aperture

        :param from_system:
        :param to_system:
        :param angle_deg:
        :param parity:
        :return:
        """

        # create the model for the transformation

        if parity is None:
            parity = getattr(self, 'DetSciParity')

        if angle_deg is None:
            angle_deg = getattr(self, 'DetSciYAngle')

        x_model_1, y_model_1 = linear_transform_model(from_system, to_system, parity, angle_deg)

        if from_system == 'det':
            # add offset model, see JWST-001550 Sect. 4.1
            x_offset = models.Shift(self.XSciRef)  # & models.Shift(self.XSciRef)
            y_offset = models.Shift(self.YSciRef)  # & models.Shift(self.YSciRef)
        elif from_system == 'sci':
            x_offset = models.Shift(self.XDetRef)  # & models.Shift(self.XDetRef)
            y_offset = models.Shift(self.YDetRef)  # & models.Shift(self.YDetRef)

        x_model = x_model_1 | x_offset  # evaluated as x_offset( x_model_1(*args) )
        y_model = y_model_1 | y_offset

        return x_model, y_model


    def distortion_transform(self, from_system, to_system, include_offset=True):
        """
        return transformation corresponding to aperture.

        adapted from https://github.com/spacetelescope/ramp_simulator/blob/master/read_siaf_table.py


        Parameters
        ----------
        from_system : str
            Starting system (e.g. "sci", "idl")
        to_system : str
            Ending coordinate system (e.g. "sci" , "idl")

        Returns
        -------
        x_model : astropy.modeling.Model
            Correction in x
        y_model : astropy.modeling.Model
            Correction in y

        """
        if from_system not in ['idl', 'sci']:
            raise ValueError('Requested from_system of {} not recognized.'.format(from_system))

        if to_system not in ['idl', 'sci']:
            raise ValueError("Requested to_system of {} not recognized.".format(to_system))

        # Generate the string corresponding to the requested coefficient labels
        if from_system == 'idl' and to_system == 'sci':
            label = 'Idl2Sci'
            #         from_units = 'arcsec'
            #         to_units = 'distorted pixels'
        elif from_system == 'sci' and to_system == 'idl':
            label = 'Sci2Idl'
            #         from_units = 'distorted pixels'
            #         to_units = 'arcsec'

        # Get the coefficients for "science" to "ideal" transformation (and back)
        # "science" is distorted pixels. "ideal" is undistorted arcsec from the
        # the reference pixel location.

        # degree of distortion polynomial
        degree = np.int(getattr(self, 'Sci2IdlDeg'))

        number_of_coefficients = np.int((degree + 1) * (degree + 2) / 2)
        all_keys = self.__dict__.keys()

        for axis in ['X', 'Y']:
            coeff_keys = np.array([c for c in all_keys if label + axis in c])
            coeff_keys.sort()
            coeff = np.array([getattr(self, c) for c in coeff_keys[0:number_of_coefficients]])
            coeffs = Table(coeff, names=(coeff_keys[0:number_of_coefficients].tolist()))

            # create the model for the transformation
            if axis == 'X':
                X_model = to_distortion_model(coeffs, degree)
            elif axis == 'Y':
                Y_model = to_distortion_model(coeffs, degree)

        if (label == 'Idl2Sci') and (include_offset):
            # add constant model, see JWST-001550 Sect. 4.2
            X_offset = models.Shift(self.XSciRef)
            Y_offset = models.Shift(self.YSciRef)

            X_model = X_model | X_offset
            Y_model = Y_model | Y_offset

        return X_model, Y_model


    def telescope_transform(self, from_system, to_system, V3IdlYAngle_deg=None, V2Ref_arcsec=None,
                            V3Ref_arcsec=None,
                            verbose=False):
        """Generate transformation model to go to/from tel (V2/V3) from
        undistorted angular distances from the reference pixel ("ideal") idl.

        adapted from https://github.com/spacetelescope/ramp_simulator/blob/master/read_siaf_table.py

        values to be transformed must be in arcseconds

        TODO
        (upgrade: use constants to get rid of this requirement)

        V3IdlYAngle_deg, V2Ref_arcsec, V3Ref_arcsec can be defined externally to override value
        saved in aperture
        :param from_system:
        :param to_system:
        :param V3IdlYAngle_deg:
        :param V2Ref_arcsec:
        :param V3Ref_arcsec:
        :param verbose:
        :return:
        """
        if verbose:
            print('Using planar approximation to convert between IDL and V2V3')

        if from_system == to_system:
            raise ValueError("WARNING, from_system and to_system must be different")

        if from_system != 'tel' and to_system != 'tel':
            raise ValueError("WARNING, either from_system or to_system must be 'V2V3'")

        # create the model for the transformation
        parity = getattr(self, 'VIdlParity')
        if V3IdlYAngle_deg is None:
            V3IdlYAngle = getattr(self, 'V3IdlYAngle')
            if np.isnan(V3IdlYAngle):
                raise RuntimeError('Attribute {} is nan'.format(V3IdlYAngle))
            V3IdlYAngle_rad = np.deg2rad(V3IdlYAngle)
        else:
            V3IdlYAngle_rad = np.deg2rad(V3IdlYAngle_deg)

        X_model, Y_model = _telescope_transform_model(from_system, to_system, parity,
                                                      V3IdlYAngle_rad)

        if from_system == 'idl':
            if V2Ref_arcsec is None:
                V2Ref_arcsec = self.V2Ref
            if V3Ref_arcsec is None:
                V3Ref_arcsec = self.V3Ref

            # add constant model, see JWST-001550 Sect. 4.3
            X_offset = models.Shift(V2Ref_arcsec)
            Y_offset = models.Shift(V3Ref_arcsec)

            X_model = X_model | X_offset
            Y_model = Y_model | Y_offset

        return X_model, Y_model


    def det_to_sci(self, XDet, YDet, *args):
        """ Detector to Science, following Section 4.1 of JWST-STScI-001550"""
        X_model, Y_model = self.detector_transform('det', 'sci', *args)
        return X_model(XDet - self.XDetRef, YDet - self.YDetRef), Y_model(XDet - self.XDetRef,
                                                                          YDet - self.YDetRef)


    def sci_to_det(self, XSci, YSci, *args):
        """ Science to Detector, following Section 4.1 of JWST-STScI-001550"""
        X_model, Y_model = self.detector_transform('sci', 'det', *args)
        return X_model(XSci - self.XSciRef, YSci - self.YSciRef), Y_model(XSci - self.XSciRef,
                                                                          YSci - self.YSciRef)


    def idl_to_tel(self, XIdl, YIdl, V3IdlYAngle_deg=None, V2Ref_arcsec=None, V3Ref_arcsec=None, method='planar_approximation', input_coordinates='tangent_plane'):
        """Convert from ideal to telescope (V2/V3) coordinate system.

        By default, this implementats the planar approximation, which is adequate for most
        purposes but may not be for all. Error is about 1.7 mas at 10 arcminutes from the tangent
        point. See JWST-STScI-1550 for more details.
        For higher accuracy, set method='spherical_transformation' in which case 3D matrix rotations
        are applied.

        Also by default, the input coordinates are in a tangent plane with a reference points at the
        origin (0,0) of the ideal frame.

        Parameters
        ----------
        XIdl: float
            ideal X coordinate in arcsec
        YIdl: float
            ideal Y coordinate in arcsec
        V3IdlYAngle_deg: float
            overwrites self.V3IdlYAngle
        V2Ref_arcsec : float
            overwrites self.V2Ref
        V3Ref_arcsec : float
            overwrites self.V3Ref
        method : str
            must be one of ['planar_approximation', 'spherical_transformation']
        input_coordinates : str
            must be one of ['tangent_plane', 'spherical']

        Returns
        -------
            tuple of floats containing V2, V3 coordinates in arcsec

        """
        if method == 'planar_approximation':
            if input_coordinates != 'tangent_plane':
                raise RuntimeError('Output has to be in tangent plane.')
            X_model, Y_model = self.telescope_transform('idl', 'tel', V3IdlYAngle_deg, V2Ref_arcsec,
                                                        V3Ref_arcsec)

            v2 = X_model(XIdl, YIdl)
            v3 = Y_model(XIdl, YIdl)

        elif method == 'spherical_transformation':

            if input_coordinates == 'spherical':
                x_idl_spherical_deg, y_idl_spherical_deg = XIdl* u.arcsec.to(u.deg), YIdl* u.arcsec.to(u.deg)

            elif input_coordinates == 'tangent_plane':
                # deproject coordinates before applying rotations
                x_idl_spherical_deg, y_idl_spherical_deg = projection.deproject_from_tangent_plane(
                    XIdl * u.arcsec.to(u.deg), YIdl * u.arcsec.to(u.deg), 0.0, 0.0)
                # x_idl_spherical_deg, y_idl_spherical_deg = projection.deproject_from_tangent_plane(
                #     XIdl * u.arcsec.to(u.deg), YIdl * u.arcsec.to(u.deg), self.V2Ref/3600., self.V3Ref/3600.)


            # only matrix rotations, this transforms from a spherical to a spherical coordinate
            # system. These matrices transform the V-reference point to the ideal reference point
            M1 = rotations.rotate(3, -1 * self.V2Ref / 3600.)
            M2 = rotations.rotate(2, self.V3Ref / 3600.)
            M3 = rotations.rotate(1, self.V3IdlYAngle)
            M4 = np.dot(M2, M1)
            M = np.dot(M3, M4)

            unit_vector = rotations.unit(x_idl_spherical_deg, y_idl_spherical_deg)

            unit_vector[1] = self.VIdlParity * unit_vector[1]
            rotated_vector = np.dot(np.linalg.inv(M), unit_vector)
            # rotated_vector[0] = -1*rotated_vector[0]
            # rotated_vector[1] = self.VIdlParity * rotated_vector[1]
            v2, v3 = rotations.v2v3(rotated_vector)

        if self._correct_dva:
            return self.correct_for_dva(v2, v3)
        else:
            return v2, v3


    def tel_to_idl(self, V2, V3, V3IdlYAngle_deg=None, V2Ref_arcsec=None, V3Ref_arcsec=None, method='planar_approximation', output_coordinates='tangent_plane'):
        """Convert from telescope (V2/V3) to ideal coordinate system.

        By default, this implementats the planar approximation, which is adequate for most
        purposes but may not be for all. Error is about 1.7 mas at 10 arcminutes from the tangent
        point. See JWST-STScI-1550 for more details.
        For higher accuracy, set method='spherical_transformation' in which case 3D matrix rotations
        are applied.

        Also by default, the output coordinates are in a tangent plane with a reference points at the
        origin (0,0) of the ideal frame.

        Parameters
        ----------
        V2 : float
            V2 coordinate in arcsec
        V3 : float
            V2 coordinate in arcsec
        V3IdlYAngle_deg : float
            overwrites self.V3IdlYAngle
        V2Ref_arcsec : float
            overwrites self.V2Ref
        V3Ref_arcsec : float
            overwrites self.V3Ref
        method : str
            must be one of ['planar_approximation', 'spherical_transformation']
        output_coordinates : str
            must be one of ['tangent_plane', 'spherical']

        Returns
        -------
            tuple of floats containing x_idl, y_idl coordinates in arcsec

        """
        if V2Ref_arcsec is None:
            V2Ref_arcsec = self.V2Ref
        if V3Ref_arcsec is None:
            V3Ref_arcsec = self.V3Ref

        if method == 'planar_approximation':
            if output_coordinates!='tangent_plane':
                raise RuntimeError('Output has to be in tangent plane.')
            X_model, Y_model = self.telescope_transform('tel', 'idl', V3IdlYAngle_deg)
            return X_model(V2 - V2Ref_arcsec, V3 - V3Ref_arcsec), Y_model(V2 - V2Ref_arcsec, V3 - V3Ref_arcsec)

        elif method == 'spherical_transformation':
            # only matrix rotations, this transforms from a spherical to a spherical coordinate
            # system
            M1 = rotations.rotate(3, -1 * self.V2Ref / 3600.)
            M2 = rotations.rotate(2, self.V3Ref / 3600.)
            M3 = rotations.rotate(1, self.V3IdlYAngle)
            M4 = np.dot(M2, M1)
            M = np.dot(M3, M4)
            unit_vector = rotations.unit(V2 / 3600., V3 / 3600.)
            rotated_vector = np.dot(M, unit_vector)
            # rotated_vector[0] = -1*rotated_vector[0]
            rotated_vector[1] = self.VIdlParity * rotated_vector[1]
            x_idl_spherical_arcsec, y_idl_spherical_arcsec = rotations.v2v3(rotated_vector)

            if output_coordinates == 'spherical':
                return x_idl_spherical_arcsec, y_idl_spherical_arcsec

            elif output_coordinates == 'tangent_plane':
                # Add tangent-plane projection with reference point at origin (0,0) of ideal system
                x_idl_tangent_deg, y_idl_tangent_deg = projection.project_to_tangent_plane(
                    x_idl_spherical_arcsec * u.arcsec.to(u.deg),
                    y_idl_spherical_arcsec * u.arcsec.to(u.deg), 0.0, 0.0)

                return x_idl_tangent_deg * u.deg.to(u.arcsec), y_idl_tangent_deg * u.deg.to(u.arcsec)


        # elif method == 'projection_and_rotate':
        #     v2_tangent_deg, v3_tangent_deg = projection.project_to_tangent_plane(
        #         V2 * u.arcsec.to(u.deg), V3 * u.arcsec.to(u.deg), self.V2Ref * u.arcsec.to(u.deg), self.V3Ref * u.arcsec.to(u.deg))
        #     if V3IdlYAngle_deg is None:
        #         V3IdlYAngle = getattr(self, 'V3IdlYAngle')
        #         V3IdlYAngle_rad = np.deg2rad(V3IdlYAngle)
        #     else:
        #         V3IdlYAngle_rad = np.deg2rad(V3IdlYAngle_deg)
        #     parity = getattr(self, 'VIdlParity')
        #     X_model, Y_model = _telescope_transform_model('tel', 'idl', parity, V3IdlYAngle_rad)
        #
        #     return X_model(v2_tangent_deg*u.deg.to(u.arcsec), v3_tangent_deg*u.deg.to(u.arcsec)), Y_model(v2_tangent_deg*u.deg.to(u.arcsec), v3_tangent_deg*u.deg.to(u.arcsec))



    def sci_to_idl(self, XSci, YSci):
        X_model, Y_model = self.distortion_transform('sci', 'idl')
        return X_model(XSci - self.XSciRef, YSci - self.YSciRef), Y_model(XSci - self.XSciRef,
                                                                          YSci - self.YSciRef)
        # return X_model(XSci, YSci), Y_model(XSci, YSci)

    def idl_to_sci(self, XIdl, YIdl):
        X_model, Y_model = self.distortion_transform('idl', 'sci')
        return X_model(XIdl, YIdl), Y_model(XIdl, YIdl)

    def det_to_idl(self, *args):
        return self.sci_to_idl(*self.det_to_sci(*args))

    def det_to_tel(self, *args):
        return self.idl_to_tel(*self.sci_to_idl(*self.det_to_sci(*args)))

    def sci_to_tel(self, *args):
        return self.idl_to_tel(*self.sci_to_idl(*args))

    def idl_to_det(self, *args):
        return self.sci_to_det(*self.idl_to_sci(*args))

    def tel_to_sci(self, *args):
        return self.idl_to_sci(*self.tel_to_idl(*args))

    def tel_to_det(self, *args):
        return self.sci_to_det(*self.idl_to_sci(*self.tel_to_idl(*args)))

    def raw_to_sci(self, x_raw, y_raw):
        """Convert from raw/native coordinates to SIAF-Science coordinates (same as DMS coordinates
        for FULLSCA apertures).

        Implements the fits_generator description described the table attached to
        https://jira.stsci.edu/browse/JWSTSIAF-25 (except for Guider 2)
        and implemented in
        https://github.com/STScI-JWST/jwst/blob/master/jwst/fits_generator/create_dms_data.py

        see e.g. https://jwst-docs.stsci.edu/display/JDAT/Coordinate+Systems+and+Transformations

        see also JWST-STScI-002566, JWST-STScI-003222 Rev A

        """
        if self.AperType == 'FULLSCA':

            if (self.DetSciYAngle == 0) and (self.DetSciParity == -1):
                if 'FGS' not in self.AperName:
                    # NRCA1, NRCA3, NRCALONG, NRCB2, NRCB4
                    # Flip in the x direction
                    x_sci = self.XDetSize - x_raw + 1
                    y_sci = y_raw
                else:
                    # FGS2, GUIDER2
                    # rotate 90 degrees anticlockwise
                    x_sci = -1*y_raw
                    y_sci = x_raw

            elif (self.DetSciYAngle == 180) and (self.DetSciParity == -1):
                # NRCA2, NRCA4, NRCB1, NRCB3, NRCBLONG
                # Flip in the y direction
                x_sci = x_raw
                y_sci = self.YDetSize - y_raw + 1

            elif (self.DetSciYAngle == 0) and (self.DetSciParity == +1):
                if 'NRS1' not in self.AperName:
                    # MIRI
                    # No flip or rotation
                    x_sci = x_raw
                    y_sci = y_raw
                else:
                    # NIRSpec NRS1
                    # Flip across line x=y
                    x_sci = y_raw
                    y_sci = x_raw

            elif (self.DetSciYAngle == 180) and (self.DetSciParity == +1):
                # Flip across line x=y, then 180 degree rotation
                # NIRISS, FGS1, NIRSpec NRS2
                x_temp = y_raw
                y_temp = x_raw
                x_sci = self.XDetSize - x_temp + 1
                y_sci = self.YDetSize - y_temp + 1

            return x_sci, y_sci

        else:
            raise NotImplementedError

    def sci_to_raw(self, x_sci, y_sci):
        """Convert from Science coordinates to raw/native coordinates.

        Implements the fits_generator description described the table attached to
        https://jira.stsci.edu/browse/JWSTSIAF-25 (except for Guider 2)
        and implemented in
        https://github.com/STScI-JWST/jwst/blob/master/jwst/fits_generator/create_dms_data.py

        see e.g. https://jwst-docs.stsci.edu/display/JDAT/Coordinate+Systems+and+Transformations

        see also JWST-STScI-002566, JWST-STScI-003222 Rev A

        """
        if self.AperType == 'FULLSCA':

            if (self.DetSciYAngle == 0) and (self.DetSciParity == -1):
                if 'FGS' not in self.AperName:
                    # Flip in the x direction
                    x_raw = self.XDetSize - x_sci + 1
                    y_raw = y_sci
                else:
                    # GUIDER2, FGS2
                    # rotate 90 degrees clockwise
                    x_raw = y_sci
                    y_raw = -1*x_sci

            elif (self.DetSciYAngle == 180) and (self.DetSciParity == -1):
                # Flip in the y direction
                x_raw = x_sci
                y_raw = self.YDetSize - y_sci + 1

            elif (self.DetSciYAngle == 0) and (self.DetSciParity == +1):
                if 'NRS1' not in self.AperName:
                    # No flip or rotation
                    x_raw = x_sci
                    y_raw = y_sci
                else:
                    # Flip across line x=y
                    x_raw = y_sci
                    y_raw = x_sci

            elif (self.DetSciYAngle == 180) and (self.DetSciParity == +1):
                # 180 degree rotation, then flip across line x=y
                # e.g. NIRISS
                x_temp = self.XDetSize - x_sci + 1
                y_temp = self.YDetSize - y_sci + 1
                x_raw = y_temp
                y_raw = x_temp

            return x_raw, y_raw

        else:
            raise NotImplementedError


    def validate(self):
        """
        Verify that the set attributes fully qualify an aperture
        :return:
        # http: // ssb.stsci.edu / doc / jwst / _modules / jwst / datamodels /
        wcs_ref_models.html  # DistortionModel.validate
        """

        # from matchcsv.py
        # #  Check some columns for allowed values
        # ApTypeList = ['FULLSCA', 'SUBARRAY', 'ROI', 'COMPOUND', 'SLIT', 'OSS', 'TRANSFORM']
        # ApShapeList = ['QUAD', 'RECT', 'CIRC']
        # for i, ap in enumerate(apNames1):
        #     if siaf1['AperType'][i] not in ApTypeList:
        #         print ('Bad AperType', ap, siaf1['AperType'][i])
        #     if siaf1['AperType'][i] != 'TRANSFORM':
        #         if siaf1['AperShape'][i] not in ApShapeList:
        #             print('Bad AperShape', ap, siaf1['AperShape'][i])
        #         if abs(siaf1['VIdlParity'][i]) != 1:
        #             print('Bad VIdlParity', ap, siaf1['VIdlParity'][i])
        #         if abs(siaf1['DetSciParity'][i]) != 1:
        #             print('Bad DetSciParity', ap, siaf1['DetSciParity'][i])

        if self.InstrName.lower() == 'nirspec':
            if self.AperName not in ['NRS1_FULL']:
                raise NotImplementedError(
                    'NIRSpec aperture {} is not implemented.'.format(self.AperName))

        # check that all validation attributes are present
        for attr in VALIDATION_ATTRIBUTES:
            assert hasattr(self, attr)

        polynomial_degree = getattr(self, 'Sci2IdlDeg')
        assert type(polynomial_degree) == int
        # polynomial_k = 2 * (polynomial_degree+1)
        # number_of_coefficients = (polynomial_degree + 1) * (polynomial_degree + 2) / 2

        # check that polynomial coefficients are set
        for ii in range(polynomial_degree + 1):
            for jj in np.arange(ii + 1):
                assert getattr(self, 'Sci2IdlX{:d}{:d}'.format(ii, jj)) is not None
                assert getattr(self, 'Sci2IdlY{:d}{:d}'.format(ii, jj)) is not None
                assert getattr(self, 'Idl2SciX{:d}{:d}'.format(ii, jj)) is not None
                assert getattr(self, 'Idl2SciY{:d}{:d}'.format(ii, jj)) is not None

        setattr(self, '_initial_attributes_validated', True)


    def verify(self):
        """Perform internal verification of aperture parameters

        Returns
        -------

        """

        # check that SciRef and DetRef coordinates refer to the same on-chip location, i.e. their
        # sum or difference should be an integer, depending on the value of DetSciYAngle
        if (self.XDetRef is not None):
            sum_x = self.XDetRef + np.cos(np.deg2rad(self.DetSciYAngle)) * self.XSciRef
            sum_y = self.YDetRef + self.YSciRef
            if not sum_x.is_integer():
                print('Verification WARNING: {} has non-integer sum XDetRef {} and XSciRef {} DetSciYAngle {}'.format(self.AperName, self.XDetRef, self.XSciRef, self.DetSciYAngle))
            if not sum_y.is_integer():
                print('Verification WARNING: {} has non-integer sum between YDetRef {} and YSciRef {} DetSciYAngle {}'.format(self.AperName, self.YDetRef, self.YSciRef, self.DetSciYAngle))





def get_hst_to_jwst_coefficient_order(polynomial_degree):
    """
    function assumes that coefficient orders are as follows
    HST:  1, y, x, y^2, xy, x^2, y^3, xy^2, x^2y, x^3 ...    (according to Cox's
    /grp/hst/OTA/alignment/FocalPlane.xlsx)
    JWST: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3 ...

    :param polynomial_degree:
    :return:
    """

    # list of tuples representing the x,y exponents in JWST order
    jwst_exponents = [(i - j, j) for i in range(polynomial_degree + 1) for j in range(i + 1)]

    # list of tuples representing the x,y exponents in HST order
    hst_exponents = [(j, i - j) for i in range(polynomial_degree + 1) for j in range(i + 1)]

    conversion_index = np.array([hst_exponents.index(t) for t in jwst_exponents])

    return conversion_index

#######################################
# support for HST apertures

HST_FLIP_1 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
HST_FLIP_2 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
HST_FLIP_3 = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])

# TVS matrices (from FGS Geometry Products on GSFC/SAC web page)
HST_TVS_FGS_1R = np.array([-0.000034791201, +0.000021915510, +0.999999999155,
                           -0.003589864866, +0.999993556171, -0.000022040265,
                           -0.999993555809, -0.003589865630, -0.000034712303]).reshape(3, 3)

HST_TVS_FGS_2R2 = np.array([-0.000001019786, +0.000025918600, +0.999999999664,
                            -0.999997535003, +0.002220357221, -0.000001077332,
                            -0.002220357248, -0.999997534668, +0.000025916272]).reshape(3, 3)

HST_TVS_FGS_3 = np.array([+0.000030012865, +0.000022612406, +0.999999999294,
                          -0.002060622958, -0.999997876657, +0.000022674204,
                          +0.999997876464, -0.002060623637, -0.000029966206]).reshape(3, 3)

class HstAperture(Aperture):
    """Class for apertures of HST instruments.

    Inherits from Aperture
    """

    _accepted_aperture_types = ['QUAD', 'RECT', 'CIRC']

    def __init__(self):
        super(HstAperture,self).__init__()
        self.observatory = 'HST'

    # dictionary that allows to set attributes using JWST naming convention
    _hst_to_jwst_keys = ({'SI_mne': 'InstrName',
                          'ap_name': 'AperName',
                          'a_shape': 'AperShape',
                          'im_par': 'VIdlParity',
                          'theta': 'V3IdlYAngle',
                          # theta = Rotation angle to orient SICS and V2/V3 (deg),
                          # same convention as on JWST (2017-10-10)
                          'a_v2_ref': 'V2Ref',
                          'a_v3_ref': 'V3Ref',
                          'xa0': 'XSciRef',
                          'ya0': 'YSciRef',
                          'ideg': 'Sci2IdlDeg'
                          })

    def __setattr__(self, key, value):
        """Set attribute in JWST convention and check format. Convert HST polynomial coefficients
        to JWST convention.

        To convert polynomial coefficient order, see JWST-STScI-001550:
        The coding of the coefficients is such that they refer to powers of x and y in the order
        1, x, y, x2, xy, y2 ... (This is a more natural order than is used on
        HST which results in 1, y, x, y2, xy, x2... The change should be noted in case
        software is reused.)

        Parameters
        ----------
        key
        value

        """

        self.__dict__[key] = value

        # set attributes using JWST naming convention
        if key in self._hst_to_jwst_keys.keys():
            new_key = self._hst_to_jwst_keys[key]
            self.__dict__[new_key] = value
            if key == 'a_shape':
                self.__dict__['AperType'] = value  # HST duplicates AperShape and AperType

        # set distortion coefficient attributes using JWST order and naming convention
        if key == 'polynomial_coefficients':

            # reorder coefficients according to JWST SIAF convention
            conversion_index = get_hst_to_jwst_coefficient_order(self.Sci2IdlDeg)
            jwst_ordered_coefficients = self.polynomial_coefficients[conversion_index, :]

            self.jwst_ordered_coefficients = jwst_ordered_coefficients
            m = 0
            for i in range(self.Sci2IdlDeg + 1):
                for j in np.arange(i + 1):
                    for k, name in enumerate(POLYNOMIAL_COEFFICIENT_NAMES):
                        self.__dict__['{}{:d}{:d}'.format(name, i, j)] = jwst_ordered_coefficients[
                            m, k]
                    m += 1

    def _tvs_parameters(self, tvs=None):
        """Compute V2_tvs, V3_tvs, and V3angle_tvs from the TVS matrices stored in the database
        TVS matrices come from GSFC/SAC web page

        # from Colin Cox's ipython notebook received 11 Dec 2017
        :return:
        """

        if tvs is None:
            if self.AperName == 'FGS1':
                tvs = HST_TVS_FGS_1R
            elif self.AperName == 'FGS2':
                tvs = HST_TVS_FGS_2R2
            elif self.AperName == 'FGS3':
                tvs = HST_TVS_FGS_3
            else:
                raise NotImplementedError

        m1f = np.dot(np.transpose(self.tvs_flip_matrix), tvs)

        v2_arcsec = 3600. * np.rad2deg(np.arctan2(m1f[0, 1], m1f[0, 0]))
        v3_arcsec = 3600. * np.rad2deg(np.arcsin(m1f[0, 2]))
        pa_deg = np.rad2deg(np.arctan2(m1f[1, 2], m1f[2, 2]))
        return v2_arcsec, v3_arcsec, pa_deg, tvs

    def closed_polygon_points(self, to_frame, rederive=False):
        """Compute closed polygon points of aperture outline. Used for plotting and path generation.

        :param to_frame:
        :return:
        """

        if self.a_shape == 'PICK':
            x0 = 0.
            y0 = 0.
            outer_points_x, outer_points_y = points_on_arc(x0, y0, self.maj,
                                                           self.pi_angle - self.pi_ext,
                                                           self.pi_angle + self.pi_ext, N=100)
            inner_points_x, inner_points_y = points_on_arc(x0, y0, self.min,
                                                           self.po_angle - self.po_ext,
                                                           self.po_angle + self.po_ext, N=100)
            x_Tel = np.hstack((outer_points_x, inner_points_x[::-1]))
            y_Tel = np.hstack((outer_points_y, inner_points_y[::-1]))

            curve = SiafCoordinates(x_Tel, y_Tel, 'tel')
            points_x, points_y = self.convert(curve.x, curve.y, curve.frame, to_frame)

        elif self.a_shape == 'QUAD':
            points_x, points_y = self.corners(to_frame)

        # return the closed polygon coordinates
        return points_x[np.append(np.arange(len(points_x)), 0)], points_y[
            np.append(np.arange(len(points_y)), 0)]

    def compute_tvs_matrix(self, v2_arcsec=None, v3_arcsec=None, pa_deg=None, verbose=False):
        """Compute the TVS matrix from 'virtual' v2,v3,pa parameters

        :param v2_arcsec:
        :param v3_arcsec:
        :param pa_deg:
        :return:
        """

        if v2_arcsec is None:
            v2_arcsec = self.db_tvs_v2_arcsec
            if verbose:
                print('{} using db_tvs_v2_arcsec'.format(self))
        if v3_arcsec is None:
            v3_arcsec = self.db_tvs_v3_arcsec
        if pa_deg is None:
            pa_deg = self.db_tvs_pa_deg

        attitude = rotations.attitude(v2_arcsec, v3_arcsec, 0.0, 0.0, pa_deg)
        tvs = np.dot(self.tvs_flip_matrix, attitude)
        return tvs

    def corners(self, to_frame, rederive=False):
        """Return coordinates of the aperture vertices in the specified frame."""

        if self.a_shape == 'PICK':
            # compute pickle corner points in tel/V2V3 frame
            x0 = 0.
            y0 = 0.
            radii = np.array([self.maj, self.maj, self.min, self.min])
            angles_rad = np.deg2rad(np.array(
                [self.pi_angle - self.pi_ext, self.pi_angle + self.pi_ext,
                 self.pi_angle - self.pi_ext, self.pi_angle + self.pi_ext]))
            x_Tel = _x_from_polar(x0, radii, angles_rad)
            y_Tel = _y_from_polar(y0, radii, angles_rad)
            corners = SiafCoordinates(x_Tel, y_Tel, 'tel')

        elif self.a_shape == 'QUAD':
            x_Idl = np.array([self.v1x, self.v2x, self.v3x, self.v4x])
            y_Idl = np.array([self.v1y, self.v2y, self.v3y, self.v4y])
            corners = SiafCoordinates(x_Idl, y_Idl, 'idl')

        return self.convert(corners.x, corners.y, corners.frame, to_frame)

    def idl_to_tel(self, XIdl, YIdl, V3IdlYAngle_deg=None, V2Ref_arcsec=None, V3Ref_arcsec=None, method='planar_approximation', input_coordinates='tangent_plane'):
        """Convert idl to  tel

        input in arcsec, output in arcsec

        WARNING (JWST case)
        --------
        This is an implementation of the planar approximation, which is adequate for most
        purposes but may not be for all. Error is about 1.7 mas at 10 arcminutes from the tangent
        point. See JWST-STScI-1550 for more details.

        HST-FGS case
        --------
        Transformation is implemented using the FGS TVS matrix. Parameter names are overloaded
        for simplicity.

        tvs_pa_deg = V3IdlYAngle_deg
        tvs_v2_arcsec = V2Ref_arcsec
        tvs_v3_arcsec  = V3Ref_arcsec


        :param XIdl:
        :param YIdl:
        :param V3IdlYAngle_deg:
        :param V2Ref_arcsec:
        :param V3Ref_arcsec:
        :return:
        """

        if 'FGS' in self.AperName:
            tvs_pa_deg = V3IdlYAngle_deg
            tvs_v2_arcsec = V2Ref_arcsec
            tvs_v3_arcsec = V3Ref_arcsec

            # treat V3IdlYAngle, V2Ref, V3Ref in the TVS-specific way
            tvs = self.compute_tvs_matrix(tvs_v2_arcsec, tvs_v3_arcsec, tvs_pa_deg)

            # unit vector
            x_rad = np.deg2rad(XIdl * u.arcsec.to(u.deg))
            y_rad = np.deg2rad(YIdl * u.arcsec.to(u.deg))
            xyz = np.array([x_rad, y_rad, np.sqrt(1. - (x_rad ** 2 + y_rad ** 2))])

            v = np.rad2deg(np.dot(tvs, xyz)) * u.deg.to(u.arcsec)
            return v[1], v[2]
        else:
            return super(HstAperture, self).idl_to_tel(XIdl, YIdl, V3IdlYAngle_deg=V3IdlYAngle_deg,
                                      V2Ref_arcsec=V2Ref_arcsec, V3Ref_arcsec=V3Ref_arcsec, method=method, input_coordinates=input_coordinates)

    def set_idl_reference_point(self, v2_ref, v3_ref, verbose=False):
        """Determine the reference point in the Ideal frame that determine V2Ref and V3Ref via
        the TVS matrix
        The tvs parameters that determine the TVS matrix itself are derived and added to the
        attribute list

        :param v2_ref: in arcsec
        :param v3_ref: in arcsec
        :return:
        """

        if self.AperName == 'FGS1':
            flip = HST_FLIP_1
        elif self.AperName == 'FGS2':
            flip = HST_FLIP_2
        elif self.AperName == 'FGS3':
            flip = HST_FLIP_3
        else:
            raise NotImplementedError

        self.tvs_flip_matrix = flip

        self.db_tvs_v2_arcsec, self.db_tvs_v3_arcsec, self.db_tvs_pa_deg, db_tvs = \
            self._tvs_parameters()

        self.db_tvs = db_tvs

        inverted_tvs = np.linalg.inv(db_tvs)

        # construct the normalized vector of the reference point in the V/tel frame
        tel_vector_rad = np.array([0., np.deg2rad(v2_ref / 3600.), np.deg2rad(v3_ref / 3600.)])
        tel_vector_rad_normalized = tel_vector_rad.copy()
        tel_vector_rad_normalized[0] = np.sqrt(1. - tel_vector_rad[1] ** 2 - tel_vector_rad[2] ** 2)

        idl_vector_rad = np.dot(inverted_tvs, tel_vector_rad_normalized)
        idl_vector_arcsec = np.rad2deg(idl_vector_rad) * 3600.

        self.idl_x_ref_arcsec = idl_vector_arcsec[0]
        self.idl_y_ref_arcsec = idl_vector_arcsec[1]
        self.idl_angle_deg = idl_vector_arcsec[2] / 3600.

        tel_vector_rad_recomputed = np.dot(db_tvs, idl_vector_rad)
        tel_vector_arcsec_recomputed = np.rad2deg(tel_vector_rad_recomputed) * 3600.

        # set V2Ref and V3Ref
        self.V2Ref = tel_vector_arcsec_recomputed[1]
        self.V3Ref = tel_vector_arcsec_recomputed[2]
        self.tel_angle_deg = tel_vector_arcsec_recomputed[0] / 3600.

        if verbose:
            print('in  V2Ref,V3Ref {} {}'.format(v2_ref, v3_ref))
            print('out V2Ref,V3Ref {} {}'.format(self.V2Ref, self.V3Ref))
            print('tel Angle {} deg'.format(self.tel_angle_deg))
            print('idl_x_ref_arcsec {}'.format(self.idl_x_ref_arcsec))
            print('idl_y_ref_arcsec {}'.format(self.idl_y_ref_arcsec))
            print('idl Angle {} deg'.format(self.idl_angle_deg))
            print('TVS parameters: {} arcsec  {} arcsec  {} deg'.format(self.db_tvs_v2_arcsec,
                                                                        self.db_tvs_v3_arcsec,
                                                                        self.db_tvs_pa_deg))

    def set_tel_reference_point(self, verbose=True):
        """Recompute and set V2Ref and V3Ref to actual position in tel/V frame after using those
        attributes for TVS matrix update

        :param verbose:
        :return:
        """

        # reference point in idl frame
        idl_vector_rad = np.deg2rad(
            [self.idl_x_ref_arcsec / 3600., self.idl_y_ref_arcsec / 3600., self.idl_angle_deg])

        # transform to tel frame
        tel_vector_rad = np.array(np.dot(self.corrected_tvs, idl_vector_rad)).flatten()

        tel_vector_arcsec = np.rad2deg(tel_vector_rad) * 3600.

        # set V2Ref and V3Ref
        self.V2Ref = copy.deepcopy(self.a_v2_ref)
        self.V3Ref = copy.deepcopy(self.a_v3_ref)
        self.V3IdlYAngle = copy.deepcopy(self.theta)

        self.V2Ref_corrected = tel_vector_arcsec[1]
        self.V3Ref_corrected = tel_vector_arcsec[2]
        self.V3IdlYAngle_corrected = self.theta  # correction = 0 by convention

        if verbose:
            for attribute_name in 'V2Ref V3Ref V3IdlYAngle'.split():
                print('Setting {} to {}'.format(attribute_name, getattr(self, attribute_name)))
                print('Setting {} to {}'.format(attribute_name + '_corrected',
                                                getattr(self, attribute_name + '_corrected')))

#######################################
# JWST apertures
class JwstAperture(Aperture):
    """Class for apertures of JWST instruments. Inherits from Aperture."""

    _accepted_aperture_types = 'FULLSCA OSS ROI SUBARRAY SLIT COMPOUND TRANSFORM'.split()

    def __init__(self):
        super(JwstAperture, self).__init__()
        self.observatory = 'JWST'

def linear_transform_model(from_system, to_system, parity, angle_deg):
    """
    Creates an astropy.modeling.Model object for linear transforms
    e.g. det <-> sci and idl <-> tel

    angle has to be in degrees

    sics_to_v2v3 (HST)
    x_v2v3 = v2_origin + parity * x_sics * np.cos(np.deg2rad(theta_deg)) + y_sics * np.sin(
    np.deg2rad(theta_deg))
    y_v2v3 = v3_origin - parity * x_sics * np.sin(np.deg2rad(theta_deg)) + y_sics * np.cos(
    np.deg2rad(theta_deg))

    """
    if type(angle_deg) not in [float, np.float64, int]:
        raise TypeError('Angle has to be a float. It is of type {} and has the value {}'.format(type(angle_deg), angle_deg))

    # check for allowed system pairs
    if (from_system == 'det' and to_system != 'sci') or (
            from_system == 'sci' and to_system != 'det') or (
            from_system == 'idl' and to_system != 'tel') or (
            from_system == 'tel' and to_system != 'idl'):
        raise ValueError('Invalid combination of frames')

    angle_rad = np.deg2rad(angle_deg)

    # cast the transform functions as 1st order polynomials
    xc = {}
    yc = {}

    # 0,0 coefficients are not used here (offsets have to be applied outside of this function)
    xc['c0_0'] = 0
    yc['c0_0'] = 0

    if to_system == 'det':
        # Section 4.1 in JWST-001550
        xc['c1_0'] = parity * np.cos(angle_rad)
        xc['c0_1'] = parity * np.sin(angle_rad)
        yc['c1_0'] = -1. * np.sin(angle_rad)
        yc['c0_1'] = +1. * np.cos(angle_rad)

    elif from_system == 'det':
        xc['c1_0'] = parity * np.cos(angle_rad)
        xc['c0_1'] = -1. * np.sin(angle_rad)
        yc['c1_0'] = parity * np.sin(angle_rad)
        yc['c0_1'] = +1. * np.cos(angle_rad)

    elif to_system == 'tel':
        # Section 5.3 in JWST-001550
        xc['c1_0'] = +1. * parity * np.cos(angle_rad)
        xc['c0_1'] = np.sin(angle_rad)
        yc['c1_0'] = -1. * parity * np.sin(angle_rad)
        yc['c0_1'] = np.cos(angle_rad)

    elif from_system == 'tel':
        xc['c1_0'] = +1. * parity * np.cos(angle_rad)
        xc['c0_1'] = -1. * parity * np.sin(angle_rad)
        yc['c1_0'] = np.sin(angle_rad)
        yc['c0_1'] = np.cos(angle_rad)

    xmodel = models.Polynomial2D(1, **xc)
    ymodel = models.Polynomial2D(1, **yc)

    return xmodel, ymodel

# NIRSpec
class NirspecAperture(JwstAperture):
    """Class for apertures of the JWST NIRSpecinstrument. Inherits from JwstAperture."""

    _accepted_aperture_types = 'FULLSCA OSS ROI SUBARRAY SLIT COMPOUND TRANSFORM'.split()

    def __init__(self, tilt=None, filter_name='CLEAR'):
        super(NirspecAperture, self).__init__()
        self.observatory = 'JWST'
        self.tilt = tilt
        self.filter_name = filter_name


    def corners(self, to_frame, rederive=True):
        return super(NirspecAperture, self).corners(to_frame, rederive=False)


    def gwa_to_ote(self, gwa_x, gwa_y):
        """NIRSpec transformation from GWA sky side to OTE frame XAN, YAN

        output is in degrees

        Parameters
        ----------
        gwa_x
        gwa_y

        Returns
        -------

        """

        filter_list = 'CLEAR F110W F140X'.split()
        if self.filter_name not in filter_list:
            raise RuntimeError(
                'Filter must be one of {} (it is {})'.format(filter_list, self.filter_name))

        transform_aperture = getattr(self, '_{}_GWA_OTE'.format(self.filter_name))
        X_model, Y_model = transform_aperture.distortion_transform('sci', 'idl', include_offset=False)
        return X_model(gwa_x, gwa_y), Y_model(gwa_x, gwa_y)


    def ote_to_gwa(self, ote_x, ote_y):
        """NIRSpec transformation from OTE frame XAN, YAN to GWA sky side

        Inputs must be in degrees

        Parameters
        ----------
        ote_x
        ote_y

        Returns
        -------

        """

        filter_list = 'CLEAR F110W F140X'.split()
        if self.filter_name not in filter_list:
            raise RuntimeError(
                'Filter must be one of {} (it is {})'.format(filter_list, self.filter_name))

        # if self.AperType in ['FULLSCA', 'OSS']:
        transform_aperture = getattr(self, '_{}_GWA_OTE'.format(self.filter_name))
        # elif self.AperType in ['SLIT']:
        #     transform_aperture = getattr(self, '_{}_GWA_OTE'.format(self.filter_name))

        X_model, Y_model = transform_aperture.distortion_transform('idl', 'sci', include_offset=False)
        return X_model(ote_x, ote_y), Y_model(ote_x, ote_y)


    def gwain_to_gwaout(self, x_gwa, y_gwa):
        """Transform from GWA detector side to GWA skyward side. Effect of mirror.

        Parameters
        ----------
        x_gwa
        y_gwa

        Returns
        -------
        
        -------
        Documents Giardino, Ferruit, & Alves de Oliveira (2014) ESA NTN-2014-005, and
        Proffitt et al. (2018), JWST-STScI-005921

        """
        if self.tilt is None:
            return -1*x_gwa, -1*y_gwa
        else:
            gwa_xtil, gwa_ytil = self.tilt
            
            filter_name = self.filter_name
            
            # need to pull correct coefficients from GWA transform row
            gwa_aperture = getattr(self, '_{}_GWA_OTE'.format(filter_name))
            rx0 = getattr(gwa_aperture, 'XSciRef')
            ry0 = getattr(gwa_aperture, 'YSciRef')
            ax = getattr(gwa_aperture, 'XSciScale')
            ay = getattr(gwa_aperture, 'YSciScale')
           
            delta_theta_x = 0.5 * ax * (gwa_ytil - rx0) * np.pi / (180. * 3600.0)
            delta_theta_y = 0.5 * ay * (gwa_xtil - ry0) * np.pi / (180. * 3600.0)
            
            v = np.abs(np.sqrt(1.0 + x_gwa*x_gwa + y_gwa*y_gwa))
            x0 = x_gwa / v
            y0 = y_gwa / v
            z0 = 1.0 / v
            # rotate to mirror reference system with small angle approx. and perform rotation
            x1 = -1 * (x0 - delta_theta_y * np.sqrt(1.0 - x0*x0 - (y0+delta_theta_x*z0)*(y0+delta_theta_x*z0)))
            y1 = -1 * (y0 + delta_theta_x * z0)
            z1 = np.sqrt(1.0 - x1*x1 - y1*y1)
            # rotate reflected ray back to ref GWA coord system with small angle approx., 
            # but first with an inverse rotation around the y-axis
            x2 = x1 + delta_theta_y * z1
            y2 = y1
            z2 = np.sqrt(1.0 - x2*x2 - y2*y2)
            # now do an inverse rotation around the x-axis
            x3 = x2
            y3 = y2 - delta_theta_x * z2     
            z3 = np.sqrt(1.0 - x3*x3 - y3*y3)
            # compute the cosines from direction cosines
            x_gwap = x3 / z3
            y_gwap = y3 / z3
 
            return x_gwap, y_gwap


    def gwaout_to_gwain(self, x_gwa, y_gwa):
        """Transform from GWA skyward side to GWA detector side. Effect of mirror.

        Parameters
        ----------
        x_gwa
        y_gwa
        tilt

        Returns
        -------
        -------
        
        Equations for the reverse transform T. Keyes (private communication, but will be
        documented in next update of JWST-STScI-005921.

        """
        if self.tilt is None:
            return -1*x_gwa, -1*y_gwa
        else:
            gwa_xtil, gwa_ytil = self.tilt

            filter_name = self.filter_name
            
            # need to pull correct coefficients from transform row
            transform_aperture = getattr(self, '_{}_GWA_OTE'.format(filter_name))
            rx0 = getattr(transform_aperture, 'XSciRef')
            ry0 = getattr(transform_aperture, 'YSciRef')
            ax = getattr(transform_aperture, 'XSciScale')
            ay = getattr(transform_aperture, 'YSciScale')
           
            delta_theta_x = 0.5 * ax * (gwa_ytil - rx0) * np.pi / (180. * 3600.0)
            delta_theta_y = 0.5 * ay * (gwa_xtil - ry0) * np.pi / (180. * 3600.0)

            # calculate direction cosines of xt, yt, (xgwa, ygwa)   
            v = np.abs(np.sqrt(1.0 + x_gwa*x_gwa + y_gwa*y_gwa))
            x3 = x_gwa / v
            y3 = y_gwa / v
            z3 = 1.0 / v
            # do inverse rotation around the x-axis
            x2 = x3
            y2 = y3 + delta_theta_x*z3
            z2 = np.sqrt(1.0 - x2*x2 - y2*y2)
            # rotate to mirror reference system with small angle approx. and perform rotation
            x1 = x2 - delta_theta_y*z2 # try changing - to +???
            y1 = y2
            z1 = np.sqrt(1.0 - x1*x1 - y1*y1)
            # rotate reflected ray back to reference GWA coordinate system (use small angle approx.),
            # first with an inverse rotation around the y-axis:
            x0 = -1.0*x1 + delta_theta_y * np.sqrt(1.0 - x1*x1 - (y1+delta_theta_x*z1)*(y1+delta_theta_x*z1))
            y0 = -1.0*y1 - delta_theta_x*z1
            z0 = np.sqrt(1.0 - x0*x0 - y0*y0)
            
            x_gwap = x0/z0
            y_gwap = y0/z0
            
            return x_gwap, y_gwap
            

    def sci_to_idl(self, x_sci, y_sci):
        """Special implementation for NIRSpec, taking detour via tel frame.

        Parameters
        ----------
        x_sci
        y_sci
        filter_name

        Returns
        -------

        """
        v2, v3 = self.sci_to_tel(x_sci, y_sci)
        return self.tel_to_idl(v2, v3)


    def idl_to_sci(self, x_idl, y_idl):
        """Special implementation for NIRSpec, taking detour via tel frame.

        Parameters
        ----------
        x_idl
        y_idl
        filter_name

        Returns
        -------

        """
        v2, v3 = self.idl_to_tel(x_idl, y_idl)
        x_sci, y_sci = self.tel_to_sci(v2, v3)
        return x_sci, y_sci


    def sci_to_gwa(self, XSci, YSci):
        """NIRSpec transformation from Science frame to GWA detector side

        Parameters
        ----------
        XSci
        YSci

        Returns
        -------

        """

        X_model, Y_model = self.distortion_transform('sci', 'idl')
        return X_model(XSci - self.XSciRef, YSci - self.YSciRef), Y_model(XSci - self.XSciRef,
                                                                          YSci - self.YSciRef)

    def gwa_to_sci(self, x_gwa, y_gwa):
        """NIRSpec transformation from GWA detector side to Science frame

        Parameters
        ----------
        XSci
        YSci

        Returns
        -------

        """

        X_model, Y_model = self.distortion_transform('idl', 'sci')
        return X_model(x_gwa, y_gwa), Y_model(x_gwa, y_gwa)


    def det_to_sci(self, XDet, YDet, *args):
        """Use parent aperture if SLIT."""
        if self.AperType == 'TRANSFORM':
            raise RuntimeError('AperType {} is not supported for this transformation'.format(self.AperType))
        elif self.AperType == 'SLIT':
            if self._parent_apertures is None:
                raise RuntimeError('Transformation not supported for this aperture: {}.'.format(self.AperName))
            else:
                return self._parent_aperture.det_to_sci(XDet, YDet, *args)
        else:
            return super(NirspecAperture, self).det_to_sci(XDet, YDet, *args)


    def sci_to_det(self, XSci, YSci, *args):
        """Use parent aperture if SLIT."""
        if self.AperType == 'TRANSFORM':
            raise RuntimeError('AperType {} is not supported for this transformation'.format(self.AperType))
        elif self.AperType == 'SLIT':
            if self._parent_apertures is None:
                raise RuntimeError('Transformation not supported for this aperture: {}.'.format(self.AperName))
            else:
                return self._parent_aperture.sci_to_det(XSci, YSci, *args)
        else:
            return super(NirspecAperture, self).sci_to_det(XSci, YSci, *args)


    def sci_to_tel(self, x_sci, y_sci):
        """Overwriting standard behaviour for NIRSpec specific transformation."""

        if self.AperType == 'SLIT':
            if self._parent_apertures is None:
                raise RuntimeError('Transformation not supported for this aperture: {}.'.format(self.AperName))
            else:
                # call transformation of parent aperture
                x_gwa_in, y_gwa_in = self._parent_aperture.sci_to_gwa(x_sci, y_sci)
        else:
            x_gwa_in, y_gwa_in = self.sci_to_gwa(x_sci, y_sci)

        # x_gwa_in, y_gwa_in = self.sci_to_gwa(x_sci, y_sci)
        x_gwa_out, y_gwa_out = self.gwain_to_gwaout(x_gwa_in, y_gwa_in)
        x_ote_deg, y_ote_deg = self.gwa_to_ote(x_gwa_out, y_gwa_out)

        return an_to_tel(x_ote_deg*3600., y_ote_deg*3600.)


    def tel_to_sci(self, x_tel, y_tel):
        """Overwriting standard behaviour for NIRSpec specific transformation.

        Parameters
        ----------
        x_tel : in arcsec
        y_tel : in arcsec
        filter_name

        Returns
        -------

        """
        x_an, y_an = tel_to_an(x_tel, y_tel)
        x_ote_deg, y_ote_deg = x_an/3600., y_an/3600.

        x_gwa_out, y_gwa_out = self.ote_to_gwa(x_ote_deg, y_ote_deg)
        x_gwa_in, y_gwa_in = self.gwaout_to_gwain(x_gwa_out, y_gwa_out)
        if self.AperType == 'SLIT':
            if self._parent_apertures is None:
                raise RuntimeError('Transformation not supported for this aperture: {}.'.format(self.AperName))
            else:
                # call transformation of parent aperture
                x_sci, y_sci = self._parent_aperture.gwa_to_sci(x_gwa_in, y_gwa_in)
        else:
            x_sci, y_sci = self.gwa_to_sci(x_gwa_in, y_gwa_in)

        return x_sci, y_sci


def points_on_arc(x0, y0, radius, phi1_deg, phi2_deg, N=100):
    """
    Compute points that lie on a circular arc between angles phi1 and phi2.
    Used for HST-FGS Pickles.
    :param x0:
    :param y0:
    :param radius:
    :param phi1_deg:
    :param phi2_deg:
    :param N:
    :return:
    """
    phi_rad = np.linspace(np.deg2rad(phi1_deg), np.deg2rad(phi2_deg), N)
    x = _x_from_polar(x0, radius, phi_rad)
    y = _y_from_polar(y0, radius, phi_rad)
    return x, y

class SiafCoordinates(object):
    """A helper class to hold coordinate arrays associated with a SIAF frame."""

    def __init__(self, x, y, frame):
        self.x = x
        self.y = y
        if frame in FRAMES:
            self.frame = frame
        else:
            raise NotImplementedError('Frame is not one of {}'.format(FRAMES))

def to_distortion_model(coefficients, degree=5):
    """
    Creates an astropy.modeling.Model object

    adapted from https://github.com/spacetelescope/ramp_simulator/blob/master/read_siaf_table.py
    Parameters
    ----------
    coefficients : array like
        Coefficients from the ISIM transformations file.
    degree : int
        Degree of polynomial.
        Default is 5 as in the ISIM file but many of the polynomials are of
        a smaller degree.

    Returns
    -------
    poly : astropy.modeling.Polynomial2D
        Polynomial model transforming one coordinate (x or y) between two systems.
    """

    # map Colin's coefficients into the order expected by Polynomial2D
    c = {}
    for cname in coefficients.colnames:
        siaf_i = int(cname[-2])
        siaf_j = int(cname[-1])
        name = 'c{0}_{1}'.format(siaf_i - siaf_j, siaf_j)
        c[name] = coefficients[cname].data[0]

    # 0,0 coefficient should not be used, according to Colin's TR    #JWST-STScI-001550
    # c['c0_0'] = 0 # JSA: commented 2018-01-21 because this could not be verified in
    # JWST-STScI-001550 rev A.

    return models.Polynomial2D(degree, **c)

def compare_apertures(reference_aperture, comparison_aperture, absolute_tolerance=None, attribute_list=None, print_file=sys.stdout, fractional_tolerance=1e-6, verbose=False, ignore_attributes=None):
    """Compare the attributes of two apertures.

    Parameters
    ----------
    reference_aperture
    comparison_aperture
    absolute_tolerance
    attribute_list
    print_file
    fractional_tolerance
    verbose

    Returns
    -------

    """
    if attribute_list is None:
        attribute_list = PRD_REQUIRED_ATTRIBUTES_ORDERED

    comparison_table = Table(names=('aperture', 'attribute', 'reference', 'comparison', 'difference', 'percent'), dtype=['S50']*6)

    add_blank_line = False
    for attribute in attribute_list:
        if (ignore_attributes is not None) and (attribute in list(ignore_attributes)):
            continue
        show = False
        reference_attr = getattr(reference_aperture, attribute)
        comparison_attr = getattr(comparison_aperture, attribute)
        if verbose:
            print('Comparing {} {}: {}{} {}{}'.format(reference_aperture, attribute, type(reference_attr), reference_attr, type(comparison_attr), comparison_attr))
        if reference_attr != comparison_attr:
            show = True
            # if isinstance(reference_attr, float) and isinstance(comparison_attr, float):
            if (type(reference_attr) in [int, float, np.float64]) and (type(comparison_attr) in [int, float, np.float64]):
                difference = np.abs(comparison_attr - reference_attr)
                fractional_difference = difference / np.max(
                    [np.abs(reference_attr), np.abs(comparison_attr)])
                if verbose:
                    print('difference={}, fractional_difference={}'.format(difference, fractional_difference))
                if (absolute_tolerance is not None) and math.isclose(reference_attr, comparison_attr, abs_tol=absolute_tolerance):
                    show = False
                elif fractional_difference <= fractional_tolerance:
                    show = False
                else:
                    fractional_difference_percent_string = '{:.4f}'.format(fractional_difference*100.)
                    difference_string = '{:.6f}'.format(difference)
            else:
                difference_string = 'N/A'
                fractional_difference_percent_string = 'N/A'

        if show:
            add_blank_line = True
            print('{:25} {:>15} {:>21} {:>21} {:>15} {:>10}'.format(reference_aperture.AperName, attribute, str(reference_attr), str(comparison_attr), difference_string, fractional_difference_percent_string), file=print_file)
            # add comparison data to table
            comparison_table.add_row([reference_aperture.AperName, attribute, str(reference_attr), str(comparison_attr), difference_string, fractional_difference_percent_string])

    if add_blank_line:
        print('', file=print_file)

    return comparison_table





