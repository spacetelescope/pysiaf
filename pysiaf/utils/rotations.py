"""A collection of basic routines for performing rotation calculations.

Authors
-------
    Colin Cox
    Johannes Sahlmann

"""
from __future__ import absolute_import, print_function, division
import copy
import numpy as np

import astropy.units as u
from astropy.modeling.rotations import rotation_matrix


def attitude(v2, v3, ra, dec, pa):
    """Return rotation matrix that transforms from v2,v3 to RA,Dec.

    Makes a 3D rotation matrix which rotates a unit vector representing a v2,v3 position
    to a unit vector representing an RA, Dec pointing with an assigned position angle
    Described in JWST-STScI-001550, SM-12, section 5.1.

    Parameters
    ----------
    v2 : float
        a position measured in arc-seconds
    v3 : float
        a position measured in arc-seconds
    ra : float
        Right Ascension on the sky in degrees
    dec : float
        Declination on the sky in degrees
    pa : float
        Position angle in degrees measured from North to V3 axis in North to East direction.

    Returns
    -------
    m : numpy matrix
        A (3 x 3) matrix represents the attitude of the telescope which points the given
        V2V3 position to the indicated RA and Dec and with the V3 axis rotated by position angle pa

    """
    v2d = v2 / 3600.0
    v3d = v3 / 3600.0

    # Get separate rotation matrices
    mv2 = rotate(3, -v2d)
    mv3 = rotate(2, v3d)
    mra = rotate(3, ra)
    mdec = rotate(2, -dec)
    mpa = rotate(1, -pa)

    # Combine as mra*mdec*mpa*mv3*mv2
    m = np.dot(mv3, mv2)
    m = np.dot(mpa, m)
    m = np.dot(mdec, m)
    m = np.dot(mra, m)

    return m


def convert_quantity(x_in, to_unit, factor=1.):
    """Check if astropy quantity and apply conversion factor

    Parameters
    ----------
    x_in : float or quantity
        input
    to_unit : astropy.units unit
        unit to convert to
    factor : float
        Factor to apply if input is not a quantity

    Returns
    -------
    x_out : float
        converted value

    """
    x = copy.deepcopy(x_in)
    if isinstance(x, u.Quantity):
        x_out = x.to(to_unit).value
    else:
        x_out = x * factor
    return x_out


def attitude_matrix(nu2, nu3, ra, dec, pa, convention='JWST'):
    """Return attitude matrix.

    Makes a 3D rotation matrix that transforms between telescope frame
    and sky. It rotates a unit vector on the idealized focal sphere
    (specified by the spherical coordinates nu2, nu3) to a unit vector
    representing an RA, Dec pointing with an assigned position angle
    measured at nu2, nu3.
    See JWST-STScI-001550, SM-12, section 5.1.

    Parameters
    ----------
    nu2 : float
        an euler angle (default unit is arc-seconds)
    nu3 : float
        an euler angle (default unit is arc-seconds)
    ra : float
        Right Ascension on the sky in degrees
    dec : float
        Declination on the sky in degrees
    pa : float
        Position angle of V3 axis at nu2,nu3 measured from
        North to East (default unit is degree)

    Returns
    -------
    m : numpy matrix
        the attitude matrix

    """
    if convention == 'JWST':
        pa_sign = -1.

    if isinstance(nu2, u.Quantity):
        nu2_value = nu2.to(u.deg).value
    else:
        nu2_value = nu2 / 3600.

    if isinstance(nu3, u.Quantity):
        nu3_value = nu3.to(u.deg).value
    else:
        nu3_value = nu3 / 3600.

    if isinstance(pa, u.Quantity):
        pa_value = pa.to(u.deg).value
    else:
        pa_value = pa
    if isinstance(ra, u.Quantity):
        ra_value = ra.to(u.deg).value
    else:
        ra_value = ra
    if isinstance(dec, u.Quantity):
        dec_value = dec.to(u.deg).value
    else:
        dec_value = dec

    # Get separate rotation matrices
    # astropy's rotation matrix takes inverse sign compared to rotations.rotate
    mv2 = rotation_matrix(-1*-nu2_value, axis='z')
    mv3 = rotation_matrix(-1*nu3_value, axis='y')
    mra = rotation_matrix(-1*ra_value, axis='z')
    mdec = rotation_matrix(-1*-dec_value, axis='y')
    mpa = rotation_matrix(-1*pa_sign*pa_value, axis='x')

    # Combine as mra*mdec*mpa*mv3*mv2
    m = np.dot(mv3, mv2)
    m = np.dot(mpa, m)
    m = np.dot(mdec, m)
    m = np.dot(mra, m)

    return m


def axial_rotation(ax, phi, vector):
    """Apply direct rotation to a vector using Rodrigues' formula.

    Parameters
    ----------
    ax : float array of size 3
        a unit vector represent a rotation axis
    phi : float
        angle in degrees to rotate original vector
    vector : float
        array of size 3 representing any vector

    Returns
    -------
    v : float
        array of size 3 representing the rotated vectot

    """
    rphi = np.radians(phi)
    v = vector*np.cos(rphi) + cross(ax, vector) * np.sin(rphi) + ax * np.dot(ax, vector) * (1-np.cos(rphi))
    return v


def sky_to_tel(attitude, ra, dec, verbose=False): #, return_cartesian=False):
    """Transform from sky (RA, Dec) to telescope (nu2, nu3) angles.

    Return nu2,nu3 position on the idealized focal sphere of any RA and
    Dec using the inverse of attitude matrix.

    Parameters
    ----------
    attitude : 3 by 3 float array
        The attitude matrix.
    ra : float (default unit is degree)
        RA of sky position
    dec : float (default unit is degree)
        Dec of sky position

    Returns
    -------
    nu2, nu3 : tuple of floats with quantity
        spherical coordinates at matching position on the idealized focal sphere

    """
    # ra = convert_quantity(ra, u.deg)
    # dec = convert_quantity(dec, u.deg)
    if attitude.shape != (3,3):
        raise ValueError('Attitude has to be 3x3 array.')

    # if return_cartesian:
    #     ra_rad = np.deg2rad(ra)
    #     dec_rad = np.deg2rad(dec)
    #     urd = np.array([np.sqrt(1. - (ra_rad ** 2 + dec_rad ** 2)), ra_rad, dec_rad])
    # else:
    #     urd = unit(ra, dec)

    unit_vector_sky_side = unit_vector_sky(ra, dec)
    if verbose:
        print('Sky-side unit vector: {}'.format(unit_vector_sky_side))
    inverse_attitude = np.transpose(attitude)

    # apply transformation
    unit_vector_tel = np.dot(inverse_attitude, unit_vector_sky_side)
    if verbose:
        print('Tel-side unit vector: {}'.format(unit_vector_tel))

    # extract spherical coordinates
    nu2, nu3 = polar_angles(unit_vector_tel)

    return nu2, nu3


def getv2v3(attitude, ra, dec):
    """Return v2,v3 position of any RA and Dec using the inverse of attitude matrix.

    Parameters
    ----------
    attitude : 3 by 3 float array
        the telescope attitude matrix
    ra : float
        RA of sky position
    dec : float
        Dec of sky position

    Returns
    -------
    v2,v3 : tuple of floats
        V2,V3 value at matching position

    """
    urd = unit(ra, dec)

    inverse_attitude = np.transpose(attitude)
    uv = np.dot(inverse_attitude, urd)

    v2, v3 = v2v3(uv)

    return v2, v3


def cross(a, b):
    """Return cross product of two vectors c = a X b.

    The order is significant. Reversing the order changes the sign of the result.

    Parameters
    ----------
    a : float array or list of length 3
        first vector
    b : float array or list of length 3
        second vector

    Returns
    -------
    c   float array of length 3
        the product vector

    """
    c = np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])
    return c


def pointing(attitude, v2, v3, positive_ra=True, input_cartesian=False):
    """Calculate where a v2v3 position points on the sky using the attitude matrix.

    Parameters
    ----------
    attitude : 3 by 3 float array
        the telescope attitude matrix
    v2 : float or array of floats
        V2 coordinate in arc-seconds
    v3 : float or array of floats
        V3 coordinate in arc-seconds
    positive_ra : bool.
        If True forces ra value to be positive

    Returns
    -------
    rd : tuple of floats
        (ra, dec) - RA and Dec in degrees

    """
    v2d = v2 / 3600.0
    v3d = v3 / 3600.0

    # compute unit vector
    if input_cartesian:
        v2_rad = np.deg2rad(v2d)
        v3_rad = np.deg2rad(v3d)
        v = np.array([np.sqrt(1. - (v2_rad ** 2 + v3_rad ** 2)), v2_rad, v3_rad])
    else:
        v = unit(v2d, v3d)

    # apply attitude transformation
    w = np.dot(attitude, v)

    # compute tuple containing ra and dec in degrees
    if input_cartesian:
        rd = np.rad2deg(w[1]), np.rad2deg(w[2])
    else:
        rd = radec(w, positive_ra=positive_ra)

    return rd


def tel_to_sky(attitude, nu2, nu3, positive_ra=True):#, input_cartesian=False):
    """Calculate where a nu2,nu3 position points on the sky.

    Parameters
    ----------
    attitude : 3 by 3 float array
        the telescope attitude matrix
    nu2 : float or array of floats (default unit is arcsecond)
        V2 coordinate in arc-seconds
    nu3 : float or array of floats (default unit is arcsecond)
        V3 coordinate in arc-seconds
    positive_ra : bool.
        If True forces ra value to be positive

    Returns
    -------
    rd : tuple of floats with quantity
        (ra, dec) - RA and Dec

    """

    nu2_deg = convert_quantity(nu2, u.deg, factor=u.arcsec.to(u.deg))
    nu3_deg = convert_quantity(nu3, u.deg, factor=u.arcsec.to(u.deg))

    # v2d = v2 / 3600.0
    # v3d = v3 / 3600.0

    # # compute unit vector
    # if input_cartesian:
    #     v2_rad = np.deg2rad(v2d)
    #     v3_rad = np.deg2rad(v3d)
    #     v = np.array([np.sqrt(1. - (v2_rad ** 2 + v3_rad ** 2)), v2_rad, v3_rad])
    # else:
    #     v = unit(v2d, v3d)
    unit_vector_tel = unit_vector_sky(nu2_deg, nu3_deg)

    # apply attitude transformation
    unit_vector_sky_side = np.dot(attitude, unit_vector_tel)

    # compute tuple containing ra and dec in degrees
    # if input_cartesian:
    #     rd = np.rad2deg(w[1]), np.rad2deg(w[2])
    # else:
    #     rd = radec(w, positive_ra=positive_ra)
    ra, dec = polar_angles(unit_vector_sky_side, positive_azimuth=positive_ra)
    return ra, dec


def posangle(attitude, v2, v3):
    """Return the V3 angle at arbitrary v2,v3 using the attitude matrix.

    This is the angle measured from North to V3 in an anti-clockwise direction i.e. North to East.
    Formulae from JWST-STScI-001550, SM-12, section 6.2.
    Subtract 1 from each index in the text to allow for python zero indexing.

    Parameters
    ----------
    attitude : 3 by 3 float array
        the telescope attitude matrix
    v2 : float
        V2 coordinate in arc-seconds
    v3 : float
        V3 coordinate in arc-seconds

    Returns
    -------
    pa : float
        Angle in degrees - the position angle at (V2,V3)

    """
    v2r = np.radians(v2 / 3600.0)
    v3r = np.radians(v3 / 3600.0)
    x = -(attitude[2, 0] * np.cos(v2r) + attitude[2, 1] * np.sin(v2r)) * np.sin(v3r) \
        + attitude[2, 2] * np.cos(v3r)
    y = (attitude[0, 0] * attitude[1, 2] - attitude[1, 0] * attitude[0, 2]) * np.cos(v2r) \
        + (attitude[0, 1] * attitude[1, 2] - attitude[1, 1] * attitude[0, 2]) * np.sin(v2r)
    pa = np.degrees(np.arctan2(y, x))
    return pa


def rodrigues(attitude):
    """Interpret rotation matrix as a single rotation by angle phi around unit length axis.

    Return axis, angle and matching quaternion.
    The quaternion is given in a slightly irregular order with the angle value preceding the axis
    information. Most of the literature shows the reverse order but JWST flight software uses the
    order given here.

    Parameters
    ----------
    attitude : 3 by 3 float array
        the telescope attitude matrix

    Returns
    -------
    axis : float array of length 3
        a unit vector which is the rotation axis
    phi : float
        angle of rotation in degrees
    q : float array of length 4
        the equivalent quaternion

    """
    cos_phi = 0.5 * (attitude[0, 0] + attitude[1, 1] + attitude[2, 2] - 1.0)
    phi = np.arccos(cos_phi)
    axis = np.array([attitude[2, 1] - attitude[1, 2], attitude[0, 2] - attitude[2, 0],
                     attitude[1, 0] - attitude[0, 1]]) / (2.0 * np.sin(phi))

    # Make corresponding quaternion
    q = np.hstack(([np.cos(phi/2.0)], axis * np.sin(phi/2.0)))
    phi = np.degrees(phi)

    return axis, phi, q


def rotate(axis, angle):
    """Implement fundamental 3D rotation matrices.

    Rotate a vector by an angle measured in degrees, about axis 1 2 or 3 in the inertial frame.
    This is an anti-clockwise rotation when sighted along the axis, commonly called a
    right-handed rotation.

    Parameters
    ----------
    axis : int
            axis number, 1, 2, or 3
    angle : float
            angle of rotation in degrees

    Returns
    -------
    r : float array
        a (3 x 3) matrix which performs the specified rotation.

    """
    assert axis in list(range(1, 4)), 'Axis must be in range 1 to 3'
    r = np.zeros((3, 3))
    theta = np.radians(angle)

    ax0 = axis-1  # Allow for zero offset numbering
    ax1 = (ax0+1) % 3  # Axes in cyclic order
    ax2 = (ax0+2) % 3
    r[ax0, ax0] = 1.0
    r[ax1, ax1] = np.cos(theta)
    r[ax2, ax2] = np.cos(theta)
    r[ax1, ax2] = -np.sin(theta)
    r[ax2, ax1] = np.sin(theta)

    return r


def rv(v2, v3):
    """Rotate from v2,v3 position to V1 axis.

    Rotate so that a  V2,V3 position ends up where V1 started.

    Parameters
    ----------
    v2 : float
        V2 position in arc-sec
    v3 : float
        V3 position in arc-sec

    Returns
    -------
    rv : a (3 x 3) array
        matrix which performs the rotation described.

    """
    v2d = v2 / 3600.0  # convert from arcsec to degrees
    v3d = v3 / 3600.0
    mv2 = rotate(3, -v2d)
    mv3 = rotate(2, v3d)
    rv = np.dot(mv3, mv2)
    return rv


def sky_posangle(attitude, ra, dec):
    """Return the V3 angle at arbitrary RA and Dec using the attitude matrix.

    This is the angle measured from North to V3 in an anti-clockwise direction.

    Parameters
    ----------
    attitude : 3 by 3 float array
        the telescope attitude matrix
    ra : float
        RA position in degrees
    dec : float
        Dec position in degrees

    Returns
    -------
    pa : float
        resulting position angle in degrees

    """
    rar = np.radians(ra)
    decr = np.radians(dec)
    # Pointing of V3 axis
    v3ra = np.arctan2(attitude[1, 2], attitude[0, 2])
    v3dec = np.arcsin(attitude[2, 2])
    x = np.sin(v3dec) * np.cos(decr) - np.cos(v3dec) * np.sin(decr) * np.cos(v3ra - rar)
    y = np.cos(v3dec) * np.sin(v3ra - rar)
    pa = np.degrees(np.arctan2(y, x))
    return pa


def slew(v2t, v3t, v2a, v3a):
    """Calculate matrix which slews from target (v2t,v3t) to aperture position (v2a, v3a) without a roll change.

    Useful for target acquisition calculations.

    Parameters
    ----------
    v2t : float
        Initial V2 position in arc-sec
    v3t : float
        Initial V3 position in arc-sec
    v2a : float
        Final V2 position in arc-sec
    v3a : float
        Final V3 position in arc-sec

    Returns
    -------
    mv : a (3 x 3) float array
        The matrix that performs the rotation described

    """
    v2td = v2t/3600.0
    v3td = v3t/3600.0
    v2ad = v2a/3600.0
    v3ad = v3a/3600.0
    r1 = rotate(3, -v2td)
    r2 = rotate(2, v3td-v3ad)
    r3 = rotate(3, v2ad)

    # Combine r3 r2 r1
    mv = np.dot(r2, r1)
    mv = np.dot(r3, mv)

    return mv


def unit(ra, dec):
    """Convert inertial frame vector expressed in polar coordinates / Euler angles to unit vector components.

    See Section 5 of JWST-STScI-001550 and Equation 4.1 of JWST-PLAN-006166.

    Parameters
    ----------
    ra : float or array of floats
        RA of sky position in degrees
    dec : float or array of floats
        Dec of sky position in degrees

    Returns
    -------
    vector : float array of length 3
        the equivalent unit vector in the inertial frame

    """
    rar = np.deg2rad(ra)
    decr = np.deg2rad(dec)
    vector = np.array([np.cos(rar)*np.cos(decr), np.sin(rar)*np.cos(decr), np.sin(decr)])
    return vector


def unit_vector_sky(ra, dec):
    """Return unit vector on the celestial sphere.

    Parameters
    ----------
    ra : float or array of floats (default unit is degree)
        RA of sky position
    dec : float or array of floats (default unit is degree)
        Dec of sky position

    Returns
    -------
    vector : float array of length 3
        the equivalent unit vector in the inertial frame

    """
    ra_rad = convert_quantity(ra, u.rad, factor=np.deg2rad(1.))
    dec_rad = convert_quantity(dec, u.rad, factor=np.deg2rad(1.))
    vector = np.array([np.cos(ra_rad)*np.cos(dec_rad), np.sin(ra_rad)*np.cos(dec_rad), np.sin(dec_rad)])
    return vector


def unit_vector_hst_fgs_object(rho, phi):
    """Return unit vector on the celestial sphere.

    This is according to the HST object space angle definitions,
    CSC/TM-82/6045 1987, Section 4.1.2.2.4

    Parameters
    ----------
    rho : float or array of floats (default unit is degree)
        RA of sky position
    phi : float or array of floats (default unit is degree)
        Dec of sky position

    Returns
    -------
    vector : float array of length 3
        the equivalent unit vector in the inertial frame

    """
    rho_rad = convert_quantity(rho, u.rad, factor=np.deg2rad(1.))
    phi_rad = convert_quantity(phi, u.rad, factor=np.deg2rad(1.))
    vector = np.array([np.sin(rho_rad)*np.cos(phi_rad), np.sin(rho_rad)*np.sin(phi_rad), np.cos(rho_rad)])
    return vector


def radec(vector, positive_ra=False):
    """Return RA and Dec in degrees corresponding to the unit vector vector.

    Parameters
    ----------
    vector : a float array or list of length 3
        represents a unit vector so should have unit magnitude
        if not, the normalization is forced within the method
    positive_ra : bool
        indicating whether to force ra to be positive

    Returns
    -------
    ra , dec : tuple of floats
        RA and Dec in degrees corresponding to the unit vector vector

    """
    if len(vector) != 3:
        raise ValueError('Input is not a 3D vector')
    norm = np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)  # Works for list or array
    ra = np.degrees(np.arctan2(vector[1], vector[0]))  # atan2 puts it in the correct quadrant
    dec = np.degrees(np.arcsin(vector[2] / norm))
    if positive_ra:
        if np.isscalar(ra) and ra < 0.0:
            ra += 360.0
        if not np.isscalar(ra) and np.any(ra < 0.0):
            index = np.where(ra < 0.0)[0]
            ra[index] += 360.0
    return ra, dec


def v2v3(vector):
    """Compute v2,v3 polar coordinates corresponding to a unit vector in the rotated (telescope) frame.

    See Section 5 of JWST-STScI-001550.

    Parameters
    ----------
    vector : float list or array of length 3
        unit vector of cartesian coordinates in the rotated (telescope) frame

    Returns
    -------
    v2, v3 : tuple of floats
        The same position represented by V2, V3 values in arc-sec.

    """
    assert len(vector) == 3, 'Not a vector'

    norm = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    v2 = 3600. * np.degrees(np.arctan2(vector[1], vector[0]))  # atan2 puts it in the correct quadrant
    v3 = 3600. * np.degrees(np.arcsin(vector[2]/norm))
    return v2, v3


def polar_angles(vector, positive_azimuth=False):
    """Compute polar coordinates of an unit vector.

    Parameters
    ----------
    vector : float list or array of length 3
        3-component unit vector
    positive_azimuth : bool
        If True, the returned nu2 value is forced to be positive.

    Returns
    -------
    nu2, nu3 : tuple of floats with astropy quantity
        The same position represented by polar coordinates

    """
    if len(vector) != 3:
        raise ValueError('Input is not a vector or an array of vectors')

    norm = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    nu2 = np.arctan2(vector[1], vector[0]) * u.rad
    nu3 = np.arcsin(vector[2]/norm) * u.rad

    if positive_azimuth:
        if np.isscalar(nu2.value) and nu2.value < 0.0:
            nu2 += 360.0 * u.deg
        if not np.isscalar(nu2.value) and np.any(nu2.value < 0.0):
            index = np.where(nu2.value < 0.0)[0]
            nu2[index] += 360.0 * u.deg
    return nu2, nu3


def unit_vector_from_cartesian(x=None, y=None, z=None):
    """Return unit vector corresponding to two cartesian coordinates.

    Array inputs are supported.

    Parameters
    ----------
    x : float or quantity
        cartesian unit vector X coordinate in radians
    y : float or quantity
        cartesian unit vector Y coordinate in radians
    z : float or quantity
        cartesian unit vector Z coordinate in radians

    Returns
    -------
    unit_vector : numpy.ndarray
        Unit vector

    """
    # check that two arguments are provided
    if x is None:
        if y is None or z is None:
            raise TypeError('Function requires axactly two arguments.')
    if y is None:
        if x is None or z is None:
            raise TypeError('Function requires axactly two arguments.')
    if z is None:
        if x is None or y is None:
            raise TypeError('Function requires axactly two arguments.')

    # convert to radian
    if isinstance(x, u.Quantity):
        x_rad = x.to(u.rad).value
    else:
        x_rad = x

    if isinstance(y, u.Quantity):
        y_rad = y.to(u.rad).value
    else:
        y_rad = y

    if isinstance(z, u.Quantity):
        z_rad = z.to(u.rad).value
    else:
        z_rad = z

    # handle array and scalar inputs
    if np.array([x_rad]).all() and np.array([y_rad]).all():
        unit_vector = np.array([x_rad, y_rad, np.sqrt(1-(x_rad**2+y_rad**2))])
    elif np.array([y_rad]).all() and np.array([z_rad]).all():
        unit_vector = np.array([np.sqrt(1-(y_rad**2+z_rad**2)), y_rad, z_rad])
    elif np.array([x_rad]).all() and np.array([z_rad]).all():
        unit_vector = np.array([x_rad, np.sqrt(1-(x_rad**2+z_rad**2)), z_rad])

    if np.any(np.isnan(unit_vector)):
        raise ValueError('Invalid arguments. Inputs should be in radians.')

    return unit_vector



def idl_to_tel_rotation_matrix(V2Ref_arcsec, V3Ref_arcsec, V3IdlYAngle_deg):
    """Return 3D rotation matrix for ideal to telescope transformation.

    Parameters
    ----------
    V2Ref_arcsec
    V3Ref_arcsec
    V3IdlYAngle_deg

    Returns
    -------

    """
    M1 = rotate(3, -1 * V2Ref_arcsec / 3600.)
    M2 = rotate(2, V3Ref_arcsec / 3600.)
    M3 = rotate(1, V3IdlYAngle_deg)
    M4 = np.dot(M2, M1)
    M = np.dot(M3, M4)

    return M
