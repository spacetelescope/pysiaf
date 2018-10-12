"""A collection of basic routines for performing rotation calculations.

Authors
-------
    Colin Cox

"""
from __future__ import absolute_import, print_function, division
import numpy as np


def attitude(v2, v3, ra, dec, pa):
    """Return rotation matrix that transforms from v2,v3 to RA,Dec.

    Makes a 3D rotation matrix which rotates a unit vector representing a v2,v3 position
    to a unit vector representing an RA, Dec pointing with an assigned position angle
    Described in JWST-STScI-001550, SM-12, section 6.1.

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
        A (3 x 3) matrixrepresents the attitude of the telescope which points the given
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


def axial_rotation(ax, phi, u):
    """Apply direct rotation to a vector using Rodrigues' formula.

    Parameters
    ----------
    ax : float array of size 3
        a unit vector represent a rotation axis
    phi : float
        angle in degrees to rotate original vector
    u : float
        array of size 3 representing any vector

    Returns
    -------
    v : float
        array of size 3 representing the rotated vectot

    """
    rphi = np.radians(phi)
    v = u*np.cos(rphi) + cross(ax, u) * np.sin(rphi) + ax * np.dot(ax, u) * (1-np.cos(rphi))
    return v


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


def pointing(attitude, v2, v3, positive_ra=True, verbose=False):
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
    if verbose:
        print('POINTING v2v3')
        print(v2)
        print(v3)
        print(v2d)
        print(v3d)
    v = unit(v2d, v3d)
    w = np.dot(attitude, v)

    # tuple containing ra and dec in degrees
    rd = radec(w, positive_ra=positive_ra)
    return rd


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


def radec(u, positive_ra=False):
    """Return RA and Dec in degrees corresponding to the unit vector u.

    Parameters
    ----------
    u : a float array or list of length 3
        represents a unit vector so should have unit magnitude
        if not, the normalization is forced within the method
    positive_ra : bool
        indicating whether to force ra to be positive

    Returns
    -------
    ra , dec : tuple of floats
        RA and Dec in degrees corresponding to the unit vector u

    """
    assert len(u) == 3, 'Not a vector'
    norm = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)  # Works for list or array
    dec = np.degrees(np.arcsin(u[2] / norm))
    ra = np.degrees(np.arctan2(u[1], u[0]))  # atan2 puts it in the correct quadrant
    if positive_ra:
        if np.isscalar(ra) and ra < 0.0:
            ra += 360.0
        if not np.isscalar(ra) and np.any(ra < 0.0):
            index = np.where(ra < 0.0)[0]
            ra[index] += 360.0
    return ra, dec


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
    """Convert vector expressed in Euler angles to unit vector components.

    Parameters
    ----------
    ra : float or array of floats
        RA of sky position in degrees
    dec : float or array of floats
        Dec of sky position in degrees

    Returns
    -------
    u : float array of length 3
        the equivalent unit vector

    """
    rar = np.radians(ra)
    decr = np.radians(dec)
    u = np.array([np.cos(rar)*np.cos(decr), np.sin(rar)*np.cos(decr), np.sin(decr)])
    return u


def v2v3(u):
    """Convert unit vector to v2v3.

    Parameters
    ----------
    u : float list or array of length 3
        a unit vector.

    Returns
    -------
    v2, v3 : tuple of floate
        The same position represented by V2,V3 values in arc-sec.

    """
    assert len(u) == 3, 'Not a vector'
    norm = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)  # Works for list or array
    v2 = 3600*np.degrees(np.arctan2(u[1], u[0]))  # atan2 puts it in the correct quadrant
    v3 = 3600*np.degrees(np.arcsin(u[2]/norm))
    return v2, v3
