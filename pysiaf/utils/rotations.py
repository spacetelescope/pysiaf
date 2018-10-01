"""A collection of basic routines for performing rotation calculations

Authors
-------

    - Colin Cox


References
----------


"""
from __future__ import absolute_import, print_function, division
import numpy as np
from math import sqrt, sin, cos, atan2, asin, acos, radians, degrees


def attitude(v2, v3, ra, dec, pa):
    """This will make a 3D rotation matrix which rotates a unit vector representing a v2,v3 position
    to a unit vector representing an RA, Dec pointing with an assigned position angle
    Described in JWST-STScI-001550, SM-12, section 6.1

    Parameters
    ----------
    v2: float
      a position measured in arc-seconds
    v3: float
      a position measured in arc-seconds
    ra: float
        Right Ascension on the sky in degrees
    dec: float
         Declination on the sky in degrees
    pa: float
        Position angle in degrees measured from North to V3 axis in North to East direction.

    Returns
    -------
    m:   a (3 x 3) matrix
        represents the attitude of the telescope which points the given
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
    """Apply direct rotation to a vector using Rodrigues' formula

    Parameters
    ----------
    ax:     float array of size 3
            a unit vector represent a rotation axis
    phi:    float
            angle in degrees to rotate original vector
    u:      float
            array of size 3 representing any vector

    Returns
    -------
    v:      float
            array of size 3 representing the rotated vectot
    """

    rphi = radians(phi)
    v = u*cos(rphi) + cross(ax, u) * sin(rphi) + ax * np.dot(ax, u) * (1-cos(rphi))
    return v


def getv2v3(attitude, ra, dec):
    """Using the inverse of attitude matrix
    find v2,v3 position of any RA and Dec

    Parameters
    ----------
    attitude:   3 by 3 float array
                the telescope attitude matrix
    ra:         float
                RA of sky position
    dec:        float
                Dec of sky position

    Returns
    -------
    v2,v3:      float
                V2,V3 value at matching position

    """

    urd = unit(ra, dec)
    inverse_attitude = np.transpose(attitude)
    uv = np.dot(inverse_attitude, urd)
    v2, v3 = v2v3(uv)

    return v2, v3


def cross(a, b):
    """cross product of two vectors c = a X b
    The order is significant. Reversing the order changes the sign of the result

    Parameters
    ----------
    a:   float array or list of length 3
         first vector
    b    float array or list of length 3
         second vector

    Returns
    -------
    c   float array of length 3
        the product vector
    """

    c = np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])
    return c


def pointing(attitude, v2, v3, positive_ra=True):
    """Using the attitude matrix to calculate where any v2v3 position points on the sky.

    Parameters
    ----------
    attitude:   3 by 3 float array
                the telescope attitude matrix
    v2 :        float
                V2 coordinate in arc-seconds
    v3 :        float
                V3 coordinate in arc-seconds
    positive_ra : bool.
                If True forces ra value to be positive

    Returns
    -------
    rd :        tuple of floats
                (ra, dec) - RA and Dec in degrees
    """

    v2d = v2 / 3600.0
    v3d = v3 / 3600.0
    v = unit(v2d, v3d)
    w = np.dot(attitude, v)

    # tuple containing ra and dec in degrees
    rd = radec(w, positive_ra=positive_ra)
    return rd


def posangle(attitude, v2, v3):
    """Using the attitude matrix find the V3 angle at arbitrary v2,v3
    This is the angle measured from North to V3 in an anti-clockwise direction
    i.e. North to East
    Formulae from JWST-STScI-001550, SM-12, section 6.2
    Subtract 1 from each index in the text to allow for python zero indexing

    Parameters
    ----------
    attitude:   3 by 3 float array
                the telescope attitude matrix
    v2 :        float
                V2 coordinate in arc-seconds
    v3 :        float
                V3 coordinate in arc-seconds

    Returns
    -------
    pa          degrees - the position angle at (V2,V3)
    """

    A = attitude  # Synonym to simplify typing
    v2r = radians(v2 / 3600.0)
    v3r = radians(v3 / 3600.0)
    x = -(A[2, 0] * cos(v2r) + A[2, 1] * sin(v2r)) * sin(v3r) + A[2, 2] * cos(v3r)
    y = (A[0, 0] * A[1, 2] - A[1, 0] * A[0, 2]) * cos(v2r) + (A[0, 1] * A[1, 2] - A[1, 1] * A[
        0, 2]) * sin(v2r)
    pa = degrees(np.arctan2(y, x))
    return pa


def radec(u, positive_ra=False):
    """

    Parameters
    ----------
    u:              a float array or list of length 3
                    represents a unit vector so should have unit magnitude
                    if not, the normalization is forced withinin the method
    positive_ra:    bool
                    indicating whether to force ra to be positive

    Returns
    -------
    ra, dec:        float
                    RA and Dec in degrees corresponding to the unit vector u
    """

    assert len(u) == 3, 'Not a vector'
    norm = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)  # Works for list or array
    dec = degrees(np.arcsin(u[2] / norm))
    ra = degrees(np.arctan2(u[1], u[0]))  # atan2 puts it in the correct quadrant
    if positive_ra:
        if np.isscalar(ra) and ra < 0.0:
            ra += 360.0
        if not np.isscalar(ra) and np.any(ra < 0.0):
            index = np.where(ra < 0.0)[0]
            ra[index] += 360.0
    return ra, dec


def rodrigues(attitude):
    """Interpret rotation matrix as a single rotation by angle phi around unit length axis
    Return axis, angle and matching quaternion
    The quaternion is given in a slightly irregular order with the angle value preceding the axis information.
    Most of the literature shows the reverse order but JWST flight software uses the order given here.

    Parameters
    ----------
    attitude:   3 by 3 float array
                the telescope attitude matrix

    Returns
    -------
    axis:   float array of length 3
            a unit vector which is the rotation axis
    phi:    float
            angle of rotation in degrees
    q:      float array of length 4
            the equivalent quaternion
    """

    A = attitude  # Synonym for clarity and to save typing
    cos_phi = 0.5 * (A[0, 0] + A[1, 1] + A[2, 2] - 1.0)
    phi = np.arccos(cos_phi)
    axis = np.array([A[2, 1] - A[1, 2], A[0, 2] - A[2, 0], A[1, 0] - A[0, 1]]) / (2.0 * sin(phi))

    # Make corresponding quaternion
    q = np.hstack(([cos(phi/2.0)], axis * sin(phi/2.0)))
    phi = degrees(phi)

    return axis, phi, q


def rotate(axis, angle):
    """Implements fundamental 3D rotation matrices.
    Rotate a vector by an angle measured in degrees, about axis 1 2 or 3 in the inertial frame.
    This is an anti-clockwise rotation when sighted along the axis, commonly called a
    right-handed rotation.

    Parameters
    ----------
    axis:   integer
            axis number, 1 2 or 3
    angle   float
            angle of rotation in degrees

    Returns
    -------
    r   a (3 x 3) float array
        matrix which performs the specified rotation.
    """

    assert axis in list(range(1, 4)), 'Axis must be in range 1 to 3'
    r = np.zeros((3, 3))
    theta = radians(angle)

    ax0 = axis-1 # Allow for zero offset numbering
    ax1 = (ax0+1) % 3 # Axes in cyclic order
    ax2 = (ax0+2) % 3
    r[ax0, ax0] = 1.0
    r[ax1, ax1] = cos(theta)
    r[ax2, ax2] = cos(theta)
    r[ax1, ax2] = -sin(theta)
    r[ax2, ax1] = sin(theta)

    return r


def rv(v2, v3):
    """Rotate from v2,v3 position to V1 axis, i.e rotate so that a  V2,V3 position
    ends up where V1 started.


    Parameters
    ----------
    v2: float
        V2 position in arc-sec
    v3: float
        V3 position in arc-sec

    Returns
    -------
    rv:     a (3 x 3) array
            matrix which performs the rotation described.

    """

    v2d = v2 / 3600.0  # convert from arcsec to degrees
    v3d = v3 / 3600.0
    mv2 = rotate(3, -v2d)
    mv3 = rotate(2, v3d)
    rv = np.dot(mv3, mv2)
    return rv


def sky_posangle(attitude, ra, dec):
    """Using the attitude matrix find the V3 angle at arbitrary RA and Dec
    This is the angle measured from North to V3 in an anti-clockwise direction

    Parameters
    ----------
    attitude:   3 by 3 float array
                the telescope attitude matrix
    ra:         float
                RA position in degrees
    dec:        float
                Dec position in degrees

    Returns
    -------
    pa:         float
                resulting position angle in degrees
    """

    rar = radians(ra)
    decr = radians(dec)
    A = attitude  # Synonym to simplify typing
    # Pointing of V3 axis
    v3ra = atan2(A[1, 2], A[0, 2])
    v3dec = asin(A[2, 2])
    x = sin(v3dec) * cos(decr) - cos(v3dec) * sin(decr) * cos(v3ra - rar)
    y = cos(v3dec) * sin(v3ra - rar)
    pa = degrees(atan2(y, x))
    return pa


def slew(v2t, v3t, v2a, v3a):
    """Calculate matrix which slews from target (v2t,v3t)
    to aperture position (v2a, v3a) without a roll change.
    Useful for target acquisition calculations.

    Parameters
    ----------
    v2t:    float
            Initial V2 position in arc-sec
    v3t:    float
            Initial V3 position in arc-sec
    v2a:    float
            Final V2 position in arc-sec
    v3a:    float
            Final V3 position in arc-sec

    Returns
    -------
    mv      a (3 x 3) float array
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
    """Converts vector expressed in Euler angles to unit vector components.

    Parameters
    ----------
    ra:     float
            RA of sky position in degrees
    dec:    float
            Dec of sky position in degrees

    Returns
    -------
    u       a float array of length 3
            the equivalent unit vector
    """

    rar = radians(ra)
    decr = radians(dec)
    u = np.array([cos(rar)*cos(decr), sin(rar)*cos(decr), sin(decr)])
    return u


def v2v3(u):
    """Convert unit vector to v2v3

    Parameters
    ----------
    u:  float list or array of length 3
        a unit vector.

    Returns
    -------
    v2, v3  The same position represented by V2,V3 values in arc-sec.
    """

    assert len(u) == 3, 'Not a vector'
    norm = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2) # Works for list or array
    v2 = 3600*degrees(np.arctan2(u[1], u[0])) # atan2 puts it in the correct quadrant
    v3 = 3600*degrees(np.arcsin(u[2]/norm))
    return v2, v3
