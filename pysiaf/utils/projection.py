"""A collection of functions to support tangent-plane de-/projections.

Authors
-------
    Johannes Sahlmann

"""

from astropy.modeling import models as astmodels
from astropy.modeling import rotations as astrotations


def project_to_tangent_plane(ra, dec, ra_ref, dec_ref, scale=1.):
    """Convert ra/dec coordinates into pixel coordinates using a tangent plane projection.

    Theprojection's reference point has to be specified.
    Scale is a convenience parameter that defaults to 1.0, in which case the returned pixel
    coordinates are also in degree. Scale can be set to a pixel scale to return detector coordinates
    in pixels

    Parameters
    ----------
    ra : float
        Right Ascension in decimal degrees

    dec: float
        declination in decimal degrees

    ra_ref : float
        Right Ascension of reference point in decimal degrees

    dec_ref: float
        declination of reference point in decimal degrees

    scale : float
        Multiplicative factor that is applied to the returned values. Default is 1.0

    Returns
    -------
    x,y : float
        pixel coordinates in decimal degrees if scale = 1.0

    """
    # for zenithal projections, i.e. gnomonic, i.e. TAN:
    lonpole = 180.

    # tangent plane projection from phi/theta to x,y
    tan = astmodels.Sky2Pix_TAN()

    # compute native coordinate rotation to obtain phi and theta
    rot_for_tan = astrotations.RotateCelestial2Native(ra_ref, dec_ref, lonpole)

    phi_theta = rot_for_tan(ra, dec)

    # pixel coordinates,  x and y are in degree-equivalent
    x, y = tan(phi_theta[0], phi_theta[1])

    x = x * scale
    y = y * scale

    return x, y


def deproject_from_tangent_plane(x, y, ra_ref, dec_ref, scale=1.):
    """Convert pixel coordinates into ra/dec coordinates using a tangent plane de-projection.

    The projection's reference point has to be specified.
    See the inverse transformation radec2Pix_TAN.

    Parameters
    ----------
    x : float
        Pixel coordinate (default is in decimal degrees, but depends on value of scale parameter)
        x/scale has to be degrees.
    y : float
        Pixel coordinate (default is in decimal degrees, but depends on value of scale parameter)
        x/scale has to be degrees.
    ra_ref : float
        Right Ascension of reference point in decimal degrees
    dec_ref: float
        declination of reference point in decimal degrees
    scale : float
        Multiplicative factor that is applied to the input values. Default is 1.0

    Returns
    -------
    ra : float
        Right Ascension in decimal degrees
    dec: float
        declination in decimal degrees

    """
    # for zenithal projections, i.e. gnomonic, i.e. TAN
    lonpole = 180.

    x = x / scale
    y = y / scale

    # tangent plane projection from x,y to phi/theta
    tan = astmodels.Pix2Sky_TAN()

    # compute native coordinate rotation to obtain ra and dec
    rot_for_tan = astrotations.RotateNative2Celestial(ra_ref, dec_ref, lonpole)

    phi, theta = tan(x, y)

    # ra and dec
    ra, dec = rot_for_tan(phi, theta)

    return ra, dec
