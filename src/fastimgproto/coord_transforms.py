import numpy as np
import astropy.units as u


def xyz_to_uvw_rotation_matrix(hour_angle, declination):
    """
    Generates rotation matrix from XYZ -> UVW co-ordinates, for given HA, Dec.

    For an explanation and derivation, see Notebook 4.1,
    Sections 4.1.1d and 4.1.2
    of https://github.com/griffinfoster/fundamentals_of_interferometry
    ("The (`u,v,w`) Space")

    The definition of XYZ used here is that:

    - X axis points towards the crossing of the celestial equator and meridian
      at hour-angle :math:`H=0h`
    - Y axis points towards hour-angle :math:`H=-6h` (Due East from telescope, in
      the local-XYZ / local hour-angle case.)
    - Z axis points towards the North Celestial Pole (NCP).

    The definition of UVW used here is that:

    - the `u`-axis lies in the celestial equatorial plane, and points toward the
      hour angle :math:`H_0-6h` (where :math:`H_0` is the hour-angle of
      the source) (i.e. East of source).
    - the `v`-axis lies in the plane of the great circle with hour angle
      :math:`H_0`, and points toward the declination
      and points toward the declination :math:`\\frac{\\pi}{2}-\\delta_0`
      (North of source).
    - the `w`-axis points in the direction of the source.


    Can be used to generate a rotation matrix for either local XYZ co-ordinates,
    or for Global XYZ (also known as ECEF or ITRF) co-ordinates.
    In the case of local-XYZ co-ords the hour-angle :math:`H_0` supplied should be
    Local-hour-angle (relative to transit, :math:`LHA = LST - RA`).
    In the case of Global XYZ co-ords it should be Greenwich hour angle
    (:math:`GHA = GST - RA`, where `GST` is Greenwich Sidereal Time).

    NB `LST = GST + Longitude`, equivalently `LHA = GHA + Longitude`
    (with the standard convention of East-is-positive)

    Args:
        hour_angle (astropy.units.Quantity): Source hour-angle.
            Dimensions of angular width (arc).
        dec (astropy.units.Quantity): Source declination.
            Dimensions of angular width (arc).

    Returns:
        numpy.ndarray: Rotation matrix for converting from XYZ to UVW.
        [Dtype ``numpy.float_``, shape ``(3, 3)``]


    """
    har = hour_angle.to(u.rad).value  # HA-radians
    dr = declination.to(u.rad).value  # Dec-radians
    rotation = np.array(
        [[np.sin(har), np.cos(har), 0],
         [-np.sin(dr) * np.cos(har), np.sin(dr) * np.sin(har), np.cos(dr)],
         [np.cos(dr) * np.cos(har), -np.cos(dr) * np.sin(har), np.sin(dr)]],
        dtype=np.float_)
    return rotation


def z_rotation_matrix(rotation_angle):
    """
    Rotation matrix for a passive transformation counter-clockwise about Z-axis

    (i.e. rotation of co-ordinate system)

    Args:
        rotation_angle (astropy.units.Quantity): Rotation-angle.
            Dimensions of angular width (arc).

    Returns:
        numpy.ndarray: Rotation matrix
    """
    ar = rotation_angle.to(u.rad).value  # Angle in radians
    rotation = np.array(
        [[np.cos(ar), np.sin(ar), 0],
         [-np.sin(ar), np.cos(ar), 0],
         [0, 0, 1], ],
        dtype=np.float_)
    return rotation