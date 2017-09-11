import astropy.units as u
import numpy as np


def time_of_next_transit(observer_longitude, target_ra, start_time,
                         tolerance=0.02 * u.arcsec):
    """
    Calculate time, ``t`` when the local-hour-angle of target will be 0.

    (Such that ``start_time <= t < start_time + 24*u.hr``.)

    NB: This gives no guarantees about visibility / horizons, as we are ignoring
    observer-declination / target-declination here. Just provides a useful way
    of setting up an observation at a good hour-angle, for a given
    time-period.

    Args:
        observer_longitude (astropy.coordinates.Longitude): Longitude
            of position on Earth
        target_ra (astropy.coordinates.Longitude): Right ascension of
            sky-target.
        start_time (astropy.time.Time): Time to start from.
        tolerance (astropy.units.Quantity): If target's LHA is within
            ``tolerance`` of zero at ``start_time``, simply return
            ``start_time``. Otherwise look for the next transit.
            Dimensions of angular width (arc).
    Returns:
        astropy.time.Time: Approximate time of the next transit
            (good to around 0.02 arcsec)
    """
    # Note, we get better looking precision we use 'mean' sidereal time,
    # i.e. the ``t.sidereal_time`` for returned ``t`` will be closer to 0.
    # However, since this ignores nutation, it will actually be a less accurate
    # value.
    lst = start_time.sidereal_time('apparent', longitude=observer_longitude)
    target_lha = lst - target_ra
    if np.fabs(target_lha) < tolerance:
        return start_time

    # Time to transit, in units of seconds:
    # (Recall,  ``LHA=-6h`` -> target is about to rise.
    time_to_transit = -1. * u.sday * (target_lha / (24. * u.hourangle))
    if time_to_transit < 0:
        time_to_transit = time_to_transit + 1. * u.sday
    return start_time + time_to_transit


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


def rotate_basis(input, rotation_angle):
    """
    Rotate the basis-vectors of a matrix by rotation_angle.

    Args:
        input (numpy.ndarray): 2-D Matrix to rotate
        rotation_angle (astropy.units.Quantity): Rotation-angle.

    Returns:
        numpy.ndarray: Matrix after change-of-basis
    """
    ar = rotation_angle.to(u.rad).value  # Angle in radians
    c = np.cos(ar)
    s = np.sin(ar)
    rotation = np.array([[c, s],
                         [-s, c]])
    inv_rotation = np.array([[c, -s],
                             [s, c]])
    return np.dot(rotation, np.dot(input, inv_rotation))
