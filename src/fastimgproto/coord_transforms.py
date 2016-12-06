import numpy as np
import astropy.units as u


def local_xyz_to_uvw(local_xyz, local_ha, dec):
    """
    Convert between local XYZ co-ordinates and UVW co-ordinates

    ... for a given source position.

    For an explanation and derivation, see Notebook 4.1, Section 4.1.2
    of https://github.com/griffinfoster/fundamentals_of_interferometry
    ("The ($u$,$v$,$w$) Space")

    The definition of UVW used here is that
    - the $u$-axis lies in the celestial equatorial plane, and points toward the
      hour angle $H_0-6^\text{h}$.
    - the $v$-axis lies in the plane of the great circle with hour angle $H_0$,
      and points toward the declination $\frac{\pi}{2}-\delta_0$.
    - the $w$-axis points in the direction of the source.


    Args:
        local_xyz (numpy.ndarray): Position co-ordinates, in metres, in the
            local XYZ frame. Should either be single position vector
            [1-d array of shape ``(3,)``], or a horizontal stack of correctly
            oriented vectors [2-d array of shape ``(3, n_positions)``].
            [Dtype: ``numpy.float_``].
        local_ha (astropy.units.Quantity): Source local-hour-angle.
            ($local_ha = LST - RA$). Dimensions of angular width (arc).
        dec (astropy.units.Quantity): Source declination.
            Dimensions of angular width (arc).

    Returns:
        numpy.ndarray: Baseline co-ordinates, in metres, in UVW with w-axis
        towards the source.
        [Dtype ``numpy.float_``, shape ``(3, n_positions)`` or just ``(3,)``
        if a 1-d array (single position-vector) was supplied.]
    """
    # Shorter aliases, ensure radians for angles:
    assert local_xyz.shape[0] == 3
    lhar = local_ha.to(u.rad).value #LHA-radians
    dr = dec.to(u.rad).value #Dec-radians
    rotation = np.array(
        [
            [np.sin(lhar), np.cos(lhar), 0],
            [-np.sin(dr) * np.cos(lhar), np.sin(dr) * np.sin(lhar), np.cos(dr)],
            [np.cos(dr) * np.cos(lhar), -np.cos(dr) * np.sin(lhar), np.sin(dr)]
        ])
    return np.dot(rotation, local_xyz)