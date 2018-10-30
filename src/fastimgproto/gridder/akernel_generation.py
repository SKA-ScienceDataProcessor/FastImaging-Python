import numpy as np
from scipy.special import sph_harm
from scipy import ndimage


def generate_primary_beam(pbeam_coefs, fov, workarea_size, rot_angle=0.0):
    """
    Generate rotated primary beam for A-projection using spherical harmonics and rotation angle provided.

    Args:
        pbeam_coefs (numpy.ndarray): Spherical harmonic coefficients that generate the primary beam.
        fov (float): Field od view, in radians
        workarea_size (int): Workarea size for kernel generation.
        rot_angle (float): Rotation angle (added to theta angle of spherical harmonics).

    Returns:
        numpy.ndarray: Rotated primary beam
    """

    distance_radians = np.linspace(-(fov/2), (fov/2), workarea_size)
    x_rad, y_rad = np.meshgrid(distance_radians, distance_radians)

    # Compute phi and theta
    phi = np.sqrt(x_rad*x_rad + y_rad*y_rad)
    x_rad[x_rad == 0] = 1e-20
    theta = np.arctan(y_rad/x_rad)

    # Degree of spherical harmonics
    sh_degree = len(pbeam_coefs) - 1

    # Create primary beam from spherical harmonics
    pbeam = np.zeros((workarea_size, workarea_size))
    for ord in range(0, len(pbeam_coefs)):
        if pbeam_coefs[ord] != 0.0:
            pbeam += pbeam_coefs[ord] * np.abs(sph_harm(ord, sh_degree, theta + rot_angle, phi).real)

    # Invert primary beam function (and normalize)
    pbeam = np.max(np.max(pbeam)) / pbeam

    return pbeam


def generate_primary_beam_for_lha(pbeam_coefs, lha, dec_rad, ra_rad, fov, workarea_size):
    """
    Generate rotated primary beam for A-projection using spherical harmonics and the specified LHA value.

    Args:
        pbeam_coefs (numpy.ndarray): Spherical harmonic coefficients that generate the primary beam.
        lha (float): Local hour angle of the object, in decimal hours (0,24)
        dec_rad (float): Declination of the observation pointing centre, in radians
        ra_rad (float): Right Ascension of the observation pointing centre, in radians
        fov (float): Field od view, in radians
        workarea_size (int): Workarea size for kernel generation.

    Returns:
        numpy.ndarray: Rotated primary beam
    """

    # Determine parallactic angle (in radians)
    pangle = parallatic_angle(lha, dec_rad, ra_rad)

    pbeam = generate_primary_beam(pbeam_coefs, fov, workarea_size, pangle)

    return pbeam


def rotate_primary_beam_for_lha(pbeam, lha, dec_rad, ra_rad):
    """
    Rotate input primary beam matrix using interpolation and the specified LHA value.

    Args:
        pbeam (numpy.ndarray): Primary beam matrix.
        lha (float): Local hour angle of the object, in decimal hours (0,24)
        dec_rad (float): Declination of the observation pointing centre, in radians
        ra_rad (float): Right Ascension of the observation pointing centre, in radians

    Returns:
        numpy.ndarray: Rotated primary beam
    """

    # Determine parallactic angle (in radians)
    pangle = parallatic_angle(lha, dec_rad, ra_rad)

    pbmin_r = pbeam[0, 0]

    # Rotate (use order-1 spline interpolation)
    a_kernel = ndimage.interpolation.rotate(pbeam, np.rad2deg(pangle), reshape=False, mode='constant', cval=pbmin_r, order=1)

    return a_kernel


def parallatic_angle(lha, dec_rad, ra_rad):
    """
    Compute parallatic angle for primary beam rotation.

    Args:
        lha (float): Local hour angle of the object, in decimal hours (0,24)
        dec_rad (float): Declination of the observation pointing centre, in radians
        ra_rad (float): The right ascension of the observation pointing centre, in radians

    Returns:
        float: parallactic angle in radians
    """
    ha_rad = np.deg2rad(lha * 15.)
    sin_eta_sin_z = np.cos(ra_rad) * np.sin(ha_rad)
    cos_eta_sin_z = np.sin(ra_rad) * np.cos(dec_rad) - np.cos(ra_rad) * np.sin(dec_rad) * np.cos(ha_rad)
    eta = np.arctan2(sin_eta_sin_z, cos_eta_sin_z)

    return eta
