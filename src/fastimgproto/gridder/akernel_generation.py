import numpy as np
from scipy.special import sph_harm
from scipy import ndimage


def generate_akernel(pbeam_coefs, fov, kernel_size, rot_angle=0.0):
    """
    Generate rotated A-kernel for A-projection using spherical harmonics and rotation angle provided.

    Args:
        pbeam_coefs (numpy.ndarray): Spherical harmonic coefficients that generate the primary beam.
        fov (float): Field od view, in radians
        kernel_size (int): Workspace size for kernel generation.
        rot_angle (float): Rotation angle (added to theta angle of spherical harmonics).

    Returns:
        numpy.ndarray: Rotated A-kernel
    """

    distance_radians = np.arange(-kernel_size/2, kernel_size/2) * fov / kernel_size
    x_rad, y_rad = np.meshgrid(distance_radians, distance_radians)

    # Compute phi and theta
    phi = np.sqrt(x_rad*x_rad + y_rad*y_rad)
    x_rad[x_rad == 0] = 1e-20
    theta = np.arctan(y_rad/x_rad)

    # Degree of spherical harmonics
    sh_degree = int((len(pbeam_coefs) - 1) / 2)

    # Create primary beam from spherical harmonics
    pbeam = np.zeros((kernel_size, kernel_size))
    for ord in range(-sh_degree, sh_degree + 1):
        if pbeam_coefs[sh_degree + ord] != 0.0:
            pbeam += pbeam_coefs[sh_degree + ord] * np.abs(sph_harm(ord, sh_degree, theta + rot_angle, phi).real)

    # Invert primary beam function (and normalize)
    akernel = 1.0 / pbeam

    return akernel


def generate_akernel_from_lha(pbeam_coefs, lha, dec_rad, ra_rad, fov, kernel_size):
    """
    Generate rotated A-kernel for A-projection using spherical harmonics and the specified LHA value.

    Args:
        pbeam_coefs (numpy.ndarray): Spherical harmonic coefficients that generate the primary beam.
        lha (float): Local hour angle of the object, in decimal hours (0,24)
        dec_rad (float): Declination of the observation pointing centre, in radians
        ra_rad (float): Right Ascension of the observation pointing centre, in radians
        fov (float): Field od view, in radians
        kernel_size (int): Workspace size for kernel generation.

    Returns:
        numpy.ndarray: Rotated A-kernel
    """

    # Determine parallactic angle (in radians)
    pangle = parallatic_angle(lha, dec_rad, ra_rad)

    akernel = generate_akernel(pbeam_coefs, fov, kernel_size, pangle)

    return akernel


def rotate_akernel_by_lha(akernel, lha, dec_rad, ra_rad):
    """
    Rotate input A-kernel matrix using interpolation and the specified LHA value.

    Args:
        akernel (numpy.ndarray): Primary beam matrix.
        lha (float): Local hour angle of the object, in decimal hours (0,24)
        dec_rad (float): Declination of the observation pointing centre, in radians
        ra_rad (float): Right Ascension of the observation pointing centre, in radians

    Returns:
        numpy.ndarray: Rotated A-kernel
    """

    # Determine parallactic angle (in radians)
    pangle = parallatic_angle(lha, dec_rad, ra_rad)

    akmin_r = akernel[0, 0]

    # Rotate (use order-1 spline interpolation)
    rot_akernel = ndimage.interpolation.rotate(akernel, np.rad2deg(pangle), reshape=False, mode='constant',
                                               cval=akmin_r, order=1)

    return rot_akernel


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
