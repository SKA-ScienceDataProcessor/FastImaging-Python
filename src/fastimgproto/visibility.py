import math

import astropy.units as u
import numpy as np

from fastimgproto.skymodel.helpers import SkySource
from fastimgproto.gridder.akernel_generation import parallatic_angle
from scipy.special import sph_harm


def l_cosine(ra, dec, ra0):
    """
    Convert a coordinate in RA,Dec into a direction cosine l (RA-direction)

    Args:
        ra,dec: Source location [rad]
        ra0: RA centre of the field [rad]

    Returns:
        l: Direction cosine

    """

    return math.cos(dec) * math.sin(ra - ra0)


def m_cosine(ra, dec, ra0, dec0):
    """
    Convert a coordinate in RA,Dec into a direction cosine m (Dec-direction)

    Args:
        ra,dec: Source location [rad]
        ra0,dec0: Centre of the field [rad]

    Returns:
        m: direction cosine
    """

    return ((math.sin(dec) * math.cos(dec0)) -
            (math.cos(dec) * math.sin(dec0) * math.cos(ra - ra0))
            )


def visibilities_for_point_source(uvw_baselines, l, m, flux):
    """
    Simulate visibilities for point source.

    Calculate visibilities for a source located at
    angular position (l,m) relative to observed phase centre
    as used for calculating baselines in UVW space.

    Note that point source is delta function, therefore
    FT relationship becomes an exponential, evaluated at
    (uvw.lmn)

    Args:
        uvw_baselines (numpy.array): Array of 3-vectors representing
            baselines in UVW space. Implicit units are
            (dimensionless) multiples of wavelength, lambda.
            [numpy shape: (n_uvw, 3), dtype=np.float_]
        l,m (float): Direction cosines (in RA, Dec directions)
        flux (float): Flux [Jy]

    Returns:
        vis (numpy.array): Array of complex visibilities
            (numpy shape: (n_uvw,))
    """
    # For background, see section 4.2 of
    # https://github.com/griffinfoster/fundamentals_of_interferometry/

    # Component of source vector along n-axis / w-axis
    # (Doubles as flux attenuation factor due to projection effects)
    src_n = np.sqrt(1 - l ** 2 - m ** 2)

    # src vec = [l,m, src_w]
    # phase centre vec = [0,0,1]
    # src - centre:
    src_offset = -np.array([l, m, src_n - 1])

    return flux * src_n * np.exp(-2j * np.pi * np.dot(uvw_baselines, src_offset))


def visibilities_for_point_source_and_pbeam(uvw_baselines, uvw_parangles, l, m, flux, pbeam_coefs):
    """
    Simulate visibilities for point source and primary beam.

    Calculate visibilities for a source located at
    angular position (l,m) relative to observed phase centre
    as used for calculating baselines in UVW space.

    Note that point source is delta function, therefore
    FT relationship becomes an exponential, evaluated at
    (uvw.lmn)

    Args:
        uvw_baselines (numpy.array): Array of 3-vectors representing
            baselines in UVW space. Implicit units are
            (dimensionless) multiples of wavelength, lambda.
            [numpy shape: (n_uvw, 3), dtype=np.float_]
        uvw_parangles (numpy.ndarray): Parallatic angles for the rotation of primary beam.
        l,m (float): Direction cosines (in RA, Dec directions)
        flux (float): Flux [Jy]
        pbeam_coefs (numpy.ndarray): Primary beam given by spherical harmonics coefficients.
            The SH degree is constant being derived from the number of coefficients minus one.

    Returns:
        vis (numpy.array): Array of complex visibilities
            (numpy shape: (n_uvw,))
    """
    # For background, see section 4.2 of
    # https://github.com/griffinfoster/fundamentals_of_interferometry/

    # Component of source vector along n-axis / w-axis
    # (Doubles as flux attenuation factor due to projection effects)
    src_n = np.sqrt(1 - l ** 2 - m ** 2)

    # src vec = [l,m, src_w]
    # phase centre vec = [0,0,1]
    # src - centre:
    src_offset = -np.array([l, m, src_n - 1])

    # Degree of spherical harmonics
    sh_degree = len(pbeam_coefs) - 1

    # Derive phi and theta of the source
    phi = np.arccos(src_n)
    if np.abs(src_n) < 1:
        theta = np.arctan(m/l)
    else:
        theta = 0.0

    # Find maximum value of primary beam - for normalisation
    pbeam_max = 0.0
    for ord in range(0, len(pbeam_coefs)):
        if pbeam_coefs[ord] != 0.0:
            pbeam_max += pbeam_coefs[ord] * np.abs(sph_harm(ord, sh_degree, 0.0, 0.0).real)

    pbeam_flux = np.empty_like(uvw_parangles)
    for idx, pa in enumerate(uvw_parangles):
        pbeam_value = 0.0
        for ord in range(0, len(pbeam_coefs)):
            if pbeam_coefs[ord] != 0.0:
                pbeam_value += pbeam_coefs[ord] * np.abs(sph_harm(ord, sh_degree, theta + pa, phi).real)
        # Normalise and append
        pbeam_flux[idx] = pbeam_value / pbeam_max

    return pbeam_flux * flux * src_n * np.exp(-2j * np.pi * np.dot(uvw_baselines, src_offset))


def calculate_direction_cosines(pointing_centre, source):
    """
    Calculate direction-cosine coefficients for the given source

    Args:
        pointing_centre (astropy.coordinates.SkyCoord): Field pointing centre.
        source (`fastimgproto.skymodel.helpers.SkySource`): Source.

    Returns:
        tuple: (l,m) cosine values.
    """
    sp = source.position
    l = l_cosine(sp.ra.rad, sp.dec.rad, pointing_centre.ra.rad)
    m = m_cosine(sp.ra.rad, sp.dec.rad, pointing_centre.ra.rad,
                 pointing_centre.dec.rad)
    return l, m


def visibilities_for_source_list(pointing_centre, source_list, uvw,
                                 progress_updater=None):
    """
    Generate noise-free visibilities from UVW baselines and point-sources.

    Args:
        pointing_centre (astropy.coordinates.SkyCoord): Pointing centre
        source_list: List of list of :class:`fastimgproto.skymodel.helpers.SkySource`
        uvw (numpy.ndarray): UVW baselines (units of lambda).
            Numpy array shape: (n_baselines, 3)
    Returns (numpy.ndarray):
        Complex visbilities sum for each baseline.
            Numpy array shape: (n_baselines,)
    """

    # Sum visibilities from all sources
    sumvis = np.zeros(len(uvw), dtype=np.dtype(complex))

    for src in source_list:
        assert isinstance(src, SkySource)
        l, m = calculate_direction_cosines(pointing_centre, src)
        sumvis += visibilities_for_point_source(uvw, l, m,
                                                flux=src.flux.to(u.Jy).value)

    return sumvis


def visibilities_for_source_list_and_pbeam(pointing_centre, source_list, uvw,
                                            lha, pbeam_coefs, progress_updater=None):
    """
    Generate noise-free visibilities from UVW baselines, point-sources and primary beam.

    Args:
        pointing_centre (astropy.coordinates.SkyCoord): Pointing centre
        source_list: List of list of :class:`fastimgproto.skymodel.helpers.SkySource`
        uvw (numpy.ndarray): UVW baselines (units of lambda).
            Numpy array shape: (n_baselines, 3)
        lha (numpy.ndarray): Local hour angle (implicit units are hours).
        pbeam_coefs (numpy.ndarray): Primary beam given by spherical harmonics coefficients.
            The SH degree is constant being derived from the number of coefficients minus one.
    Returns (numpy.ndarray):
        Complex visbilities sum for each baseline.
            Numpy array shape: (n_baselines,)
    """

    # Sum visibilities from all sources
    sumvis = np.zeros(len(uvw), dtype=np.dtype(complex))
    uvw_parangles = np.empty_like(lha)

    # Determine parallatic angles for primary beam rotation
    for idx, ha in enumerate(lha):
        uvw_parangles[idx] = parallatic_angle(ha, pointing_centre.dec.value, pointing_centre.ra.value)

    for src in source_list:
        assert isinstance(src, SkySource)
        l, m = calculate_direction_cosines(pointing_centre, src)
        sumvis += visibilities_for_point_source_and_pbeam(uvw, uvw_parangles, l, m,
                                                           src.flux.to(u.Jy).value, pbeam_coefs)

    return sumvis


def add_gaussian_noise(noise_level, vis, seed=None):
    """
    Add random Gaussian distributed noise to the visibilities.

    Adds jointly-normal (i.e. independent) Gaussian noise to both the real
    and imaginary components of the visibilities.

    Args:
        noise_level (astropy.units.Quantity): Noise level, in units equivalent
            to Jansky. This defines the std. dev. / sigma of the Gaussian
            distribution.
        vis (numpy.ndarray): The array of (noise-free) complex visibilities.
            (Does not alter - assign the returned array if you wish to replace.)
        seed (int): Optional -  can be used to seed the random number generator
            to ensure reproducible results.
    Returns (numpy.ndarray):
        Visibilities with complex Gaussian noise added.

    """
    sigma = noise_level.to(u.Jansky).value
    rstate = np.random.RandomState(seed)
    noise = rstate.normal(loc=0, scale=sigma, size=(len(vis), 2))
    return vis + (noise[:, 0] + 1j * noise[:, 1])
