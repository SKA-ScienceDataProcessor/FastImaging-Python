import math
import numpy as np
import astropy.units as u
from fastimgproto.skymodel.helpers import SkySource


def l_cosine(ra, dec, ra0):
    """
    Convert a coordinate in RA,Dec into a direction cosine l (RA-direction)

    Args:
        ra,dec: Source location [rad]
        ra0: RA centre of the field [rad]

    Returns:
        l: Direction cosine

    """

    return (math.cos(dec) * math.sin(ra - ra0))


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


def visibilities_for_point_source(dist_uvw, l, m, flux):
    """
    Simulate visibilities for point source.

    Calculate visibilities for a source located at
    angular position (l,m) relative to observed phase centre
    as used for calculating baselines in UVW space.

    Note that point source is delta function, therefore
    FT relationship becomes an exponential, evaluated at
    (uvw.lmn)

    Args:
        dist_uvw (numpy.array): Array of 3-vectors representing
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

    return flux * src_n * np.exp(-2j * np.pi * np.dot(dist_uvw, src_offset))


def visibilities_for_source_list(pointing_centre, source_list, uvw):
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
        sp = src.position
        l = l_cosine(sp.ra.rad, sp.dec.rad, pointing_centre.ra.rad)
        m = m_cosine(sp.ra.rad, sp.dec.rad, pointing_centre.ra.rad,
                     pointing_centre.dec.rad)

        vis = visibilities_for_point_source(uvw, l, m,
                                            flux=src.flux.to(u.Jy).value)
        sumvis += vis

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
    noise = rstate.normal(loc=0, scale=sigma, size=(len(vis),2))
    return vis + (noise[:,0] + 1j*noise[:,1])

