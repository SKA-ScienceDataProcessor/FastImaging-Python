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
            baselines in UVW space [lambda]. (numpy shape: (n_uvw, 3))
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


def calculated_summed_vis(pointing_centre, source_list, uvw):
    """
    Generate noise-free visibilities from UVW baselines and point-sources.

    Args:
        pointing_centre (astropy.coordinates.SkyCoord): Pointing centre
        source_list: List of list of :class:`fastimgproto.skymodel.helpers.SkySource`
        uvw (numpy.array): UVW baselines (units of lambda).
            Numpy array shape: (n_baselines, 3)
    Returns (numpy.array):
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