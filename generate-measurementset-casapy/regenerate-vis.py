import math
import numpy as np
from astropy.coordinates import SkyCoord

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


def genvis(dist_uvw, l, m, flux):
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
    src_offset = np.array([l, m, src_n - 1])

    return flux * src_n * np.exp(-2j * np.pi * np.dot(dist_uvw, src_offset))

uvw = np.loadtxt('uvw-vla-sim.txt')

pointing_centre = SkyCoord('12h00m00s 34d00m00s')

src1 = pointing_centre
src2 = SkyCoord(pointing_centre.ra.deg + 0.1, pointing_centre.dec.deg + 0.1,
                unit=('deg', 'deg'))


sumvis = np.zeros(len(uvw), dtype=np.dtype(complex))

src_fluxes = [1.0, 0.25]

for idx, s in enumerate((src1,src2)):
    l = l_cosine(s.ra.rad, s.dec.rad, pointing_centre.ra.rad)
    m = m_cosine(s.ra.rad, s.dec.rad, pointing_centre.ra.rad,
              pointing_centre.dec.rad)

    vis = genvis(uvw, l, m, flux=src_fluxes[idx])
    sumvis += vis

np.savetxt('data-vla-resim.txt', sumvis.view(float).reshape(-1,2))