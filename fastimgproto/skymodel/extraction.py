"""
Local sky model extraction code.

Currently SUMSS specific, to be refactored in future as required.
"""
from __future__ import print_function
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
import math
import numpy as np
import pandas
from collections import OrderedDict
from fastimgproto.skymodel.helpers import PositionError, SkySource


# http://www.astrop.physics.usyd.edu.au/sumsscat/description.html
sumss_colnames = (
    'ra_h', 'ra_m', 'ra_s',
    'dec_d', 'dec_m', 'dec_s',
    'ra_err_arcsec',
    'dec_err_arcsec',
    'peak_flux_mjy', 'peak_flux_err_mjy',
    'int_flux_mjy', 'int_flux_err_mjy',
    'beam_major_axis_arcseconds',
    'beam_minor_axis_arcseconds',
    'beam_angle_deg',
    'deconv_major_axis_arcseconds',
    'deconv_minor_axis_arcseconds',
    'deconv_angle_deg',
    'mosaic_name',
    'n_mosaics_src_present',
    'pixel_x',
    'pixel_y',
)


class SumssSrc(SkySource):
    """
    Represents an entry from the SUMSS catalog.

    (Sydney University Molonglo Southern Sky Survey)
    """

    def __init__(self, row):
        position = SkyCoord(
            (row.ra_h, row.ra_m, row.ra_s),
            (row.dec_d, row.dec_m, row.dec_s),
            unit=(u.hourangle, u.deg),
        )
        self.peak_flux = row.peak_flux_mjy * u.mJy
        self.peak_flux_err = row.peak_flux_err_mjy * u.mJy
        self.variable = row.variable
        self.position_err = PositionError(
            ra_err=Angle(row.ra_err_arcsec * u.arcsecond),
            dec_err=Angle(row.dec_err_arcsec * u.arcsecond)
        )

        super(SumssSrc, self).__init__(
            position=position, flux=self.peak_flux, variable=self.variable)


    def to_ordereddict(self):
        od = OrderedDict()
        od['ra'] = self.position.ra.deg
        od['dec'] = self.position.dec.deg
        od['ra_err'] = self.position_err.ra.deg
        od['dec_err'] = self.position_err.dec.deg
        od['peak_flux_mjy'] = self.peak_flux.to(u.mJy).value
        od['peak_flux_err_mjy'] = self.peak_flux_err.to(u.mJy).value
        od['variable'] = self.variable
        return od


def sumss_file_to_dataframe(catalog_file):
    df = pandas.read_csv(
        catalog_file, sep='\s+', header=None, names=sumss_colnames,
        compression='gzip',
    )
    return df


def alpha_factor(dec_deg, radius_deg):
    """
    Convert angular separation to delta-RA (dec-dependent 'inflated radius').

    cf
    http://research.microsoft.com/pubs/64524/tr-2006-52.pdf
    """
    if math.fabs(dec_deg) + radius_deg>89.9:
        return 180.

    alpha = math.degrees(math.fabs(math.atan(
        math.sin(math.radians(radius_deg)) /
        math.sqrt(
            math.fabs(
                math.cos(math.radians(dec_deg-radius_deg))*
                math.cos(math.radians(dec_deg+radius_deg))
            )
        )
    )))
    return alpha

def lsm_extract(skyregion, full_catalog,
                variable_portion=0.1, seed=42):
    """
    Extract a local sky model from a given catalog

    Args:
    - skyregion (:class:`.SkyRegion`): Area of sky to extract catalog for
    - full_catalog (DataFrame): pandas.DataFrame loaded from the catalog.
    - variable_portion (float): Proportion of sources to be randomly assigned
        'variable=True'.
    - seed: Seed to use for random state generator. We set a default,
        static value for this since we generally want consistent variability
        results from one run to the next.

    Returns:
        matches: A list of :class:`SumssSrc` in the region.

    """
    ra_centre = skyregion.centre.ra.deg
    dec_centre = skyregion.centre.dec.deg
    radius_deg = skyregion.radius.deg
    # Shouldn't be possible if using SkyCoord, but leave sanity-check in anyway:
    if not (ra_centre>=0. and ra_centre<360):
        raise ValueError("Please use central RA in range [0,360).")

    full_cat = full_catalog

    rstate = np.random.RandomState(42)
    random_floats =  rstate.rand(len(full_cat))
    variable_col = random_floats < variable_portion
    full_cat['variable'] = variable_col

    dec_floor = math.floor(dec_centre - radius_deg)
    dec_ceil = math.ceil(dec_centre + radius_deg)

    # We filter on declination. Easy since there's no wraparound,
    # and we don't even need to convert from DMS, we just filter at the degree
    # level.
    dec_filtered = pandas.DataFrame(full_cat[(full_cat.dec_d >= dec_floor) &
                                    (full_cat.dec_d <= dec_ceil)])

    # OK, now we calculate RA in decimal degress for our Dec-filtered sources:
    dec_filtered['ra_degrees'] = 15.*(dec_filtered['ra_h']
                                      + dec_filtered['ra_m']/60.
                                      + dec_filtered['ra_s']/3600)

    # Calculate RA limits taking into account longitude convergence near poles:
    alpha = alpha_factor(dec_centre, radius_deg)
    ra_floor = ra_centre - alpha
    ra_ceil = ra_centre + alpha


    if ra_floor > 0 and ra_ceil < 360:
        #simplecase
        ra_filtered = dec_filtered[
            (dec_filtered.ra_degrees > ra_floor) &
            (dec_filtered.ra_degrees < ra_ceil)
        ]
    elif ra_floor<0 and ra_ceil<360:
        # low end wraparound
        ra_floor += 360.
        ra_filtered = dec_filtered[
            (dec_filtered.ra_degrees > ra_floor) |
            (dec_filtered.ra_degrees < ra_ceil)
        ]
    elif ra_floor > 0 and ra_ceil > 360:
        # high end wraparound
        ra_ceil -= 360.
        ra_filtered = dec_filtered[
            (dec_filtered.ra_degrees > ra_floor) |
            (dec_filtered.ra_degrees < ra_ceil)
        ]
    else:
        # Extreme RA wraparound, possible near pole.
        # Don't filter any further:
        ra_filtered = dec_filtered


    # print(len(dec_filtered), 'source matched by declination.')
    matches = []
    centre = SkyCoord(ra_centre, dec_centre, unit='deg')
    for row in ra_filtered.itertuples():
        src = SumssSrc(row)
        sep_degrees = centre.separation(src.position).deg
        if sep_degrees < radius_deg:
            matches.append(src)
        # print(count)
    return matches