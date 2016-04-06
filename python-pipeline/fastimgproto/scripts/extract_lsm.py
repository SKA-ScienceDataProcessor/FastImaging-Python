#!/usr/bin/env python
from __future__ import print_function
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
import math
import click
import sys
import pandas
import csv
import numpy as np
from collections import OrderedDict


class PositionError():
    def __init__(self, ra_err, dec_err):
        self.ra = ra_err
        self.dec = dec_err

    def __str__(self):
        return "<PositionError: (ra, dec) in deg ({}, {})>".format(
            self.ra.deg, self.dec.deg)


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


class SumssSrc():
    """
    Represents an entry from the SUMSS catalog.

    (Sydney University Molonglo Southern Sky Survey)
    """

    def __init__(self, row):
        self.position = SkyCoord(
            (row.ra_h, row.ra_m, row.ra_s),
            (row.dec_d, row.dec_m, row.dec_s),
            unit=(u.hourangle, u.deg),
        )

        self.position_err = PositionError(
            ra_err=Angle(row.ra_err_arcsec * u.arcsecond),
            dec_err=Angle(row.dec_err_arcsec * u.arcsecond)
        )
        self.peak_flux = row.peak_flux_mjy * u.mJy
        self.peak_flux_err = row.peak_flux_err_mjy * u.mJy
        self.variable_source = False

    def to_ordereddict(self):
        od = OrderedDict()
        od['ra'] = self.position.ra.deg
        od['dec'] = self.position.dec.deg
        od['ra_err'] = self.position_err.ra.deg
        od['dec_err'] = self.position_err.dec.deg
        od['peak_flux'] = self.peak_flux.to(u.mJy).value
        od['peak_flux_err'] = self.peak_flux_err.to(u.mJy).value
        od['variable'] = self.variable_source
        return od


def sumss_file_to_dataframe(catalog_file):
    df = pandas.read_csv(
        catalog_file, sep='\s+', header=None, names=sumss_colnames)
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

def lsm_extract(ra_centre, dec_centre, radius, full_catalog):
    """
    Args:
    - ra_centre (float): RA of centre (decimal degrees, J2000)
    - dec_centre (float): Dec of centre (decimal degrees, J2000)
    - radius (float): Cone-radius (decimal degrees)
    - full_catalog (DataFrame): pandas.DataFrame loaded from the catalog.
    """
    full_cat = full_catalog
    dec_floor = math.floor(dec_centre - radius)
    dec_ceil = math.ceil(dec_centre + radius)

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
    alpha = alpha_factor(dec_centre, radius)
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
        if sep_degrees < radius:
            matches.append(src)
        # print(count)
    return matches


@click.command()
@click.argument('ra', type=click.FLOAT)
@click.argument('dec', type=click.FLOAT)
@click.argument('radius', type=click.FLOAT)
@click.argument('outfile', type=click.File(mode='w'), default='-')
@click.option('--catalog-file', type=click.Path(exists=True),
              default='./sumsscat.Mar-11-2008.gz')
def cli(ra, dec, radius, outfile, catalog_file):
    """
    Extracts sources from the catalog within a circular region

    \b
    Example usage:
    fastimg_extract_lsm -- 189.2 -45.6 1.5 lsm.csv

    \b
    Args:
    - ra (float): RA of centre (decimal degrees, J2000)
    - dec (float): Dec of centre (decimal degrees, J2000)
    - radius (float): Cone-radius (decimal degrees)
    - outfile (filepath): Filename to write filtered catalog to.
        (Default is stdout)


    \b
    Options:
    - catalog_file (path): Path to sumsscat. Default: ./sumsscat.Mar-11-2008.gz

    Outputs:
        An list of matching sources in GSM-format.

    """
    if not (ra>=0. and ra<360):
        raise ValueError("Please use central RA in range [0,360).")

    full_cat = sumss_file_to_dataframe(catalog_file)
    lsm_cat = lsm_extract(ra, dec, radius, full_cat)
    if lsm_cat:
        fieldnames = lsm_cat[0].to_ordereddict().keys()
        dw = csv.DictWriter(outfile,
                            fieldnames=fieldnames,
                            delimiter='\t')
        dw.writeheader()
        for src in lsm_cat:
            dw.writerow(src.to_ordereddict())
    click.echo("{} sources matched".format(len(lsm_cat)))
    sys.exit(0)


if __name__ == '__main__':
    cli()
