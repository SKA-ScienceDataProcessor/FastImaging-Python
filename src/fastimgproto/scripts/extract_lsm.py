#!/usr/bin/env python
from __future__ import print_function
from astropy.utils.data import download_file
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from fastimgproto.skymodel.helpers import SkyRegion
from fastimgproto.skymodel.extraction import (
    sumss_file_to_dataframe,
    lsm_extract
)
import click
import sys
import csv

def write_catalog(src_list, filehandle):
    if src_list:
        fieldnames = src_list[0].to_ordereddict().keys()
        dw = csv.DictWriter(filehandle,
                            fieldnames=fieldnames,
                            delimiter='\t')
        dw.writeheader()
        for src in src_list:
            dw.writerow(src.to_ordereddict())

@click.command()
@click.argument('ra', type=click.FLOAT)
@click.argument('dec', type=click.FLOAT)
@click.argument('radius', type=click.FLOAT)
@click.argument('outfile', type=click.File(mode='w'), default='-')
@click.option('--vcat', type=click.File(mode='w'))
@click.option('--catalog-file', type=click.Path(exists=True))
def cli(ra, dec, radius, outfile, vcat, catalog_file):
    """
    Extracts sources from the catalog within a circular region

    \b
    Example usage:

        fastimg_extract_lsm -- 189.2 -45.6 1.5 lsm.csv

    or with separate variables catalog:

        fastimg_extract_lsm --vcat var.csv -- 189.2 -45.6 1.5 lsm.csv

    \b
    Args:
    - ra (float): RA of centre (decimal degrees, J2000)
    - dec (float): Dec of centre (decimal degrees, J2000)
    - radius (float): Cone-radius (decimal degrees)
    - outfile (filepath): Filename to write filtered catalog to.
        (Default is stdout)


    \b
    Options:
    - vcat (path): Path to output a separate catalog containing only
        variable sources.
    - catalog_file (path): Path to SUMS catalog file (sumsscat.Mar-11-2008.gz).
        If unsupplied, will attempt to download / use cached version via the
        Astropy download cache.

    Outputs:
        An list of matching sources in GSM-format.

    """

    field_of_view = SkyRegion(SkyCoord(ra*u.deg, dec*u.deg),
                              Angle(radius*u.deg))

    if catalog_file is None:
        catalog_file = download_file(
            'http://www.astrop.physics.usyd.edu.au/sumsscat/sumsscat.Mar-11-2008.gz',
            cache=True)

    full_cat = sumss_file_to_dataframe(catalog_file)
    lsm_cat = lsm_extract(field_of_view, full_cat)
    variable_cat = [s for s in lsm_cat if s.variable]

    write_catalog(lsm_cat, outfile)
    if vcat:
        write_catalog(variable_cat, vcat)

    click.echo("{} sources matched".format(len(lsm_cat)))
    click.echo("Of which {} variable.".format(len(variable_cat)))
    sys.exit(0)


if __name__ == '__main__':
    cli()
