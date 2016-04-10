#!/usr/bin/env python
from __future__ import print_function
from astropy.utils.data import download_file
from fastimgproto.skymodel.extraction import (
    sumss_file_to_dataframe,
    lsm_extract
)
import click
import sys
import csv


@click.command()
@click.argument('ra', type=click.FLOAT)
@click.argument('dec', type=click.FLOAT)
@click.argument('radius', type=click.FLOAT)
@click.argument('outfile', type=click.File(mode='w'), default='-')
@click.option('--catalog-file', type=click.Path(exists=True))
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
    - catalog_file (path): Path to SUMS catalog file (sumsscat.Mar-11-2008.gz).
        If unsupplied, will attempt to download / use cached version via the
        Astropy download cache.

    Outputs:
        An list of matching sources in GSM-format.

    """
    if catalog_file is None:
        catalog_file = download_file(
            'http://www.astrop.physics.usyd.edu.au/sumsscat/sumsscat.Mar-11-2008.gz',
            cache=True)

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
