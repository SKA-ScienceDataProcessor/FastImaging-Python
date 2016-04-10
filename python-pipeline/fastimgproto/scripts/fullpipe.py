import click
from astropy.utils.data import download_file
from fastimgproto.skymodel.extraction import (
    sumss_file_to_dataframe, lsm_extract
)


def main():
    ra_centre = 180
    dec_centre = -45.
    radius = 0.2

    catalog_file = download_file(
        'http://www.astrop.physics.usyd.edu.au/sumsscat/sumsscat.Mar-11-2008.gz',
        cache=True)
    full_cat = sumss_file_to_dataframe(catalog_file)
    lsm_cat = lsm_extract(ra_centre, dec_centre, radius, full_cat)

    # Load UVW, simulate visibilities



@click.command()
def cli():
    main()
