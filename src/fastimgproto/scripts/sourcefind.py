"""
Run sourcefinding on a FITS image
"""

import click
from astropy.io import fits
# from fastimgproto.sourcefind import extract_sources


def main(fits_image_path, detection_n_sigma, analysis_n_sigma):
    hdu0 = fits.open(fits_image_path)[0]
    imgdata = hdu0.data.squeeze()
    sources = extract_sources(imgdata, detection_n_sigma, analysis_n_sigma)
    return sources


@click.command()
@click.argument('fitsfile', type=click.Path(exists=True))
@click.option('-d', '--detection', type=click.FLOAT, default=5.,
              help="Detection threshold (multiple of RMS above background)")
@click.option('-a', '--analysis', type=click.FLOAT, default=3.,
              help="Analysis threshold (multiple of RMS above background)")
def cli(fitsfile, detection, analysis):
    sources = main(fitsfile, detection, analysis)
    click.echo("Found the following sources:")
    for s in sources:
        peak_pixel, peak_value = s
        y,x = peak_pixel
        click.echo("({},{}):\t{}".format(x,y,peak_value))
