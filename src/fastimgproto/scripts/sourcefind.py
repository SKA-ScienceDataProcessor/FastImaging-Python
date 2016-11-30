"""
Run sourcefinding on a FITS image
"""

import click
from astropy.io import fits
from fastimgproto.sourcefind.image import SourceFindImage, IslandParams

# from fastimgproto.sourcefind import extract_sources


def main(fits_image_path, detection_n_sigma, analysis_n_sigma):
    hdu0 = fits.open(fits_image_path)[0]
    imgdata = hdu0.data.squeeze()
    sfimage = SourceFindImage(data=imgdata,
                              detection_n_sigma=detection_n_sigma,
                              analysis_n_sigma=analysis_n_sigma,
                              )
    return sfimage


@click.command()
@click.argument('fitsfile', type=click.Path(exists=True))
@click.option('-d', '--detection', type=click.FLOAT, default=5.,
              help="Detection threshold (multiple of RMS above background)")
@click.option('-a', '--analysis', type=click.FLOAT, default=3.,
              help="Analysis threshold (multiple of RMS above background)")
def cli(fitsfile, detection, analysis):
    sfimage = main(fitsfile, detection, analysis)
    click.echo("Found the following sources:")
    for src in sfimage.islands:
        assert isinstance(src, IslandParams)
        click.echo("({},{}):\t{}".format(src.xbar, src.ybar, src.extrema_val))
