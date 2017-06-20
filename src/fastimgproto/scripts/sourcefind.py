"""
Run sourcefinding on an image
"""

import click
import numpy as np

from fastimgproto.sourcefind.image import IslandParams, SourceFindImage


def main(image_path, detection_n_sigma, analysis_n_sigma):
    # hdu0 = fits.open(image_path)[0]
    # imgdata = hdu0.data.squeeze()
    with open(image_path, 'rb') as f:
        data_dict = np.load(f)
        imgdata = data_dict['image']
    sfimage = SourceFindImage(data=imgdata,
                              detection_n_sigma=detection_n_sigma,
                              analysis_n_sigma=analysis_n_sigma,
                              )
    return sfimage


@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('-d', '--detection', type=click.FLOAT, default=25.,
              help="Detection threshold (multiple of RMS above background)")
@click.option('-a', '--analysis', type=click.FLOAT, default=15.,
              help="Analysis threshold (multiple of RMS above background)")
def cli(image_path, detection, analysis):
    sfimage = main(image_path, detection, analysis)
    click.echo("Found the following sources:")
    for src in sfimage.islands:
        assert isinstance(src, IslandParams)
        click.echo("({},{}):\t{}".format(src.xbar, src.ybar, src.extremum_val))
