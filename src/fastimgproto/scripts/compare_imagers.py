#!/usr/bin/env python
from __future__ import print_function

import logging
import os
import sys
import numpy as np

import astropy.units as u
import click
import fastimgproto.casa.io as casa_io
import fastimgproto.casa.reduction as casa_reduce
import fastimgproto.imager as imager
from fastimgproto.gridder.conv_funcs import GaussianSinc
from fastimgproto.pipeline.io import save_as_fits



@click.command()
@click.argument('casavis', type=click.Path(exists=True))
@click.argument('outdir', type=click.Path())
def cli(casavis, outdir):
    logging.basicConfig(level=logging.DEBUG)
    """
    Create a dirty image of a CASA measurementset, using CASA & fastimgproto
    """

    image_size = 1024 * u.pixel
    cell_size = 3 * u.arcsecond
    # dirty_fits_path, clean_fits_path = casa_reduce.make_image_map_fits(
    #     casavis, outdir, image_size, cell_size)

    vis = casa_io.get_stokes_i_vis(casavis)
    uvw_lambda = casa_io.get_uvw_in_lambda(casavis)
    kernel_support = 3
    kernel_func = GaussianSinc(trunc=kernel_support)
    image, beam = imager.image_visibilities(vis, uvw_lambda,
                                            image_size=image_size,
                                            cell_size=cell_size,
                                            kernel_func=kernel_func,
                                            kernel_support=kernel_support,
                                            kernel_oversampling=None)

    fits_path = os.path.join(outdir, 'pyimage.fits')
    save_as_fits(np.real(image), fits_path=fits_path)
    sys.exit(0)


if __name__ == '__main__':
    cli()
