#!/usr/bin/env python
from __future__ import print_function
import click
import logging
import sys
import json
import astropy.units as u
import numpy as np
import fastimgproto.imager as imager
from fastimgproto.gridder.conv_funcs import GaussianSinc


class ConfigKeys:
    """
    Define the string literals to be used as keys in the JSON config file.
    """
    image_size_pix = 'image_size_pix'
    cell_size_arcsec = 'cell_size_arcsec'



@click.command()
@click.argument('config_json', type=click.File(mode='r'))
@click.argument('in_vis', type=click.File(mode='rb'))
@click.argument('out_img', type=click.File(mode='wb'))
def cli(config_json, in_vis, out_img):
    logging.basicConfig(level=logging.DEBUG)
    """
    Load uvw / vis data from npz, produce a dirty image , save in npz format.

    UVW data should be in units of wavelength multiples. (key 'uvw_lambda')

    Outputs an npz file containing dirty image and synthesized beam
    (keys 'image', 'beam').
    """
    config = json.load(config_json)
    cell_size = config[ConfigKeys.cell_size_arcsec] * u.arcsec
    image_size = config[ConfigKeys.image_size_pix] * u.pix

    npz_data_dict = np.load(in_vis)
    uvw_lambda = npz_data_dict['uvw_lambda']
    vis = npz_data_dict['vis']

    # Will move this to a config option later
    kernel_support = 3
    kernel_func = GaussianSinc(trunc=kernel_support)
    image, beam = imager.image_visibilities(vis, uvw_lambda,
                              image_size=image_size, cell_size=cell_size,
                              kernel_func=kernel_func,
                              kernel_support=kernel_support,
                              kernel_oversampling=None)

    np.savez(out_img, image=image, beam=beam)
    sys.exit(0)


if __name__ == '__main__':
    cli()
