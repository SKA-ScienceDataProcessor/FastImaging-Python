#!/usr/bin/env python
from __future__ import print_function

import json
import logging
import sys

import astropy.units as u
import click
import numpy as np
from tqdm import tqdm as Tqdm

import fastimgproto.imager as imager
from fastimgproto.gridder.conv_funcs import PSWF

from .config import ConfigKeys, default_config_path


@click.command()
@click.option(
    '-c', '--config_json', type=click.File(mode='r'), default=default_config_path,
    help="Path to config file. Default:'{}' ".format(default_config_path))
@click.argument('in_vis', type=click.File(mode='rb'))
@click.argument('out_img', type=click.File(mode='wb'))
def cli(config_json, in_vis, out_img):
    """
    Load uvw / vis data from npz, produce a dirty image , save in npz format.

    UVW data should be in units of wavelength multiples. (key 'uvw_lambda')

    Outputs an npz file containing dirty image and synthesized beam
    (keys 'image', 'beam').
    """
    logging.basicConfig(level=logging.DEBUG)
    config = json.load(config_json)
    imager_settings = config[ConfigKeys.imager_settings]
    cell_size = imager_settings[ConfigKeys.cell_size_arcsec] * u.arcsec
    image_size = imager_settings[ConfigKeys.image_size_pix] * u.pix

    npz_data_dict = np.load(in_vis)
    uvw_lambda = npz_data_dict['uvw_lambda']
    vis = npz_data_dict['vis']
    snr_weights = np.ones(vis.shape)

    # Will move this to a config option later
    kernel_support = 3
    kernel_func = PSWF(trunc=kernel_support)

    with Tqdm() as progress_bar:
        image, beam = imager.image_visibilities(vis, vis_weights=snr_weights,
                                                uvw_lambda=uvw_lambda,
                                                image_size=image_size,
                                                cell_size=cell_size,
                                                kernel_func=kernel_func,
                                                kernel_support=kernel_support,
                                                kernel_exact=True,
                                                kernel_oversampling=None,
                                                gridding_correction=False,
                                                analytic_gcf=False,
                                                progress_bar=progress_bar)

    np.savez(out_img, image=image, beam=beam)
    sys.exit(0)


if __name__ == '__main__':
    cli()
