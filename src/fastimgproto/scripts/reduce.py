"""
Simulated pipeline run
"""
from __future__ import print_function

import json
import logging

import astropy.units as u
import click
import numpy as np
from tqdm import tqdm as Tqdm

import fastimgproto.imager as imager
import fastimgproto.visibility as visibility
from fastimgproto.gridder.conv_funcs import GaussianSinc
from fastimgproto.sourcefind.image import SourceFindImage

from .config import ConfigKeys, default_config_path

logger = logging.getLogger()


# @click.argument(
#     'output_img', type=click.File(mode='wb'))
@click.command(context_settings=dict(max_content_width=120))
@click.option(
    '-c', '--config_json', type=click.File(mode='r'),
    default=default_config_path,
    help="Path to config file. Default: '{}' ".format(default_config_path))
@click.argument('input_npz', type=click.File(mode='rb'))
def cli(config_json, input_npz, ):
    """
    Load simulated data from INPUT_NPZ, and search for transients as follows:

    \b
     * Apply difference imaging (subtract model visibilities from data, apply
       synthesis-imaging).
     * Run sourcefinding on the resulting diff-image.
    """

    npz_content = np.load(input_npz)
    uvw_lambda = npz_content['uvw_lambda']
    model_vis = npz_content['model']
    data_vis = npz_content['vis']

    config = json.load(config_json)
    cell_size = config[ConfigKeys.cell_size_arcsec] * u.arcsec
    image_size = config[ConfigKeys.image_size_pix] * u.pix
    detection_n_sigma = config[ConfigKeys.sourcefind_detection]
    analysis_n_sigma = config[ConfigKeys.sourcefind_analysis]

    sfimage = main(
        uvw_lambda=uvw_lambda,
        model_vis=model_vis,
        data_vis=data_vis,
        image_size=image_size,
        cell_size=cell_size,
        detection_n_sigma=detection_n_sigma,
        analysis_n_sigma=analysis_n_sigma,
    )

    print("Found residual sources")
    for found_src in sfimage.islands:
        print(found_src)


def main(uvw_lambda,
         model_vis,
         data_vis,
         image_size,
         cell_size,
         detection_n_sigma,
         analysis_n_sigma,
         ):
    """
    Represents the difference-image + detect stages of the FastImaging pipeline.
    """

    # Subtract model-generated visibilities from incoming data
    residual_vis = data_vis - model_vis

    # Kernel generation - might configure this via config-file in future.
    kernel_support = 3
    kernel_func = GaussianSinc(trunc=kernel_support)
    image, beam = imager.image_visibilities(residual_vis, uvw_lambda,
                                            image_size=image_size,
                                            cell_size=cell_size,
                                            kernel_func=kernel_func,
                                            kernel_support=kernel_support,
                                            kernel_exact=True)

    sfimage = SourceFindImage(data=np.real(image),
                              detection_n_sigma=detection_n_sigma,
                              analysis_n_sigma=analysis_n_sigma,
                              )

    return sfimage
