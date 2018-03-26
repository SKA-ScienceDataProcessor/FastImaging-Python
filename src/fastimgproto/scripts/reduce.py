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
    logging.basicConfig(level=logging.DEBUG)

    npz_content = np.load(input_npz)
    uvw_lambda = npz_content['uvw_lambda']
    local_skymodel = npz_content['skymodel']
    data_vis = npz_content['vis']
    snr_weights = npz_content['snr_weights']

    config = json.load(config_json)
    cell_size = config[ConfigKeys.cell_size_arcsec] * u.arcsec
    image_size = config[ConfigKeys.image_size_pix] * u.pix
    detection_n_sigma = config[ConfigKeys.sourcefind_detection]
    analysis_n_sigma = config[ConfigKeys.sourcefind_analysis]

    sfimage = main(
        uvw_lambda=uvw_lambda,
        skymodel=local_skymodel,
        data_vis=data_vis,
        snr_weights=snr_weights,
        image_size=image_size,
        cell_size=cell_size,
        detection_n_sigma=detection_n_sigma,
        analysis_n_sigma=analysis_n_sigma,
    )

    logger.info("Found residual sources")
    for found_src in sfimage.islands:
        logger.info(found_src)


def generate_visibilities_from_local_skymodel(skymodel, uvw_baselines):
    """
    Generate a set of model visibilities given a skymodel and UVW-baselines.

    Args:
        skymodel (numpy.ndarray): The local skymodel.
            Array of triples ``[l,m,flux_jy]`` , where ``l,m`` are the
            directional cosines for this source, ``flux_jy`` is flux in Janskys.
            Numpy array shape: (n_baselines, 3)
        uvw_baselines (numpy.ndarray): UVW baselines (units of lambda).
            Numpy array shape: (n_baselines, 3)
    Returns (numpy.ndarray):
        Complex visbilities sum for each baseline.
            Numpy array shape: (n_baselines,)
    """
    model_vis = np.zeros(len(uvw_baselines), dtype=np.dtype(complex))
    for src_entry in skymodel:
        model_vis += visibility.visibilities_for_point_source(
            uvw_baselines=uvw_baselines,
            l=src_entry[0],
            m=src_entry[1],
            flux=src_entry[2],
        )
    return model_vis


def main(uvw_lambda,
         skymodel,
         data_vis,
         snr_weights,
         image_size,
         cell_size,
         detection_n_sigma,
         analysis_n_sigma,
         ):
    """
    Represents the difference-image + detect stages of the FastImaging pipeline.
    """

    logger.info("Generating model visibilities from skymodel")
    model_vis = generate_visibilities_from_local_skymodel(
        skymodel=skymodel, uvw_baselines=uvw_lambda)
    # Subtract model-generated visibilities from incoming data
    residual_vis = data_vis - model_vis

    # Kernel generation - might configure this via config-file in future.
    kernel_support = 3
    kernel_func = GaussianSinc(trunc=kernel_support)
    logger.info("Imaging residual visibilities")
    with Tqdm() as progress_bar:
        image, beam = imager.image_visibilities(residual_vis,
                                                vis_weights=snr_weights,
                                                uvw_lambda=uvw_lambda,
                                                image_size=image_size,
                                                cell_size=cell_size,
                                                kernel_func=kernel_func,
                                                kernel_support=kernel_support,
                                                kernel_exact=False,
                                                kernel_oversampling=8,
                                                num_wplanes=16,
                                                max_wpconv_support=20,
                                                progress_bar=progress_bar)
    logger.info("Running sourcefinder on image")
    sfimage = SourceFindImage(data=np.real(image),
                              detection_n_sigma=detection_n_sigma,
                              analysis_n_sigma=analysis_n_sigma,
                              )

    return sfimage
