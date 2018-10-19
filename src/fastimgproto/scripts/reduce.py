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
import fastimgproto.gridder.conv_funcs as kfuncs
from fastimgproto.sourcefind.image import SourceFindImage

from .config import ConfigKeys, default_config_path

logger = logging.getLogger()

# Mapping kernel function strings to python implementation
FUNCTION_KERNELS = {
    'PSWF': kfuncs.PSWF,
    'GaussianSinc': kfuncs.GaussianSinc,
    'Gaussian': kfuncs.Gaussian,
    'Sinc': kfuncs.Sinc,
    'TopHat': kfuncs.Pillbox,
    'Triangle': kfuncs.Triangle,
}


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

    if 'uvw_lambda' in npz_content:
        uvw_lambda = npz_content['uvw_lambda']
    else:
        assert False

    if 'skymodel' in npz_content:
        local_skymodel = npz_content['skymodel']
    else:
        assert False

    if 'vis' in npz_content:
        data_vis = npz_content['vis']
    else:
        assert False

    if 'lha' in npz_content:
        lha = npz_content['lha']
    else:
        lha = np.zeros_like(data_vis)

    if 'snr_weights' in npz_content:
        snr_weights = npz_content['snr_weights']
    else:
        snr_weights = np.ones_like(data_vis)

    config = json.load(config_json)
    imager_config = config[ConfigKeys.imager_settings]
    wprojection_config = config[ConfigKeys.wprojection_settings]
    aprojection_config = config[ConfigKeys.aprojection_settings]
    sourcefind_config = config[ConfigKeys.sourcefind_settings]

    sfimage = main(
        uvw_lambda=uvw_lambda,
        skymodel=local_skymodel,
        data_vis=data_vis,
        snr_weights=snr_weights,
        lha=lha,
        imager_config=imager_config,
        wprojection_config=wprojection_config,
        aprojection_config=aprojection_config,
        sourcefind_config=sourcefind_config,
    )

    if sfimage.islands:
        logger.info("Found residual sources:")
        for found_src in sfimage.islands:
            logger.info(found_src.params)
    else:
        logger.info("No residual source found.")


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
         lha,
         imager_config,
         wprojection_config,
         aprojection_config,
         sourcefind_config,
         ):
    """
    Represents the difference-image + detect stages of the FastImaging pipeline.
    """

    logger.info("Generating model visibilities from skymodel")
    model_vis = generate_visibilities_from_local_skymodel(
        skymodel=skymodel, uvw_baselines=uvw_lambda)
    # Subtract model-generated visibilities from incoming data
    residual_vis = data_vis - model_vis

    # Imager settings
    cell_size = imager_config[ConfigKeys.cell_size_arcsec] * u.arcsec
    image_size = imager_config[ConfigKeys.image_size_pix] * u.pix
    kernel_function_str = imager_config[ConfigKeys.kernel_function]
    kernel_support = imager_config[ConfigKeys.kernel_support]
    kernel_exact = imager_config[ConfigKeys.kernel_exact]
    oversampling = imager_config[ConfigKeys.oversampling]
    gridding_correction = imager_config[ConfigKeys.gridding_correction]
    analytic_gcf = imager_config[ConfigKeys.analytic_gcf]

    # Wprojection settings
    num_wplanes = wprojection_config[ConfigKeys.num_wplanes]
    wplanes_median = wprojection_config[ConfigKeys.wplanes_median]
    max_wpconv_support = wprojection_config[ConfigKeys.max_wpconv_support]
    hankel_opt = wprojection_config[ConfigKeys.hankel_opt]
    undersampling_opt = wprojection_config[ConfigKeys.undersampling_opt]
    kernel_trunc_perc = wprojection_config[ConfigKeys.kernel_trunc_perc]
    interp_type = wprojection_config[ConfigKeys.interp_type]

    # Aprojection settings
    aproj_numtimesteps = aprojection_config[ConfigKeys.aproj_numtimesteps]
    obs_dec = aprojection_config[ConfigKeys.obs_dec]
    obs_ra = aprojection_config[ConfigKeys.obs_ra]
    pbeam_coefs = aprojection_config[ConfigKeys.pbeam_coefs]

    # Sourcefind settings
    detection_n_sigma = sourcefind_config[ConfigKeys.sourcefind_detection]
    analysis_n_sigma = sourcefind_config[ConfigKeys.sourcefind_analysis]

    # Kernel generation - might configure this via config-file in future.
    kernel_funct = FUNCTION_KERNELS[kernel_function_str](kernel_support)

    logger.info("Imaging residual visibilities")
    with Tqdm() as progress_bar:
        image, beam = imager.image_visibilities(residual_vis,
                                                vis_weights=snr_weights,
                                                uvw_lambda=uvw_lambda,
                                                image_size=image_size,
                                                cell_size=cell_size,
                                                kernel_func=kernel_funct,
                                                kernel_support=kernel_support,
                                                kernel_exact=kernel_exact,
                                                kernel_oversampling=oversampling,
                                                gridding_correction=gridding_correction,
                                                analytic_gcf=analytic_gcf,
                                                num_wplanes=num_wplanes,
                                                wplanes_median=wplanes_median,
                                                max_wpconv_support=max_wpconv_support,
                                                hankel_opt=hankel_opt,
                                                undersampling_opt=undersampling_opt,
                                                kernel_trunc_perc=kernel_trunc_perc,
                                                interp_type=interp_type,
                                                aproj_numtimesteps=aproj_numtimesteps,
                                                obs_dec=obs_dec,
                                                obs_ra=obs_ra,
                                                lha=lha,
                                                pbeam_coefs=pbeam_coefs,
                                                progress_bar=progress_bar)
    logger.info("Running sourcefinder on image")
    sfimage = SourceFindImage(data=np.real(image),
                              detection_n_sigma=detection_n_sigma,
                              analysis_n_sigma=analysis_n_sigma,
                              )

    return sfimage
