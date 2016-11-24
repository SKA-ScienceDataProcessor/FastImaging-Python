"""
Simulated pipeline run
"""
from __future__ import print_function
import logging
import os

import astropy.units as u
import click
import fastimgproto.casa.io as casa_io
import fastimgproto.casa.simulation as casa_sim
import fastimgproto.imager as imager
import fastimgproto.visibility as visibility
from astropy.coordinates import Angle, SkyCoord
from fastimgproto.gridder.conv_funcs import GaussianSinc
from fastimgproto.skymodel.helpers import SkyRegion, SkySource
from fastimgproto.sourcefind.image import SourceFindImage
import numpy as np


DEFAULT_CASAVIS_PATH ='/tmp/fastimgproto_simpipe_vis.ms'
DEFAULT_UVW_PATH ='./uvw_lambda.npz'
@click.command()
@click.option('--load-uvw/--no-load-uvw', default=False,
              help="Load UVW-baseline data from a previous run rather than "
                   "generate new with CASA.")
@click.option('--casavis', type=click.Path(),
              default=DEFAULT_CASAVIS_PATH,
              help="Path where CASA-generated visibilities will be written out,"
                   " default: '{}'".format(DEFAULT_CASAVIS_PATH))
@click.option('--uvw', type=click.Path(),
              default=DEFAULT_UVW_PATH,
              help="Path where UVW-baseline data will be written to / read from,"
                   " default: '{}'".format(DEFAULT_UVW_PATH))
def cli(load_uvw, uvw, casavis):
    """
    Define source pattern, generate uvw data, then pass on to the main pipeline.
    """
    logging.basicConfig(level=logging.DEBUG)
    uvw_path = uvw
    casavis_path = casavis
    pointing_centre = SkyCoord(180 * u.deg, 8 * u.deg)
    field_of_view = SkyRegion(pointing_centre,
                              radius=Angle(1 * u.deg))

    # source_list = get_lsm(field_of_view)
    # source_list = get_spiral_source_test_pattern(field_of_view)
    northeast_of_centre = SkyCoord(
        ra=pointing_centre.ra + 0.01 * u.deg,
        dec=pointing_centre.dec + 0.01 * u.deg, )
    steady_source_list = [
        SkySource(position=pointing_centre, flux=1 * u.Jy),
        SkySource(position=northeast_of_centre, flux=0.4 * u.Jy),
    ]

    southwest_of_centre = SkyCoord(
        ra=field_of_view.centre.ra - 0.05 * u.deg,
        dec=field_of_view.centre.dec - 0.05 * u.deg)
    transient_src_list = [
        SkySource(position=southwest_of_centre, flux=0.5 * u.Jy),
    ]

    # Std. dev of Gaussian noise added to visibilities for each baseline:
    # (Jointly normal, i.e. independently added to real / imaginary components.)
    vis_noise_level = 0.001 * u.Jy

    # Simulate visibilities using casapy to generate a set of UVW baselines.
    # (This is next on the list for a 'from scratch' implementation,
    # at which point the CASA / casacore dependency becomes purely optional, for
    # cross-validation purposes.)
    if not load_uvw:
        for path in uvw_path, casavis_path:
            output_dir = os.path.dirname(os.path.abspath(path))
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
        casa_output = casa_sim.simulate_vis_with_casa(pointing_centre,
                                                      steady_source_list,
                                                      # source_list_w_transient,
                                                      noise_std_dev=vis_noise_level,
                                                      vis_path=casavis_path)
        uvw_lambda = casa_io.get_uvw_in_lambda(casavis_path)
        with open(uvw_path, 'wb') as f:
            np.savez(f, uvw_lambda=uvw_lambda)
    else:
        with open(uvw_path, 'rb') as f:
            npz_content = np.load(f)
            uvw_lambda = npz_content['uvw_lambda']

    sfimage = main(
        steady_source_list=steady_source_list,
        transient_source_list=transient_src_list,
        pointing_centre=pointing_centre,
        vis_noise_level=vis_noise_level,
        uvw_lambda=uvw_lambda,
    )
    print("Inserted transients:")
    for insert_src in transient_src_list:
        print(insert_src)
    print("Found residual sources")
    for found_src in sfimage.islands:
        print(found_src)


def main(steady_source_list,
         transient_source_list,
         pointing_centre,
         vis_noise_level,
         uvw_lambda,
         image_size=1024 * u.pixel,
         cell_size=3 * u.arcsecond,
         detection_n_sigma=50,
         analysis_n_sigma=25,
         ):
    """
    Represents the image + detect stages of the FastImaging pipeline.

    This includes simulating the incoming data, and therefore this script
    isn't ideal for benchmarking 'as is'. On the other hand it allows for
    quick and easy variations of the source pattern, exposure length, etc.
    """
    source_list_w_transient = steady_source_list + transient_source_list

    # Now use UVW to generate visibilities from scratch...
    # Represent incoming data; includes transient sources, noise:
    data_vis = visibility.calculated_summed_vis(
        pointing_centre, source_list_w_transient, uvw_lambda)
    data_vis = visibility.add_gaussian_noise(vis_noise_level, data_vis)

    # Model vis; only steady sources from the catalog, noise free.
    model_vis = visibility.calculated_summed_vis(
        pointing_centre, steady_source_list, uvw_lambda)

    # Subtract model-generated visibilities from incoming data
    residual_vis = data_vis - model_vis

    # Will move this to a config option later
    kernel_support = 3
    kernel_func = GaussianSinc(trunc=kernel_support)
    image, beam = imager.image_visibilities(residual_vis, uvw_lambda,
                                            image_size=image_size,
                                            cell_size=cell_size,
                                            kernel_func=kernel_func,
                                            kernel_support=kernel_support,
                                            kernel_oversampling=None)

    sfimage = SourceFindImage(data=np.real(image),
                              detection_n_sigma=detection_n_sigma,
                              analysis_n_sigma=analysis_n_sigma,
                              )

    return sfimage
