"""
Simulated pipeline run
"""
import logging

import astropy.constants as const
import astropy.units as u
from astropy.time import Time
import click
import numpy as np
from astropy.coordinates import Angle, SkyCoord

import fastimgproto.visibility as visibility
from fastimgproto.skymodel.helpers import SkyRegion, SkySource
from fastimgproto.telescope.readymade import Meerkat

default_n_timestep = 1000


@click.command()
@click.argument('output_npz', type=click.File('wb'))
@click.option(
    '--nstep', type=click.INT, default=default_n_timestep,
    help="Number of integration timesteps to simulate (default:{})".format(
        default_n_timestep))
def cli(output_npz, nstep):
    """
    Simulates UVW-baselines, data-visibilities and model-visibilities.

    Resulting numpy arrays are saved in npz format.
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    pointing_centre = SkyCoord(0 * u.deg, -30 * u.deg)

    telescope = Meerkat()
    obs_central_frequency = 3. * u.GHz
    wavelength = const.c / obs_central_frequency
    transit_time = telescope.next_transit(pointing_centre.ra,
                                          start_time=Time('2017-01-01'))
    obs_times = transit_time + np.linspace(-1, 1, nstep) * u.hr
    logger.info("Generating UVW-baselines for {} timesteps".format(nstep))
    with click.progressbar(length=len(obs_times),
                           label='Generating UVW-baselines') as pbar:
        uvw_m = telescope.uvw_tracking_skycoord(
            pointing_centre, obs_times,
            progress_update=pbar.update
        )
    uvw_lambda = uvw_m / wavelength.to(u.m).value
    vis_noise_level = 0.001 * u.Jy

    # source_list = get_lsm(field_of_view)
    # source_list = get_spiral_source_test_pattern(field_of_view)
    extra_src_position = SkyCoord(ra=pointing_centre.ra + 0.01 * u.deg,
                                  dec=pointing_centre.dec + 0.01 * u.deg, )

    steady_sources = [
        SkySource(pointing_centre, flux=1 * u.Jy),
        SkySource(extra_src_position, flux=0.4 * u.Jy),
    ]

    transient_posn = SkyCoord(
        ra=pointing_centre.ra - 0.05 * u.deg,
        dec=pointing_centre.dec - 0.05 * u.deg)
    transient_sources = [
        SkySource(position=transient_posn, flux=0.5 * u.Jy),
    ]

    all_sources = steady_sources + transient_sources

    # Now use UVW to generate visibilities from scratch...
    # Simulate model data - known sources only, noise-free:
    model_vis = visibility.visibilities_for_source_list(
        pointing_centre, steady_sources, uvw_lambda)

    # Simulate incoming data; includes transient sources, noise:
    logger.info("Simulating visibilities")
    data_vis = visibility.visibilities_for_source_list(
        pointing_centre, all_sources, uvw_lambda)
    data_vis = visibility.add_gaussian_noise(vis_noise_level, data_vis)

    # with open(output, 'wb') as outfile:
    np.savez(output_npz,
             uvw_lambda=uvw_lambda,
             model=model_vis,
             vis=data_vis,
             )
