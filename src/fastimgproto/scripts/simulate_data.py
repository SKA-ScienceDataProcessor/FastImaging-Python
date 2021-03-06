"""
Simulated pipeline run
"""
import logging

import astropy.constants as const
import astropy.units as u
import click
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from scipy.stats import lognorm
from tqdm import tqdm as Tqdm

import fastimgproto.visibility as visibility
from fastimgproto.skymodel.helpers import SkySource
from fastimgproto.telescope.readymade import Meerkat


default_n_timestep = 100


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
    with Tqdm() as progress_bar:
        uvw_m = telescope.uvw_tracking_skycoord(
            pointing_centre, obs_times,
            progress_bar=progress_bar
        )
    # From here on we use UVW as multiples of wavelength, lambda:
    uvw_lambda = (uvw_m / wavelength).to(u.dimensionless_unscaled).value
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
    # Store l,m cosines & fluxes for skymodel (known sources only):
    local_skymodel = []
    for src in steady_sources:
        l, m = visibility.calculate_direction_cosines(pointing_centre, src)
        local_skymodel.append((l,m,src.flux.to(u.Jy).value))
    local_skymodel = np.asarray(local_skymodel,dtype=np.float_)

    # Simulate incoming data; includes transient sources, noise:
    logger.info("Simulating visibilities")
    data_vis = visibility.visibilities_for_source_list(
        pointing_centre, all_sources, uvw_lambda)
    data_vis = visibility.add_gaussian_noise(vis_noise_level, data_vis)

    # Generate some SNR weights
    # Current implementation has no theoretical basis, just picked an arbitrary
    # plausible-looking distribution:
    snr_weights = np.minimum(1.0, lognorm.rvs(s=0.25, size=len(data_vis)))

    #Alternatively, just assume perfect data:
    # snr_weights = np.ones_like(data_vis,dtype=np.float_)

    # with open(output, 'wb') as outfile:
    np.savez(output_npz,
             uvw_lambda=uvw_lambda,
             skymodel=local_skymodel,
             vis=data_vis,
             snr_weights=snr_weights,
             )
