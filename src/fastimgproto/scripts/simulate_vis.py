"""
Simulated pipeline run
"""
import logging

import astropy.constants as const
import astropy.units as u
import click
import numpy as np
from astropy.coordinates import Angle, SkyCoord

import fastimgproto.visibility as visibility
from fastimgproto.skymodel.helpers import SkyRegion, SkySource
from fastimgproto.telescope.readymade import meerkat


@click.command()
@click.argument('outpath', type=click.Path(exists=False))
def cli(outpath):
    logging.basicConfig(level=logging.DEBUG)
    pointing_centre = SkyCoord(0 * u.deg, 8 * u.deg)
    field_of_view = SkyRegion(pointing_centre,
                              radius=Angle(1 * u.deg))
    telescope = meerkat
    obs_central_frequency = 3. * u.GHz
    wavelength = const.c / obs_central_frequency
    uvw_m = meerkat.uvw_at_local_hour_angle(local_hour_angle=pointing_centre.ra,
                                            dec = pointing_centre.dec)
    uvw_lambda = uvw_m / wavelength.to(u.m).value
    vis_noise_level = 0.001 * u.Jy
    image_size = 1024 * u.pixel
    cell_size = 3 * u.arcsecond

    # source_list = get_lsm(field_of_view)
    # source_list = get_spiral_source_test_pattern(field_of_view)
    extra_src_position = SkyCoord(ra=pointing_centre.ra + 0.01 * u.deg,
                                  dec=pointing_centre.dec + 0.01 * u.deg, )
    extra_src = SkySource(position=extra_src_position,
                          flux=0.4 * u.Jy)
    source_list = [SkySource(position=pointing_centre, flux=1 * u.Jy),
                   extra_src,
                   ]

    transient_posn = SkyCoord(
        ra=field_of_view.centre.ra - 0.05 * u.deg,
        dec=field_of_view.centre.dec - 0.05 * u.deg)
    transient = SkySource(position=transient_posn, flux=0.5 * u.Jy)

    source_list_w_transient = source_list + [transient]

    # Now use UVW to generate visibilities from scratch...
    # Represent incoming data; includes transient sources, noise:
    data_vis = visibility.visibilities_for_source_list(
        pointing_centre, source_list_w_transient, uvw_lambda)
    data_vis = visibility.add_gaussian_noise(vis_noise_level, data_vis)
    with open(outpath,'wb') as outfile:
        np.savez(outfile, uvw_lambda=uvw_lambda, vis=data_vis)
