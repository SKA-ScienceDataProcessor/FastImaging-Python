"""
Simulated pipeline run
"""
import logging
import os

import astropy.units as u
import click
from astropy.coordinates import Angle, SkyCoord

import fastimgproto.casa.simulation as casa_sim
from fastimgproto.pipeline.skymodel import get_spiral_source_test_pattern
from fastimgproto.skymodel.helpers import SkyRegion, SkySource


@click.command()
@click.argument('outpath', type=click.Path(exists=False))
def cli(outpath):
    logging.basicConfig(level=logging.DEBUG)
    pointing_centre = SkyCoord(180 * u.deg, 8 * u.deg)
    field_of_view = SkyRegion(pointing_centre,
                              radius=Angle(1 * u.deg))

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

    # source_list_w_transient = source_list + [transient]

    casa_sim.simulate_vis_with_casa(pointing_centre,
                                    source_list,
                                    # source_list_w_transient,
                                    vis_path=outpath,
                                    echo=True)
