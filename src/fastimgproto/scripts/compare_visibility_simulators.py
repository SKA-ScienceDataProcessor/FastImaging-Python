#!/usr/bin/env python
from __future__ import print_function

import logging
import sys

import astropy.units as u
import click
from astropy.coordinates import Angle, SkyCoord
from fastimgproto.skymodel.helpers import SkyRegion, SkySource
import fastimgproto.casa.io as casa_io
import fastimgproto.casa.reduction as casa_reduce
import fastimgproto.casa.simulation as casa_sim



@click.command()
@click.argument('casavis', type=click.Path(exists=True))
@click.argument('outdir', type=click.Path())
def cli(casavis, outdir):
    logging.basicConfig(level=logging.DEBUG)
    """
    Create a dirty image of a CASA measurementset, using CASA & fastimgproto
    """

    pointing_centre = SkyCoord(180 * u.deg, 8 * u.deg)
    field_of_view = SkyRegion(pointing_centre,
                              radius=Angle(1 * u.deg))

    image_size = 1024 * u.pixel
    cell_size = 3 * u.arcsecond

    # source_list = get_lsm(field_of_view)
    # source_list = get_spiral_source_test_pattern(field_of_view)
    northeast_of_centre = SkyCoord(
        ra=pointing_centre.ra + 0.01 * u.deg,
        dec=pointing_centre.dec + 0.01 * u.deg, )
    steady_source_list = [
        SkySource(position=pointing_centre, flux=1 * u.Jy),
        SkySource(position=northeast_of_centre, flux=0.4 * u.Jy),
    ]


    # Std. dev of Gaussian noise added to visibilities for each baseline:
    # (Jointly normal, i.e. independently added to real / imaginary components.)
    vis_noise_level = 0.001 * u.Jy

    # Simulate visibilities using casapy to generate a set of UVW baselines.
    # (This is next on the list for a 'from scratch' implementation,
    # at which point the CASA dependency becomes purely optional, for
    # cross-validation purposes.)
    vis_path = casa_sim.simulate_vis_with_casa(pointing_centre,
                                               steady_source_list,
                                               # source_list_w_transient,
                                               noise_std_dev=vis_noise_level,
                                               vis_path=output_dir)

    model_vis_path = os.path.join(output_dir, 'modelvis.ms')
    casa_io.copy_measurementset(vis_path, model_vis_path)
    casa_io.replace_corrected_data_vis(model_vis_path, model_vis)
    residuals_vis_path = os.path.join(output_dir, 'residuals.ms')
    casa_io.copy_measurementset(vis_path, residuals_vis_path)
    casa_io.replace_corrected_data_vis(residuals_vis_path, residual_stokes)
    sys.exit(0)


if __name__ == '__main__':
    cli()
