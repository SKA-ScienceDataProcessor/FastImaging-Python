"""
Simulated pipeline run
"""
import click
import astropy.units as u
import fastimgproto.casa.io as casa_io
import fastimgproto.casa.reduction as casa_reduce
import fastimgproto.casa.simulation as casa_sim
import fastimgproto.visibility as visibility
import logging
import os
from astropy.coordinates import Angle, SkyCoord
from fastimgproto.skymodel.helpers import SkyRegion, SkySource
from fastimgproto.pipeline.skymodel import (get_spiral_source_test_pattern)


def main():
    output_dir = './simulation_output'
    pointing_centre = SkyCoord(180 * u.deg, 34 * u.deg)
    field_of_view = SkyRegion(pointing_centre,
                              radius=Angle(1 * u.deg))

    # source_list = get_lsm(field_of_view)
    source_list = get_spiral_source_test_pattern(field_of_view)

    transient_posn = SkyCoord(
        ra=field_of_view.centre.ra - 0.05 * u.deg,
        dec=field_of_view.centre.dec - 0.05 * u.deg)
    transient = SkySource(position=transient_posn, flux=0.5 * u.Jy)

    source_list_w_transient = source_list + [transient]

    # Simulate 'incoming data' using casapy, this gives us UVW for free
    vis_path = casa_sim.simulate_vis_with_casa(pointing_centre,
                                               source_list_w_transient,
                                               output_dir=output_dir)
    uvw = casa_io.get_uvw_in_lambda(vis_path)
    stokes_i = casa_io.get_stokes_i_vis(vis_path)

    # # Replace vis with copy that has **only** stokes-I component:
    # casa_io.replace_corrected_data_vis(vis_path, stokes_i)

    # Use UVW to generate visibilities according to skymodel
    modelvis = visibility.calculated_summed_vis(
        pointing_centre, source_list, uvw)

    model_vis_path = os.path.join(output_dir, 'modelvis.ms')
    casa_io.copy_measurementset(vis_path, model_vis_path)
    casa_io.replace_corrected_data_vis(model_vis_path, modelvis)

    # Subtract model-generated visibilities from incoming data
    residual_stokes = stokes_i - modelvis
    # Dump to new ms
    residuals_vis_path = os.path.join(output_dir, 'residuals.ms')
    casa_io.copy_measurementset(vis_path, residuals_vis_path)
    casa_io.replace_corrected_data_vis(residuals_vis_path, residual_stokes)

    # Image the difference
    # (also image incoming / model data separately for desk-checking).
    for vp in (vis_path, model_vis_path, residuals_vis_path,):
        casa_reduce.make_image_map_fits(vp, output_dir)


        # Source find / verify output


@click.command()
def cli():
    logging.basicConfig(level=logging.DEBUG)
    main()
