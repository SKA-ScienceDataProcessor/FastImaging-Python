from __future__ import print_function
import os
import drivecasa
import drivecasa.commands.simulation as sim
from drivecasa.utils import ensure_dir
import astropy.units as u
from astropy.time import Time
from fastimgproto.pipeline.data import vla_c_antennalist_path

def simulate_vis_with_casa(pointing_centre, source_list, output_dir):
    """
    Use casapy to simulate a visibility measurementset with noise.

    (This also produces an initial set of UVW data)

    Args:
        pointing_centre (:class:`astropy.coordinates.SkyCoord`)
        source_list: list of :class:`fastimgproto.skymodel.helpers.SkySource`
        output_dir (str): Output directory which will contain `vis.ms`

    Returns (str): Full path to `vis.ms`.
    """

    ensure_dir(output_dir)

    commands_logfile = os.path.join(output_dir, "./casa-visibilities_for_point_source-commands.log")
    component_list_path = os.path.join(output_dir, './sources.cl')
    output_visibility = os.path.abspath(os.path.join(output_dir, './vis.ms'))

    if os.path.isfile(commands_logfile):
        os.unlink(commands_logfile)
    casa = drivecasa.Casapy(commands_logfile=commands_logfile)
    script = []

    # For VLA reference numbers, see:
    # https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/fov
    # https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/resolution


    # Define some observation parameters:
    obs_central_frequency = 3. * u.GHz
    obs_frequency_bandwidth = 0.125 * u.GHz
    primary_beam_fwhm = (45. * u.GHz / obs_central_frequency) * u.arcmin

    # Convert the sources to a CASA 'componentlist'
    component_list_path = sim.make_componentlist(
        script,
        source_list=[(s.position, s.flux, s.frequency) for s in source_list],
        out_path=component_list_path)

    # Open the visibility file
    sim.open_sim(script, output_visibility)

    # Configure the virtual telescope
    # sim.setpb(script,
    #           telescope_name='VLA',
    #           primary_beam_hwhm=primary_beam_fwhm * 0.5,
    #           frequency=obs_central_frequency)
    sim.setconfig(script,
                  telescope_name='VLA',
                  antennalist_path=vla_c_antennalist_path)
    sim.setspwindow(script,
                    freq_start=obs_central_frequency - 0.5 * obs_frequency_bandwidth,
                    freq_resolution=obs_frequency_bandwidth,
                    freq_delta=obs_frequency_bandwidth,
                    n_channels=1,
                    )
    sim.setfeed(script, )
    sim.setfield(script, pointing_centre)
    sim.setlimits(script)
    sim.setauto(script)
    ref_time = Time('2014-05-01T19:55:45', format='isot', scale='tai')
    sim.settimes(script, integration_time=10 * u.s, reference_time=ref_time)

    # Generate the visibilities
    sim.observe(script, stop_delay=10 * u.s)

    sim.predict(script, component_list_path)

    sim.set_simplenoise(script, noise_std_dev=1 * u.mJy)
    sim.corrupt(script)
    sim.close_sim(script)

    casa.run_script(script)
    return output_visibility