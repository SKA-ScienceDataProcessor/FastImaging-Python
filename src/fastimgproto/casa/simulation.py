from __future__ import print_function

import os
import shutil

import astropy.units as u
import drivecasa
import drivecasa.commands.simulation as sim
import drivecasa.commands.subroutines
from astropy.time import Time
from drivecasa.utils import ensure_dir
from fastimgproto.pipeline.data import vla_c_antennalist_path


def simulate_vis_with_casa(pointing_centre, source_list, noise_std_dev, vis_path,
                           overwrite=True, echo=False):
    """
    Use casapy to simulate a visibility measurementset with noise.

    (This also produces an initial set of UVW data)

    Args:
        pointing_centre (:class:`astropy.coordinates.SkyCoord`)
        source_list: list of :class:`fastimgproto.skymodel.helpers.SkySource`
        noise_std_dev (astropy.units.Quantity): Standard deviation of the noise
            (units of Jy).
        vis_path (str): Path to visibilities generated.
        echo (bool): Echo the CASA output to terminal during processing.

    Returns (str): Full path to `vis.ms`.
    """


    vis_abspath = os.path.abspath(vis_path)
    commands_logfile = vis_abspath + "_casa_commands.log"
    casa_logfile = vis_abspath + "_casa_log.log"
    component_list_path = vis_abspath + "_sources.cl"

    for outdir in vis_abspath, component_list_path:
        if os.path.isdir(outdir):
            if overwrite:
                shutil.rmtree(outdir)
            else:
                raise IOError("{} already present and overwrite==False.".format(
                    outdir
                ))
    if os.path.isfile(commands_logfile):
        os.remove(commands_logfile)
    ensure_dir(os.path.dirname(vis_abspath))

    casa = drivecasa.Casapy(commands_logfile=commands_logfile,
                            casa_logfile=casa_logfile,
                            echo_to_stdout=echo)
    script = []
    # Add subroutine definition, for manual reproduction with CASA:
    script.append(drivecasa.drivecasa.commands.subroutines.def_load_antennalist)

    # Define some observation parameters...
    # For VLA reference numbers, see:
    # https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/fov
    # https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/resolution
    obs_central_frequency = 3. * u.GHz
    obs_frequency_bandwidth = 0.125 * u.GHz
    primary_beam_fwhm = (45. * u.GHz / obs_central_frequency) * u.arcmin

    # Convert the sources to a CASA 'componentlist'
    component_list_path = sim.make_componentlist(
        script,
        source_list=[(s.position, s.flux, s.frequency) for s in source_list],
        out_path=component_list_path)

    # Open the visibility file
    sim.open_sim(script, vis_abspath)

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

    sim.set_simplenoise(script, noise_std_dev=noise_std_dev)
    sim.corrupt(script)
    sim.close_sim(script)

    casa_output = casa.run_script(script)
    return casa_output
