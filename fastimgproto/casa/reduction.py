import drivecasa
from drivecasa.utils import ensure_dir
import os


def make_image_map_fits(vis_ms_path, output_dir,
                        niter=150, threshold_in_jy=0.3):
    ensure_dir(output_dir)


    script = []

    clean_args = {
        "imsize": [1024, 1024],
        "cell": ['3.0arcsec'],
        "weighting": 'briggs',
        "robust": 0.5,
    }
    maps = drivecasa.commands.clean(script,
                                    vis_paths=vis_ms_path,
                                    niter=niter,
                                    threshold_in_jy=threshold_in_jy,
                                    other_clean_args=clean_args,
                                    out_dir=output_dir,
                                    overwrite=True)

    fits_path = drivecasa.commands.export_fits(script, maps.image,
                                               overwrite=True)

    logfile_basename = os.path.basename(maps.image)+".casa-clean-commands.log"
    commands_logfile = os.path.join(output_dir, logfile_basename)
    if os.path.isfile(commands_logfile):
        os.unlink(commands_logfile)
    casa = drivecasa.Casapy(commands_logfile=commands_logfile)
    casa.run_script(script)
    return fits_path
