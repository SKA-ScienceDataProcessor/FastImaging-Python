import os

import astropy.units as u
import drivecasa
from drivecasa.utils import ensure_dir


def make_image_map_fits(vis_ms_path, output_dir,
                        image_size, cell_size,
                        niter=150, threshold_in_jy=0.1):
    ensure_dir(output_dir)


    script = []

    img_n_pix = int(image_size.to(u.pixel).value)
    cell_arcsec = cell_size.to(u.arcsec).value
    clean_args = {
        "imsize": [img_n_pix, img_n_pix],
        "cell": [str(cell_arcsec)+'arcsec'],
        "weighting": 'briggs',
        "robust": 0.5,
    }
    dirty_maps = drivecasa.commands.clean(script,
                                    vis_paths=vis_ms_path,
                                    niter=0,
                                    threshold_in_jy=threshold_in_jy,
                                    other_clean_args=clean_args,
                                    out_dir=output_dir,
                                    overwrite=True)

    dirty_fits_path = drivecasa.commands.export_fits(script, dirty_maps.image,
                                               overwrite=True)

    clean_maps = drivecasa.commands.clean(script,
                                    vis_paths=vis_ms_path,
                                    niter=niter,
                                    threshold_in_jy=threshold_in_jy,
                                    other_clean_args=clean_args,
                                    out_dir=output_dir,
                                    overwrite=True)

    clean_fits_path = drivecasa.commands.export_fits(script, clean_maps.image,
                                               overwrite=True)

    logfile_basename = os.path.basename(vis_ms_path)+".casa-clean-commands.log"
    commands_logfile = os.path.join(output_dir, logfile_basename)
    if os.path.isfile(commands_logfile):
        os.unlink(commands_logfile)
    casa = drivecasa.Casapy(commands_logfile=commands_logfile)
    casa.run_script(script)
    return dirty_fits_path, clean_fits_path
