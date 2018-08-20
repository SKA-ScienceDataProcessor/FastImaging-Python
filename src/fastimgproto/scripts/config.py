import json
from collections import OrderedDict

import click


class ConfigKeys:
    """
    Define the string literals to be used as keys in the JSON config file.
    """
    imager_settings = 'imager_settings'
    wprojection_settings = 'wprojection_settings'
    sourcefind_settings = 'sourcefind_settings'

    image_size_pix = 'image_size_pix'
    cell_size_arcsec = 'cell_size_arcsec'
    kernel_function = 'kernel_function'
    kernel_support = 'kernel_support'
    kernel_exact = 'kernel_exact'
    oversampling = 'oversampling'
    gridding_correction = 'gridding_correction'
    analytic_gcf = 'analytic_gcf'
    sourcefind_detection = 'sourcefind_detection'
    sourcefind_analysis = 'sourcefind_analysis'

    num_wplanes='num_wplanes'
    wplanes_median='wplanes_median'
    max_wpconv_support='max_wpconv_support'
    hankel_opt='hankel_opt'
    undersampling_opt='undersampling_opt'
    kernel_trunc_perc='kernel_trunc_perc'
    interp_type='interp_type'


imager_config = OrderedDict((
    (ConfigKeys.image_size_pix, 1024),
    (ConfigKeys.cell_size_arcsec, 1),
    (ConfigKeys.kernel_function, 'pswf'),
    (ConfigKeys.kernel_support, 3),
    (ConfigKeys.kernel_exact, False),
    (ConfigKeys.oversampling, 8),
    (ConfigKeys.gridding_correction, True),
    (ConfigKeys.analytic_gcf, True),
))

wprojection_config = OrderedDict((
    (ConfigKeys.num_wplanes, 0),
    (ConfigKeys.wplanes_median, False),
    (ConfigKeys.max_wpconv_support, 0),
    (ConfigKeys.hankel_opt, False),
    (ConfigKeys.undersampling_opt, 0),
    (ConfigKeys.kernel_trunc_perc, 0),
    (ConfigKeys.interp_type, 'linear'),
))

sourcefind_config = OrderedDict((
    (ConfigKeys.sourcefind_detection, 50),
    (ConfigKeys.sourcefind_analysis, 50),
))

default_config = OrderedDict((
    (ConfigKeys.imager_settings, imager_config),
    (ConfigKeys.wprojection_settings, wprojection_config),
    (ConfigKeys.sourcefind_settings, sourcefind_config),
))

default_config_path = 'fastimg_config.json'


@click.command()
@click.argument('config_path', type=click.File(mode='w'),
                default=default_config_path)
def cli(config_path):
    make_config(config_path)


def make_config(config_file, **kwargs):
    """
    Write out a json file with default fastimg config settings.
    """
    config = default_config.copy()
    for k, v in kwargs.items():
        config[k] = v
    json.dump(config, config_file, indent=2)
