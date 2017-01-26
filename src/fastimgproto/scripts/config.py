import json

import click

from collections import OrderedDict


class ConfigKeys:
    """
    Define the string literals to be used as keys in the JSON config file.
    """
    image_size_pix = 'image_size_pix'
    cell_size_arcsec = 'cell_size_arcsec'
    kernel_support = "kernel_support"
    kernel_exact = "kernel_exact"
    sourcefind_detection = 'sourcefind_detection'
    sourcefind_analysis = 'sourcefind_analysis'


default_config = OrderedDict((
    (ConfigKeys.image_size_pix, 1024),
    (ConfigKeys.cell_size_arcsec, 1),
    (ConfigKeys.kernel_support, 3),
    (ConfigKeys.kernel_exact, True),
    (ConfigKeys.sourcefind_detection, 50),
    (ConfigKeys.sourcefind_analysis, 50),
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
