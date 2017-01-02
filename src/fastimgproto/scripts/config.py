import json

import click

from collections import OrderedDict


class ConfigKeys:
    """
    Define the string literals to be used as keys in the JSON config file.
    """
    image_size_pix = 'image_size_pix'
    cell_size_arcsec = 'cell_size_arcsec'
    sourcefind_detection = 'sourcefind_detection'
    sourcefind_analysis = 'sourcefind_analysis'


default_config_path = 'fastimg_config.json'


@click.command()
@click.argument('config_path', type=click.File(mode='wb'),
                default=default_config_path)
def make_config(config_path):
    """
    Write out a json file with default fastimg config settings.
    """
    default_config = OrderedDict((
        (ConfigKeys.image_size_pix, 1024),
        (ConfigKeys.cell_size_arcsec, 1),
        (ConfigKeys.sourcefind_detection, 50),
        (ConfigKeys.sourcefind_analysis, 50),
    ))
    json.dump(default_config, config_path, indent=2)
