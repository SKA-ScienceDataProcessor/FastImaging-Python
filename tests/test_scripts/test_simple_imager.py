import json
import numpy as np

from click.testing import CliRunner
from fastimgproto.resources.testdata import simple_vis_npz_filepath
from fastimgproto.scripts.simple_imager import ConfigKeys, \
    cli as simple_imager_cli


def test_simple_imager():
    runner = CliRunner()
    with runner.isolated_filesystem():
        config_filename = 'imager_conf.json'
        output_filename = 'image.npz'
        conf = {
            ConfigKeys.image_size_pix: 1024,
            ConfigKeys.cell_size_arcsec: 3.
        }
        with open(config_filename, 'w') as f:
            json.dump(conf, f)

        result = runner.invoke(simple_imager_cli,
                               [
                                   config_filename,
                                   simple_vis_npz_filepath,
                                   output_filename
                               ])
        assert result.exit_code == 0
        with open(output_filename, 'rb') as f:
            output_data = np.load(f)
            assert 'image' in output_data
            assert 'beam' in output_data