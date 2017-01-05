import json

import numpy as np
from click.testing import CliRunner
from fastimgproto.fixtures.data import simple_vis_npz_filepath
from fastimgproto.scripts.config import (
    default_config_path,
    make_config as makeconfig_cli,
)
from fastimgproto.scripts.image import (
    cli as  imager_cli
)

def test_imager():
    runner = CliRunner()
    output_filename = 'image.npz'
    with runner.isolated_filesystem():
        runner.invoke(makeconfig_cli)
        result = runner.invoke(imager_cli,
                               [
                                   '-c', default_config_path,
                                   simple_vis_npz_filepath,
                                   output_filename,
                               ])
        assert result.exit_code == 0
        with open(output_filename, 'rb') as f:
            output_data = np.load(f)
            assert 'image' in output_data
            assert 'beam' in output_data
