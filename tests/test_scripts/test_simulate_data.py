import json

import numpy as np
from click.testing import CliRunner
from fastimgproto.scripts.simulate_data import cli as sim_cli


def test_simulate_data():
    runner = CliRunner()
    with runner.isolated_filesystem():
        output_filename = 'simdata.npz'

        result = runner.invoke(sim_cli,
                               [output_filename,
                                '--nstep','5'
                                ])
        assert result.exit_code == 0
        with open(output_filename, 'rb') as f:
            output_data = np.load(f)
            expected_keys = ('uvw_lambda', 'model', 'vis')
            for k in expected_keys:
                assert k in output_data