from __future__ import print_function
import json
import numpy as np

from click.testing import CliRunner
from fastimgproto.resources.testdata import simple_vis_npz_filepath
from fastimgproto.scripts.extract_lsm import cli as extract_lsm_cli
from fastimgproto.skymodel.extraction import SumssSrc
import csv


def test_extract_lsm():
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog_output_filename = 'foo.csv'
        args = '-- 189.2 -45.6 0.2'.split()
        # Invoke with output to stdout
        result = runner.invoke(extract_lsm_cli,
                               args=args)
        assert result.exit_code == 0

        # And output to TSV file:
        args.append(catalog_output_filename)
        result = runner.invoke(extract_lsm_cli,
                               args=args)
        print(result.output)

        with open(catalog_output_filename, 'rb') as tsvfile:
            dr=csv.DictReader(tsvfile, delimiter='\t')
            rows = [r for r in dr]

        assert len(rows)==1

        for key in SumssSrc._list_dictkeys():
            assert key in rows[0].keys()
