#!/usr/bin/env python
from __future__ import print_function

import logging
import sys

import click
import numpy as np

import fastimgproto.casa.io as casa_io


@click.command()
@click.argument('casavis', type=click.Path(exists=True))
@click.argument('outfile', type=click.File(mode='w'))
def cli(casavis, outfile):
    logging.basicConfig(level=logging.DEBUG)
    """
    Extracts uvw / vis data from a CASA measurementset, save in npz format.
    """
    vis = casa_io.get_stokes_i_vis(casavis)
    uvw_lambda = casa_io.get_uvw_in_lambda(casavis)
    np.savez(outfile, uvw_lambda=uvw_lambda, vis=vis)
    sys.exit(0)

if __name__ == '__main__':
    cli()
