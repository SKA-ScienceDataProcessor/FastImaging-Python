from __future__ import print_function

import os
import shutil
from contextlib import closing

import astropy.constants as const
import numpy as np
from drivecasa.utils import ensure_dir

import casacore.tables as casatables


def get_stokes_i_vis(vis_ms_path):
    """
    Export 'CORRECTED_DATA' Stokes I complex visibilities to numpy-array

    Args:
        vis_ms_path (str): Path to visibility MeasurementSet
    Returns (numpy.array): visibility data.
        Array of complex, shape: (n_baseline_samples,)
    """
    with closing(casatables.table(vis_ms_path)) as tbl:
        stokes_i = tbl.getcol('CORRECTED_DATA').squeeze()[:, 0]
    return stokes_i


def get_uvw(vis_ms_path):
    """
    Extract uvw data as squeezed numpy array. Units of metres.

    Args:
        vis_ms_path (str): Path to visibility MeasurementSet
    Returns (numpy.array): uvw data.
        Array of floats, shape: (n_baseline_samples, 3)
    """
    with closing(casatables.table(vis_ms_path)) as tbl:
        uvw_metres = tbl.getcol('UVW')
    return uvw_metres


def get_uvw_in_lambda(vis_ms_path):
    """
    Extract uvw data as squeezed numpy array. Units of lambda.

    Args:
        vis_ms_path (str): Path to visibility MeasurementSet
    Returns (numpy.array): uvw data. Array shape: (n_baseline_samples, 3)
    """
    uvw_metres = get_uvw(vis_ms_path)
    spw_path = os.path.join(vis_ms_path, 'SPECTRAL_WINDOW')
    with closing(casatables.table(spw_path)) as spw:
        freq_hz = spw.getcol('CHAN_FREQ').squeeze()
    wavelength = const.c.value / freq_hz
    uvw_lambda = uvw_metres / wavelength
    return uvw_lambda

def replace_corrected_data_vis(vis_ms_path, stokes_i):
    """
    Replace 'CORRECTED_DATA' Stokes I+V complex visibilities with given array

    Args:
        vis_ms_path (str): Path to visibility MeasurementSet
        stokes_i (numpy.array): visibility data.
            Array of complex, shape: (n_baseline_samples,)
    """
    with closing(casatables.table(vis_ms_path,readonly=False)) as tbl:
        corrdata = tbl.getcol('CORRECTED_DATA')
        corrdata= np.zeros_like(corrdata)
        colshape = corrdata[...,0].shape
        corrdata[..., 0] = stokes_i.reshape(colshape)
        corrdata[..., 3] = stokes_i.reshape(colshape)
        tbl.putcol('CORRECTED_DATA', corrdata)


def copy_measurementset(src, dest):
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    ensure_dir(os.path.dirname(dest))
    shutil.copytree(src, dest)
