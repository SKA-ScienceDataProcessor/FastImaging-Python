import logging

import astropy.units as u
import numpy as np
import pytest

from fastimgproto.bindings import CPP_BINDINGS_PRESENT
from fastimgproto.bindings.imager import (
    PYTHON_KERNELS,
    CppKernelFuncs,
    cpp_image_visibilities,
)
from fastimgproto.fixtures.data import simple_vis_npz_filepath
from fastimgproto.imager import image_visibilities as py_image_visibilities

logger = logging.getLogger(__name__)


def compare_imagers(kwargs):
    pyargs = kwargs.copy()
    py_kernel_func = PYTHON_KERNELS[pyargs.pop('kernel_func_name')]
    trunc = pyargs.pop('kernel_trunc_radius')
    py_kernel = py_kernel_func(trunc)
    pyargs['kernel_func'] = py_kernel

    display_kwargs = kwargs.copy()
    del display_kwargs['vis']
    del display_kwargs['uvw_lambda']

    logger.info("KWARGS: {}".format(display_kwargs))

    logger.debug("Running Python imager...")
    py_img, py_beam = py_image_visibilities(**pyargs)
    logger.debug("Done")
    logger.debug("Running C++ imager...")
    cpp_img, cpp_beam = cpp_image_visibilities(**kwargs)
    logger.debug("Done")

    # cpp_img = np.real(cpp_img)
    # py_img = np.real(py_img)
    # print()
    # print("REAL", np.max(np.absolute(np.real((cpp_img-py_img)))))
    # print("IMAG", np.max(np.absolute(np.imag((cpp_img-py_img)))))
    print("ABS", np.max(np.absolute((cpp_img-py_img))))
    assert (np.absolute(cpp_img - py_img) < 2E-14).all()


@pytest.mark.skipif(not CPP_BINDINGS_PRESENT,
                    reason="C++ bindings not present")
def test_cpp_imager_results_parity():
    with open(simple_vis_npz_filepath, 'rb') as f:
        npz_data_dict = np.load(f)
        uvw_lambda = npz_data_dict['uvw_lambda']
        vis = npz_data_dict['vis']

    trunc = 3.0
    support = 3
    kernel_func_name = CppKernelFuncs.gauss_sinc

    image_size = 1024 * u.pix
    cell_size = 1. * u.arcsec
    kwargs = dict(
        vis=vis,
        uvw_lambda=uvw_lambda,
        image_size=image_size,
        cell_size=cell_size,
        kernel_func_name=CppKernelFuncs.gauss_sinc,
        kernel_trunc_radius=trunc,
        kernel_support=support,
        kernel_exact=True,
        normalize=True
    )
    compare_imagers(kwargs)

    kwargs['image_size']=2048* u.pix
    kwargs['cell_size'] = 0.5*u.arcsec
    compare_imagers(kwargs)

    kwargs['image_size'] = 1024 * u.pix
    kwargs['cell_size'] = 0.5 * u.arcsec

    # kwargs['kernel_func_name'] = CppKernelFuncs.triangle
    # compare_imagers(kwargs)
    # kwargs['kernel_func_name'] = CppKernelFuncs.gauss
    # compare_imagers(kwargs)

    kwargs['kernel_exact'] = False
    kwargs['kernel_oversampling'] = 9
    compare_imagers(kwargs)
