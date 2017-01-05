import numpy as np
import astropy.units as u
from fastimgproto.fixtures.data import simple_vis_npz_filepath
from fastimgproto.imager import image_visibilities
from fastimgproto.bindings.imager import cpp_image_visibilities, CppKernelFuncs
from fastimgproto.gridder.conv_funcs import GaussianSinc


def test_cpp_imager_results_parity():
    with open(simple_vis_npz_filepath, 'rb') as f:
        npz_data_dict = np.load(f)
        uvw_lambda = npz_data_dict['uvw_lambda']
        vis = npz_data_dict['vis']

    trunc = 3.0
    support = 3

    kernel_func = GaussianSinc(trunc=trunc)
    kernel_func_name = CppKernelFuncs.gauss_sinc

    image_size = 1024 * u.pix
    cell_size = 1. * u.arcsec
    py_img, py_beam = image_visibilities(
        vis=vis,
        uvw_lambda=uvw_lambda,
        image_size= image_size,
        cell_size=cell_size,
        kernel_func = kernel_func,
        kernel_support=support,
        kernel_exact=True,
        normalize=True,
    )

    cpp_img, cpp_beam = cpp_image_visibilities(
        vis=vis,
        uvw_lambda=uvw_lambda,
        image_size=image_size,
        cell_size=cell_size,
        kernel_func_name=kernel_func_name,
        kernel_trunc_radius=trunc,
        kernel_support=support,
        kernel_exact=True,
        normalize=True
    )

    assert (np.abs(cpp_img - py_img)< 1E-5).all()
