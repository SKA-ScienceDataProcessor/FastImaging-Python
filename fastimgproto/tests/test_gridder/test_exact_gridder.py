import fastimgproto.gridder.conv_funcs as conv_funcs
from fastimgproto.gridder.exact_gridder import exact_convolve_to_grid

import numpy as np



# def test_exact_gridder():
#     n_image = 32
#     grid = np.zeros((n_image, n_image), dtype=np.complex_)
#     g_centre = (n_image//2, n_image//2)
#
#     uv = np.array([(10.5,10.5)])
#     vis = np.array([1+0j])
#
#     kernel_func = conv_funcs.Pillbox(0.75)
#
#     exact_convolve_to_grid(kernel_func, support=1, grid=grid,
#                            uv=uv, vis=vis)

