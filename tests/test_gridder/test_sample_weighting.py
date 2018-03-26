import numpy as np
import pytest

import fastimgproto.gridder.conv_funcs as conv_funcs
from fastimgproto.gridder.gridder import convolve_to_grid


def test_zero_weighting():
    """
    Sanity check that the weights are having some effect
    """
    n_image = 8
    support = 1
    uvw = np.array([(-2., 0, 0),
                   (-2., 0, 0)])
    # Real vis will be complex_, but we can substitute float_ for testing:
    vis_amplitude = 42.123
    vis = vis_amplitude * np.ones(len(uvw), dtype=np.float_)
    vis_weights = np.zeros_like(vis)
    kernel_func = conv_funcs.Pillbox(0.5)

    zeros_array = np.zeros((8, 8), dtype=vis.dtype)

    # Exact gridding
    vis_grid, sampling_grid = convolve_to_grid(kernel_func,
                                               support=support,
                                               image_size=n_image,
                                               uvw=uvw,
                                               vis=vis,
                                               vis_weights=vis_weights,
                                               exact=True,
                                               oversampling=None)
    assert vis.sum() != 0
    assert (vis_grid == zeros_array).all()

    # Kernel-cache
    vis_grid, sampling_grid = convolve_to_grid(kernel_func,
                                               support=support,
                                               image_size=n_image,
                                               uvw=uvw,
                                               vis=vis,
                                               vis_weights=vis_weights,
                                               exact=False,
                                               oversampling=5)
    assert (vis_grid == zeros_array).all()


def test_natural_weighting():
    """
    Confirm natural weighting works as expected in most basic non-zero case
    """
    n_image = 8
    support = 1
    uvw = np.array([(-2., 0, 0),
                   (-2., 0, 0)])
    # Real vis will be complex_, but we can substitute float_ for testing:
    vis = np.asarray([3. / 2., 3.])
    vis_weights = np.asarray([2. / 3., 1. / 3])
    vis_weights = np.asarray([2., 3.])
    # vis_weights = np.ones_like(vis)
    kernel_func = conv_funcs.Pillbox(0.5)
    # Exact gridding
    vis_grid, sampling_grid = convolve_to_grid(kernel_func,
                                               support=support,
                                               image_size=n_image,
                                               uvw=uvw,
                                               vis=vis,
                                               vis_weights=vis_weights,
                                               exact=True,
                                               oversampling=None)

    natural_weighted_sum = (vis * vis_weights).sum() / vis_weights.sum()
    # print("Natural estimate", natural_weighted_sum)
    # print("SAMPLES\n", sampling_grid)
    # print("VIS\n", vis_grid)

    expected_sample_locations = np.array(
        [[0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 1., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ]]
    )
    assert (
    expected_sample_locations == sampling_grid/sampling_grid.sum()).all()
    assert ((expected_sample_locations * natural_weighted_sum) ==
                abs(vis_grid) / abs(sampling_grid.sum())).all()
