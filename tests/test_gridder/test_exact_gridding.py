import fastimgproto.gridder.conv_funcs as conv_funcs
from fastimgproto.gridder.kernel_generation import Kernel
from fastimgproto.gridder.gridder import convolve_to_grid

import numpy as np
import pytest


def test_single_pixel_overlap_pillbox():
    # NB Grid uv-coordinates are np.arange(n_image) - n_image/2, so e.g. if
    # n_image = 8 then the co-ords in the u/x-direction are:
    # [-4, -3, -2, -1, 0, 1, 2, 3 ]
    n_image = 8
    support = 1
    uv = np.array([(-2., 0), (-2., 0)])
    # Real vis will be complex_, but we can substitute float_ for testing:
    vis_amplitude = 42.123
    vis = vis_amplitude * np.ones(len(uv), dtype=np.float_)
    kernel_func = conv_funcs.Pillbox(0.5)

    vis_grid, sampling_grid = convolve_to_grid(kernel_func,
                                               support=support,
                                               image_size=n_image,
                                               uv=uv, vis=vis,
                                               oversampling=None)
    assert vis_grid.sum() == vis.sum()
    expected_result = np.array(
        [[0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 1., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ]]
    )
    assert (expected_result * vis.sum() == vis_grid).all()
    assert (expected_result * len(uv) == sampling_grid).all()


def test_bounds_checking():
    n_image = 8
    support = 2
    # n_image = 8 then the co-ords in the u/x-direction are:
    # [-4, -3, -2, -1, 0, 1, 2, 3 ]
    # Support = 2 takes this over the edge (since -3 -2 = -5):
    bad_uv = np.array([(-3., 0)])
    vis = np.ones(len(bad_uv), dtype=np.float_)
    kernel_func = conv_funcs.Pillbox(1.5)

    with pytest.raises(ValueError):
        grid = convolve_to_grid(kernel_func,
                                support=support,
                                image_size=n_image,
                                uv=bad_uv, vis=vis)

    grid, _ = convolve_to_grid(kernel_func,
                               support=support,
                               image_size=n_image,
                               uv=bad_uv, vis=vis,
                               raise_bounds=False
                               )
    assert grid.sum() == 0.

    # Now check we're filtering indices in the correct order
    # The mixed
    good_uv = np.array([(0., 0.)])
    mixed_uv = np.array([(-3., 0),
                         (0., 0.)])
    good_grid, _ = convolve_to_grid(kernel_func,
                                    support=support,
                                    image_size=n_image,
                                    uv=good_uv,
                                    vis=np.ones(len(good_uv), dtype=np.float_),
                                    raise_bounds=False
                                    )
    mixed_grid, _ = convolve_to_grid(kernel_func,
                                     support=support,
                                     image_size=n_image,
                                     uv=mixed_uv,
                                     vis=np.ones(len(mixed_uv), dtype=np.float_),
                                     raise_bounds=False
                                     )

    assert (good_grid == mixed_grid).all()


def test_multi_pixel_pillbox():
    n_image = 8
    support = 1

    uv = np.array([(-2., 0)])
    vis = np.ones(len(uv), dtype=np.float_)
    kernel_func = conv_funcs.Pillbox(1.1)

    vis_grid, sampling_grid = convolve_to_grid(kernel_func,
                                               support=support,
                                               image_size=n_image,
                                               uv=uv, vis=vis)
    assert vis_grid.sum() == vis.sum()

    # Since uv is precisely on a sampling point, we'll get a
    # 3x3 pillbox
    v = 1. / 9.
    expected_result = np.array(
        [[0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., v, v, v, 0., 0., 0., 0., ],
         [0., v, v, v, 0., 0., 0., 0., ],
         [0., v, v, v, 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ]]
    )
    assert (expected_result == vis_grid).all()
    # In this case we expect sampling == vis, since we only have one vis == 1.0
    assert (expected_result == sampling_grid).all()


def test_small_pillbox():
    n_image = 8
    support = 1

    uv = np.array([(-1.5, 0.5)])
    vis = np.ones(len(uv), dtype=np.float_)
    kernel_func = conv_funcs.Pillbox(0.55)

    grid, _ = convolve_to_grid(kernel_func,
                               support=support,
                               image_size=n_image,
                               uv=uv, vis=vis)
    assert grid.sum() == vis.sum()
    # This time we're on a mid-point, with a smaller pillbox
    # so we should get a 2x2 output
    v = 1. / 4.
    expected_result = np.array(
        # [-4, -3, -2, -1, 0, 1, 2, 3 ]
        [[0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., v, v, 0., 0., 0., 0., ],
         [0., 0., v, v, 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., ]]
    )
    assert (expected_result == grid).all()


def test_multiple_complex_vis():
    # Quick sanity check for multiple visibilities, complex_ this time:
    n_image = 8
    support = 2

    uv = np.array([(-2., 1),
                   (1., -1),
                   ])
    # vis = np.ones(len(uv), dtype=np.float_)
    vis = np.ones(len(uv), dtype=np.complex_)
    kernel_func = conv_funcs.Pillbox(1.1)

    vis_grid, sampling_grid = convolve_to_grid(kernel_func,
                                               support=support,
                                               image_size=n_image,
                                               uv=uv, vis=vis)
    assert vis_grid.sum() == vis.sum()

    # Since uv is precisely on a sampling point, we'll get a
    # 3x3 pillbox
    v = 1. / 9. + 0j

    expected_result = np.array(
        # [-4, -3, -2, -1, 0, 1, 2, 3 ]
        [[0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., v, v, v, 0.],
         [0., 0., 0., 0., v, v, v, 0.],
         [0., v, v, v, v, v, v, 0.],
         [0., v, v, v, 0., 0., 0., 0.],
         [0., v, v, v, 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.]]
    )
    assert (expected_result == vis_grid).all()
    assert (expected_result == sampling_grid).all()


def test_nearby_complex_vis():
    # Quick sanity check for multiple visibilities, kernel footprints
    # overlapping:
    n_image = 8
    support = 2

    uv = np.array([(-2., 1),
                   (0., -1),
                   ])
    # vis = np.ones(len(uv), dtype=np.float_)
    vis = np.ones(len(uv), dtype=np.complex_)
    kernel_func = conv_funcs.Pillbox(1.1)

    vis_grid, sampling_grid = convolve_to_grid(kernel_func,
                                               support=support,
                                               image_size=n_image,
                                               uv=uv, vis=vis)
    assert vis_grid.sum() == vis.sum()

    # Since uv is precisely on a sampling point, we'll get a
    # 3x3 pillbox
    v = 1. / 9. + 0j

    expected_vis_grid = np.array(
        # [-4, -3, -2, -1, 0, 1, 2, 3 ]
        [[0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., v, v, v, 0., 0.],
         [0., 0., 0., v, v, v, 0., 0.],
         [0., v, v, 2. * v, v, v, 0., 0.],
         [0., v, v, v, 0., 0., 0., 0.],
         [0., v, v, v, 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.]]
    )
    assert (expected_vis_grid == vis_grid).all()
    assert (expected_vis_grid == sampling_grid).all()


def test_triangle():
    # Now let's try a triangle function, larger support this time:
    n_image = 8
    support = 2
    uv = np.array([(1.0, 0.0)])
    subpix_offset = np.array([(0.1, -0.15)])
    vis = np.ones(len(uv), dtype=np.float_)

    # offset = np.array([(0.0, 0.0)])
    uv += subpix_offset
    kernel_func = conv_funcs.Triangle(2.0)

    grid, _ = convolve_to_grid(kernel_func,
                               support=support,
                               image_size=n_image,
                               uv=uv, vis=vis)

    kernel = Kernel(kernel_func=kernel_func, support=support,
                    offset=subpix_offset[0],
                    oversampling=1)

    assert grid.sum() == vis.sum()
    # uv location of sample is 1, therefore pixel index = n_image/2 +1
    xrange = slice(n_image // 2 + 1 - support, n_image // 2 + 1 + support + 1)
    yrange = slice(n_image // 2 - support, n_image // 2 + support + 1)
    assert (grid[yrange, xrange] == (kernel.array / kernel.array.sum())).all()
