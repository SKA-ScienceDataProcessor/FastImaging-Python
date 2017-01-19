"""
Test the oversampling / downsampling routines used to generate Kernels at
regularly spaced subpixel offsets
"""

import fastimgproto.gridder.conv_funcs as conv_funcs
from fastimgproto.gridder.kernel_generation import Kernel
from fastimgproto.gridder.gridder import (
    calculate_oversampled_kernel_indices,
    populate_kernel_cache,
)
from fastimgproto.gridder.gridder import convolve_to_grid
import numpy as np


def test_fractional_coord_to_oversampled_index_math():
    """
    Sanity check the calculations for oversampling of subpixel offsets

    """

    ##NB Example edge case:
    oversampling = 7
    subpix_offset = 0.5
    # Returns index value greater than ``oversampling//2`` :
    assert np.around(subpix_offset * oversampling).astype(int) == 4
    assert (calculate_oversampled_kernel_indices(
        subpixel_coord=subpix_offset, oversampling=oversampling) == 3).all()

    # OK, now demonstrate values with oversampling of 5, which has easy
    # numbers to calculate since 1/5 = 0.2
    oversampling = 5
    io_pairs = np.array([
        [-0.5, -2],
        [-0.4999, -2],
        [-0.4, -2],
        [-0.35, -2],
        [-0.3, -2],  # <-- numpy.around favours even roundings
        [-0.2999, -1],
        [-0.2, -1],
        [-0.11, -1],
        [-0.1, 0],  # <-- numpy.around favours even roundings
        [-0.09, 0],
        [-0.01, 0],
        [0.0, 0],
    ])

    outputs = calculate_oversampled_kernel_indices(io_pairs[:, 0], oversampling)
    assert (io_pairs[:, 1] == outputs).all()
    # symmetry:
    io_pairs *= -1.
    outputs = calculate_oversampled_kernel_indices(io_pairs[:, 0], oversampling)
    assert (io_pairs[:, 1] == outputs).all()

    ## Check it works as expected when the inputs are co-ordinate pairs:
    inputs = np.array([(0.3, 0.3), ])
    outputs = np.array([(2, 2), ])
    assert (calculate_oversampled_kernel_indices(inputs, oversampling) ==
            outputs).all()


def test_fractional_coord_in_2d_case():
    """
    Sanity check everything works OK when input is 2d array (i.e. UV-coords)

    """

    # OK, now demonstrate values with oversampling of 5, which has easy
    # numbers to calculate since 1/5 = 0.2
    oversampling = 5
    io_pairs = np.array([
        [-0.5, -2],
        [-0.4999, -2],
        [-0.4, -2],
        [-0.35, -2],
        [-0.3, -2],  # <-- numpy.around favours even roundings
        [-0.2999, -1],
        [-0.2, -1],
        [-0.11, -1],
        [-0.1, 0],  # <-- numpy.around favours even roundings
        [-0.09, 0],
        [-0.01, 0],
        [0.0, 0],
    ])
    input = io_pairs[:, 0]
    input = np.vstack((input, input)).T
    assert (input.shape == (len(io_pairs), 2))

    output = calculate_oversampled_kernel_indices(input, oversampling)
    assert output.shape == input.shape
    assert (io_pairs[:, 1] == output[:,0]).all()
    assert (io_pairs[:, 1] == output[:,1]).all()


def test_kernel_caching():
    """
    Test generation of cached (offset) kernels, and demonstrate correct usage.

    In this test, we assume an oversampling of 5, resulting in
    step-widths of 0.2 regular pixels. We then iterate through a bunch of
    possible sub-pixel offsets, checking that we pick the nearest (closest to
    exact-positioned) cached kernel correctly.
    """
    # We use a triangle to compare, since even a tiny pixel offset should
    # result in differing values when using exact convolution,
    # this makes it easier to verify that the 'stepped' kernel is behaving
    # as expected.

    n_image = 8
    support = 3
    kernel_func = conv_funcs.Triangle(half_base_width=2.5)
    oversampling = 5

    # Choose sub-pixel steps that align with oversampling grid:
    steps = np.array([-0.4, 0.2, 0.0, 0.2, 0.4])
    substeps = np.linspace(-0.099999, 0.099999, num=15)

    kernel_cache = populate_kernel_cache(
        kernel_func=kernel_func, support=support, oversampling=oversampling)

    for x_offset in steps:
        offset = (x_offset, 0.0)
        aligned_exact_kernel = Kernel(kernel_func=kernel_func, support=support,
                                      offset=offset)
        # Generate an index into the kernel-cache at the precise offset
        # (i.e. a multiple of 0.2-regular-pixel-widths)
        aligned_cache_idx = calculate_oversampled_kernel_indices(offset,
                                                                 oversampling)
        cached_kernel = kernel_cache[tuple(aligned_cache_idx)]

        # Check that for oversampling-grid aligned positions, cache == exact
        assert (aligned_exact_kernel.array == cached_kernel.array).all()

        # Now check the results for non-aligned positions
        for sub_offset in substeps:
            offset = (x_offset + sub_offset, 0.0)
            if sub_offset != 0.0:
                unaligned_exact_kernel = Kernel(kernel_func=kernel_func,
                                                support=support, offset=offset)
                # Check that the irregular position resolves to the correct
                # nearby aligned position:
                unaligned_cache_idx = calculate_oversampled_kernel_indices(
                    offset, oversampling)
                assert (unaligned_cache_idx == aligned_cache_idx).all()

                ## Demonstrate retrieval of the cached kernel:
                cached_kernel = kernel_cache[tuple(unaligned_cache_idx)]
                assert (aligned_exact_kernel.array == cached_kernel.array).all()

                ## Sanity check - we expect the exact-calculated kernel to
                ## be different by a small amount
                diff = (
                aligned_exact_kernel.array - unaligned_exact_kernel.array)
                eps = 10e-9
                assert not (np.fabs(diff) < eps).all()


def test_oversampled_gridding():
    """
    Integration test of the convolve_to_grid function with oversampling

    Mostly tests same functionality as ``test_kernel_caching``.

    """
    # Let's grid a triangle function
    n_image = 8
    support = 2
    uv = np.array([(1.0, 0.0),
                   (1.3, 0.0),
                   (.01, -1.32),
                   ])

    vis = np.ones(len(uv), dtype=np.float_)

    kernel_func = conv_funcs.Triangle(2.0)

    grid, _ = convolve_to_grid(kernel_func,
                               support=support,
                               image_size=n_image,
                               uv=uv, vis=vis,
                               exact=False,
                               oversampling=9
                               )

    assert grid.sum() == vis.sum()
