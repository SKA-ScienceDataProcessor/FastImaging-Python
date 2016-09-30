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

    # OK, now demonstrate values with oversampling of 0.5, which has easy
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


def test_stepped_vs_exact_convolution():
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
    substeps = np.linspace(-0.099999, 0.099999, num=50)

    kernel_cache = populate_kernel_cache(
        kernel_func=kernel_func, support=support, oversampling=oversampling)

    for x_offset in steps:
        offset = (x_offset, 0.0)
        aligned_exact_kernel = Kernel(kernel_func=kernel_func, support=support,
                                      offset=offset)
        aligned_cache_idx = calculate_oversampled_kernel_indices(offset, oversampling)
        cached_kernel = kernel_cache[tuple(aligned_cache_idx)]

        # Check that for oversampling-grid aligned positions, cache == exact
        assert (aligned_exact_kernel.array == cached_kernel.array).all()

        # Now check the results for non-aligned positions
        for sub_offset in substeps:
            offset = (x_offset + sub_offset, 0.0)
            if sub_offset != 0.0:
                unaligned_exact_kernel = Kernel(kernel_func=kernel_func,
                                                support=support, offset=offset)
                unaligned_cache_idx = calculate_oversampled_kernel_indices(offset, oversampling)
                # We expect to return the same kernel as nearest grid-point
                assert (unaligned_cache_idx==aligned_cache_idx).all()

                ## No need to actually check the result, since we're fetching
                ## from a fixed dict, but this is true if uncommented:
                # cached_kernel = kernel_cache[tuple(unaligned_cache_idx)]
                # assert (aligned_exact_kernel.array == cached_kernel.array).all()

                ## Check that the exact kernel really does differ from the
                ## nearest oversampled kernel (again, bit superfluous)
                # diff = (aligned_exact_kernel.array - unaligned_exact_kernel.array)
                # eps = 10e-10
                # assert not (np.fabs(diff) <eps).all()
