"""
Convolutional gridding of visibilities.
"""
import logging

import numpy as np
import tqdm

from fastimgproto.gridder.kernel_generation import Kernel

logger = logging.getLogger(__name__)


def convolve_to_grid(kernel_func, support,
                     image_size,
                     uv, vis,
                     exact=True,
                     oversampling=0,
                     raise_bounds=True,
                     pbar=None):
    """
    Grid visibilities, calculating the exact kernel distribution for each.

    If ``exact == True`` then exact gridding is used, i.e. the kernel is
    recalculated for each visibility, with precise sub-pixel offset according to
    that visibility's UV co-ordinates. Otherwise, instead of recalculating the
    kernel for each sub-pixel location, we pre-generate an oversampled kernel
    ahead of time - so e.g. for an oversampling of 5, the kernel is
    pre-generated at 0.2 pixel-width offsets. We then pick the pre-generated
    kernel corresponding to the sub-pixel offset nearest to that of the
    visibility.

    Kernel pre-generation results in improved performance, particularly with
    large numbers of visibilities and complex kernel functions, at the cost of
    introducing minor aliasing effects due to the 'step-like' nature of the
    oversampled kernel. This in turn can be minimised (at the cost of longer
    start-up times and larger memory usage) by pre-generating kernels with a
    larger oversampling ratio, to give finer interpolation.


    Args:
        kernel_func (callable): Callable object,
            (e.g. :class:`.conv_funcs.Pillbox`,)
            that returns a convolution
            co-efficient for a given distance in pixel-widths.
        support (int): Defines the 'radius' of the bounding box within
            which convolution takes place. `Box width in pixels = 2*support+1`.
            (The central pixel is the one nearest to the UV co-ordinates.)
            (This is sometimes known as the 'half-support')
        image_size (int): Width of the image in pixels. NB we assume
            the pixel `[image_size//2,image_size//2]` corresponds to the origin
            in UV-space.
        uv (numpy.ndarray): UV-coordinates of visibilities.
            2d array of `float_`, shape: `(n_vis, 2)`.
            assumed ordering is u-then-v, i.e. `u, v = uv[idx]`
        vis (numpy.ndarray): Complex visibilities.
            1d array, shape: `(n_vis,)`.
        exact (bool): Calculate exact kernel-values for every UV-sample.
        oversampling (int): Controls kernel-generation if ``exact==False``.
            Larger values give a finer-sampled set of pre-cached kernels.
        raise_bounds (bool): Raise an exception if any of the UV
            samples lie outside (or too close to the edge) of the grid.
        pbar (tqdm.tqdm): [Optional] progressbar to update.

    Returns:
        tuple: (vis_grid, sampling_grid)
            Tuple of ndarrays representing the gridded visibilities and the
            sampling weights.
            These are 2d arrays of same dtype as **vis**,
            shape ``(image_size, image_size)``.
            Note numpy style index-order, i.e. access like ``vis_grid[v,u]``.

    """
    assert len(uv) == len(vis)
    # Check for sensible combinations of exact / oversampling parameter-values:
    if not exact:
        assert oversampling >= 1

    # Calculate nearest integer pixel co-ords ('rounded positions')
    uv_rounded = np.around(uv)
    # Calculate sub-pixel vector from rounded-to-precise positions
    # ('fractional coords'):
    uv_frac = uv - uv_rounded
    uv_rounded_int = uv_rounded.astype(np.int)
    # Now get the corresponding grid-pixel indices by adding the origin offset
    kernel_centre_on_grid = uv_rounded_int + (image_size // 2, image_size // 2)

    # Check if any of our kernel placements will overlap / lie outside the
    # grid edge.
    good_vis_idx = _bounds_check_kernel_centre_locations(
        uv, kernel_centre_on_grid,
        support=support, image_size=image_size,
        raise_if_bad=raise_bounds)

    vis_grid = np.zeros((image_size, image_size), dtype=vis.dtype)
    # At the same time as we grid the visibilities, we track the grid-sampling
    # weights:
    sampling_grid = np.zeros_like(vis_grid)
    # Use either `1.0` or `1.0 +0j` depending on input dtype:
    typed_one = np.array(1, dtype=vis.dtype)

    if not exact:
        kernel_cache = populate_kernel_cache(
            kernel_func, support, oversampling)
        oversampled_offset = calculate_oversampled_kernel_indices(
            uv_frac, oversampling)
    logger.debug("Gridding {} visibilities".format(len(good_vis_idx)))
    if pbar is not None:
        pbar.total = len(good_vis_idx)
        pbar.n = 0
        pbar.set_description('Gridding visibilities')
    for idx in good_vis_idx:
        gc_x, gc_y = kernel_centre_on_grid[idx]
        # Generate a convolution kernel with the precise offset required:
        xrange = slice(gc_x - support, gc_x + support + 1)
        yrange = slice(gc_y - support, gc_y + support + 1)
        if exact:
            kernel = Kernel(kernel_func=kernel_func, support=support,
                            offset=uv_frac[idx])
            normed_kernel_array = kernel.array
        else:
            normed_kernel_array = kernel_cache[
                tuple(oversampled_offset[idx])].array

        vis_grid[yrange, xrange] += vis[idx] * normed_kernel_array
        sampling_grid[yrange, xrange] += typed_one * normed_kernel_array
        if pbar is not None:
            pbar.update(1)
    return vis_grid, sampling_grid


def _bounds_check_kernel_centre_locations(uv, kernel_centre_indices,
                                          support, image_size,
                                          raise_if_bad):
    """
    Vectorized bounds check, returns index of good positions in the uv array.

    Check if kernel over-runs the image boundary for any of the chosen central
    pixels

    Args:
        uv (numpy.ndarray): Array of uv co-ordinates
        kernel_centre_indices(numpy.ndarray): Corresponding array of
            nearest-pixel grid-locations, which will be the centre position
            of a kernel placement.
        support (int): Kernel support size in regular pixels.
        image_size (int): Image width in pixels
        raise_if_bad (bool): If true, throw a ValueError if any bad locations
            are found, otherwise just log a warning message.

    Return:
        list: List of indices for 'good' (in-bounds) positions. Note this is
        a list of integer index values, of length `n_good_positions`.
        (Not to be confused with a boolean mask of length `n_vis`).
    """

    out_of_bounds_bool = (
        (kernel_centre_indices[:, 0] - support < 0)
        | (kernel_centre_indices[:, 1] - support < 0)
        | (kernel_centre_indices[:, 0] + support >= image_size)
        | (kernel_centre_indices[:, 1] + support >= image_size)
    )
    out_of_bounds_idx = np.nonzero(out_of_bounds_bool)[0]
    good_vis_idx = np.nonzero(np.invert(out_of_bounds_bool))[0]

    if out_of_bounds_bool.any():
        bad_uv = uv[out_of_bounds_idx]
        msg = "{} UV locations are out-of-grid or too close to edge:{}".format(
            len(bad_uv), bad_uv)
        if raise_if_bad:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    return good_vis_idx


def calculate_oversampled_kernel_indices(subpixel_coord, oversampling):
    """
    Find the nearest oversampled gridpoint for given sub-pixel offset.

    Effectively we are mapping the range ``[-0.5, 0.5]`` to the integer range
    ``[-oversampling//2, ..., oversampling//2]``.

    Inputs will be between -0.5 and 0.5 inclusive. This is an issue,
    because inputs at the extreme (e.g. 0.5) might round *UP*, taking them
    outside the desired integer output range. We simply correct this edge-case
    by replacing outlier values before returning.


    Args:
        subpixel_coord (numpy.ndarray): Array of 'fractional' co-ords, that is the
            subpixel offsets from nearest pixel on the regular grid.
            dtype: float, shape: `(n_vis, 2)`.
        oversampling (int): How many oversampled pixels to one regular pixel.
    Returns:
        numpy.ndarray: Corresponding oversampled pixel indexes. These are in oversampled pixel
        widths from the kernel centre pixel, to a maximum of half a regular
        pixel, so they have integer values ranging  from ``-oversampling/2`` to
        ``oversampling/2``. [Dtype: ``int``, shape: ``(n_vis, 2)``].
    """
    subpixel_coord = np.atleast_1d(subpixel_coord)
    assert (-0.5 <= subpixel_coord).all()
    assert (subpixel_coord <= 0.5).all()
    oversampled_k_idx = np.around(subpixel_coord * oversampling).astype(int)
    range_max = oversampling // 2
    range_min = -1 * range_max
    oversampled_k_idx[oversampled_k_idx == (range_max + 1)] = range_max
    oversampled_k_idx[oversampled_k_idx == (range_min - 1)] = range_min
    return oversampled_k_idx


def populate_kernel_cache(kernel_func, support, oversampling):
    """
    Generate a cache of normalised kernels at oversampled-pixel offsets.

    We need kernels for offsets of up to ``oversampling//2`` oversampling-pixels
    in any direction, in steps of one oversampling-pixel
    (i.e. steps of width ``1/oversampling`` in the original co-ordinate system).

    Args:
        kernel_func (callable): Callable object,
            (e.g. :class:`.conv_funcs.Pillbox`,)
            that returns a convolution
            co-efficient for a given distance in pixel-widths.
        support (int): See kernel generation routine.
        oversampling (int): Oversampling ratio.
            cache_size = ((oversampling // 2 * 2) + 1)**2

    Returns:
        dict: Dictionary mapping oversampling-pixel offsets to normalised kernels.
    """
    # We use floordiv and multiply to give sensible results for both odd / even
    # oversampling values:
    cache_size = (oversampling // 2 * 2) + 1
    oversampled_pixel_offsets = np.arange(cache_size) - oversampling // 2
    cache = dict()
    for x_step in oversampled_pixel_offsets:
        for y_step in oversampled_pixel_offsets:
            subpixel_offset = (np.array((x_step, y_step),
                                        dtype=np.float_) / oversampling)
            kernel = Kernel(kernel_func=kernel_func, support=support,
                            offset=subpixel_offset,
                            oversampling=None,
                            normalize=True
                            )
            cache[(x_step, y_step)] = kernel

    return cache
