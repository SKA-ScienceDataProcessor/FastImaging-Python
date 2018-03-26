"""
Convolutional gridding of visibilities.
"""
import logging

import numpy as np
import tqdm
from copy import deepcopy

from fastimgproto.gridder.wkernel_generation import WKernel
from fastimgproto.gridder.kernel_generation import Kernel
from fastimgproto.utils import reset_progress_bar

logger = logging.getLogger(__name__)


def convolve_to_grid(kernel_func,
                     aa_support,
                     image_size,
                     cell_size,
                     uvw_lambda,
                     vis,
                     vis_weights,
                     exact=True,
                     oversampling=0,
                     num_wplanes=0,
                     max_conv_support=0,
                     raise_bounds=False,
                     progress_bar=None):
    """
    Grid visibilities using convolutional gridding.

    Returns the **un-normalized** weighted visibilities; the
    weights-renormalization factor can be calculated by summing the sample grid.

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

        uvw (numpy.ndarray): UVW-coordinates of visibilities.
            2d array of `float_`, shape: `(n_vis, 3)`.
            assumed ordering is u-then-v, i.e. `u, v = uv[idx]`
        vis (numpy.ndarray): Complex visibilities.
            1d array, shape: `(n_vis,)`.
        vis_weights (numpy.ndarray): Visibility weights.
            1d array, shape: `(n_vis,)`.
        exact (bool): Calculate exact kernel-values for every UV-sample.
        oversampling (int): Controls kernel-generation if ``exact==False``.
            Larger values give a finer-sampled set of pre-cached kernels.
        num_wplanes (int): Number of planes for W-Projection. Set to zero or
            None to disable W-projection.
        raise_bounds (bool): Raise an exception if any of the UV
            samples lie outside (or too close to the edge) of the grid.
        progress_bar (tqdm.tqdm): [Optional] progressbar to update.

    Returns:
        tuple: (vis_grid, sampling_grid)
            Tuple of ndarrays representing the gridded visibilities and the
            sampling weights.
            These are 2d arrays of same dtype as **vis**,
            shape ``(image_size, image_size)``.
            Note numpy style index-order, i.e. access like ``vis_grid[v,u]``.

    """
    if num_wplanes is None:
        num_wplanes = 0
    if num_wplanes > 0:
        assert exact is False
    assert len(uvw_lambda) == len(vis)

    # Check for sensible combinations of exact / oversampling parameter-values:
    if not exact:
        assert oversampling >= 1
        # We use floordiv and multiply to convert to an even oversampling value
        if oversampling > 2:
            oversampling = (oversampling // 2) * 2

    # Sort uvw lambda data by increasing w-value
    sort_arg = uvw_lambda[:, 2].argsort()
    uvw_lambda = uvw_lambda[sort_arg]
    vis = vis[sort_arg]
    vis_weights = vis_weights[sort_arg]

    # Size of a UV-grid pixel, in multiples of wavelength (lambda):
    uv = uvw_lambda[:, :2]
    w = uvw_lambda[:, 2]
    grid_pixel_width_lambda = 1.0 / (cell_size * image_size)
    uv_in_pixels = (uv / grid_pixel_width_lambda)

    # Calculate nearest integer pixel co-ords ('rounded positions')
    uv_rounded = np.around(uv_in_pixels)
    # Calculate sub-pixel vector from rounded-to-precise positions
    # ('fractional coords'):
    uv_frac = uv_in_pixels - uv_rounded
    uv_rounded_int = uv_rounded.astype(np.int)
    # Now get the corresponding grid-pixel indices by adding the origin offset
    kernel_centre_on_grid = uv_rounded_int + (image_size // 2, image_size // 2)

    # Set convolution kernel support for gridding
    if num_wplanes > 0:
        conv_support = max_conv_support
    else:
        conv_support = aa_support

    # Check if any of our kernel placements will overlap / lie outside the
    # grid edge.
    good_vis_idx = _bounds_check_kernel_centre_locations(
        uv, kernel_centre_on_grid,
        support=conv_support, image_size=image_size,
        raise_if_bad=raise_bounds)

    if not exact:
        oversampled_offset = calculate_oversampled_kernel_indices(
            uv_frac, oversampling)

        # Compute W-Planes and convolution kernels for W-Projection
        if num_wplanes > 0:
            num_gvis = len(good_vis_idx)

            # Define w-planes
            plane_size = np.ceil(num_gvis / num_wplanes)
            w_avg_values = np.empty(num_wplanes)
            w_planes_idx = np.empty_like(w)

            for idx in range(0, num_wplanes):
                begin = int(np.round(idx * plane_size))
                end = int(min(np.round((idx + 1) * plane_size), num_gvis))
                indexes = good_vis_idx[begin:end]
                w_avg_values[idx] = np.average(w[indexes])
                w_planes_idx[indexes] = idx
                if end >= num_gvis:
                    break

            # Calculate kernel working area size
            scale = 1
            if (max_conv_support * 2 + 1) * oversampling >= image_size:
                while (image_size * scale) <= (max_conv_support * 2 + 1) * oversampling:
                    scale *= 2
                workarea_size = image_size * scale
                scale = 1.0 / scale
            else:
                while (image_size // (scale * 2)) > (max_conv_support * 2 + 1) * oversampling:
                    scale *= 2
                workarea_size = image_size // scale

            workarea_centre = workarea_size // 2
            aa_kernel_array = np.zeros((workarea_size, workarea_size), dtype=np.complex)
            # Generate oversampled AA kernel
            aa_kernel_oversampled = Kernel(kernel_func=kernel_func, support=aa_support,
                                           offset=(0, 0),
                                           oversampling=oversampling,
                                           normalize=True
                                           )
            # Copy AA kernel to working area
            aa_support = aa_kernel_oversampled.array_size // 2
            xrange = slice(workarea_centre - aa_support, workarea_centre + aa_support + 1)
            yrange = slice(workarea_centre - aa_support, workarea_centre + aa_support + 1)
            aa_kernel_array[yrange, xrange] = aa_kernel_oversampled.array
            # iFFT - transform AA kernel to image domain
            kernel_img_array = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(aa_kernel_array))))

        else:
            # Compute AA kernel cache
            kernel_cache = populate_kernel_cache(kernel_func, aa_support, oversampling)

    # Create gridding arrays
    vis_grid = np.zeros((image_size, image_size), dtype=np.complex)
    # At the same time as we grid the visibilities, we track the grid-sampling
    # weights:
    sampling_grid = np.zeros_like(vis_grid)
    # We will compose the sample grid of floats or complex to match input dtype:
    typed_one = np.array(1, dtype=np.complex)

    logger.debug("Gridding {} visibilities".format(len(good_vis_idx)))
    if progress_bar is not None:
        reset_progress_bar(progress_bar, len(good_vis_idx), 'Gridding visibilities')

    wplane = -1
    for idx in good_vis_idx:
        weight = vis_weights[idx]
        if weight == 0.:
            continue  # Skip this visibility if zero-weighted

        if num_wplanes > 0:
            # Get wplane idx
            if wplane != w_planes_idx[idx]:
                wplane = w_planes_idx[idx]
                # Generate W-kernel
                w_kernel = WKernel(w_value=w_avg_values[wplane], array_size=workarea_size, cell_size=cell_size,
                                   scale=scale*oversampling)
                kernel_cache = generate_kernel_cache_wprojection(w_kernel, kernel_img_array, conv_support, oversampling)

        # Integer positions of the kernel
        gc_x, gc_y = kernel_centre_on_grid[idx]
        if exact:
            normed_kernel_array = Kernel(kernel_func=kernel_func, support=aa_support, offset=uv_frac[idx],
                                         normalize=True).array
        else:
            normed_kernel_array = kernel_cache[tuple(oversampled_offset[idx])].array

        # Generate a convolution kernel with the precise offset required:
        xrange = slice(gc_x - conv_support, gc_x + conv_support + 1)
        yrange = slice(gc_y - conv_support, gc_y + conv_support + 1)

        downweighted_kernel = weight * normed_kernel_array
        vis_grid[yrange, xrange] += vis[idx] * downweighted_kernel
        sampling_grid[yrange, xrange] += typed_one * downweighted_kernel
        if progress_bar is not None:
            progress_bar.update(1)

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


def generate_kernel_cache_wprojection(w_kernel, aa_kernel_img, conv_support, oversampling):
    """
    Generate a cache of kernels at oversampled-pixel offsets for W-Projection.

    We need kernels for offsets of up to ``oversampling//2`` oversampling-pixels
    in any direction, in steps of one oversampling-pixel
    (i.e. steps of width ``1/oversampling`` in the original co-ordinate system).

    Args:
        w_kernel (numpy.ndarray): Sampled W-kernel function.
        aa_kernel_img (numpy.ndarray): Sampled image-domain anti-aliasing kernel.
        conv_support (int): Convolution kernel support size.
        oversampling (int): Oversampling ratio.

    Returns:
        dict: Dictionary mapping oversampling-pixel offsets to normalised gridding
            kernels for the W-plane associated to the input W-kernel.
            cache_size = ((oversampling // 2 * 2) + 1)**2
    """
    kernel_cache = dict()
    workarea_size = w_kernel.array_size
    workarea_centre = workarea_size // 2
    cache_size = (oversampling // 2 * 2) + 1
    oversampled_pixel_offsets = np.arange(cache_size) - oversampling // 2
    # Multiply AA and W kernels on image domain
    comb_kernel_img_array = w_kernel.array * aa_kernel_img
    # FFT - transform kernel to UV domain
    comb_kernel_array = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(comb_kernel_img_array)))
    assert (conv_support * 2 + 1) * oversampling < workarea_size

    # Generate kernel cache
    for x_step in oversampled_pixel_offsets:
        for y_step in oversampled_pixel_offsets:
            xrange = slice(workarea_centre - conv_support * oversampling - x_step,
                           workarea_centre + conv_support * oversampling - x_step + 1, oversampling)
            yrange = slice(workarea_centre - conv_support * oversampling - y_step,
                           workarea_centre + conv_support * oversampling - y_step + 1, oversampling)
            w_kernel.array = comb_kernel_array[yrange, xrange]
            # Store kernel on cache
            kernel_cache[(x_step, y_step)] = deepcopy(w_kernel)

    return kernel_cache
