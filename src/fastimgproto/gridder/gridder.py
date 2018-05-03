"""
Convolutional gridding of visibilities.
"""
import logging

import numpy as np
import tqdm
from scipy.special import jn
from scipy.interpolate import interp1d

from fastimgproto.gridder.wkernel_generation import WKernel
from fastimgproto.gridder.kernel_generation import Kernel, ImgDomKernel
from fastimgproto.utils import reset_progress_bar

logger = logging.getLogger(__name__)


def construct_dht_matrix(N, r, k):
    rn = np.concatenate(((r[1:(N)] + r[0:(N-1)]) / 2, [r[N-1]]))
    kn = np.empty_like(k)
    kn[1:] = 2 * np.pi / k[1:]
    kn[0] = 0
    I = np.outer(kn, rn) * jn(1, np.outer(k, rn))
    I[0, :] = np.pi*rn*rn
    I[:, 1:N] = I[:, 1:N] - I[:, 0:(N - 1)]
    return I


def dht(f):
    """
    Hankel Transform of order 0.
    Args:
        f (numpy.ndarray): Input vector be transformed.
    Returns:
        numpy.ndarray: Hankel transform output array.
    """
    N = f.shape[0]
    r = np.arange(0, N)
    k = (np.pi / N) * r
    I = construct_dht_matrix(N, r, k)
    return np.tensordot(I, f, axes=([1], [0]))


def convolve_to_grid(kernel_func,
                     aa_support,
                     image_size,
                     uv,
                     vis,
                     vis_weights,
                     exact=True,
                     oversampling=None,
                     num_wplanes=None,
                     cell_size=None,
                     w_lambda=None,
                     wplanes_median=False,
                     max_wpconv_support=0,
                     analytic_gcf=False,
                     hankel_opt=0,
                     undersampling_opt=0,
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
        uv (numpy.ndarray): UV-coordinates of visibilities.
            2d array of `float_`, shape: `(n_vis, 2)`.
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
        cell_size (astropy.units.Quantity): Angular-width of a synthesized pixel
            in the image to be created, e.g. ``3.5 * u.arcsecond``.
        w_lambda (numpy.ndarray): W-coordinate of visibilities. Units are
            multiples of wavelength. 1d array of `float_`, shape: `(n_vis, 1)`.
        wplanes_median (bool): Use median to compute w-planes, otherwise use mean.
        max_wpconv_support (int): Defines the maximum 'radius' of the bounding box
            within which convolution takes place when W-Projection is used.
            `Box width in pixels = 2*support+1`.
        analytic_gcf (bool): Compute approximation of image-domain kernel from
            analytic expression of DFT.
        hankel_opt (int): Use Hankel Transform (HT) optimization for quicker
            execution of W-Projection. Set 0 to disable HT and 1 or 2 to enable HT.
            The larger non-zero value increases HT accuracy, by using an extended
            W-kernel workarea size.
        undersampling_opt (int): Use W-kernel undersampling for faster kernel
            generation. Set 0 to disable undersampling and 1 to enable maximum
            undersampling. Reduce the level of undersampling by increasing the
            integer value.
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
    if oversampling is None:
        oversampling = 1
    if num_wplanes is None:
        num_wplanes = 0
    if num_wplanes > 0:
        assert exact is False
    if w_lambda is not None:
        assert len(w_lambda) == len(vis)
    assert len(uv) == len(vis)

    # Check for sensible combinations of exact / oversampling parameter-values:
    if not exact:
        assert oversampling >= 1
        # We use floordiv and multiply to convert to an even oversampling value
        if oversampling > 2:
            oversampling = (oversampling // 2) * 2

    # Sort uvw lambda data by increasing absolute value of w-lambda
    if num_wplanes > 0:
        sort_arg = np.abs(w_lambda).argsort()
        w_lambda = w_lambda[sort_arg]
        uv = uv[sort_arg]
        vis = vis[sort_arg]
        vis_weights = vis_weights[sort_arg]

    # Calculate nearest integer pixel co-ords ('rounded positions')
    uv_rounded = np.around(uv)
    # Calculate sub-pixel vector from rounded-to-precise positions
    # ('fractional coords'):
    uv_frac = uv - uv_rounded
    uv_rounded_int = uv_rounded.astype(np.int)
    # Now get the corresponding grid-pixel indices by adding the origin offset
    kernel_centre_on_grid = uv_rounded_int + (image_size // 2, image_size // 2)

    # Set convolution kernel support for gridding
    if num_wplanes > 0:
        conv_support = max_wpconv_support
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
            w_planes_idx = np.empty_like(w_lambda, dtype=int)

            for idx in range(0, num_wplanes):
                begin = int(idx * plane_size)
                end = int(min((idx + 1) * plane_size, num_gvis))
                indexes = good_vis_idx[begin:end]
                if wplanes_median is True:
                    w_avg_values[idx] = np.median(np.abs(w_lambda[indexes]))
                else:
                    w_avg_values[idx] = np.average(np.abs(w_lambda[indexes]))

                w_planes_idx[indexes] = idx
                if end >= num_gvis:
                    break

            # Calculate kernel working area size
            undersampling_scale = 1
            workarea_size = image_size * oversampling

            # Kernel undersampling optimization for speedup
            if undersampling_opt > 0:
                while (image_size // (undersampling_scale * 2 * undersampling_opt)) > (max_wpconv_support * 2 + 1):
                    undersampling_scale *= 2
                # Check required conditions for the used size
                assert (workarea_size % undersampling_scale) == 0
                assert ((workarea_size // undersampling_scale) % 2) == 0
                # Compute workarea size
                workarea_size = workarea_size // undersampling_scale

            if hankel_opt > 0:
                radial_line = True
                workarea_size = workarea_size * hankel_opt
                if undersampling_scale > 1:
                    undersampling_scale = undersampling_scale // 2
                    workarea_size = workarea_size * 2
            else:
                radial_line = False

            # Compute image-domain AA kernel
            aa_kernel_img_array = ImgDomKernel(kernel_func, workarea_size, oversampling=oversampling,
                                               normalize=False, radial_line=radial_line,
                                               analytic_gcf=analytic_gcf).array

        else:
            # Compute oversampled AA kernel cache
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
                                   oversampling=oversampling, scale=undersampling_scale, normalize=False,
                                   radial_line=radial_line)

                kernel_cache, conv_support = \
                    generate_kernel_cache_wprojection(w_kernel, aa_kernel_img_array, workarea_size,
                                                      max_wpconv_support, oversampling, hankel_opt)

        # Integer positions of the kernel
        gc_x, gc_y = kernel_centre_on_grid[idx]
        if exact:
            normed_kernel_array = Kernel(kernel_func=kernel_func, support=aa_support, offset=uv_frac[idx],
                                         normalize=True).array
        else:
            if num_wplanes > 0:
                normed_kernel_array = kernel_cache[tuple(oversampled_offset[idx])]
                # If w is negative, we must use the conjugate kernel
                if w_lambda[idx] < 0:
                    normed_kernel_array = np.conj(normed_kernel_array)
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


def generate_kernel_cache_wprojection(w_kernel, aa_kernel_img, workarea_size, conv_support, oversampling, hankel_opt=0):
    """
    Generate a cache of kernels at oversampled-pixel offsets for W-Projection.

    We need kernels for offsets of up to ``oversampling//2`` oversampling-pixels
    in any direction, in steps of one oversampling-pixel
    (i.e. steps of width ``1/oversampling`` in the original co-ordinate system).

    Args:
        w_kernel (numpy.ndarray): Sampled W-kernel function.
        aa_kernel_img (numpy.ndarray): Sampled image-domain anti-aliasing kernel.
        workarea_size (int): Workarea size for kernel generation.
        conv_support (int): Convolution kernel support size.
        oversampling (int): Oversampling ratio.
        hankel_opt (int): Use hankel transform optimisation.

    Returns:
        dict: Dictionary mapping oversampling-pixel offsets to normalised gridding
            kernels for the W-plane associated to the input W-kernel.
            cache_size = ((oversampling // 2 * 2) + 1)**2
    """
    if oversampling > 1:
        assert (oversampling % 2) == 0

    kernel_cache = dict()
    workarea_centre = workarea_size // 2
    cache_size = (oversampling // 2 * 2) + 1
    oversampled_pixel_offsets = np.arange(cache_size) - oversampling // 2

    assert (conv_support * 2 + 1) * oversampling < workarea_size

    if hankel_opt == 0:
        kernel_centre = workarea_centre
        # Multiply AA and W kernels on image domain
        comb_kernel_img_array = w_kernel.array * aa_kernel_img
        # FFT - transform kernel to UV domain
        comb_kernel_array = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(comb_kernel_img_array)))
    else:
        ### Alternative method using Hankel transform ###
        comb_kernel_img_radius = w_kernel.array * aa_kernel_img
        comb_kernel_radius = dht(comb_kernel_img_radius)

        max_kernel_size = (conv_support * 2 + 1 + 1) * oversampling
        kernel_centre = max_kernel_size // 2
        x, y = np.meshgrid(range(max_kernel_size + 1), range(max_kernel_size + 1))
        r = np.sqrt((x - kernel_centre) ** 2 + (y - kernel_centre) ** 2)
        f = interp1d(np.arange(0, workarea_centre / (np.sqrt(2) * hankel_opt), 1 / (np.sqrt(2) * hankel_opt)),
                     comb_kernel_radius, copy=False, kind="cubic", bounds_error=False, fill_value=0.0,
                     assume_sorted=True)
        comb_kernel_array = f(r.flat).reshape(r.shape)

    xrange = slice(kernel_centre - conv_support * oversampling,
                   kernel_centre + conv_support * oversampling + 1, oversampling)
    yrange = slice(kernel_centre - conv_support * oversampling,
                   kernel_centre + conv_support * oversampling + 1, oversampling)
    tmp_array = comb_kernel_array[yrange, xrange]

    min_value = np.abs(tmp_array[conv_support, conv_support]) * 0.01
    for pos in range(0, conv_support + 1):
        if np.abs(tmp_array[conv_support, pos]) > min_value:
            break
    trunc_conv_support = max(1, conv_support - pos)
    assert trunc_conv_support <= conv_support

    # Generate kernel cache
    for x_step in oversampled_pixel_offsets:
        for y_step in oversampled_pixel_offsets:
            xrange = slice(kernel_centre - trunc_conv_support * oversampling - x_step,
                           kernel_centre + trunc_conv_support * oversampling - x_step + 1, oversampling)
            yrange = slice(kernel_centre - trunc_conv_support * oversampling - y_step,
                           kernel_centre + trunc_conv_support * oversampling - y_step + 1, oversampling)
            conv_kernel_array = comb_kernel_array[yrange, xrange]
            array_sum = np.real(conv_kernel_array.sum())
            conv_kernel_array = conv_kernel_array / array_sum

            # Store kernel on cache
            kernel_cache[(x_step, y_step)] = conv_kernel_array

    return kernel_cache, trunc_conv_support
