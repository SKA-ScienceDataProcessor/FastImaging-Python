"""
Convolutional gridding of visibilities.
"""
import logging

import numpy as np
import astropy.units as u
import tqdm
from scipy.special import jn
from scipy.interpolate import interp1d
from scipy import ndimage

from fastimgproto.gridder.wkernel_generation import WKernel
from fastimgproto.gridder.kernel_generation import Kernel, ImgDomKernel
from fastimgproto.gridder import akernel_generation

from fastimgproto.utils import reset_progress_bar

logger = logging.getLogger(__name__)

aproj_interp_rotation = False


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
                     hankel_opt=False,
                     interp_type="linear",
                     undersampling_opt=0,
                     kernel_trunc_perc=1.0,
                     aproj_numtimesteps=0,
                     obs_dec=0.0,
                     obs_ra=0.0,
                     lha=np.ones(1, ),
                     pbeam_coefs=np.array([1]),
                     aproj_opt=False,
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
        aa_support (int): Defines the 'radius' of the bounding box within
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
        hankel_opt (bool): Use Hankel Transform (HT) optimization for quicker
            execution of W-Projection.
        interp_type (string): Interpolation method (use "linear" or "cubic").
        undersampling_opt (int): Use W-kernel undersampling for faster kernel
            generation. Set 0 to disable undersampling and 1 to enable maximum
            undersampling. Reduce the level of undersampling by increasing the
            integer value.
        kernel_trunc_perc (float): Percentage of maximum amplitude from which
            convolution kernel is truncated.
        aproj_numtimesteps (int): Number of time steps used for A-projection.
            Set zero to disable A-projection.
        obs_dec (float): Declination of observation pointing centre (in degrees)
        obs_ra (float): Right Ascension of observation pointing centre (in degrees)
        lha (numpy.ndarray): Local hour angle of visibilities.
            1d array, shape: `(n_vis,)`.
        pbeam_coefs (numpy.ndarray): Primary beam given by spherical harmonics coefficients.
            The SH degree is constant being derived from the number of coefficients minus one.
        aproj_opt (bool): Use A-projection optimisation which rotates the convolution
            kernel rather than the A-kernel.
        raise_bounds (bool): Raise an exception if any of the UV samples lie
            outside (or too close to the edge) of the grid.
        progress_bar (tqdm.tqdm): [Optional] progressbar to update.

    Returns:
        tuple: (vis_grid, sampling_grid, lha_planes)
            Tuple of ndarrays representing the gridded visibilities, the
            sampling weights, and the list of LHA planes (when A-projection
            is used).
            The first two are 2d arrays of same dtype as **vis**,
            shape ``(image_size, image_size)``.
            Note numpy style index-order, i.e. access like ``vis_grid[v,u]``.
            The last one is 1d array os shape `(aproj_numtimesteps,)`

    """
    if oversampling is None:
        oversampling = 1

    if num_wplanes is None:
        num_wplanes = 0

    if num_wplanes > 0:
        use_wproj = True
        assert exact is False
    else:
        use_wproj = False

    if aproj_numtimesteps > 0:
        use_aproj = True
        assert exact is False
        assert use_wproj is True
        assert hankel_opt is False
    else:
        use_aproj = False

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
    if use_wproj is True:
        sort_arg = np.abs(w_lambda).argsort()
        w_lambda = w_lambda[sort_arg]
        uv = uv[sort_arg]
        vis = vis[sort_arg]
        vis_weights = vis_weights[sort_arg]
        if use_aproj is True:
            lha = lha[sort_arg]

    # Calculate nearest integer pixel co-ords ('rounded positions')
    uv_rounded = np.around(uv)
    # Calculate sub-pixel vector from rounded-to-precise positions
    # ('fractional coords'):
    uv_frac = uv - uv_rounded
    uv_rounded_int = uv_rounded.astype(np.int)
    # Now get the corresponding grid-pixel indices by adding the origin offset
    kernel_centre_on_grid = uv_rounded_int + (image_size // 2, image_size // 2)

    # Set convolution kernel support for gridding
    if use_wproj is True:
        conv_support = max_wpconv_support
    else:
        conv_support = aa_support

    assert((conv_support * 2 + 1) < image_size)

    # Check if any of our kernel placements will overlap / lie outside the
    # grid edge.
    good_vis_idx = _bounds_check_kernel_centre_locations(
        uv, kernel_centre_on_grid,
        support=conv_support, image_size=image_size,
        raise_if_bad=raise_bounds)

    # Compute W-Planes and image-domain AA-kernels for W-Projection
    if use_wproj is True:

        w_avg_values, w_planes_gvidx = compute_wplanes(good_vis_idx, num_wplanes, w_lambda, wplanes_median)

        # Update number of planes
        num_wplanes = len(w_avg_values)

        # Calculate kernel working area size
        undersampling_ratio = 1
        workarea_size = image_size

        # Kernel undersampling optimization for speedup
        if undersampling_opt > 0:
            while (image_size // (undersampling_ratio * 2 * undersampling_opt)) > (max_wpconv_support * 2 + 1):
                undersampling_ratio *= 2
            # Check required conditions for the used size
            assert (workarea_size % undersampling_ratio) == 0
            assert ((workarea_size // undersampling_ratio) % 2) == 0
            # Compute workarea size
            workarea_size = image_size // undersampling_ratio

        if hankel_opt is True:
            radial_line = True
            if undersampling_ratio > 1:
                undersampling_ratio = undersampling_ratio // 2
                workarea_size = workarea_size * 2
        else:
            radial_line = False

        # Compute image-domain AA kernel
        aa_kernel_img_array = ImgDomKernel(kernel_func, workarea_size, normalize=False, radial_line=radial_line,
                                           analytic_gcf=analytic_gcf).array

    else:
        # Set 1 plane when W-projection is False (just to enter in gridding loop. W-projection is not performed)
        num_wplanes = 1
        w_planes_gvidx = [0, len(good_vis_idx)]


    # Compute A-Projection time intervals
    lha_planes = []
    if use_aproj is True:
        min_time = np.min(lha)
        max_time = np.max(lha)
        num_timesteps = aproj_numtimesteps
        tstep_size = (max_time-min_time)/num_timesteps
        time_intervals = np.linspace(min_time-tstep_size/2, max_time+tstep_size/2, aproj_numtimesteps+1)
        # Compute average lha value for each interval
        vis_timestep = np.zeros_like(vis, dtype=int)
        for ts in range(num_timesteps):
            # Get visibilities within the current time range
            targs = np.where(np.logical_and(lha[good_vis_idx] >= time_intervals[ts],
                                            lha[good_vis_idx] < time_intervals[ts + 1]))
            if np.size(targs) > 0:
                lha_planes.append(np.mean(lha[good_vis_idx[targs]]))
                vis_timestep[good_vis_idx[targs]] = ts
            else:
                lha_planes.append(0.0)

        # If matrix rotation is enabled, we generate A-kernel once before the gridding procedure
        if aproj_opt or aproj_interp_rotation is True:
            fov = image_size * cell_size.to(u.rad).value
            pbeam = akernel_generation.generate_akernel(pbeam_coefs, fov, workarea_size)
    else:
        # Set 1 time step when A-projection is False (just to enter in gridding loop. A-projection is not performed)
        num_timesteps = 1

    # Not exact gridding
    if not exact:
        # Compute oversampled AA kernel cache
        kernel_cache = populate_kernel_cache(kernel_func, aa_support, oversampling)
        oversampled_offset = calculate_oversampled_kernel_indices(uv_frac, oversampling)

    # Create gridding arrays
    vis_grid = np.zeros((image_size, image_size), dtype=np.complex)
    # At the same time as we grid the visibilities, we track the grid-sampling weights:
    sampling_grid = np.zeros_like(vis_grid)
    # We will compose the sample grid of floats or complex to match input dtype:
    typed_one = np.array(1, dtype=np.complex)

    logger.debug("Gridding {} visibilities".format(len(good_vis_idx)))
    if progress_bar is not None:
        reset_progress_bar(progress_bar, len(good_vis_idx), 'Gridding visibilities')

    # Iterate through each w-plane
    for wplane in range(num_wplanes):
        w_gvlow = w_planes_gvidx[wplane]
        w_gvhigh = w_planes_gvidx[wplane + 1]

        if use_wproj is True:
            # Generate W-kernel
            w_kernel = WKernel(w_value=w_avg_values[wplane], array_size=workarea_size,
                               cell_size=cell_size.to(u.rad).value, undersampling=undersampling_ratio,
                               radial_line=radial_line)
            # If aproj-optimisation is set then generate the oversampled convolution kernel using unrotated a-kernel
            if use_aproj and aproj_opt is True:
                oversampled_conv_kernel, conv_support = \
                    generate_oversampled_convolution_kernel(w_kernel, aa_kernel_img_array, workarea_size,
                                                            max_wpconv_support, oversampling, pbeam, hankel_opt,
                                                            interp_type, kernel_trunc_perc)
        # Iterate through each time step
        for ts in range(num_timesteps):
            if use_aproj is True:
                # Generate the AW-kernels
                if aproj_opt is True:
                    # Due to aproj-optimisation we just need to rotate the convolution kernel for each LHA
                    kernel_cache = populate_kernel_cache_awprojection(oversampled_conv_kernel, conv_support,
                                                                      oversampling, True, lha_planes[ts],
                                                                      np.deg2rad(obs_dec), np.deg2rad(obs_ra))
                else:
                    # Non-optimised A-projection: rotate a-kernel before multiplying by w- and aa-kernels
                    if aproj_interp_rotation is True:
                        a_kernel = akernel_generation.rotate_akernel_by_lha(pbeam, lha_planes[ts], np.deg2rad(obs_dec),
                                                                       np.deg2rad(obs_ra))
                    else:
                        fov = image_size * cell_size.to(u.rad).value
                        a_kernel = akernel_generation.generate_akernel_from_lha(pbeam_coefs, lha_planes[ts], np.deg2rad(obs_dec),
                                                                         np.deg2rad(obs_ra), fov, workarea_size)

                    # Multiply a-, w- and aa-kernels, and determine FFT
                    oversampled_conv_kernel, conv_support = \
                        generate_oversampled_convolution_kernel(w_kernel, aa_kernel_img_array, workarea_size,
                                                                max_wpconv_support, oversampling, a_kernel,
                                                                hankel_opt, interp_type, kernel_trunc_perc)
                    # Generate kernel cache from oversampled convolution kernel
                    kernel_cache = populate_kernel_cache_awprojection(oversampled_conv_kernel, conv_support,
                                                                      oversampling, False)
            else:
                if use_wproj is True:
                    # Multiply w- and aa-kernels, and determine FFT
                    oversampled_conv_kernel, conv_support = \
                        generate_oversampled_convolution_kernel(w_kernel, aa_kernel_img_array, workarea_size,
                                                                max_wpconv_support, oversampling, None,
                                                                hankel_opt, interp_type, kernel_trunc_perc)
                    # Generate kernel cache from oversampled convolution kernel
                    kernel_cache = populate_kernel_cache_awprojection(oversampled_conv_kernel, conv_support,
                                                                      oversampling, False)

            # Iterate through each visibility
            for idx in good_vis_idx[w_gvlow:w_gvhigh]:
                weight = vis_weights[idx]
                # Skip this visibility if zero-weighted
                if weight == 0.:
                    continue

                # Skip this visibility if using A-projection and it does not belong to the time interval
                if use_aproj is True:
                    if not vis_timestep[idx] == ts:
                        continue

                # Integer positions of the kernel
                gc_x, gc_y = kernel_centre_on_grid[idx]
                if exact:
                    normed_kernel_array = Kernel(kernel_func=kernel_func, support=aa_support, offset=uv_frac[idx],
                                                 normalize=True).array
                else:
                    if use_wproj is True:
                        normed_kernel_array = kernel_cache[tuple(oversampled_offset[idx])]
                        # If w is negative, we must use the conjugate kernel
                        if w_lambda[idx] < 0.0:
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

    return vis_grid, sampling_grid, lha_planes


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
            (kernel_centre_indices[:, 0] - support <= 0)
            | (kernel_centre_indices[:, 1] - support <= 0)
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


def construct_dht_matrix(N, r, k):
    rn = np.concatenate(((r[1:(N)] + r[0:(N - 1)]) / 2, [r[N - 1]]))
    kn = np.empty_like(k)
    kn[1:] = 2 * np.pi / k[1:]
    kn[0] = 0
    I = np.outer(kn, rn) * jn(1, np.outer(k, rn))
    I[0, :] = np.pi * rn * rn
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


def generate_oversampled_convolution_kernel(w_kernel, aa_kernel_img, workarea_size, conv_support, oversampling,
                                            a_kernel=None, hankel_opt=False, interp_type="linear",
                                            kernel_trunc_perc=1.0):
    """
    Generate convolution kernel at oversampled-pixel offsets for A- and W-Projection.

    We need kernels for offsets of up to ``oversampling//2`` oversampling-pixels
    in any direction, in steps of one oversampling-pixel
    (i.e. steps of width ``1/oversampling`` in the original co-ordinate system).

    Args:
        w_kernel (numpy.ndarray): Sampled W-kernel function.
        aa_kernel_img (numpy.ndarray): Sampled image-domain anti-aliasing kernel.
        workarea_size (int): Workarea size for kernel generation.
        conv_support (int): Convolution kernel support size.
        oversampling (int): Oversampling ratio.
        a_kernel (numpy.ndarray): Sampled A-kernel function (optional).
        hankel_opt (bool): Use hankel transform optimisation.
        interp_type (string): Interpolation method (use "linear" or "cubic").
        kernel_trunc_perc (float): Percentage of maximum amplitude from which
            convolution kernel is truncated.

    Returns:
        dict: Dictionary mapping oversampling-pixel offsets to normalised gridding
            kernels for the W-plane associated to the input W-kernel.
            cache_size = ((oversampling // 2 * 2) + 1)**2
    """
    if oversampling > 1:
        assert (oversampling % 2) == 0
    if a_kernel is not None:
        assert hankel_opt is False

    assert (conv_support * 2 + 1) < workarea_size

    wkernel_size = workarea_size
    workarea_size = workarea_size * oversampling
    workarea_centre = workarea_size // 2
    trunc_conv_sup = conv_support

    if hankel_opt is False:
        kernel_centre = workarea_centre
        comb_kernel_img_array = np.zeros((workarea_size, workarea_size), dtype=np.complex)
        offset = wkernel_size*(oversampling-1)//2
        if a_kernel is not None:
            # Multiply AA, W and A kernels on image domain
            comb_kernel_img_array[offset:(offset + wkernel_size), offset:(offset + wkernel_size)] = \
                w_kernel.array * aa_kernel_img * a_kernel
        else:
            # Multiply AA and W kernels on image domain
            comb_kernel_img_array[offset:(offset + wkernel_size), offset:(offset + wkernel_size)] = \
                w_kernel.array * aa_kernel_img

        # FFT - transform kernel to UV domain
        comb_kernel_array = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(comb_kernel_img_array)))

        # Find truncate position
        if kernel_trunc_perc > 0.0:
            min_value = np.abs(comb_kernel_array[kernel_centre, kernel_centre]) * kernel_trunc_perc / 100.0
            for pos in reversed(range(1, conv_support + 1)):
                if np.abs(comb_kernel_array[kernel_centre, kernel_centre + pos * oversampling]) > min_value:
                    break
            trunc_conv_sup = pos

    else:
        # Alternative method using Hankel transform #
        comb_kernel_img_radius = np.zeros(workarea_centre, dtype=np.complex)
        comb_kernel_img_radius[0:wkernel_size//2] = w_kernel.array * aa_kernel_img
        comb_kernel_radius = dht(comb_kernel_img_radius)

        # Find truncate position
        if kernel_trunc_perc > 0.0:
            min_value = np.abs(comb_kernel_radius[0]) * kernel_trunc_perc / 100.0
            for pos in reversed(range(1, conv_support + 1)):
                if np.abs(comb_kernel_radius[pos * oversampling]) > min_value:
                    break
            trunc_conv_sup = min(int(np.ceil(pos * np.sqrt(2) / 2)), conv_support)

        # Interpolate
        max_kernel_size = (trunc_conv_sup * 2 + 1 + 1) * oversampling
        kernel_centre = max_kernel_size // 2
        x, y = np.meshgrid(range(max_kernel_size), range(max_kernel_size))
        r = np.sqrt((x - kernel_centre) ** 2 + (y - kernel_centre) ** 2)
        f = interp1d(np.arange(0, workarea_centre / (np.sqrt(2)), 1 / np.sqrt(2)),
                     comb_kernel_radius, copy=False, kind=interp_type, bounds_error=False, fill_value=0.0,
                     assume_sorted=True)
        comb_kernel_array = f(r.flat).reshape(r.shape)

    assert trunc_conv_sup <= conv_support

    return comb_kernel_array, trunc_conv_sup


def populate_kernel_cache_awprojection(oversampled_conv_kernel, trunc_conv_sup, oversampling, rotate=False,
                                       lha=0.0, dec_rad=0.0, ra_rad=0.0):
    """
    Generate a cache of normalised kernels at oversampled-pixel offsets for A- and W-Projection

    We need kernels for offsets of up to ``oversampling//2`` oversampling-pixels
    in any direction, in steps of one oversampling-pixel
    (i.e. steps of width ``1/oversampling`` in the original co-ordinate system).

    Args:
        w_kernel (numpy.ndarray): Sampled W-kernel function.
        aa_kernel_img (numpy.ndarray): Sampled image-domain anti-aliasing kernel.
        workarea_size (int): Workarea size for kernel generation.
        conv_support (int): Convolution kernel support size.
        oversampling (int): Oversampling ratio.
        a_kernel (numpy.ndarray): Sampled A-kernel function.
        hankel_opt (bool): Use hankel transform optimisation.
        interp_type (string): Interpolation method (use "linear" or "cubic").
        kernel_trunc_perc (float): Percentage of maximum amplitude from which
            convolution kernel is truncated.

    Returns:
        dict: Dictionary mapping oversampling-pixel offsets to normalised gridding
            kernels for the W-plane associated to the input W-kernel.
            cache_size = ((oversampling // 2 * 2) + 1)**2
    """
    if oversampling > 1:
        assert (oversampling % 2) == 0

    kernel_centre = oversampled_conv_kernel.shape[0] // 2
    kernel_cache = dict()
    cache_size = (oversampling // 2 * 2) + 1
    oversampled_pixel_offsets = np.arange(cache_size) - oversampling // 2

    if rotate is True:
        # Determine parallactic angle (in radians)
        pangle = akernel_generation.parallatic_angle(lha, dec_rad, ra_rad)

        min_val = oversampled_conv_kernel[0, 0]
        rot_oversampled_conv_kernel = np.empty_like(oversampled_conv_kernel)
        rot_oversampled_conv_kernel.real = ndimage.interpolation.rotate(
            oversampled_conv_kernel.real, np.rad2deg(pangle), reshape=False, mode='constant', cval=min_val, order=1)
        rot_oversampled_conv_kernel.imag = ndimage.interpolation.rotate(
            oversampled_conv_kernel.imag, np.rad2deg(pangle), reshape=False, mode='constant', cval=min_val, order=1)
    else:
        rot_oversampled_conv_kernel = oversampled_conv_kernel

    # Generate kernel cache
    for x_step in oversampled_pixel_offsets:
        for y_step in oversampled_pixel_offsets:
            xrange = slice(kernel_centre - trunc_conv_sup * oversampling - x_step,
                           kernel_centre + trunc_conv_sup * oversampling - x_step + 1, oversampling)
            yrange = slice(kernel_centre - trunc_conv_sup * oversampling - y_step,
                           kernel_centre + trunc_conv_sup * oversampling - y_step + 1, oversampling)
            conv_kernel_array = rot_oversampled_conv_kernel[yrange, xrange]
            array_sum = np.real(conv_kernel_array.sum())
            conv_kernel_array = conv_kernel_array / array_sum

            # Store kernel on cache
            kernel_cache[(x_step, y_step)] = conv_kernel_array

    return kernel_cache


def compute_wplanes(good_vis_idx, num_wplanes, w_lambda, wplanes_median):
    """
    Determine central W-plane values for wide field imaging

    Args:
        good_vis_idx (numpy.array): List of good visibility indexes.
        num_wplanes (int): Number of W-planes to compute.
        w_lambda (numpy.array): W-lambda values for each visibility.
        wplanes_median (bool): If true, determine W-plane values using the median rather than mean.

    Returns:
        tuple: (w_avg_values, w_planes_gvidx)
            Tuple of ndarrays representing the determined W-planes and the index of the first visibility assigned to
            the plane.
    """
    num_gvis = len(good_vis_idx)

    if num_gvis < num_wplanes:
        num_wplanes = num_gvis

    # Define w-planes
    plane_size = np.ceil(num_gvis / num_wplanes)

    w_avg_values = []
    w_planes_gvidx = []

    for idx in range(0, num_wplanes):
        begin = int(idx * plane_size)
        end = int(min((idx + 1) * plane_size, num_gvis))
        indexes = good_vis_idx[begin:end]
        if wplanes_median is True:
            w_avg_values.append(np.median(np.abs(w_lambda[indexes])))
        else:
            w_avg_values.append(np.average(np.abs(w_lambda[indexes])))
        w_planes_gvidx.append(begin)

        if end >= num_gvis:
            break

    w_planes_gvidx.append(num_gvis)

    return w_avg_values, w_planes_gvidx
