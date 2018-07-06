import astropy.units as u
import numpy as np

from fastimgproto.gridder.gridder import convolve_to_grid
from fastimgproto.gridder.kernel_generation import ImgDomKernel


def fft_to_image_plane(uv_grid):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid)))


def image_visibilities(
        vis,
        vis_weights,
        uvw_lambda,
        image_size,
        cell_size,
        kernel_func,
        kernel_support,
        kernel_exact=True,
        kernel_oversampling=None,
        num_wplanes=0,
        wplanes_median=False,
        max_wpconv_support=0,
        analytic_gcf=False,
        hankel_opt=False,
        interp_type="linear",
        undersampling_opt=0,
        kernel_trunc_perc=1.0,
        aproj_numtimesteps=0,
        obs_dec=0.0,
        obs_lat=0.0,
        lha=np.ones(1,),
        mueller_term=np.ones((1, 1)),
        progress_bar=None):
    """
    Args:
        vis (numpy.ndarray): Complex visibilities.
            1d array, shape: `(n_vis,)`.
        vis_weights (numpy.ndarray): Visibility weights.
            1d array, shape: `(n_vis,)`.
        uvw_lambda (numpy.ndarray): UVW-coordinates of visibilities. Units are
            multiples of wavelength.
            2d array of ``np.float_``, shape: ``(n_vis, 3)``.
            Assumed ordering is u,v,w i.e. ``u,v,w = uvw[idx]``
        image_size (astropy.units.Quantity): Width of the image in pixels.
            e.g. ``1024 * u.pixel``.
            NB we assume the pixel ``[image_size//2,image_size//2]``
            corresponds to the origin in UV-space.
        cell_size (astropy.units.Quantity): Angular-width of a synthesized pixel
            in the image to be created, e.g. ``3.5 * u.arcsecond``.
        kernel_func (callable): Callable object,
            (e.g. :class:`.conv_funcs.Pillbox`,)
            that returns a convolution
            co-efficient for a given distance in pixel-widths.
        kernel_support (int): Defines the 'radius' of the bounding box within
            which the anti-aliasing kernel is generated.
            `Box width in pixels = 2*support+1`.
            (The central pixel is the one nearest to the UV co-ordinates.)
            (This is sometimes known as the 'half-support')
        kernel_exact (bool): Calculate exact kernel-values for every UV-sample.
        kernel_oversampling (int): Controls kernel-generation if
            ``exact==False``. Larger values give a finer-sampled set of
            pre-cached kernels.
        num_wplanes (int): Number of planes for W-Projection. Set to zero or None
            to disable W-projection.
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
        obs_lat (float): Latitude of observation pointing centre (in degrees)
        lha (numpy.ndarray): Local hour angle of visibilities.
            LHA=0 is transit, LHA=-6h is rising, LHA=+6h is setting.
            1d array, shape: `(n_vis,)`.
        mueller_term (numpy.array):  Mueller matrix term (defined each image
            coordinate) used for A-projection.
        progress_bar (tqdm.tqdm): [Optional] progressbar to update.

    Returns:
        tuple: (image, beam)
            Tuple of ndarrays representing the image map and beam model.
            These are 2d arrays of same dtype as ``vis``,
            (typically ``np._complex``),  shape ``(image_size, image_size)``.
            Note numpy style index-order, i.e. access like ``image[y,x]``.
    """

    assert isinstance(hankel_opt, bool)
    assert isinstance(undersampling_opt, int)
    assert isinstance(wplanes_median, bool)
    assert isinstance(analytic_gcf, bool)

    # Convert undersampling_opt to power of two value
    if undersampling_opt > 0:
        undersampling_opt = pow(2, undersampling_opt - 1)

    image_size = image_size.to(u.pix)
    image_size_int = int(image_size.value)
    if image_size_int != image_size.value:
        raise ValueError("Please supply an integer-valued image size")

    if num_wplanes is None:
        num_wplanes = 0
        # Hankel transform only can be used when w-projection is enabled
        hankel_opt = False

    if num_wplanes > 0 and kernel_exact is True:
        msg = "W-Projection (set by num_wplanes > 0) cannot be used when 'kernel_exact' is True. " \
              "Please disable 'kernel_exact' or W-Projection."
        raise ValueError(msg)

    # Size of a UV-grid pixel, in multiples of wavelength (lambda):
    grid_pixel_width_lambda = 1.0 / (cell_size.to(u.rad) * image_size)
    uvw_in_pixels = (uvw_lambda / grid_pixel_width_lambda).value
    uv_in_pixels = uvw_in_pixels[:, :2]

    vis_grid, sample_grid = convolve_to_grid(kernel_func,
                                             aa_support=kernel_support,
                                             image_size=image_size_int,
                                             uv=uv_in_pixels,
                                             vis=vis,
                                             vis_weights=vis_weights,
                                             exact=kernel_exact,
                                             oversampling=kernel_oversampling,
                                             num_wplanes=num_wplanes,
                                             cell_size=cell_size.to(u.rad).value,
                                             w_lambda=uvw_lambda[:, 2],
                                             wplanes_median=wplanes_median,
                                             max_wpconv_support=max_wpconv_support,
                                             analytic_gcf=analytic_gcf,
                                             hankel_opt=hankel_opt,
                                             interp_type=interp_type,
                                             undersampling_opt=undersampling_opt,
                                             kernel_trunc_perc=kernel_trunc_perc,
                                             aproj_numtimesteps=aproj_numtimesteps,
                                             obs_dec=obs_dec,
                                             obs_lat=obs_lat,
                                             lha=lha,
                                             mueller_term=mueller_term,
                                             progress_bar=progress_bar
                                             )

    # Perform FFT step
    image = np.real(fft_to_image_plane(vis_grid))
    beam = np.real(fft_to_image_plane(sample_grid))

    # Generate gridding correction kernel
    gcf_array = ImgDomKernel(kernel_func, image_size_int, normalize=False, radial_line=False,
                             analytic_gcf=analytic_gcf).array

    # Normalization factor:
    # We correct for the FFT scale factor of 1/image_size**2 by dividing by the image-domain AA-kernel
    # (cf https://docs.scipy.org/doc/numpy/reference/routines.fft.html#implementation-details)
    image = image / gcf_array
    beam = beam / gcf_array

    # And then divide by the total sample weight to renormalize the visibilities.
    if analytic_gcf is True:
        total_sample_weight = np.real(sample_grid.sum()) / (image_size_int*image_size_int)
    else:
        total_sample_weight = np.real(sample_grid.sum())

    if total_sample_weight != 0:
        beam /= total_sample_weight
        image /= total_sample_weight

    return image, beam
