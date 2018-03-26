import astropy.units as u
import numpy as np

from fastimgproto.gridder.gridder import convolve_to_grid
from fastimgproto.gridder.kernel_generation import Kernel


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
        kernel_oversampling=0,
        num_wplanes=0,
        max_wpconv_support=0,
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
        max_wkernel_support (int): Defines the maximum 'radius' of the bounding box
            within which convolution takes place when W-Projection is used.
            `Box width in pixels = 2*support+1`.
        progress_bar (tqdm.tqdm): [Optional] progressbar to update.

    Returns:
        tuple: (image, beam)
            Tuple of ndarrays representing the image map and beam model.
            These are 2d arrays of same dtype as ``vis``,
            (typically ``np._complex``),  shape ``(image_size, image_size)``.
            Note numpy style index-order, i.e. access like ``image[y,x]``.
    """

    image_size = image_size.to(u.pix)
    image_size_int = int(image_size.value)
    if image_size_int != image_size.value:
        raise ValueError("Please supply an integer-valued image size")

    if num_wplanes is None:
        num_wplanes = 0

    if num_wplanes > 0 and kernel_exact is True:
        msg = "W-Projection (set by num_wplanes > 0) cannot be used when 'kernel_exact' is True. " \
              "Please disable 'kernel_exact' or W-Projection."
        raise ValueError(msg)

    # The convolution kernel support must be at least equal to the AA kernel support
    if max_wpconv_support < kernel_support:
        max_wpconv_support = kernel_support

    vis_grid, sample_grid = convolve_to_grid(kernel_func,
                                             aa_support=kernel_support,
                                             max_conv_support=max_wpconv_support,
                                             image_size=image_size_int,
                                             cell_size=cell_size.to(u.rad).value,
                                             uvw_lambda=uvw_lambda,
                                             vis=vis,
                                             vis_weights=vis_weights,
                                             exact=kernel_exact,
                                             oversampling=kernel_oversampling,
                                             num_wplanes=num_wplanes,
                                             progress_bar=progress_bar
                                             )

    image = np.real(fft_to_image_plane(vis_grid))
    beam = np.real(fft_to_image_plane(sample_grid))

    # Computed images need to be divided by the image-domain AA-kernel
    # First, generate image-domain AA-kernel
    kernel_img_array = np.zeros_like(image)
    aa_kernel = Kernel(kernel_func=kernel_func, support=kernel_support,
                       offset=(0, 0),
                       oversampling=None,
                       normalize=True
                       )
    xrange = slice(image_size_int//2 - kernel_support, image_size_int//2 + kernel_support + 1)
    yrange = slice(image_size_int//2 - kernel_support, image_size_int//2 + kernel_support + 1)
    kernel_img_array[yrange, xrange] = aa_kernel.array
    kernel_img_array = np.real(fft_to_image_plane(kernel_img_array))

    # Normalization factor:
    # We correct for the FFT scale factor of 1/image_size**2 by dividing by the image-domain AA-kernel
    # (cf https://docs.scipy.org/doc/numpy/reference/routines.fft.html#implementation-details)
    image = image / kernel_img_array
    beam = beam / kernel_img_array

    # And then divide by the total sample weight to renormalize the visibilities.
    total_sample_weight = np.real(sample_grid.sum())
    if total_sample_weight != 0:
        beam /= total_sample_weight
        image /= total_sample_weight

    return image, beam
