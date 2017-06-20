import astropy.units as u
import numpy as np

from fastimgproto.gridder.gridder import convolve_to_grid


def fft_to_image_plane(uv_grid):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid)))


def image_visibilities(vis, uvw_lambda,
                       image_size, cell_size,
                       kernel_func, kernel_support,
                       kernel_exact=True,
                       kernel_oversampling=0,
                       normalize=True,
                       pbar=None):
    """
    Args:
        vis (numpy.ndarray): Complex visibilities.
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
            which convolution takes place. `Box width in pixels = 2*support+1`.
            (The central pixel is the one nearest to the UV co-ordinates.)
            (This is sometimes known as the 'half-support')
        kernel_exact (bool): Calculate exact kernel-values for every UV-sample.
        kernel_oversampling (int): Controls kernel-generation if
            ``exact==False``. Larger values give a finer-sampled set of
            pre-cached kernels.
        raise_bounds (bool): Raise an exception if any of the UV
        kernel_oversampling (int): (Or None). Controls kernel-generation,
            see :func:`fastimgproto.gridder.gridder.convolve_to_grid` for
            details.
        normalize (bool): Whether or not the returned image and beam
            should be normalized such that the beam peaks at a value of
            1.0 Jansky. You normally want this to be true, but it may be
            interesting to check the raw values for debugging purposes.
        pbar (tqdm.tqdm): [Optional] progressbar to update.

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

    # Size of a UV-grid pixel, in multiples of wavelength (lambda):
    grid_pixel_width_lambda = 1.0 / (cell_size.to(u.rad) * image_size)
    uvw_in_pixels = (uvw_lambda / grid_pixel_width_lambda).value

    uv_in_pixels = uvw_in_pixels[:, :2]
    vis_grid, sample_grid = convolve_to_grid(kernel_func,
                                             support=kernel_support,
                                             image_size=image_size_int,
                                             uv=uv_in_pixels,
                                             vis=vis,
                                             exact=kernel_exact,
                                             oversampling=kernel_oversampling,
                                             pbar=pbar
                                             )
    image = fft_to_image_plane(vis_grid)
    beam = fft_to_image_plane(sample_grid)
    if normalize:
        beam_max = np.max(np.real(beam))
        beam /= beam_max
        image /= beam_max
    return (image, beam)
