import astropy.units as u
import numpy as np

from fastimgproto.gridder.gridder import convolve_to_grid
from fastimgproto.imager import fft_to_image_plane
from fastimgproto.gridder.conv_funcs import GaussianSinc


class CppKernelFuncs(object):
    gauss_sinc = 'gauss_sinc'


def cpp_image_visibilities(vis, uvw_lambda,
                           image_size, cell_size,
                           kernel_func_name=CppKernelFuncs.gauss_sinc,
                           kernel_trunc_radius=3.0,
                           kernel_support=3,
                           kernel_oversampling=None,
                           normalize=True):
    """
    Convenience wrapper over _cpp_image_visibilities.

    Functionality largely mirrors
    :func:`fastimgproto.imager.image_visibilities`, but the key difference is
    that instead of passing a callable kernel-function, you must choose
    ``kernel_func_name`` from a limited selection of kernel-functions
    implemented in the C++ code. Currently, choices are limited to:

        - ``gauss_sinc``


    Performs the following tasks before handing over to C++ bindings:
    - Checks arguments are of correct type / units
    - Converts uvw-array from wavelength (lambda) units to pixel units

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
        kernel_func_name (str): Choice of kernel function from limited C++ selection.
        kernel_trunc_radius (float): Truncation radius of the kernel to be used.
        kernel_support (int): Defines the 'radius' of the bounding box within
            which convolution takes place. `Box width in pixels = 2*support+1`.
            (The central pixel is the one nearest to the UV co-ordinates.)
            (This is sometimes known as the 'half-support')
        kernel_oversampling (int): (Or None). Controls kernel-generation,
            see :func:`fastimgproto.gridder.gridder.convolve_to_grid` for
            details.
        normalize (bool): Whether or not the returned image and beam
            should be normalized such that the beam peaks at a value of
            1.0 Jansky. You normally want this to be true, but it may be
            interesting to check the raw values for debugging purposes.

    Returns:
        tuple: (image, beam)
            Tuple of ndarrays representing the image map and beam model.
            These are 2d arrays of same dtype as ``vis``,
            (typically ``np._complex``),  shape ``(image_size, image_size)``.
            Note numpy style index-order, i.e. access like ``image[y,x]``.

    """
    if kernel_func_name not in (CppKernelFuncs.gauss_sinc,):
        raise ValueError(
            "kernel function of type {} not recognised".format(
                kernel_func_name))

    image_size = image_size.to(u.pix)
    # Size of a UV-grid pixel, in multiples of wavelength (lambda):
    grid_pixel_width_lambda = 1.0 / (cell_size.to(u.rad) * image_size)
    uvw_in_pixels = (uvw_lambda / grid_pixel_width_lambda).value
    uv_in_pixels = uvw_in_pixels[:, :2]

    # subroutine = _cpp_image_visibilities
    subroutine = _python_image_visibilities
    (image, beam) = _python_image_visibilities(
        vis=vis,
        uv_pixels=uv_in_pixels,
        image_size=int(image_size.value),
        kernel_func_name=kernel_func_name,
        kernel_trunc_radius=kernel_trunc_radius,
        kernel_support=kernel_support,
        kernel_oversampling=kernel_oversampling,
        normalize=normalize,
    )

    return image, beam


def _cpp_image_visibilities(vis,
                            uv_pixels,
                            image_size,
                            kernel_func_name,
                            kernel_trunc_radius,
                            kernel_support,
                            kernel_oversampling,
                            normalize=True
                            ):
    pass
    # C++ Bindings here


def _python_image_visibilities(vis,
                               uv_pixels,
                               image_size,
                               kernel_func_name,
                               kernel_trunc_radius,
                               kernel_support,
                               kernel_oversampling,
                               normalize=True
                               ):
    """
    Equivalent Python code for validation of _cpp_image_visibilities

    Args:
        vis (numpy.ndarray): Complex visibilities.
            1d array, shape: `(n_vis,)`.
        uv_pixels (numpy.ndarray): UV-coordinates of visibilities. Units are
            pixel-widths relative to the grid being sampled onto.
            2d array of ``np.float_``, shape: ``(n_vis, 2)``.
            Assumed ordering is u,v i.e. ``u,v = uv[idx]``
        image_size (int): Width of the image in pixels.
            NB we assume the pixel ``[image_size//2,image_size//2]``
            corresponds to the origin in UV-space.
        kernel_func_name (str): Choice of kernel function from limited C++ selection.
        kernel_trunc_radius (float): Truncation radius of the kernel to be used.
        kernel_support (int): Defines the 'radius' of the bounding box within
            which convolution takes place. `Box width in pixels = 2*support+1`.
            (The central pixel is the one nearest to the UV co-ordinates.)
            (This is sometimes known as the 'half-support')
        kernel_oversampling (int): (Or None). Controls kernel-generation,
            see :func:`fastimgproto.gridder.gridder.convolve_to_grid` for
            details.
        normalize (bool): Whether or not the returned image and beam
            should be normalized such that the beam peaks at a value of
            1.0 Jansky. You normally want this to be true, but it may be
            interesting to check the raw values for debugging purposes.

    Returns:
        tuple: (image, beam)
            Tuple of ndarrays representing the image map and beam model.
            These are 2d arrays of same dtype as ``vis``,
            (typically ``np._complex``),  shape ``(image_size, image_size)``.
            Note numpy style index-order, i.e. access like ``image[y,x]``.

    """
    assert kernel_func_name == CppKernelFuncs.gauss_sinc
    kernel_func = GaussianSinc(trunc=kernel_trunc_radius)

    vis_grid, sample_grid = convolve_to_grid(kernel_func,
                                             support=kernel_support,
                                             image_size=image_size,
                                             uv=uv_pixels,
                                             vis=vis,
                                             oversampling=kernel_oversampling
                                             )
    image = fft_to_image_plane(vis_grid)
    beam = fft_to_image_plane(sample_grid)
    if normalize:
        beam_max = np.max(np.real(beam))
        beam /= beam_max
        image /= beam_max

    return (image, beam)
