import astropy.units as u

import fastimgproto.gridder.conv_funcs as kfuncs
from .present import CPP_BINDINGS_PRESENT


class CppKernelFuncs(object):
    """
    A simple namespace / enum structure for listing the available kernels.
    """
    gauss = 'gauss'
    gauss_sinc = 'gauss_sinc'
    sinc = 'sinc'
    triangle = 'triangle'
    tophat = 'tophat'


# Mapping to equivalent implementation in pure Python
PYTHON_KERNELS = {
    CppKernelFuncs.gauss: kfuncs.Gaussian,
    CppKernelFuncs.gauss_sinc: kfuncs.GaussianSinc,
    CppKernelFuncs.sinc: kfuncs.Sinc,
    CppKernelFuncs.tophat: kfuncs.Pillbox,
    CppKernelFuncs.triangle: kfuncs.Triangle,
}

if CPP_BINDINGS_PRESENT:
    import stp_python

    # Mapping from name to stp function:
    CPP_KERNELS = {
        CppKernelFuncs.gauss_sinc: stp_python.KernelFunction.GaussianSinc,
        CppKernelFuncs.gauss: stp_python.KernelFunction.Gaussian,
        CppKernelFuncs.sinc: stp_python.KernelFunction.Sinc,
        CppKernelFuncs.triangle: stp_python.KernelFunction.Triangle,
        CppKernelFuncs.tophat: stp_python.KernelFunction.TopHat,
    }


def cpp_image_visibilities(vis,
                           uvw_lambda,
                           image_size,
                           cell_size,
                           kernel_func_name,
                           kernel_trunc_radius=3.0,
                           kernel_support=3,
                           kernel_exact=True,
                           kernel_oversampling=0,
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
    - Checks CPP bindings are available
    - Checks arguments are of correct type / units

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
    if not CPP_BINDINGS_PRESENT:
        raise OSError("Cannot import stp_python (C++ bindings module)")

    stp_kernel = CPP_KERNELS[kernel_func_name]

    if kernel_oversampling is None:
        kernel_oversampling = 0
    if not kernel_exact:
        assert kernel_oversampling >= 1

    (image, beam) = stp_python.image_visibilities_wrapper(
        vis,
        uvw_lambda,
        int(image_size.to(u.pix).value),
        cell_size.to(u.arcsec).value,
        stp_kernel,
        kernel_trunc_radius,
        int(kernel_support),
        kernel_exact,
        kernel_oversampling,
        normalize,
    )

    return image, beam
