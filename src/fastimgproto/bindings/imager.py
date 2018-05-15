import astropy.units as u

import fastimgproto.gridder.conv_funcs as kfuncs

from .present import CPP_BINDINGS_PRESENT


class CppKernelFuncs(object):
    """
    A simple namespace / enum structure for listing the available kernels.
    """
    pswf = 'pswf'
    gauss = 'gauss'
    gauss_sinc = 'gauss_sinc'
    sinc = 'sinc'
    triangle = 'triangle'
    tophat = 'tophat'


# Mapping to equivalent implementation in pure Python
PYTHON_KERNELS = {
    CppKernelFuncs.pswf: kfuncs.PSWF,
    CppKernelFuncs.gauss_sinc: kfuncs.GaussianSinc,
    CppKernelFuncs.gauss: kfuncs.Gaussian,
    CppKernelFuncs.sinc: kfuncs.Sinc,
    CppKernelFuncs.tophat: kfuncs.Pillbox,
    CppKernelFuncs.triangle: kfuncs.Triangle,
}

if CPP_BINDINGS_PRESENT:
    import stp_python

    # Mapping from name to stp function:
    CPP_KERNELS = {
        CppKernelFuncs.pswf: stp_python.KernelFunction.PSWF,
        CppKernelFuncs.gauss_sinc: stp_python.KernelFunction.GaussianSinc,
        CppKernelFuncs.gauss: stp_python.KernelFunction.Gaussian,
        CppKernelFuncs.sinc: stp_python.KernelFunction.Sinc,
        CppKernelFuncs.triangle: stp_python.KernelFunction.Triangle,
        CppKernelFuncs.tophat: stp_python.KernelFunction.TopHat,
    }


def cpp_image_visibilities(vis,
                           vis_weights,
                           uvw_lambda,
                           image_size,
                           cell_size,
                           kernel_func_name,
                           kernel_trunc_radius=3.0,
                           kernel_support=3,
                           kernel_exact=True,
                           kernel_oversampling=0,
                           generate_beam=False,
                           fft_routine = stp_python.FFTRoutine.FFTW_ESTIMATE_FFT,
                           fft_wisdom_filename="",
                           fft_wisdom_1D_filename="",
                           num_wplanes=0,
                           wplanes_median=False,
                           max_wpconv_support=0,
                           analytic_gcf=False,
                           hankel_opt=False,
                           undersampling_opt=0,
                           kernel_trunc_perc=0.0,
                           interp_type=stp_python.InterpType.LINEAR):
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
        num_wplanes (int): Number of planes for W-Projection. Set zero or None
            to disable W-projection.
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
        kernel_trunc_perc (float): Percentage of the kernel peak at which the
            W-projection convolution kernel is trucanted. If 0 use kernel size
            computed from max_wpconv_support.
        interp_type (enum): Interpolation type to be used for kernel generation
            step in the Hankel Transorm. Available options are: "LINEAR", "COSINE"
            and "CUBIC".

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
        vis_weights,
        uvw_lambda,
        int(image_size.to(u.pix).value),
        cell_size.to(u.arcsec).value,
        stp_kernel,
        kernel_trunc_radius,
        int(kernel_support),
        kernel_exact,
        kernel_oversampling,
        generate_beam,
        fft_routine,
        fft_wisdom_filename,
        fft_wisdom_1D_filename,
        num_wplanes,
        wplanes_median,
        max_wpconv_support,
        analytic_gcf,
        hankel_opt,
        undersampling_opt,
        kernel_trunc_perc,
        interp_type
    )

    return image, beam
