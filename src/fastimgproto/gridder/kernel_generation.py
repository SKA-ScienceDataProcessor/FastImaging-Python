"""
Generation of convolution kernel arrays, with optional sub-pixel origin offset
and oversampling.
"""

import numpy as np


class Kernel(object):
    """
    Generates a 2D array representing a sampled kernel function.

    Args:
        kernel_func (callable): Callable object,
            (e.g. :class:`.conv_funcs.Pillbox`,)
            that returns a convolution
            co-efficient for a given distance in pixel-widths.
        support (int): Defines the 'radius' of the bounding box within
            which convolution takes place. `Box width in pixels = 2*support+1`.
            (The central pixel is the one nearest to the UV co-ordinates.)
            For a kernel_func with truncation radius `trunc`, the support
            should be set to `ceil(trunc+0.5)` to ensure that the kernel
            function is fully supported for all valid subpixel offsets.
        offset (tuple): 2-vector subpixel offset from the sampling position of the
            central pixel to the origin of the kernel function.
            Ordering is (x_offset,y_offset). Should have values such that
            `fabs(offset) <= 0.5`
            otherwise the nearest integer grid-point would be different!
        oversampling (int): Oversampling ratio, how many kernel pixels
            to each UV-grid pixel.
            Defaults to 1 if not given or ``oversampling=None`` is passed.
        pad (bool): Whether to pad the array by an extra pixel-width.
            This is used when generating an oversampled kernel that will be used
            for interpolation.
        normalize (bool): Whether to normalize the kernel.
        radial_line (bool): Computes only the semi-diagonal line of the kernel rather
            than the 2D matrix.

    Attributes:
        array (numpy.ndarray): The sampled kernel function.
        centre_idx (int): Index of the central pixel
        kernel_func, support, offset, oversampling : See params.


    """

    def __init__(self, kernel_func, support, offset=(0.0, 0.0),
                 oversampling=None, pad=False, normalize=True, radial_line=False):
        if oversampling is None:
            oversampling = 1

        assert isinstance(oversampling, int)
        assert isinstance(support, int)
        assert support >= 1
        assert len(offset) == 2
        for off_val in offset:
            assert -0.5 <= off_val <= 0.5

        self.oversampling = oversampling
        self.kernel_func = kernel_func
        self.offset = offset
        self.support = support

        if pad:
            padding = 1
        else:
            padding = 0

        if radial_line is False:
            self.array_size = 2 * (self.support + padding) * self.oversampling + 1
            self.centre_idx = (self.support + padding) * self.oversampling

            # Distance from each pixel's sample position to kernel-centre position:
            # (units of oversampled pixels)
            oversampled_xy = np.arange(self.array_size,
                                       dtype=np.float_) - self.centre_idx

            # Translate that to distance from sampling origin, in units of
            # regular pixels (x):
            self.x_distance_vec = oversampled_xy / oversampling - offset[0]
            self.y_distance_vec = oversampled_xy / oversampling - offset[1]

            x_kernel_coeffs = self.kernel_func(self.x_distance_vec)
            y_kernel_coeffs = self.kernel_func(self.y_distance_vec)

            # Now multiply separable components to get the 2-d kernel:
            self.array = np.outer(y_kernel_coeffs, x_kernel_coeffs)

        else:
            self.array_size = (self.support + padding) * self.oversampling + 1

            # Distance from each pixel's sample position to kernel-centre position:
            # (units of oversampled pixels)
            oversampled_xy = np.arange(self.array_size, dtype=np.float_)

            # Translate that to distance from sampling origin, in units of
            # regular pixels (x):
            self.x_distance_vec = oversampled_xy / oversampling - offset[0]
            kernel_1d = self.kernel_func(self.x_distance_vec)
            # Semi-diagonal of the kernel:
            self.array = kernel_1d[(array_size//2):] * kernel_1d[(array_size//2):]

        if normalize:
            array_sum = self.array.sum()
            if array_sum > 0.0:
                self.array /= array_sum


class ImgDomKernel(object):
    """
    Generates a 2D array representing a sampled kernel function in the image domain.

    Args:
        kernel_func (callable): Callable object,
            (e.g. :class:`.conv_funcs.Pillbox`,)
            that returns a convolution
            co-efficient for a given distance in pixel-widths.
        array_size (int): Image domain kernel width in pixels.
        oversampling (int): Oversampling ratio, how many kernel pixels
            to each UV-grid pixel.
            Defaults to 1 if not given or ``oversampling=None`` is passed.
        normalize (bool): Whether to normalize the kernel.
        radial_line (bool): Computes only the semi-diagonal line of the kernel rather
            than the 2D matrix.
        analytic_gcf (bool): Whether to compute the kernel from analytic expression
            or using FFT of UV-domain kernel.

    Attributes:
        array (numpy.ndarray): The sampled kernel function.
        kernel_func, array_size, oversampling : See params.


    """

    def __init__(self, kernel_func, array_size, oversampling=None, normalize=False,
                 radial_line=False, analytic_gcf=False):
        if oversampling is None:
            oversampling = 1

        assert (array_size % 2) == 0
        assert (array_size % oversampling) == 0
        assert ((array_size / oversampling) % 2) == 0

        self.array_size = array_size
        self.oversampling = oversampling
        self.kernel_func = kernel_func
        kernel_size = int(array_size // oversampling)
        centre_idx = kernel_size // 2
        array_offset = array_size // 2 - centre_idx

        if analytic_gcf is True:
            self.radius = (np.arange(kernel_size, dtype=np.float_) - centre_idx) / centre_idx
            kernel_img_1d = np.zeros((array_size,), dtype=np.float)
            kernel_img_1d[range(array_offset, array_offset+kernel_size)] = self.kernel_func.gcf(self.radius)
        else:
            # Distance from each pixel's sample position to kernel-centre position:
            self.distance_vec = (np.arange(kernel_size, dtype=np.float_) - centre_idx)
            kernel_1d = self.kernel_func(self.distance_vec)
            array_sum = kernel_1d.sum()
            if array_sum > 0.0:
                kernel_1d /= array_sum
            kernel_img_1d_aux = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kernel_1d))))
            kernel_img_1d = np.zeros((array_size,), dtype=np.float)
            kernel_img_1d[range(array_offset, array_offset + kernel_size)] = kernel_img_1d_aux

        if radial_line is False:
            # Multiply separable components to get the 2-d kernel:
            self.array = np.outer(kernel_img_1d, kernel_img_1d)
        else:
            # Semi-diagonal of the kernel:
            self.array = kernel_img_1d[(array_size//2):] * kernel_img_1d[(array_size//2):]

        if normalize:
            array_sum = self.array.sum()
            if array_sum > 0.0:
                self.array /= array_sum
