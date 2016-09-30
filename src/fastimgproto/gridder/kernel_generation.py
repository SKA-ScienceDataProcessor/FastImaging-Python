"""
Generation of convolution kernel arrays, with optional sub-pixel origin offset
and and oversampling.
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

    Attributes:
        array (numpy.ndarray): The sampled kernel function.
        centre_idx (int): Index of the central pixel
        kernel_func, support, offset, oversampling : See params.


    """

    def __init__(self, kernel_func, support, offset=(0.0, 0.0),
                 oversampling=None, pad=False, normalize=True):
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

        array_size = 2 * (self.support + padding) * self.oversampling + 1
        self.centre_idx = (self.support + padding) * self.oversampling

        # Distance from each pixel's sample position to kernel-centre position:
        # (units of oversampled pixels)
        oversampled_xy = np.arange(array_size,
                                   dtype=np.float_) - self.centre_idx

        # Now translate that to distance from sampling origin, in units of
        # regular pixels:
        self.x_distance_vec = oversampled_xy / oversampling - offset[0]
        self.y_distance_vec = oversampled_xy / oversampling - offset[1]
        # Re-orient y_vec as a column
        self.y_distance_vec = np.atleast_2d(self.y_distance_vec).T

        x_kernel_coeffs = self.kernel_func(self.x_distance_vec)
        y_kernel_coeffs = self.kernel_func(self.y_distance_vec)

        # Now multiply separable components to get the 2-d kernel:
        self.array = x_kernel_coeffs * y_kernel_coeffs
        if normalize:
            self.array/=self.array.sum()
