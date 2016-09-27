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

    Attributes:
        array (numpy.ndarray): The sampled kernel function.
        kernel_func, support, offset, oversampling : See params.


    """
    def __init__(self, kernel_func, support, offset=(0.0, 0.0), oversampling=1):


        self.oversampling = oversampling
        self.kernel_func = kernel_func
        self.offset = offset
        self.support = support
        assert isinstance(support, int)
        assert support >= 1
        assert len(offset) == 2
        for off_val in offset:
            assert -0.5 <= off_val <= 0.5
        array_size = 2 * self.support * self.oversampling + 1
        self.centre_idx = self.support*self.oversampling

        #Offset from array[0,0] sample-position to kernel origin, in units of
        # regular (i.e. not oversampled!) pixels:
        origin_offset_x = self.support + offset[0]
        origin_offset_y = self.support + offset[1]

        # Distance from each pixel's sample position to kernel-origin position,
        # in x/y axis (units of regular, non-oversampled, pixels, since those
        # are what the kernel_func expects):
        self.x_distance_vec = np.arange(array_size,dtype=np.float_)/oversampling - origin_offset_x
        self.y_distance_vec = np.arange(array_size,dtype=np.float_)/oversampling - origin_offset_y
        #Re-orient y_vec as a column
        self.y_distance_vec = np.atleast_2d(self.y_distance_vec).T

        x_kernel_coeffs = self.kernel_func(self.x_distance_vec)
        y_kernel_coeffs = self.kernel_func(self.y_distance_vec)

        # Now multiply separable components to get the 2-d kernel:
        self.array = x_kernel_coeffs * y_kernel_coeffs




