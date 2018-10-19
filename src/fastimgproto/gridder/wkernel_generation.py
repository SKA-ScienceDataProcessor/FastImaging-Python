"""
Generation of W-Projection kernel arrays.
"""
import numpy as np


class WKernel(object):
    """
    Generates a 2D array representing a sampled kernel function for W-Projection.

    Args:
        w_value (float): W lambda value used to generate the W-Projection kernel.
        array_size (int): Defines the size of the kernel.
        cell_size (float): Angular-width in radians of a synthesized pixel
            in the image to be created.
        undersampling (int): Scale ratio for generating the kernel distance positions.
            The kernel shape shrinks when scale is larger than 1.
            Defaults to 1 if not given or ``scale=None`` is passed.
        radial_line (bool): Computes only the semi-diagonal line of the kernel rather
            than the 2D matrix.

    Attributes:
        array (numpy.ndarray): The sampled kernel function.
        w_value, array_size, oversampling, cell_size, scale : See params.

    """

    def __init__(self, w_value, array_size, cell_size, undersampling=None, radial_line=False):

        if undersampling is None:
            undersampling = 1

        assert isinstance(array_size, int)
        assert array_size >= 1
        assert (array_size % 2) == 0

        self.w_value = w_value
        self.cell_size = cell_size
        self.undersampling_ratio = undersampling

        if radial_line is False:
            self.array_size = array_size
            centre_idx = array_size // 2

            # Distance from each pixel's sample position to kernel-centre position (units of pixels):
            xy_pixels = np.arange(array_size, dtype=np.float_) - centre_idx

            # Now translate that to distance in terms of radians
            self.distance_vec = xy_pixels * cell_size * undersampling

            # Create empty 2-D kernel array
            self.array = np.zeros((self.array_size, self.array_size), dtype=np.complex)

            # Generate kernel coefficients
            for y in range(1, array_size):
                for x in range(1, array_size):
                    rsquared_radians = self.distance_vec[x] * self.distance_vec[x] + \
                                       self.distance_vec[y] * self.distance_vec[y]
                    if rsquared_radians >= 1.0:
                        self.array[y, x] = 1.0
                    else:
                        n = np.sqrt(1 - rsquared_radians)
                        self.array[y, x] = np.exp(-2 * np.pi * 1j * w_value * (n - 1)) / n
        else:
            array_size = array_size // 2
            self.array_size = array_size

            # Distance from each pixel in the diagonal direction to the kernel-centre position:
            # (units of pixels)
            xy_pixels = np.arange(array_size, dtype=np.float_) * undersampling * np.sqrt(2)

            # Now translate that to distance in terms of direction cosines
            self.distance_vec = xy_pixels * cell_size

            # Create empty 1-D kernel array
            self.array = np.zeros((self.array_size,), dtype=np.complex)

            # Generate kernel coefficients
            for x in range(array_size):
                rsquared_radians = self.distance_vec[x] * self.distance_vec[x]
                if rsquared_radians >= 1.0:
                    self.array[x] = 1.0
                else:
                    n = np.sqrt(1 - rsquared_radians)
                    self.array[x] = np.exp(-2 * np.pi * 1j * w_value * (n - 1)) / n
