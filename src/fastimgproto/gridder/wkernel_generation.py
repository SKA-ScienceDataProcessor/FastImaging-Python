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
        scale (int): Scale ratio for generating the kernel distance positions.
            The kernel shape shrinks when scale is larger than 1.
            Defaults to 1 if not given or ``scale=None`` is passed.
        radial_line (bool): Computes only the semi-diagonal line of the kernel rather
            than the 2D matrix.

    Attributes:
        array (numpy.ndarray): The sampled kernel function.
        w_value, array_size, oversampling, cell_size, scale : See params.

    """

    def __init__(self, w_value, array_size, cell_size, scale=None, radial_line=False):

        if scale is None:
            scale = 1

        assert isinstance(array_size, int)
        assert array_size >= 1
        assert (array_size % 2) == 0

        self.w_value = w_value
        self.cell_size = cell_size
        self.scale = scale

        if radial_line is False:
            self.array_size = array_size
            centre_idx = array_size // 2

            # Distance from each pixel's sample position to kernel-centre position:
            # (units of pixels)
            xy_pixels = np.arange(array_size, dtype=np.float_) - centre_idx

            # Now translate that to distance in terms of direction cosines and scale according the scale ratio
            self.distance_vec = xy_pixels * cell_size * scale

            # Create empty 2-D kernel array
            self.array = np.zeros((self.array_size, self.array_size), dtype=np.complex)

            # Generate kernel coefficients
            for y in range(1, array_size):
                for x in range(1, array_size):
                    squared_radians = self.distance_vec[x] * self.distance_vec[x] + self.distance_vec[y] * self.distance_vec[y]
                    if squared_radians > 1.0:
                        self.array[y, x] = 1.0
                    else:
                        n = np.sqrt(1 - squared_radians)
                        self.array[y, x] = np.exp(-2 * np.pi * 1j * w_value * (n - 1)) / n
        else:
            array_size = array_size // 2
            self.array_size = array_size

            # Distance from each pixel in the diagonal direction to the kernel-centre position:
            # (units of pixels)
            xy_pixels = np.arange(array_size, dtype=np.float_) * scale * np.sqrt(2)

            # Now translate that to distance in terms of direction cosines and scale according the scale ratio
            self.distance_vec = xy_pixels * cell_size

            # Create empty 1-D kernel array
            self.array = np.zeros((self.array_size,), dtype=np.complex)

            # Generate kernel coefficients
            for x in range(array_size):
                squared_radians = self.distance_vec[x] * self.distance_vec[x]
                if squared_radians > 1.0:
                    self.array[x] = 1.0
                else:
                    n = np.sqrt(1 - squared_radians)
                    self.array[x] = np.exp(-2 * np.pi * 1j * w_value * (n - 1)) / n
