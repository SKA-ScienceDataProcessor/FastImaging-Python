"""
Convolution functions.

Used for generating the kernel used in convolutional gridding.

We actually paramerize the functions at initialization and return a simple
callable with one parameter, the distance in pixels.

This allows us to pass the convolution routine the minimum of extra parameters.
"""
from attr import attrs, attrib
import numpy as np


@attrs
class Triangle(object):
    """
    Linearly declines from 1.0 at origin to 0.0 at **half_base_width**, zero thereafter.
    "

    Symmetric about the origin.

    Makes a terrible anti-aliasing function. But, because it's so
    simple, it's easy to verify and therefore a useful tool in verifying
    convolution codes.

    Attributes:
        half_base_width (float): Half-base width of the triangle.

    """
    half_base_width = attrib()

    def __call__(self, radius_in_pix):
        return np.maximum(
            1.0 - np.fabs(radius_in_pix) / self.half_base_width,
            np.zeros_like(radius_in_pix)
        )


@attrs
class Pillbox(object):
    """
    Valued 1.0 from origin to **half_base_width**, zero thereafter.

    AKA 'TopHat' function.

    Symmetric about the origin.

    Makes a terrible anti-aliasing function. But, because it's so
    simple, it's easy to verify and therefore a useful tool in verifying
    convolution codes.

    Attributes:
        half_base_width (float): Half-base width pillbox.
    """
    half_base_width = attrib()

    def __call__(self, radius_in_pix):
        return np.where(np.fabs(radius_in_pix) < self.half_base_width, 1.0, 0.0)
