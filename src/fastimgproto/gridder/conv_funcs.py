"""
Convolution functions.

Used for generating the kernel used in convolutional gridding.

We actually paramerize the functions at initialization and return a simple
callable with one parameter, the distance in pixels.

This allows us to pass the convolution routine the minimum of extra parameters.
"""
from attr import attrs, attrib
import numpy as np
from six import add_metaclass
from abc import ABCMeta, abstractmethod


@add_metaclass(ABCMeta)
class ConvFuncBase(object):
    """
    Implements truncation (via __call__), numpy array reshaping.

    Always returns 0 outside truncation radius, i.e.::

        if np.fabs(x) > trunc:
            conv_func(x)==0 # True

    Args:
        trunc: truncation radius.
    """

    def __init__(self, trunc):
        self.trunc = trunc

    @abstractmethod
    def f(self, radius):
        """The convolution function to be evaluated and truncated"""
        pass

    def __call__(self, radius_in_pix):
        radius_in_pix = np.atleast_1d(radius_in_pix)
        output = np.zeros_like(radius_in_pix)
        inside_trunc_radius = np.fabs(radius_in_pix) < self.trunc
        output[inside_trunc_radius] = self.f(radius_in_pix[inside_trunc_radius])
        return output


class Triangle(ConvFuncBase):
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

    def __init__(self, half_base_width):
        self.half_base_width = half_base_width
        super(Triangle, self).__init__(half_base_width)

    def f(self, radius_in_pix):
        return np.maximum(
            1.0 - np.fabs(radius_in_pix) / self.half_base_width,
            np.zeros_like(radius_in_pix)
        )


class Pillbox(ConvFuncBase):
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

    def __init__(self, half_base_width):
        self.half_base_width = half_base_width
        super(Pillbox, self).__init__(half_base_width)

    def f(self, radius_in_pix):
        return np.where(np.fabs(radius_in_pix) < self.half_base_width, 1.0, 0.0)


class Sinc(ConvFuncBase):
    """
    Sinc function, truncated beyond **trunc** pixels from centre.


    Attributes:
        trunc (float): Truncation radius
    """
    trunc = attrib(default=3.0)

    def __init__(self, trunc):
        super(Sinc, self).__init__(trunc)

    def f(self, radius_in_pix):
        return np.sinc(radius_in_pix)


class Sinc(ConvFuncBase):
    """
    Sinc function, truncated beyond **trunc** pixels from centre.


    Attributes:
        trunc (float): Truncation radius
    """
    trunc = attrib(default=3.0)

    def __init__(self, trunc):
        super(Sinc, self).__init__(trunc)

    def f(self, radius_in_pix):
        return np.sinc(radius_in_pix)
