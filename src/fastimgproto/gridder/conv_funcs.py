"""
Convolution functions.

Used for generating the kernel used in convolutional gridding.

We actually paramerize the functions at initialization and return a simple
callable with one parameter, the distance in pixels.

This allows us to pass the convolution routine the minimum of extra parameters.
"""
from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy as np
from six import add_metaclass


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

    @abstractmethod
    def gcf(self, radius):
        """The gridding correction function"""
        pass

    def __call__(self, radius_in_pix):
        radius_in_pix = np.atleast_1d(radius_in_pix)
        output = np.zeros_like(radius_in_pix, dtype=np.float)
        inside_trunc_radius = ~(np.fabs(radius_in_pix) > self.trunc)
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

    Args:
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

    def gcf(self, radius):
        assert False
        return 0


class Pillbox(ConvFuncBase):
    """
    Valued 1.0 from origin to **half_base_width**, zero thereafter.

    AKA 'TopHat' function.

    Symmetric about the origin.

    Makes a terrible anti-aliasing function. But, because it's so
    simple, it's easy to verify and therefore a useful tool in verifying
    convolution codes.

    Args:
        half_base_width (float): Half-base width pillbox.
    """

    def __init__(self, half_base_width):
        self.half_base_width = half_base_width
        super(Pillbox, self).__init__(half_base_width)

    def f(self, radius_in_pix):
        return np.where(np.fabs(radius_in_pix) < self.half_base_width, 1.0, 0.0)

    def gcf(self, radius):
        assert False
        return 0


class Sinc(ConvFuncBase):
    """
    Sinc function (with truncation).
    Args:
        w (float): Width normalization of the sinc. Default = 1.55
    """

    def __init__(self, trunc, w=1.55):
        super(Sinc, self).__init__(trunc)
        self.w = w

    def f(self, radius_in_pix):
        return np.sinc(radius_in_pix / self.w)

    def gcf(self, radius):
        """
        Args:
            radius: normalized radius (between -1 and 1).

        Returns the 1D grid correction function (gcf)
        """
        nu = radius
        return np.where(np.fabs(nu) < 1.0 * self.w, 1.0, 0.0)


class Gaussian(ConvFuncBase):
    """
    Gaussian function (with truncation).

    evaluates the function::

        exp(-(x/w)**2)

    (Using the notation of Taylor 1998, p143, where x = u/delta_u and alpha==2.
    Default value of ``w=1``).

    Args:
        trunc: truncation radius.
        w (float): Width normalization of the Gaussian. Default = 1.0

    """

    def __init__(self, trunc, w=1.0):
        super(Gaussian, self).__init__(trunc)
        self.w = w

    def f(self, radius_in_pix):
        radius_div_w = radius_in_pix / self.w
        return np.exp(-1. * (radius_div_w * radius_div_w))

    def gcf(self, radius):
        """
        Args:
            radius: normalized radius (between -1 and 1).

        Returns the 1D grid correction function (gcf)
        """
        radius_div_w = radius * np.pi / (self.w * 2)
        return np.exp(-1. * (radius_div_w * radius_div_w))


class GaussianSinc(ConvFuncBase):
    """
    Gaussian times sinc function (with truncation).

    evaluates the function::

        exp(-(x/w1)**2) * sinc(x/w2)

    (Using the notation of Taylor 1998, p143, where x = u/delta_u and alpha==2.
    Default values for w1,w2 are chosen according to recommendation therein).

    Args:
        trunc: truncation radius.
        w1 (float): Width normalization of the Gaussian. Default = 2.52
        w2 (float): Width normalization of the sinc. Default = 1.55

    """

    def __init__(self, trunc, w1=2.52, w2=1.55):
        super(GaussianSinc, self).__init__(trunc)
        self.w1 = w1
        self.w2 = w2

    def f(self, radius_in_pix):
        radius_div_w1 = radius_in_pix / self.w1
        return (
            np.exp(-1. * (radius_div_w1 * radius_div_w1)) *
            np.sinc(radius_in_pix / self.w2)
        )

    def gcf(self, radius):
        assert False
        return 0


class PSWF(ConvFuncBase):
    """
    Compute the 1D prolate spheroidal anti-aliasing function

    The kernel is to be used in gridding visibility data onto a grid.
    The gridding correction function (gcf) is used to correct the image for
    decorrelation due to gridding.

    2D Prolate spheroidal angular function is separable

    Args:
        trunc: truncation radius.
    """

    def __init__(self, trunc):
        super(PSWF, self).__init__(trunc)

    def f(self, radius_in_pix):
        """
        Args:
            radius: radius in pixels

        Returns the 1D convolving kernel
        """
        nu = radius_in_pix / self.trunc
        # Compute the gridding function
        kernel1d = self.__grdsf(nu) * (1 - nu ** 2)
        return kernel1d

    def gcf(self, radius):
        """
        Args:
            radius: normalized radius (between -1 and 1).

        Returns the 1D grid correction function (gcf)
        """
        nu = radius
        # Compute the grid correction function:
        gcf1d = self.__grdsf(nu)
        return gcf1d

    def __grdsf(self, nu):
        """Calculate PSWF using an old SDE routine re-written in Python
        Find Spheroidal function with M = 6, alpha = 1 using the rational
        approximations discussed by Fred Schwab in 'Indirect Imaging'.
        This routine was checked against Fred's SPHFN routine, and agreed
        to about the 7th significant digit.
        The gridding function is (1-NU**2)*GRDSF(NU) where NU is the distance
        to the edge. The grid correction function is just 1/GRDSF(NU) where NU
        is now the distance to the edge of the image.
        Author: Tim Cornwell
        """
        p = np.array([[8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
                         [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])
        q = np.array([[1.0000000e0, 8.212018e-1, 2.078043e-1], [1.0000000e0, 9.599102e-1, 2.918724e-1]])

        _, n_p = p.shape
        _, n_q = q.shape

        nu = np.abs(nu)

        nuend = np.zeros_like(nu)
        part = np.zeros(len(nu), dtype='int')
        part[(nu >= 0.0) & (nu < 0.75)] = 0
        part[(nu >= 0.75) & (nu <= 1.0)] = 1
        nuend[(nu >= 0.0) & (nu < 0.75)] = 0.75
        nuend[(nu >= 0.75) & (nu <= 1.0)] = 1.0

        delnusq = nu ** 2 - nuend ** 2

        top = p[part, 0]
        for k in range(1, n_p):
            top += p[part, k] * np.power(delnusq, k)

        bot = q[part, 0]
        for k in range(1, n_q):
            bot += q[part, k] * np.power(delnusq, k)

        grdsf = np.zeros_like(nu)
        ok = (bot > 0.0)
        grdsf[ok] = top[ok] / bot[ok]
        ok = np.abs(nu > 1.0)
        grdsf[ok] = 0.0

        # Return the grid correction function
        return grdsf
