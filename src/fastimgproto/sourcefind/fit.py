"""
Routines for fitting source profiles (currently just 2d-Gaussian)

We borrow from the Astropy modeling routines, since they provide a tested
implementation of both the 2d Gaussian and its Jacobian
(first partial derivatives).

We add routines for estimation of initial fitting parameters (method of moments,
using the TraP (https://github.com/transientskp/tkp routines for reference),
and for processing the fits to determine which is the semimajor/minor
axis, and constrain the rotation angle to within 180 degrees.

To do:
If time allows, understand, document and re-implement the error-estimation
routines currently implemented in the TraP (Condon 1995, Spreeuw's Thesis).
Initial implementation will have to make do with naive errors.
"""

import math

import astropy.units as u
import numpy as np
import pytest
from attr import attrib, attrs

from fastimgproto.coords import rotate_basis


def _valid_semimajor(instance, attribute, value):
    if not value > 0.:
        raise ValueError("Semimajor axis value must be positive.")


def _valid_semiminor(instance, attribute, value):
    """
    Check if the semiminor axis is smaller than semimajor.

    We leave a little bit of wiggle room (`rel_tol`) to ignore values
    that are almost within numerical precision limits
    """
    if not value > 0.:
        raise ValueError("Semiminor axis value must be positive.")
    rel_tol = 1e-12
    tol_factor = 1. + rel_tol
    if value > instance.semimajor * tol_factor:
        raise ValueError("Semiminor axis should be smaller than semimajor.")


def _valid_theta(instance, attribute, value):
    """
    Check if theta lies in the range (-pi/2,pi/2].
    """
    half_pi = np.pi / 2.
    if (value <= -half_pi) or (value > half_pi):
        raise ValueError("Theta should lie in the range (-pi/2,pi/2].")


@attrs(frozen=True)
class Gaussian2dParams(object):
    """
    Data structure for representing a 2d Gaussian profile.

    Similar to an astropy Gaussian2d parameter set, but we refer to
    semimajor/semiminor axis length rather than x_std_dev / y_std_dev.

    Otherwise all values have the same meaning - we just always assume
    that `x_std_dev > y_std_dev`, or equivalently, that theta describes the
    rotation in the counterclockwise sense of the semimajor axis from
    the positive x-direction. (For fits returned where this does not happen
    to be true, we swap semi-major/minor and adjust theta accordingly.)

    NB - we don't use astropy.units / astropy.units.Quantity parameters here,
    to ease interfacing with the scipy fitting routines. Most of the parameters
    are obvious anyway - pixels - but take care that theta is in radians.

    All values are in units of pixels, except for theta which has units of
    radians.

    The data-structure is 'frozen' to avoid inadvertent modification of values,
    we don't expect to need to modify a returned fit often.
    """
    x_centre = attrib(convert=float)
    y_centre = attrib(convert=float)
    amplitude = attrib(convert=float)
    semimajor = attrib(convert=float, validator=_valid_semimajor)
    semiminor = attrib(convert=float, validator=_valid_semiminor)
    theta = attrib(convert=float, validator=_valid_theta)

    @property
    def covariance(self):
        """
        Reference covariance matrix

        Returns:
            numpy.ndarray: 2x2 matrix representing covariance matrix in the
            reference x-y frame. Covariance matrix elements are ordered
            like::

                [[sigma_x**2, rho*sigma_x*sigma_y],
                 [rho*sigma_x*sigma_y, sigma_y**2]]
        """
        rotated_cov = np.array([[self.semimajor ** 2, 0],
                                [0, self.semiminor ** 2]],
                               dtype=np.float_)
        ref_cov = rotate_basis(rotated_cov, -self.theta * u.rad)
        return ref_cov

    @property
    def correlation(self):
        """
        Correlation co-efficient between x and y

        (This is effectively a proxy for rotation angle - much easier to
        compare fits with since it does not suffer from degeneracies that
        are inherent to rotation angle.)

        Returns:
            float: Correlation coefficient in the range (-1,1).

        """
        cov_matrix = self.covariance
        rho = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
        return rho

    @staticmethod
    def from_unconstrained_parameters(x_centre, y_centre, amplitude, semimajor,
                                      semiminor, theta):
        """
        Construct from unconstrained parameters, e.g. from a fitting routine.

        If necessary this will swap semimajor / semiminor so that
        semimajor is always the larger of the two, and shift the rotation
        angle appropriately. Also shifts theta to lie within
        (-pi/2,pi/2].


        Args:
            x_centre:
            y_centre:
            amplitude:
            semimajor:
            semiminor:
            theta:

        Returns:
            Gaussian2dParams
        """
        # Semimajor / minor are only evaluated as squares, so unconstrained
        # fits can easily stray into negative values:
        semimajor = np.fabs(semimajor)
        semiminor = np.fabs(semiminor)
        half_pi = np.pi / 2.
        if semimajor < semiminor:
            semimajor, semiminor = semiminor, semimajor
            theta = theta + half_pi
        mod_theta = math.fmod(theta, np.pi)  # Rotations by pi are degeneracies
        # This gets us to the range (-pi,pi). Now we add/subtract an additional
        # pi as required to get down to (-pi/2, pi/2).
        if mod_theta <= -half_pi:
            mod_theta += np.pi
        elif mod_theta > half_pi:
            mod_theta -= np.pi
        return Gaussian2dParams(x_centre, y_centre, amplitude, semimajor,
                                semiminor, mod_theta)

    @property
    def comparable_params(self):
        """
        A tuple of values for easy comparison - replace theta with correlation.
        """
        return (self.x_centre,
                self.y_centre,
                self.amplitude,
                self.semimajor,
                self.semiminor,
                self.correlation,
                )

    def approx_equal_to(self, other, rel_tol=1e-8, abs_tol=1e-12):
        """
        Determine if two Gaussian fits are approximately equivalent.
        """
        return self.comparable_params == pytest.approx(other.comparable_params,
                                                       rel=rel_tol, abs=abs_tol)


def gaussian2d(x, y, x_centre, y_centre, amplitude, x_stddev, y_stddev, theta):
    """
    Two dimensional Gaussian function for use in source-fitting

    A tested implementation of a 2d Gaussian function, with rotation of
    axes. Original Source code:
    https://github.com/astropy/astropy/blob/3b1de6ee3165d176c3e2901028f86be60b4b0f4d/astropy/modeling/functional_models.py#L446

    Wikipedia article on the formula:
    https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function

    Args:
        x (numpy.ndarray): Datatype int. X-Pixel indices to calculate
            Gaussian values for.
        y (numpy.ndarray): Datatype int. Y-Pixel indices to calculate
            Gaussian values for.
        x_centre (float): Mean of the Gaussian in x.
        y_centre (float): Mean of the Gaussian in y.
        amplitude(float): Amplitude of the Gaussian.
        x_stddev(float): Standard deviation of the Gaussian in x before rotating
            by theta.
        y_stddev(float): Standard deviation of the Gaussian in y before rotating
            by theta.
        theta(float): Rotation angle in radians. The rotation angle increases
            counterclockwise.

    Returns:
        numpy.ndarray: Datatype np.float_. m values, one for each pixel fitted.
    """

    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xstd2 = x_stddev ** 2
    ystd2 = y_stddev ** 2
    xdiff = x - x_centre
    ydiff = y - y_centre
    a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
    b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
    c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
    return amplitude * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) +
                                (c * ydiff ** 2)))


def gaussian2d_jac(x, y, x_centre, y_centre, amplitude, x_stddev, y_stddev,
                   theta):
    """
    Jacobian of Gaussian2d.

    (Two dimensional Gaussian function derivative with respect to parameters)

    See :ref:`.gaussian2d` for arg details

    """

    cost = np.cos(theta)
    sint = np.sin(theta)
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    cos2t = np.cos(2. * theta)
    sin2t = np.sin(2. * theta)
    xstd2 = x_stddev ** 2
    ystd2 = y_stddev ** 2
    xstd3 = x_stddev ** 3
    ystd3 = y_stddev ** 3
    xdiff = x - x_centre
    ydiff = y - y_centre
    xdiff2 = xdiff ** 2
    ydiff2 = ydiff ** 2
    a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
    b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
    c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
    g = amplitude * np.exp(-((a * xdiff2) + (b * xdiff * ydiff) +
                             (c * ydiff2)))
    da_dtheta = (sint * cost * ((1. / ystd2) - (1. / xstd2)))
    da_dx_stddev = -cost2 / xstd3
    da_dy_stddev = -sint2 / ystd3
    db_dtheta = (cos2t / xstd2) - (cos2t / ystd2)
    db_dx_stddev = -sin2t / xstd3
    db_dy_stddev = sin2t / ystd3
    dc_dtheta = -da_dtheta
    dc_dx_stddev = -sint2 / xstd3
    dc_dy_stddev = -cost2 / ystd3
    dg_dA = g / amplitude
    dg_dx_mean = g * ((2. * a * xdiff) + (b * ydiff))
    dg_dy_mean = g * ((b * xdiff) + (2. * c * ydiff))
    dg_dx_stddev = g * (-(da_dx_stddev * xdiff2 +
                          db_dx_stddev * xdiff * ydiff +
                          dc_dx_stddev * ydiff2))
    dg_dy_stddev = g * (-(da_dy_stddev * xdiff2 +
                          db_dy_stddev * xdiff * ydiff +
                          dc_dy_stddev * ydiff2))
    dg_dtheta = g * (-(da_dtheta * xdiff2 +
                       db_dtheta * xdiff * ydiff +
                       dc_dtheta * ydiff2))

    return np.array([dg_dx_mean,
                     dg_dy_mean,
                     dg_dA,
                     dg_dx_stddev,
                     dg_dy_stddev,
                     dg_dtheta]).T
