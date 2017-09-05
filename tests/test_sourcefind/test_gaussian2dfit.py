"""
Check if the Gaussian2dFit class methods work as expected
"""
import attr
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from pytest import approx

from fastimgproto.sourcefind.fit import Gaussian2dParams


def test_correlation_coefficient():
    smaj = 1.5
    smin = 1.
    g1 = Gaussian2dParams(x_centre=0,
                          y_centre=0,
                          amplitude=1,
                          semimajor=smaj,
                          semiminor=smin,
                          theta=0.,
                          )

    assert g1.correlation == 0

    cov = np.array([[smaj * smaj, 0],
                    [0, smin * smin]])
    assert_equal(g1.covariance, cov)

    cov_rotated = np.array([[smin * smin, 0],
                            [0, smaj * smaj]])
    g1 = attr.evolve(g1, theta=np.pi / 2)
    assert_allclose(g1.covariance, cov_rotated, atol=1e-10)

    smaj = 1e5
    g2 = Gaussian2dParams(x_centre=0,
                          y_centre=0,
                          amplitude=1,
                          semimajor=smaj,
                          semiminor=smin,
                          theta=0.,
                          )
    assert g2.correlation == 0
    g2 = attr.evolve(g2, theta=np.pi / 4)
    # print()
    # print(g2.correlation)
    # print(1.0 - g2.correlation)
    assert g2.correlation == approx(1.0, rel=1e-7)
    g2 = attr.evolve(g2, theta=-np.pi / 4)
    assert g2.correlation == approx(-1.0, rel=1e-7)


def test_approx_equality():
    init_pars = dict(x_centre=48.24, y_centre=52.66,
                     amplitude=42.,
                     semimajor=1.5,
                     semiminor=1.4,
                     theta=1.1,
                     )
    g1 = Gaussian2dParams(**init_pars)
    g2 = Gaussian2dParams(**init_pars)
    assert g1 == g2
    g2 = attr.evolve(g2,
                     y_centre=g2.y_centre + 5e-9,
                     amplitude=g2.amplitude - 5e-9,
                     )

    assert g1 != g2
    assert g2.approx_equal_to(g1)

    g2 = attr.evolve(g2,
                     theta=g2.theta + g2.theta * 1e-4)
    # It appears that the correlation sensitivity matches reasonably well
    # to the order-of-change to theta (at least for these semimajor/minor):
    assert g2.comparable_params == approx(g1.comparable_params, rel=2e-4)
    assert g2.comparable_params != approx(g1.comparable_params, rel=1e-4)


def test_validation_after_evolve_call():
    smaj = 1.5
    smin = 1.

    g1 = Gaussian2dParams(x_centre=0,
                          y_centre=0,
                          amplitude=1,
                          semimajor=smaj,
                          semiminor=smin,
                          theta=0.,
                          )
    with pytest.raises(ValueError):
        g2 = attr.evolve(g1, semimajor=smin - 0.1)


def test_unconstrained_initialization_theta_oob():
    pars = dict(x_centre=48.24, y_centre=52.66,
                  amplitude=42.,
                  semimajor=1.5,
                  semiminor=1.4,
                  theta=np.pi / 4,
                  )
    g1 = Gaussian2dParams(**pars)

    pars.update(theta=g1.theta + np.pi)
    g2 = Gaussian2dParams.from_unconstrained_parameters(**pars)
    assert g1 == g2

    pars.update(theta=g1.theta + 2*np.pi)
    g3 = Gaussian2dParams.from_unconstrained_parameters(**pars)
    assert g1 == g3

    pars.update(theta=g1.theta - np.pi)
    g4 = Gaussian2dParams.from_unconstrained_parameters(**pars)
    assert g1 == g4

    # Check initialization at bounds:
    # Include upper bound
    pars.update(theta=np.pi/2.)
    g5 = Gaussian2dParams(**pars)

    # Shouldn't work - exclude lower bound
    pars.update(theta=-np.pi/2.)
    with pytest.raises(ValueError):
        g6 = Gaussian2dParams(**pars)
    # Should flip lower bound to upper bound:
    g6 = Gaussian2dParams.from_unconstrained_parameters(**pars)
    assert g6.theta == np.pi / 2.



def test_unconstrained_initialization_flipped_major_minor():
    smaj = 1.5
    smin = 1.3
    g1 = Gaussian2dParams(x_centre=0,
                          y_centre=0,
                          amplitude=1,
                          semimajor=smaj,
                          semiminor=smin,
                          theta=np.pi / 4.,
                          )

    g2 = Gaussian2dParams.from_unconstrained_parameters(
        g1.x_centre, g1.y_centre, g1.amplitude,
        semimajor=g1.semiminor,
        semiminor=g1.semimajor,
        theta=g1.theta + np.pi/2.
    )
    assert g1 == g2
