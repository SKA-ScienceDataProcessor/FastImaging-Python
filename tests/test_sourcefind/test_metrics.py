import math

import attr
import numpy as np
from pytest import approx

from fastimgproto.sourcefind.fit import Gaussian2dParams
from fastimgproto.sourcefind.metrics import (
    mahalanobis_distance,
    pixel_distance,
    shape_difference,
)


def test_distance_metrics():
    x_0, y_0 = 1., 4.
    g2d_x_long = Gaussian2dParams(x_centre=x_0,
                                  y_centre=y_0,
                                  amplitude=1,
                                  semimajor=2.,
                                  semiminor=1.,
                                  theta=0.,
                                  )

    # A Gaussian elongated in the y-direction (Note theta)
    g2d_y_long = attr.evolve(g2d_x_long, theta=np.pi / 2)

    x, y = x_0 + 2, y_0
    assert mahalanobis_distance(g2d_x_long, x=x, y=y) == approx(1.0)
    assert mahalanobis_distance(g2d_y_long, x=x, y=y) == approx(2.0)
    assert pixel_distance(g2d_x_long, x, y) == pixel_distance(g2d_y_long, x, y)
    assert pixel_distance(g2d_x_long, x, y) == 2.

    x, y = x_0, y_0 + 1
    assert mahalanobis_distance(g2d_x_long, x=x, y=y) == approx(1.0)
    assert mahalanobis_distance(g2d_y_long, x=x, y=y) == approx(0.5)
    assert pixel_distance(g2d_x_long, x, y) == pixel_distance(g2d_y_long, x, y)
    assert pixel_distance(g2d_x_long, x, y) == 1.

    g2d_diag_long = attr.evolve(g2d_x_long, theta=np.pi / 4.)
    x, y = x_0 + np.cos(np.pi / 4), y_0 + np.sin(np.pi / 4)
    assert mahalanobis_distance(g2d_diag_long, x=x, y=y) == approx(0.5)
    assert pixel_distance(g2d_diag_long, x, y) == approx(1.)


def test_shape_metric():
    g_symmetric_1 = Gaussian2dParams(x_centre=0.,
                                     y_centre=0.,
                                     amplitude=1,
                                     semimajor=1.,
                                     semiminor=1.,
                                     theta=0.,
                                     )

    assert shape_difference(g_symmetric_1, g_symmetric_1) == 0

    translated_sym = attr.evolve(g_symmetric_1, x_centre=123,
                                 y_centre=456, )
    assert shape_difference(g_symmetric_1, translated_sym) == 0

    # These are all the same 'shape distance' from the symmetric circular
    # Gaussian:
    g_x_2 = attr.evolve(g_symmetric_1, semimajor=2., semiminor=1.)
    g_y_2 = attr.evolve(g_x_2, theta=np.pi / 2.)
    g_diag_2 = attr.evolve(g_x_2, theta=np.pi / 4.)
    g_x_half = attr.evolve(g_symmetric_1, semiminor=0.5)

    symmetric_to_elongated_by_2 = shape_difference(g_symmetric_1, g_x_2)
    for g_pars in (g_x_2, g_y_2, g_diag_2, g_x_half):
        assert shape_difference(g_symmetric_1,
                                g_pars) == symmetric_to_elongated_by_2

    # Twice the elongation factor -> Twice the distance metric
    assert shape_difference(g_x_half, g_x_2) == 2 * symmetric_to_elongated_by_2

    # Elongated in both directions -> Twice the distance metric
    g_symmetric_2 = attr.evolve(g_symmetric_1, semimajor=2., semiminor=2.)
    assert (shape_difference(g_symmetric_1, g_symmetric_2) ==
            approx(2 * symmetric_to_elongated_by_2))

    # Elongated and rotated by 90-degrees:
    # (Equivalent to shrink in x, stretch in y)
    # -> Twice the distance metric
    assert (shape_difference(g_x_2, g_y_2) ==
            approx(2 * symmetric_to_elongated_by_2))

    # Rotation by 45-degrees:
    assert (symmetric_to_elongated_by_2 <
            shape_difference(g_diag_2, g_x_2) <
            2 * symmetric_to_elongated_by_2)
