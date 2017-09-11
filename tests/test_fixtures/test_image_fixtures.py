import numpy as np
from pytest import approx

from fastimgproto.fixtures.image import (
    add_gaussian2d_to_image,
    gaussian_point_source,
)


def test_model_generation_and_evaluation():
    ydim = 128
    xdim = 64
    img = np.zeros((ydim, xdim))
    src = gaussian_point_source(x_centre=32, y_centre=64)
    add_gaussian2d_to_image(src, img)
    assert img[0, 0] == 0.0
    assert img[int(src.y_centre), int(src.x_centre)]== approx(src.amplitude)
