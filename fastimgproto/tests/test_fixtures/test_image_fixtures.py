from fastimgproto.tests.fixtures.image import (
    add_models_to_background,
    gaussian_point_source,
)
import numpy as np


def test_model_generation_and_evaluation():
    ydim = 128
    xdim = 64
    img = np.zeros((ydim, xdim))

    src = gaussian_point_source(x_centre=32, y_centre=64)
    add_models_to_background(img, [src])
    assert img[0, 0] == 0.0
    assert img[src.y_mean, src.x_mean] - src.amplitude < 0.01
