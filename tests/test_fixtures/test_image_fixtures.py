import numpy as np

from fastimgproto.fixtures.image import (
    evaluate_model_on_pixel_grid,
    gaussian_point_source,
)


def test_model_generation_and_evaluation():
    ydim = 128
    xdim = 64
    img = np.zeros((ydim, xdim))
    src = gaussian_point_source(x_centre=32, y_centre=64)
    img += evaluate_model_on_pixel_grid(img.shape, src)
    assert img[0, 0] == 0.0
    assert np.abs(img[int(src.y_mean.value), int(src.x_mean.value)]
                    - src.amplitude) < 0.01
