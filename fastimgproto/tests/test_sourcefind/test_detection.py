from fastimgproto.tests.fixtures.image import (
    evaluate_model_on_pixel_grid,
    gaussian_point_source,
    uncorrelated_gaussian_noise_background
)
import fastimgproto.sourcefind.detect as detect
import numpy as np

ydim = 128
xdim = 64
rms = 1.0
bright_src = gaussian_point_source(x_centre=48, y_centre=52, amplitude=10.0)
faint_src = gaussian_point_source(x_centre=32, y_centre=64, amplitude=3.5)


def test_rms_estimation():
    img = uncorrelated_gaussian_noise_background(shape=(ydim, xdim),
                                                 sigma=rms)
    img += evaluate_model_on_pixel_grid(img.shape, bright_src)
    img += evaluate_model_on_pixel_grid(img.shape, faint_src)
    rms_est= detect.estimate_rms(img)
    # print "RMS EST:", rms_est
    assert np.abs((rms_est - rms) / rms) < 0.05

def test_basic_source_detection():
    """
    We use a flat background (rather than noisy) to avoid random-noise fluctuations
    causing erroneous detections (and test-failures).
    """
    img = np.zeros((ydim, xdim))
    img += evaluate_model_on_pixel_grid(img.shape, bright_src)
    # img += evaluate_model_on_pixel_grid(img.shape, faint_src)

    srcs = detect.extract_sources(img, detection_n_sigma=4,
                                  analysis_n_sigma=3,
                                  rms_estimate=rms)
    assert len(srcs) == 1
    pixel_idx, pixel_value = srcs[0]
    y_idx, x_idx = pixel_idx
    assert np.abs(y_idx - bright_src.y_mean) <1.1
    assert np.abs(x_idx - bright_src.x_mean) < 1.1

    # We expect to detect the bright source, but not the faint one.
    img += evaluate_model_on_pixel_grid(img.shape, faint_src)
    srcs = detect.extract_sources(img, detection_n_sigma=4,
                                  analysis_n_sigma=3,
                                  rms_estimate=rms)
    assert len(srcs) == 1
    # Unless we add it again and effectively double the faint_src flux
    img += evaluate_model_on_pixel_grid(img.shape, faint_src)
    srcs = detect.extract_sources(img, detection_n_sigma=4,
                                  analysis_n_sigma=3,
                                  rms_estimate=rms)
    assert len(srcs) == 2

