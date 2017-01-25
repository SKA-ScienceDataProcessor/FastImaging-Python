from __future__ import print_function

import numpy as np

from fastimgproto.fixtures.image import (
    evaluate_model_on_pixel_grid,
    gaussian_point_source,
    uncorrelated_gaussian_noise_background
)
from fastimgproto.sourcefind.image import (SourceFindImage, _estimate_rms)

ydim = 128
xdim = 64
rms = 1.0
bright_src = gaussian_point_source(x_centre=48.24, y_centre=52.66,
                                   amplitude=10.0)
faint_src = gaussian_point_source(x_centre=32, y_centre=64, amplitude=3.5)
negative_src = gaussian_point_source(x_centre=24.31, y_centre=28.157,
                                     amplitude=-10.0)


def test_rms_estimation():
    img = uncorrelated_gaussian_noise_background(shape=(ydim, xdim),
                                                 sigma=rms)
    img += evaluate_model_on_pixel_grid(img.shape, bright_src)
    img += evaluate_model_on_pixel_grid(img.shape, faint_src)
    rms_est = _estimate_rms(img)
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

    sf = SourceFindImage(img, detection_n_sigma=4,
                         analysis_n_sigma=3,
                         rms_est=rms,
                         find_negative_sources=False)
    assert len(sf.islands) == 1
    found_src = sf.islands[0]
    # print(bright_src)
    # print(src)
    assert np.abs(found_src.extremum_x_idx - bright_src.x_mean) < 0.5
    assert np.abs(found_src.extremum_y_idx - bright_src.y_mean) < 0.5
    assert np.abs(found_src.xbar - bright_src.x_mean) < 0.1
    assert np.abs(found_src.ybar - bright_src.y_mean) < 0.1

    # We expect to detect the bright source, but not the faint one.
    img += evaluate_model_on_pixel_grid(img.shape, faint_src)
    sf = SourceFindImage(img, detection_n_sigma=4,
                         analysis_n_sigma=3,
                         rms_est=rms,
                         find_negative_sources=False)
    assert len(sf.islands) == 1
    # Unless we add it again and effectively double the faint_src flux
    img += evaluate_model_on_pixel_grid(img.shape, faint_src)
    sf = SourceFindImage(img, detection_n_sigma=4,
                         analysis_n_sigma=3,
                         rms_est=rms,
                         find_negative_sources=False)
    assert len(sf.islands) == 2


def test_negative_source_detection():
    """
    Also need to detect 'negative' sources, i.e. where a source in the
    subtraction model is not present in the data, creating a trough in the
    difference image.
    Again, start by using a flat background (rather than noisy) to avoid
    random-noise fluctuations causing erroneous detections (and test-failures).
    """
    img = np.zeros((ydim, xdim))
    img += evaluate_model_on_pixel_grid(img.shape, negative_src)
    # img += evaluate_model_on_pixel_grid(img.shape, faint_src)

    sf = SourceFindImage(img, detection_n_sigma=4,
                         analysis_n_sigma=3,
                         rms_est=rms)
    assert len(sf.islands) == 1
    found_src = sf.islands[0]
    print()
    print(negative_src)
    print(found_src)
    assert np.abs(found_src.extremum_x_idx - negative_src.x_mean) < 0.5
    assert np.abs(found_src.extremum_y_idx - negative_src.y_mean) < 0.5
    assert np.abs(found_src.xbar - negative_src.x_mean) < 0.1
    assert np.abs(found_src.ybar - negative_src.y_mean) < 0.1

    img += evaluate_model_on_pixel_grid(img.shape, bright_src)
    sf = SourceFindImage(img, detection_n_sigma=4,
                         analysis_n_sigma=3,
                         rms_est=rms)
    assert len(sf.islands) == 2
    positive_islands = [i for i in sf.islands if i.sign == 1]
    negative_islands = [i for i in sf.islands if i.sign == -1]
    assert len(positive_islands) == 1
    assert len(negative_islands) == 1
    assert negative_islands[0] == found_src


def test_memory_usage():
    """
    Double check that the sourcefinder code behaves as expected,
    i.e. the same image-data is passed around and reused rather than
    being extraneously copied.

    This is non-obvious because the Island analysis creates a masked-array
    version of the original image data - is that a view or a copy?

    """
    img = np.zeros((ydim, xdim))
    img += evaluate_model_on_pixel_grid(img.shape, bright_src)
    # img += evaluate_model_on_pixel_grid(img.shape, faint_src)

    sf = SourceFindImage(img, detection_n_sigma=4,
                         analysis_n_sigma=3,
                         rms_est=rms,
                         find_negative_sources=False)
    assert len(sf.islands) == 1
    island = sf.islands[0]
    assert island.parent is sf

    # island.data is the masked array (image masked to just this island)
    island_mask = island.data
    # Check if the underlying data is the same:
    assert np.may_share_memory(sf.data, island_mask.data)

    assert (sf.data == island_mask.data).all()
    # Check our initial conditions
    assert sf.data[0][0] != 42.
    # Alter the masked-array underlying data:
    island_mask.data[0][0] = 42.
    # Check if the original image data has been altered accordingly:
    assert sf.data[0][0] == 42.
    