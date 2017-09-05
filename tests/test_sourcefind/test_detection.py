from __future__ import print_function

import attr
import numpy as np
import scipy.ndimage
from pytest import approx

from fastimgproto.fixtures.image import (
    add_gaussian2d_to_image,
    gaussian_point_source,
    uncorrelated_gaussian_noise_background,
)
from fastimgproto.fixtures.sourcefits import (
    check_single_source_extraction_successful,
)
from fastimgproto.sourcefind.fit import Gaussian2dParams
from fastimgproto.sourcefind.image import (
    SourceFindImage,
    estimate_rms,
    extremum_pixel_index,
)

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
    add_gaussian2d_to_image(bright_src, img)
    add_gaussian2d_to_image(faint_src, img)
    rms_est = estimate_rms(img)
    # print "RMS EST:", rms_est
    assert np.abs((rms_est - rms) / rms) < 0.05


def test_basic_source_detection():
    """
    We use a flat background (rather than noisy) to avoid random-noise fluctuations
    causing erroneous detections (and test-failures).

    Check the correct values are returned for a single source, then
    start adding secondary sources and check we get multiple finds as
    expected...
    """
    img = np.zeros((ydim, xdim))
    add_gaussian2d_to_image(bright_src, img)

    sf = SourceFindImage(img, detection_n_sigma=4,
                         analysis_n_sigma=3,
                         rms_est=rms,
                         find_negative_sources=False)
    assert len(sf.islands) == 1
    found_src = sf.islands[0].params

    # Check the max-pixel is correct:
    assert found_src.extremum.value == np.max(img)
    max_pixel_index = extremum_pixel_index(img, 1)
    assert (max_pixel_index == attr.astuple(found_src.extremum.index))

    # For a *compact* Gaussian profile this should also be close to the
    # true subpixel position:
    assert np.abs(found_src.extremum.index.x - bright_src.x_centre) < 0.5
    assert np.abs(found_src.extremum.index.y - bright_src.y_centre) < 0.5

    # And peak-pixel value within a reasonable range of the amplitude:
    # - to within 10%, anyway
    assert found_src.extremum.value == approx(bright_src.amplitude, rel=0.1)

    # We expect to detect the bright source, but not the faint one.
    add_gaussian2d_to_image(faint_src, img)
    sf = SourceFindImage(img, detection_n_sigma=4,
                         analysis_n_sigma=3,
                         rms_est=rms,
                         find_negative_sources=False)
    assert len(sf.islands) == 1
    # Unless we add it again and effectively double the faint_src flux
    add_gaussian2d_to_image(faint_src, img)
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
    add_gaussian2d_to_image(negative_src, img)
    # img += evaluate_model_on_pixel_grid(img.shape, faint_src)

    sf = SourceFindImage(img, detection_n_sigma=4,
                         analysis_n_sigma=3,
                         rms_est=rms)
    assert len(sf.islands) == 1
    found_island = sf.islands[0]
    # print()
    # print(negative_src)
    # print(found_island.params)
    assert np.abs(found_island.extremum.index.x - negative_src.x_centre) < 0.5
    assert np.abs(found_island.extremum.index.y - negative_src.y_centre) < 0.5

    add_gaussian2d_to_image(bright_src, img)
    sf = SourceFindImage(img, detection_n_sigma=4,
                         analysis_n_sigma=3,
                         rms_est=rms)
    assert len(sf.islands) == 2
    positive_islands = [i for i in sf.islands if i.sign == 1]
    negative_islands = [i for i in sf.islands if i.sign == -1]
    assert len(positive_islands) == 1
    assert len(negative_islands) == 1
    assert negative_islands[0] == found_island

    neg_island = negative_islands[0]
    pos_island = positive_islands[0]


    min_pixel_index = extremum_pixel_index(img, -1)
    assert attr.astuple(neg_island.extremum.index) == min_pixel_index

    max_pixel_index = extremum_pixel_index(img, 1)
    assert attr.astuple(pos_island.extremum.index) == max_pixel_index

    # Sanity check that the island masks look sensible
    # Check that the mask==False regions are disjoint - taking the boolean OR
    # on both masks should result in a fully `True` mask-array.
    assert (np.logical_or(neg_island.data.mask,pos_island.data.mask).all())

    # And that the true/false regions look sensible for the extremum pixels:
    assert neg_island.data.mask[min_pixel_index] == False
    assert neg_island.data.mask[max_pixel_index] == True
    assert pos_island.data.mask[min_pixel_index] == True
    assert pos_island.data.mask[max_pixel_index] == False


def test_memory_usage():
    """
    Double check that the sourcefinder code behaves as expected,
    i.e. the same image-data is passed around and reused rather than
    being extraneously copied.

    This is non-obvious because the Island analysis creates a masked-array
    version of the original image data - is that a view or a copy?

    """
    img = np.zeros((ydim, xdim))
    add_gaussian2d_to_image(bright_src, img)

    # img += evaluate_model_on_pixel_grid(img.shape, faint_src)

    sf = SourceFindImage(img, detection_n_sigma=4,
                         analysis_n_sigma=3,
                         rms_est=rms,
                         find_negative_sources=False)
    assert len(sf.islands) == 1
    island = sf.islands[0]
    assert island.parent_data is sf.data

    # island.data is the masked array (image masked to just this island)
    island_view = island.data
    # Check if the underlying data is the same:
    assert np.may_share_memory(sf.data, island_view.data)

    assert (sf.data == island_view.data).all()
    # Check our initial conditions
    assert sf.data[0][0] != 42.
    # Alter the masked-array underlying data:
    island_view.data[0][0] = 42.
    # Check if the original image data has been altered accordingly:
    assert sf.data[0][0] == 42.


def test_connectivity():
    """
    Illustrative test-case where the connectivity structure is crucial.

    As per scipy docs...
    https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.ndimage.label.html#scipy.ndimage.label
    ...the default structuring element is::

        [[0,1,0],
        [1,1,1],
        [0,1,0]]

    This favours connectivity along the x and y axes. However, it's quite easy
    to find (e.g. certainly within a 1000 random realisations) a source with
    parameters which produces 'two islands' as seen by this connectivity
    structure - all you need is a reasonably narrow / elongated Gaussian with
    semimajor-axis aligned to the diagonal, with a certain combination of
    analysis threshold / source-amplitude such that the off-axis pixels don't
    quite make the cut to connect the diagonal ridge.

    The alternative - a 3x3 connectivity search - is also imperfect, since
    the diagonal sample-spacing is larger than the on-axis sample-spacing,
    so we're not exactly comparing like with like. But on balance it's probably
    the better option - sophisticated sourcefinders can always attempt to
    'deblend' connected regions, anyway.
    """
    test_source = Gaussian2dParams(x_centre=18.68,
                                   y_centre=34.55,
                                   amplitude=6,
                                   semimajor=1.2,
                                   semiminor=0.5,
                                   theta=np.pi / 4.)
    ydim = 64
    xdim = 32
    image_shape = (ydim, xdim)
    img = np.zeros(image_shape)
    add_gaussian2d_to_image(test_source, img)

    sfimage = SourceFindImage(img, detection_n_sigma=4,
                              analysis_n_sigma=3,
                              rms_est=rms,
                              find_negative_sources=False)

    binary_map = sfimage.data > sfimage.analysis_n_sigma * sfimage.rms_est

    default_structure_element = np.array([[0, 1, 0],
                                          [1, 1, 1],
                                          [0, 1, 0], ],
                                         dtype=int)

    diagonal_structure_element = np.array([[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1], ],
                                          dtype=int)

    label_map, n_labels = scipy.ndimage.label(binary_map,
                                              structure=default_structure_element)
    assert n_labels == 2
    label_map, n_labels = scipy.ndimage.label(binary_map,
                                              structure=diagonal_structure_element)
    assert n_labels == 1

    # Fails with default connectivity structure:
    assert len(sfimage.islands) == 1


def test_peak_pixel_offset():
    """
    Another test-case. This one demonstrates that, given just the right
    conditions, an extended profile can result in the peak-pixel being
    surprisingly far from the true centre.

    I have my suspicions that you can find (vanishingly rare) cases of this
    occurring even for 2 or 3 pixels apart, if you have a large, elongated
    profile.

    Illustrates that moments are definitely the way to go, even for basic
    property checking in the *absence* of noise!
    """
    test_source = Gaussian2dParams(x_centre=18.472842883054525,
                                   y_centre=34.48307160810558,
                                   amplitude=12.428984813225547,
                                   semimajor=8.334020542093349,
                                   semiminor=1.7607968749558363,
                                   theta=-1.3864202588398769)
    img = np.zeros((64, 32))
    add_gaussian2d_to_image(test_source, img)
    sf_img = SourceFindImage(img, detection_n_sigma=4., analysis_n_sigma=3.,
                             rms_est=1.)
    check_single_source_extraction_successful(test_source, sf_img)
    assert len(sf_img.islands) == 1
    island_pars = sf_img.islands[0].params
    # This may come as a bit of a surprise - it's not even in the directly
    # adjacent pixel
    assert np.abs(island_pars.extremum.index.y - test_source.y_centre) > 1.5
