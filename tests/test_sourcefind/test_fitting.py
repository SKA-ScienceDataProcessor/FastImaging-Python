from __future__ import print_function

import inspect
import logging

import attr
import numpy as np
from pytest import approx

from fastimgproto.fixtures.image import add_gaussian2d_to_image
from fastimgproto.fixtures.sourcefits import (
    check_single_source_extraction_successful,
    generate_random_source_params,
)
from fastimgproto.sourcefind.fit import gaussian2d, gaussian2d_jac
from fastimgproto.sourcefind.image import SourceFindImage

logger = logging.getLogger(__name__)


def test_gauss2d_func_and_jacobian():
    assert (inspect.getargspec(gaussian2d).args ==
            inspect.getargspec(gaussian2d_jac).args)


def test_basic_source_detection_and_fitting():
    ydim = 64
    xdim = 32
    image_shape = (ydim, xdim)
    seed = 123456

    base_x = 18
    base_y = 34
    n_sources = 100
    positive_sources = generate_random_source_params(n_sources=n_sources,
                                                     base_x=base_x,
                                                     base_y=base_y,
                                                     amplitude_range=(4.5, 42.),
                                                     semiminor_range=(0.4, 2.),
                                                     axis_ratio_range=(1., 5.),
                                                     )
    islands = []
    for src in positive_sources:
        img = np.zeros(image_shape)
        add_gaussian2d_to_image(src, img)
        detection_thresh = 4.
        sfimg = SourceFindImage(img, detection_n_sigma=detection_thresh,
                                analysis_n_sigma=3.,
                                rms_est=1.,
                                find_negative_sources=True)

        check_single_source_extraction_successful(src, sfimg)
        if sfimg.islands:
            assert len(sfimg.islands) == 1
            sfimg.islands[0].fit_gaussian_2d(verbose=1)
            islands.append(sfimg.islands[0])

    fits = [i.fit for i in islands]

    success = np.zeros_like(positive_sources, dtype=int)
    for idx, _ in enumerate(positive_sources):
        if (positive_sources[idx].comparable_params ==
                approx(islands[idx].fit.comparable_params)):
            success[idx] = 1

    logger.debug("{} of {} fits successful".format(success.sum(), len(islands)))

    bad_idx = np.where(success == 0)
    bad_truth = np.array([attr.astuple(s) for s in positive_sources])[bad_idx]
    bad_fits = np.array([attr.astuple(s) for s in fits])[bad_idx]


    #
    # half_diagonal = math.sqrt(2.) / 2
    # assert np.abs(found_src.extremum_x_idx - source.x_centre) < half_diagonal
    # assert np.abs(found_src.extremum_y_idx - source.y_centre) < half_diagonal
    # assert np.abs(found_src.xbar - source.x_centre) < 0.1
    # assert np.abs(found_src.ybar - source.y_centre) < 0.1
    #
    # # Now, do the fitting routines work?
    # assert isinstance(fit, Gaussian2dFit)
    # assert source.comparable_params == approx(fit.comparable_params)
    # logger.debug("Fit success")
