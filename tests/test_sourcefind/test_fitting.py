from __future__ import print_function

import inspect
import logging

import attr
import numpy as np
import scipy.optimize
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
    scipy.optimize.check_grad(gaussian2d, gaussian2d_jac)



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
                                                     seed=123456,
                                                     )
    n_islands = 0
    fits = []
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
            n_islands += 1
            assert len(sfimg.islands) == 1
            lsq_fit = sfimg.fit_gaussian_2d(sfimg.islands[0], verbose=1)
            fits.append(lsq_fit)
        else:
            fits.append(None)

    n_completed_fits = sum(1 for f in fits if f)  # Count where f is not False
    logger.debug(
        "{} of {} island-fits completed".format(n_completed_fits, n_islands))

    success = np.zeros_like(positive_sources, dtype=int)
    for idx, lsq_fit in enumerate(fits):
        if lsq_fit is None:
            success[idx] = 1 # Trivial no-island case. Count as OK.
        elif (positive_sources[idx].comparable_params ==
                approx(lsq_fit.comparable_params,rel=1e-2)):
            success[idx] = 1

    n_successful = success.sum()
    logger.debug("{} of {} sources fitted accurately".format(
        n_successful, len(positive_sources)
    ))
    # assert n_successful == len(positive_sources )


    bad_idx = np.where(success == 0)
    bad_truth = np.array([attr.astuple(s) for s in positive_sources])[bad_idx]
    bad_fits = np.array([attr.astuple(s) for s in fits])[bad_idx]
