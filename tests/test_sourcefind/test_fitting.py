from __future__ import print_function

import inspect
import logging
from functools import partial

import attr
import numpy as np
import scipy.optimize
from pytest import approx

from fastimgproto.fixtures.image import add_gaussian2d_to_image
from fastimgproto.fixtures.sourcefits import (
    check_single_source_extraction_successful,
    generate_random_source_params,
)
from fastimgproto.sourcefind.fit import (
    Gaussian2dParams,
    gaussian2d,
    gaussian2d_jac,
)
from fastimgproto.sourcefind.image import SourceFindImage

logger = logging.getLogger(__name__)


def test_gauss2d_func_and_jacobian():
    assert (inspect.getargspec(gaussian2d).args ==
            inspect.getargspec(gaussian2d_jac).args)

    profile = Gaussian2dParams(x_centre=4,
                               y_centre=9,
                               amplitude=10.,
                               semimajor=1.5,
                               semiminor=1.2,
                               theta=np.pi / 4.
                               )
    # Generate some random positions to test
    rstate = np.random.RandomState(seed=42)
    n_samples = 10

    pars = attr.astuple(profile)
    n_tests_run = 0

    x_samples = rstate.uniform(profile.x_centre - profile.semimajor * 5.,
                               profile.x_centre + profile.semimajor * 5.,
                               size=n_samples
                               )
    y_samples = rstate.uniform(profile.y_centre - profile.semimajor * 5.,
                               profile.y_centre + profile.semimajor * 5.,
                               size=n_samples
                               )

    xy_pairs = np.column_stack((x_samples, y_samples))

    looped_g2d = np.array(
        [gaussian2d(*tuple(list(pair) + list(attr.astuple(profile))))
         for pair in xy_pairs])
    vectored_g2d = gaussian2d(x_samples, y_samples, *attr.astuple(profile))
    assert (looped_g2d == vectored_g2d).all()

    looped_jac = np.array(
        [gaussian2d_jac(*tuple(list(pair) + list(attr.astuple(profile))))
         for pair in xy_pairs])
    vectored_jac = gaussian2d_jac(x_samples, y_samples, *attr.astuple(profile))

    print(looped_jac.shape)
    print(vectored_jac.shape)
    assert (looped_jac.shape == vectored_jac.shape)
    assert (looped_jac == vectored_jac).all()

    # assert

    for x in x_samples:
        for y in y_samples:
            # Need to partially bind the functions for each x/y location
            def located_g2d(x_centre, y_centre, amplitude, x_stddev, y_stddev,
                            theta):
                return gaussian2d(x, y, x_centre, y_centre, amplitude,
                                  x_stddev, y_stddev,
                                  theta)

            def located_g2d_jac(x_centre, y_centre, amplitude, x_stddev,
                                y_stddev,
                                theta):
                return gaussian2d_jac(x, y, x_centre, y_centre, amplitude,
                                      x_stddev, y_stddev,
                                      theta)

            def wrapped_gaussian2d(pars):
                assert len(pars) == 6
                return located_g2d(*pars)

            def wrapped_gaussian2d_jac(pars):
                assert len(pars) == 6
                return located_g2d_jac(*pars)

            n_tests_run += 1
            logger.debug("Gradient check {} of {}".format(
                n_tests_run, n_samples ** 2))
            logger.debug("Checking gradient at x/y position {} {}".format(x, y))

            eps = np.sqrt(np.finfo(float).eps)  # default for check_grad
            # eps = 1e-12

            numerical = scipy.optimize.approx_fprime(pars, wrapped_gaussian2d,
                                                     epsilon=eps, )
            analytic = wrapped_gaussian2d_jac(pars)
            check_grad_result = scipy.optimize.check_grad(
                wrapped_gaussian2d,
                wrapped_gaussian2d_jac,
                x0=pars,
                epsilon=eps,
            )
            # logger.debug("Analytic {}".format(analytic))
            # logger.debug("Numerical {}".format(numerical))
            #
            # logger.debug("Diffs: manual {}, check grad {}".format(
            #     np.abs(analytic - numerical),
            #     check_grad_result))
            # logger.debug("")

            np.testing.assert_allclose(analytic, numerical,
                                       atol=1e-7)
            assert check_grad_result == approx(0., abs=2e-6)

    def test_basic_source_detection_and_fitting():
        ydim = 64
        xdim = 32
        image_shape = (ydim, xdim)
        seed = 123456

        base_x = 18
        base_y = 34
        n_sources = 10
        positive_sources = generate_random_source_params(n_sources=n_sources,
                                                         base_x=base_x,
                                                         base_y=base_y,
                                                         amplitude_range=(
                                                             4.5, 42.),
                                                         semiminor_range=(
                                                             0.4, 2.),
                                                         axis_ratio_range=(
                                                             1., 5.),
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

        n_completed_fits = sum(
            1 for f in fits if f)  # Count where f is not False
        logger.debug(
            "{} of {} island-fits completed".format(n_completed_fits,
                                                    n_islands))

        success = np.zeros_like(positive_sources, dtype=int)
        for idx, lsq_fit in enumerate(fits):
            if lsq_fit is None:
                success[idx] = 1  # Trivial no-island case. Count as OK.
            elif (positive_sources[idx].comparable_params ==
                      approx(lsq_fit.comparable_params, rel=1e-2)):
                success[idx] = 1

        n_successful = success.sum()
        logger.debug("{} of {} sources fitted accurately".format(
            n_successful, len(positive_sources)
        ))
        # assert n_successful == len(positive_sources )


        bad_idx = np.where(success == 0)
        bad_truth = np.array([attr.astuple(s) for s in positive_sources])[
            bad_idx]
        bad_fits = np.array([attr.astuple(s) for s in fits])[bad_idx]
