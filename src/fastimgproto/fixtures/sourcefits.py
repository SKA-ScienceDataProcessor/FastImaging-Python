from __future__ import print_function

import inspect
import logging

import attr
import numpy as np
from pytest import approx

from fastimgproto.fixtures.image import (
    add_gaussian2d_to_image,
    gaussian_point_source,
)
from fastimgproto.sourcefind.fit import (
    Gaussian2dParams,
    gaussian2d,
    gaussian2d_jac,
)
from fastimgproto.sourcefind.image import SourceFindImage

logger = logging.getLogger(__name__)


def generate_random_source_params(n_sources,
                                  base_x,
                                  base_y,
                                  amplitude_range,
                                  semiminor_range,
                                  axis_ratio_range,
                                  seed=None):
    """
    Set up random, reasonably valued parameters for n_sources

    Args:
        n_sources:
        base_x:
        base_y:
        amplitude_range:
        seed:

    Returns:
        list[Gaussian2dParams]: List of pseudo-randomly generated sources.
    """
    if axis_ratio_range[0] < 1.:
        raise ValueError("Axis ratios must be 1. at minimum "
                         "(ensures 'semimajor > semiminor')")

    rstate = np.random.RandomState(seed)
    x = base_x + rstate.uniform(0., 1., n_sources)
    y = base_y + rstate.uniform(0., 1., n_sources)
    amplitude = rstate.uniform(amplitude_range[0],
                               amplitude_range[1],
                               n_sources)
    semiminor = rstate.uniform(semiminor_range[0],
                               semiminor_range[1],
                               n_sources)
    semimajor = semiminor * rstate.uniform(axis_ratio_range[0],
                                           axis_ratio_range[1],
                                           size=n_sources)
    # For numpy.random.uniform
    # "Samples are uniformly distributed over the half-open interval
    # [low, high) (includes low, but excludes high)."
    # This is the opposite of what we want, so we simply flip the sign.
    theta = -1. * rstate.uniform(-np.pi / 2, np.pi / 2, size=n_sources)

    param_stack = np.column_stack(
        (x, y, amplitude, semimajor, semiminor, theta))

    # Did I get the numpy syntax right?
    assert param_stack.shape == (n_sources, 6)
    return [Gaussian2dParams(*tuple(param_row)) for param_row in param_stack]


def check_single_source_extraction_successful(source_params, sf_image):
    """
    Check if a single-source-image was extracted successfully

    Only checks island-detection, not fitting.

    Assumes that the source is in a sensible location (not on the edge of
    the image).

    Args:
        source_params (Gaussian2dParams):
        sf_image (SourceFindImage):

    Returns:
        bool
    """
    if not sf_image.islands:
        # Check if the source simply didn't meet the detection criteria
        # This can happen even if the amplitude is above threshold,
        # if it's at a subpixel offset that spreads the peak-flux sufficiently
        # such that the pixel value is lower than amplitude.
        detection_amplitude = sf_image.detection_n_sigma * sf_image.rms_est

        # Positive case
        if (source_params.amplitude > 0 and
                    np.max(sf_image.data) < detection_amplitude):
            return True
        # Negative case
        if (source_params.amplitude < 0 and
                    np.min(sf_image.data) > -detection_amplitude):
            return True

        # Otherwise, something funny is going on:
        raise RuntimeError(
            "No island detected for source: {}".format(source_params))

    # Bad connectivity
    if len(sf_image.islands) > 1:
        raise RuntimeError(
            "{} islands detected for source: {}".format(
                len(sf_image.islands), source_params)
        )
    return True
