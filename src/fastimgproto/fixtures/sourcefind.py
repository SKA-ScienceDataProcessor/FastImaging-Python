from __future__ import division, print_function

import inspect
import logging
import math

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
    Set up random, reasonably valued parameters for n_sources.

    See Args for details of random-generation for amplitude, axes-lengths.
    Rotation-angle is uniform-randomly allocated in the range `(-pi/2,pi/2]`.

    Args:
        n_sources (int): Number of sources to generate
        base_x (int): Pixel position. A random subpixel offset in range (0,1.)
            will be added.
        base_y (int): Pixel position. A random subpixel offset in range (0,1.)
            will be added.
        amplitude_range (tuple): `(min, max)`. Amplitudes will be uniformly
            allocated in this range.
        semiminor_range (tuple): `(min, max)`. Minor axis (sigma_{minor}) length
            in pixels. Values will be uniformly allocated in this range.
        axis_ratio_range (tuple): '(min, max)'. For each source, major-axis
            length is assigned as `semimajor = semiminor * axis_ratio_range`.
            Ratio values will be uniformly allocated in this range.
        seed (int): RandomState seed.

    Returns:
        list[Gaussian2dParams]: List of pseudo-randomly generated sources.
    """
    if axis_ratio_range[0] < 1.:
        raise ValueError("Axis ratios must be 1. at minimum "
                         "(ensures 'semimajor > semiminor')")

    if (amplitude_range[1] < amplitude_range[0]
        or semiminor_range[1] < semiminor_range[0]
        or axis_ratio_range[1] < axis_ratio_range[0]
        ):
        raise ValueError("Range-valued tuples must be in order (min,max)")

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


def calculate_sourcegrid_base_positions(image_size,
                                        n_sources):
    sources_per_row = int(math.ceil(math.sqrt(n_sources)))
    source_spacing = image_size / sources_per_row
    base_positions = np.mgrid[
                     0: (sources_per_row) * source_spacing:source_spacing,
                     0: (sources_per_row) * source_spacing:source_spacing,
                     ]
    base_positions = base_positions.astype(np.float_)
    base_positions += source_spacing / 2.
    return source_spacing, base_positions


def random_sources_on_grid(image_size,
                           n_sources,
                           amplitude_range,
                           semiminor_range,
                           axis_ratio_range,
                           seed=None):
    """
    Generate a list of sources that are approximately located on a square grid.

    Assumes a square image / grid. Sources will be placed on a regularly spaced
    grid, but with random sub-pixel offsets applied.

    Raises a ValueError if the generated sources could be at a spacing less
    than `10*sigma`, where `sigma=max_semiminor*max_axis_ratio_range`.

    Other source parameters (amplitude, axes-size / rotation) are randomly
    generated according to the bounds given, as in
    `generate_random_source_params`.

    Args:
        image_size (int): Size of image in pixels
        n_sources (int): Number of sources to generate
        amplitude_range (tuple): See `generate_random_source_params` for details.
        semiminor_range (tuple): See `generate_random_source_params` for details
        axis_ratio_range (tuple): See `generate_random_source_params` for details
        seed (int): See `generate_random_source_params` for details.

    Returns:
        list[Gaussian2dParams]: List of pseudo-randomly generated sources.
    """
    zero_positioned_sources = generate_random_source_params(
        n_sources=n_sources,
        base_x=0, base_y=0,
        amplitude_range=amplitude_range,
        semiminor_range=semiminor_range,
        axis_ratio_range=axis_ratio_range,
        seed=seed
    )

    source_spacing, basegrid = calculate_sourcegrid_base_positions(
        image_size, n_sources)
    min_sensible_spacing = 10. * max(semiminor_range) * max(axis_ratio_range)
    if source_spacing < min_sensible_spacing:
        raise ValueError(
            "Sources would be spaced too closely to be reliably non-blended.")

    y_grid, x_grid = basegrid
    y_offsets = y_grid.ravel()
    x_offsets = x_grid.ravel()
    randomly_offset_sources = []
    for idx in range(n_sources):
        input_src = zero_positioned_sources[idx]
        randomly_offset_sources.append(
            attr.evolve(input_src,
                        y_centre=input_src.y_centre + y_offsets[idx],
                        x_centre=input_src.x_centre + x_offsets[idx],
                        ))
    return randomly_offset_sources


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
