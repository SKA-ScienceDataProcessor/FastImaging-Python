import math

import astropy.stats
import attr
import attr.validators
import numpy as np
from attr import attrib, attrs
from scipy import ndimage
from scipy.optimize import least_squares

from fastimgproto.sourcefind.fit import Gaussian2dFit, gaussian2d
from fastimgproto.utils import (
    nonzero_bounding_slice_2d,
    positive_negative_sign_validator,
)

_STRUCTURE_ELEMENT = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1], ],
                              dtype=int)


@attrs
class IslandParams(object):
    """
    Data structure for representing source 'islands'.

    Can be used to represent positive or negative sources (peaks or troughs),
    so we use the generic term 'extremum' in place of peak / trough / min/max.

    Initialized with parent image, label index, sign and extremum-pixel value.

    Attributes:
        parent (SourceFindImage): The image the island was detected from.
        label_number (int): Label of this sub-region in label-map of source
            image.
        sign (int): +/-1. Whether this is a positive or negative source.
        extremum_val (float): Value of min / max pixel in island
        extremum_x_idx (int): X-index  of min / max pixel
        extremum_y_idx (int): Y-index of min / max pixel
        xbar (float): Barycentric centre in x-pixel index
        ybar(float): Barycentric centre in y-pixel index
    """

    # Required for initialization
    parent = attrib(cmp=False)
    label_number = attrib(cmp=False)
    sign = attrib(validator=positive_negative_sign_validator)
    extremum_val = attrib(validator=attr.validators.instance_of(float))

    # Configured post-init, doesn't make sense to init these manually.
    data = attrib(init=False, cmp=False)

    # Set by 'calculate_params'. Typically not set at init-time, but we
    # leave that option open as it might be useful for mocking out tests.
    extremum_x_idx = attrib(default=None)
    extremum_y_idx = attrib(default=None)
    xbar = attrib(default=None)
    ybar = attrib(default=None)

    # `Gaussian2dFit`, set by 'fit_gaussian' method.
    fit = attrib(default=None)

    def __attrs_post_init__(self):
        """
        Initialize the `data` attribute from the parent-array and label-info.
        """
        self.data = np.ma.MaskedArray(
            self.parent.data,
            mask=_label_mask(self.parent.label_map, self.label_number),
        )

    @property
    def unmasked_pixel_indices(self):
        """
        Return the indices of the elements that are unmasked for this island.

        Uses numpy.nonzero together with the mask, to fetch a list of valid
        pixel indices
        """
        return np.nonzero(~self.data.mask)

    @property
    def bounding_slice(self):
        return nonzero_bounding_slice_2d(~self.data.mask)

    def calculate_params(self):
        """
        Analyses an 'island' to extract further parameters.
        """

        self.extremum_y_idx, self.extremum_x_idx = _extremum_pixel_index(
            self.data, self.sign)
        sum = self.sign * np.ma.sum(self.data)
        self.xbar = np.ma.sum(self.parent.xgrid * self.sign * self.data) / sum
        self.ybar = np.ma.sum(self.parent.ygrid * self.sign * self.data) / sum

    def fit_gaussian_2d(self, verbose=0):
        # x, y, x_centre, y_centre, amplitude, x_stddev, y_stddev, theta
        ygrid, xgrid = self.unmasked_pixel_indices
        data = self.data[ygrid, xgrid]

        def island_residuals(x_centre,
                             y_centre,
                             amplitude,
                             semimajor,
                             semiminor,
                             theta):
            """
            A wrapped version of gaussian_2d applied to this island's unmasked
            pixels, then subtracting the island values

            Same args as :ref:`.gaussian2d`, except without the `xgrid,ygrid`
            parameters.

            Returns:
                numpy.ndarray: vector of residuals

            """

            model_vals = gaussian2d(xgrid, ygrid,
                                    x_centre=x_centre,
                                    y_centre=y_centre,
                                    amplitude=amplitude,
                                    x_stddev=semimajor,
                                    y_stddev=semiminor,
                                    theta=theta,
                                    )
            assert model_vals.shape == data.shape
            return data - model_vals

        def wrapped_island_residuals(pars):
            """
            Wrapped version of `island_residuals` that takes a single argument

            (a tuple of the varying parameters).

            Args:
                pars (tuple):
                    (x_centre, y_centre, amplitude, x_stddev, y_stddev, theta)

            Returns:
                numpy.ndarray: vector of residuals

            """
            assert len(pars) == 6
            return island_residuals(*pars)

        initial_params = Gaussian2dFit(x_centre=self.xbar,
                                       y_centre=self.ybar,
                                       amplitude=self.extremum_val,
                                       semimajor=1.,
                                       semiminor=1.,
                                       theta=0
                                       )
        lsq_result = least_squares(fun=wrapped_island_residuals,
                                   x0=attr.astuple(initial_params),
                                   # method='lm',
                                   verbose=verbose,
                                   )
        self.fit = Gaussian2dFit.from_unconstrained_parameters(
            *tuple(lsq_result.x))
        return self.fit


def _label_mask(labels_map, label_num):
    return ~(labels_map == label_num)


def _extremum_pixel_index(masked_image, sign):
    """
    Returns max/min pixel index in np array ordering, i.e. (y_max, x_max)
    """
    if sign == 1:
        extremum_func = np.ma.argmax
    elif sign == -1:
        extremum_func = np.ma.argmin
    return np.unravel_index(extremum_func(masked_image),
                            masked_image.shape)


class SourceFindImage(object):
    """
    Data structure for collecting intermediate results from source-extraction.

    This can be useful for verifying / debugging the sourcefinder results,
    and intermediate results can also be reused to save recalculation.

    Args:
        data (array_like): numpy.ndarray or numpy.ma.MaskedArray containing
            image data.
        detection_n_sigma (float): Detection threshold as multiple of RMS
        analysis_n_sigma (float): Analysis threshold as multiple of RMS
        rms_est (float): RMS estimate (may be `None`, in which case RMS is
            estimated from the image data via sigma-clipping).
    """

    def __init__(self, data, detection_n_sigma, analysis_n_sigma,
                 rms_est=None, find_negative_sources=True):
        self.data = data
        self.detection_n_sigma = detection_n_sigma
        self.analysis_n_sigma = analysis_n_sigma

        self.rms_est = rms_est
        if self.rms_est is None:
            self.rms_est = _estimate_rms(self.data)
        self.bg_level = np.ma.median(self.data)

        self.ygrid, self.xgrid = np.indices(self.data.shape)

        # Label connected regions

        self.label_map, region_extrema = self._label_detection_islands(1)
        if find_negative_sources:
            neg_label_map, neg_label_extrema = self._label_detection_islands(-1)
            self.label_map += neg_label_map
            region_extrema.update(neg_label_extrema)

        self.islands = []
        for l_num, l_extremum in region_extrema.items():
            # Determine if the label index is positive or negative:
            l_sign = int(math.copysign(1, l_num))
            self.islands.append(
                IslandParams(parent=self, label_number=l_num, sign=l_sign,
                             extremum_val=l_extremum))
        for island in self.islands:
            island.calculate_params()

    def _label_detection_islands(self, sign):
        """
        Find connected regions which peak above/below a given threshold.

        In the positive case, when ``sign=1``, this routine will find regions
        which are more than ``analysis_n_sigma*rms`` greater than the background
        level, and then filter that list of regions to only include those which have
        a peak value more than ``detection_n_sigma*rms`` greater than the background.

        Alternatively if ``sign=-1``, we create negative thresholds, and look for
        regions which are below those thresholds accordingly.

        Args:
            sign (int): Should have value in ``(-1,1)``. Determines whether to
                search for positive or negative islands.

        Returns:
            tuple (array_like, dict): Tuple of `(label_map, valid_labels)`.
            `label_map` is an ndarray of dtype int. The background is
            zero-valued and each connected sub-region has a single integer-value
            throughout.
            `valid_labels` is a dict mapping valid label numbers to the maximum
            pixel-value in that label-region.

        """
        assert sign in (-1, 1)
        if sign == 1:
            comparison_op = np.greater
            find_local_extrema = ndimage.maximum
        elif sign == -1:
            comparison_op = np.less
            find_local_extrema = ndimage.minimum

        analysis_thresh = self.bg_level + sign * self.analysis_n_sigma * self.rms_est
        detection_thresh = self.bg_level + sign * self.detection_n_sigma * self.rms_est
        analysis_map = comparison_op(self.data, analysis_thresh)
        label_map, n_labels = ndimage.label(analysis_map,
                                            structure=_STRUCTURE_ELEMENT)

        # Get a list of the extrema for each labelled-subregion:
        region_extrema_vals = find_local_extrema(self.data, label_map,
                                                 index=range(1, n_labels + 1)
                                                 )

        valid_label_extrema = {}
        # Delabel islands that don't meet detection threshold:
        for zero_idx, ex_val in enumerate(region_extrema_vals):
            label = zero_idx + 1
            if comparison_op(ex_val, detection_thresh):
                valid_label_extrema[label] = ex_val
            else:
                label_map[label_map == label] = 0.

        if sign == -1:
            # If extracting negative sources, flip the sign of the indices
            valid_label_extrema = {-1 * k: valid_label_extrema[k]
                                   for k in valid_label_extrema}
            # ... and the corresponding label map, to keep them in sync:
            label_map = -1 * label_map
            # (this allows us to merge negative and positive island results)
        return label_map, valid_label_extrema


def _estimate_rms(image):
    clipped_data = astropy.stats.sigma_clip(image)
    clipped_stddev = np.ma.std(clipped_data)
    return clipped_stddev
