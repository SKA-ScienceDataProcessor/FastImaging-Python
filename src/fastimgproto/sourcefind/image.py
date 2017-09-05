import math

import astropy.stats
import attr
import attr.validators
import numpy as np
from attr import attrib, attrs
from scipy import ndimage
from scipy.optimize import least_squares

from fastimgproto.sourcefind.fit import Gaussian2dParams, gaussian2d
from fastimgproto.utils import (
    nonzero_bounding_slice_2d,
    positive_negative_sign_validator,
)

_STRUCTURE_ELEMENT = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1], ],
                              dtype=int)


def estimate_rms(image):
    clipped_data = astropy.stats.sigma_clip(image)
    clipped_stddev = np.ma.std(clipped_data)
    return clipped_stddev


def extremum_pixel_index(a, sign):
    """
    Returns max/min pixel index in np array ordering, i.e. (y_max, x_max)

    Args:
        a (numpy.ma.MaskedArray): array or masked-array. If masked, only the
            unmasked region is searched for an extremum.
        sign (int): +1 or -1, determines whether to search for max or min
            respectively.
    Returns:
         tuple: index of extremum-pixel.
    """
    if sign == 1:
        extremum_func = np.ma.argmax
    elif sign == -1:
        extremum_func = np.ma.argmin
    return np.unravel_index(extremum_func(a),
                            a.shape)


@attrs
class PixelIndex(object):
    """
    Simple struct for hanging together x/y pixel-index values.

    (2d arrays only!)

    Attrib ordering is (y,x) so we can init-from/convert-to numpy tuple
    indices with the default ordering.
    """
    y = attrib(validator=attr.validators.instance_of((int, np.int_)))
    x = attrib(validator=attr.validators.instance_of((int, np.int_)))


@attrs
class Pixel(object):
    """
    Simple struct for hanging together a PixelIndex with the pixel-value.

    Used for passing around 'max-pixel' / 'min-pixel' results.
    """
    index = attrib(validator=attr.validators.instance_of(PixelIndex))
    value = attrib(validator=attr.validators.instance_of((float, np.float_)))


@attrs
class IslandParams(object):
    """
    Data structure for representing source 'islands'.

    Can be used to represent positive or negative sources (peaks or troughs),
    so we use the generic term 'extremum' in place of peak / trough / min/max.

    Provides a holding structure for basic info (sign, min/max pixel val,idx),
    moments fit, and least-squares fit.

    Args:
        sign (int): +/-1. Whether this is a positive or negative source.
        extremum (Pixel): min / max pixel in island
        leastsq_fit(None or Gaussian2dParams or False): Default is 'None',
            which signifies no least-squares fit has been attempted.
            If a least-squares fit has been run successfully, the
            results are stored here.
            'False' signifies that a least-squares fit was tried and failed for
             some reason.
    """

    # Required for initialization
    sign = attrib(validator=positive_negative_sign_validator)
    extremum = attrib(validator=attr.validators.instance_of(Pixel))

    # Optional
    leastsq_fit = attrib(
        default=None,
        validator=attr.validators.optional(
            attr.validators.instance_of((Gaussian2dParams, bool))))


@attrs
class Island(object):
    """
    Represents a labelled island-region of an image.

    Pairs an instance of IslandParams with an array-mask, and provides a
    convenience attribute ('.data') for accessing the appropriately masked
    view of the parent-image data.

    Args:
        parent (SourceFindImage): The image the island was detected from.
        mask (numpy.ndarray): Boolean mask - mask is false for the region
            representing this island of above-threshold connectivity, true
            elsewhere.
        params (IslandParams): Parameter-set representing the island.
    """
    parent_data = attrib(cmp=False,
                         validator=attr.validators.instance_of(np.ndarray))
    mask = attrib(cmp=False,
                  validator=attr.validators.instance_of(np.ndarray))
    params = attrib(validator=attr.validators.instance_of(IslandParams))

    # xbar = attrib(default=None)
    # ybar = attrib(default=None)
    # `Gaussian2dFit`, set by 'fit_gaussian' method.


    @property
    def bounding_slice(self):
        return nonzero_bounding_slice_2d(~self.data.mask)

    @property
    def data(self):
        """
        Return a MaskedArray view of the parent-array.
        """
        return np.ma.MaskedArray(
            self.parent_data.data,
            mask=self.mask,
        )

    @property
    def extremum(self):
        """
        Convenience method for accessing `self.params.extremum`
        """
        return self.params.extremum

    @property
    def sign(self):
        """
        Convenience method for accessing `self.params.sign`
        """
        return self.params.sign

    @property
    def unmasked_pixel_indices(self):
        """
        Return the indices of the elements that are unmasked for this island.

        Uses numpy.nonzero together with the mask, to fetch a list of valid
        pixel indices
        """
        return np.nonzero(~self.mask)


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
            self.rms_est = estimate_rms(self.data)
        self.bg_level = np.ma.median(self.data)

        self.ygrid, self.xgrid = np.indices(self.data.shape)

        # Label connected regions
        self.label_map, region_extrema = self._label_detection_islands(1)
        if find_negative_sources:
            neg_label_map, neg_label_extrema = self._label_detection_islands(-1)
            self.label_map += neg_label_map
            region_extrema.update(neg_label_extrema)

        self.islands = []
        for label_num, extremum in region_extrema.items():
            # Determine if the label number is positive or negative:
            label_sign = int(math.copysign(1, label_num))
            island_mask = ~(self.label_map == label_num)

            init_params = IslandParams(sign=label_sign,
                                       extremum=extremum)
            island = Island(parent_data=self.data,
                            mask=island_mask,
                            params=init_params)
            self.calculate_moments(island)
            self.islands.append(island)

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
            zero-valued and each connected sub-region has a particular
            integer-value (the 'label number').
            `valid_labels` is a dict mapping valid label numbers to the extremum
            pixel in that label-region.

        """
        assert sign in (-1, 1)
        if sign == 1:
            comparison_op = np.greater
            find_local_extrema = ndimage.maximum_position
        elif sign == -1:
            comparison_op = np.less
            find_local_extrema = ndimage.minimum_position

        analysis_thresh = self.bg_level + sign * self.analysis_n_sigma * self.rms_est
        detection_thresh = self.bg_level + sign * self.detection_n_sigma * self.rms_est
        analysis_map = comparison_op(self.data, analysis_thresh)
        label_map, n_labels = ndimage.label(analysis_map,
                                            structure=_STRUCTURE_ELEMENT)

        # Get a list of the extrema for each labelled-subregion:
        region_extrema_posns = find_local_extrema(self.data, label_map,
                                                  index=range(1, n_labels + 1)
                                                  )

        valid_label_extrema = {}
        # Delabel islands that don't meet detection threshold:
        for zero_idx, extremum_posn in enumerate(region_extrema_posns):
            label = zero_idx + 1
            extremum = Pixel(
                index=PixelIndex(*extremum_posn),
                value=self.data[extremum_posn],
            )
            if comparison_op(extremum.value, detection_thresh):
                valid_label_extrema[label] = extremum
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


        # Set by 'calculate_params'. Typically not set at init-time, but we

    def calculate_moments(self, island):
        """
        Analyses an island to extract further parameters.
        """
        sign = island.sign
        sum = sign * np.ma.sum(island.data)
        island.xbar = np.ma.sum(self.xgrid * sign * island.data) / sum
        island.ybar = np.ma.sum(self.ygrid * sign * island.data) / sum

    def fit_gaussian_2d(self, island, verbose=0):
        # x, y, x_centre, y_centre, amplitude, x_stddev, y_stddev, theta
        ygrid, xgrid = island.unmasked_pixel_indices
        fitting_data = island.data[ygrid, xgrid]

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
            assert model_vals.shape == fitting_data.shape
            return fitting_data - model_vals

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

        initial_params = Gaussian2dParams(x_centre=island.xbar,
                                          y_centre=island.ybar,
                                          amplitude=island.extremum.value,
                                          semimajor=1.,
                                          semiminor=1.,
                                          theta=0
                                          )
        lsq_result = least_squares(fun=wrapped_island_residuals,
                                   x0=attr.astuple(initial_params),
                                   # method='lm',
                                   verbose=verbose,
                                   )
        island.params.leastsq_fit = Gaussian2dParams.from_unconstrained_parameters(
            *tuple(lsq_result.x))
        return island.params.leastsq_fit
