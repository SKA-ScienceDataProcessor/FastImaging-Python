import astropy.stats
import numpy as np
from scipy import ndimage
import math
from attr import attrs, attrib


def _positive_negative_sign_validator(instance, attribute, value):
    if value not in (-1, 1):
        raise ValueError("'sign' should be +1 or -1")


@attrs
class IslandParams(object):
    """
    Data structure for representing source 'islands'.

    Can be used to represent positive or negative sources (peaks or troughs),
    so we use the generic term 'extremum' in place of peak / trough / min/max.

    Initialized with parent image, label index, sign and extremum-pixel value.

    Attributes:
        parent (SourceFindImage): The image the island was detected from.
        label_idx (int): Index of region in label-map of source image.
        sign (
        extremum_val (float): Value of min / max pixel in island
        extremum_x_idx (int): X-index  of min / max pixel
        extremum_y_idx (int): Y-index of min / max pixel
        xbar (float): Barycentric centre in x-pixel index
        ybar(float): Barycentric centre in y-pixel index
    """
    parent = attrib(cmp=False)
    label_idx = attrib(cmp=False)
    sign = attrib(validator=_positive_negative_sign_validator)
    extremum_val = attrib()
    extremum_x_idx = attrib(default=None)
    extremum_y_idx = attrib(default=None)
    xbar = attrib(default=None)
    ybar = attrib(default=None)

    def calculate_params(self):
        """
        Analyses an 'island' to extract further parameters
        """
        self.data = np.ma.MaskedArray(
            self.parent.data,
            mask=_label_mask(self.parent.label_map, self.label_idx),

        )
        self.extremum_y_idx, self.extremum_x_idx = _extremum_pixel_index(
            self.data, self.sign)
        sum = self.sign * np.ma.sum(self.data)
        self.xbar = np.ma.sum(self.parent.xgrid * self.sign * self.data) / sum
        self.ybar = np.ma.sum(self.parent.ygrid * self.sign * self.data) / sum


def _label_mask(labels_map, label_num):
    return ~(labels_map == label_num)


def _extremum_pixel_index(masked_image,sign):
    """
    Returns max/min pixel index in np array ordering, i.e. (y_max, x_max)
    """
    if sign==1:
        extremum_func = np.ma.argmax
    elif sign==-1:
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

        self.label_map, label_extrema = self._label_detection_islands(1)
        if find_negative_sources:
            neg_label_map, neg_label_extrema = self._label_detection_islands(-1)
            self.label_map += neg_label_map
            label_extrema.update(neg_label_extrema)

        self.islands = []
        for l_idx, l_extremum in label_extrema.items():
            # Determine if the label index is positive or negative:
            l_sign = int(math.copysign(1, l_idx))
            self.islands.append(
                IslandParams(parent=self, label_idx=l_idx, sign=l_sign,
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
            `label_map` is an ndarray containing the pixel labels.
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
        label_map, n_labels = ndimage.label(analysis_map)

        all_label_extrema = find_local_extrema(self.data, label_map,
                                               index=range(1, n_labels + 1)
                                               )

        valid_label_extrema = {}
        # Delabel islands that don't meet detection threshold:
        for zero_idx, ex_val in enumerate(all_label_extrema):
            label = zero_idx + 1
            if comparison_op(ex_val, detection_thresh):
                valid_label_extrema[label] = ex_val
            else:
                label_map[label_map == label] = 0.

        if sign == -1:
            # If extracting negative sources, flip the sign of the indices
            valid_label_extrema = {-1 * k: valid_label_extrema[k]
                                   for k in valid_label_extrema}
            # ... and the corresponding label map:
            label_map = -1 * label_map
        return label_map, valid_label_extrema


def _estimate_rms(image):
    clipped_data = astropy.stats.sigma_clip(image)
    clipped_stddev = np.ma.std(clipped_data)
    return clipped_stddev
