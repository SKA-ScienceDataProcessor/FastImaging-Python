import astropy.stats
import numpy as np
from scipy import ndimage
from attr import attributes, attr


@attributes
class IslandParams(object):
    """
    Data structure for representing source 'islands'

    Initialized with parent image, label index, and peak-pixel value.

    Attributes:
        parent (SourceFindImage): The image the island was detected from.
        label_idx (int): Index of region in label-map of source image.
        peak_val (float): Peak pixel value
        peak_x_idx (int): Peak pixel x-index
        peak_y_idx (int): Peak pixel y-index
        xbar (float): Barycentric centre in x-pixel index
        ybar(float): Barycentric centre in y-pixel index
    """
    parent = attr()
    label_idx = attr()
    peak_val = attr()
    peak_x_idx = attr(default=None)
    peak_y_idx = attr(default=None)
    xbar = attr(default=None)
    ybar = attr(default=None)

    def calculate_params(self):
        """
        Analyses an 'island' to extract further parameters
        """
        self.data = np.ma.MaskedArray(
            self.parent.data,
            mask=_label_mask(self.parent.label_map, self.label_idx),

        )
        self.peak_y_idx, self.peak_x_idx = _max_pixel_index(self.data)
        sum = np.ma.sum(self.data)
        self.xbar = np.ma.sum(self.parent.xgrid * self.data) / sum
        self.ybar = np.ma.sum(self.parent.ygrid * self.data) / sum


def _label_mask(labels_map, label_num):
    return ~(labels_map == label_num)


def _max_pixel_index(masked_image):
    """
    Returns max pixel index in np array ordering, i.e. (y_max, x_max)
    """
    return np.unravel_index(np.ma.argmax(masked_image),
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
                 rms_est=None):
        self.data = data
        self.detection_n_sigma = detection_n_sigma
        self.analysis_n_sigma = analysis_n_sigma

        self.rms_est = rms_est
        if self.rms_est is None:
            self.rms_est = _estimate_rms(self.data)
        self.bg_level = np.ma.median(self.data)
        self.analysis_thresh = self.bg_level + self.analysis_n_sigma * self.rms_est
        self.detection_thresh = self.bg_level + self.detection_n_sigma * self.rms_est

        self.ygrid, self.xgrid = np.indices(self.data.shape)

        # Label connected regions
        self.label_map, label_maxvals = _label_detection_islands(
            self.data, self.analysis_thresh, self.detection_thresh
        )

        self.islands = []
        for l_idx, l_maxval in label_maxvals.items():
            self.islands.append(
                IslandParams(parent=self, label_idx=l_idx, peak_val=l_maxval))
        for island in self.islands:
            island.calculate_params()


def _estimate_rms(image):
    clipped_data = astropy.stats.sigma_clip(image)
    clipped_stddev = np.ma.std(clipped_data)
    return clipped_stddev


def _label_detection_islands(data, analysis_thresh, detection_thresh):
    """
    Find regions which are above analysis_thresh and peak above detection_thresh.

    Args:
        data (array_like): Raw data (may be masked)
        analysis_thresh (float): Analysis ('single connected region') threshold
        detection_thresh (float): Detections threshold

    Returns:
        tuple (array_like, dict): Tuple of `(label_map, valid_labels)`.
            `label_map` is an ndarray containing the pixel labels.
            `valid_labels` is a dict mapping valid label numbers to the maximum
            pixel-value in that label-region.

    """
    analysis_map = data > analysis_thresh
    label_map, n_labels = ndimage.label(analysis_map)

    label_maxvals = ndimage.maximum(data, label_map,
                                    index=range(1, n_labels + 1))

    valid_label_maxvals = {}
    # Delabel islands that don't meet detection threshold:
    for zero_idx, maxval in enumerate(label_maxvals):
        label = zero_idx + 1
        if maxval < detection_thresh:
            label_map[label_map == label] = 0.
        else:
            valid_label_maxvals[label] = maxval
    return label_map, valid_label_maxvals
