import astropy.stats
import numpy as np
from scipy import ndimage

def extract_sources(image, detection_n_sigma, analysis_n_sigma):
    rms = estimate_rms(image)
    bg_level = np.ma.median(image)
    detection_thresh = bg_level + detection_n_sigma*rms
    analysis_thresh = bg_level + analysis_n_sigma*rms
    analysis_map = image > analysis_thresh
    labels_map, n_labels = ndimage.label(analysis_map)
    detections = []
    for i in range(n_labels):
        extraction =  process_island(image,labels_map, i, detection_thresh)
        if extraction:
            detections.append(extraction)
    return detections

def estimate_rms(image):
    clipped_data = astropy.stats.sigma_clip(image)
    clipped_stddev = np.ma.std(clipped_data)
    return clipped_stddev

def label_mask(labels_map, label_num):
    return ~(labels_map == label_num)

def process_island(image, labels_map, label_index, detection_thresh):
    # image masked except for label 'island':
    island = np.ma.MaskedArray(image, mask=label_mask(labels_map, label_index))
    if (island > detection_thresh).any():
        peak_pixel = max_pixel_index(island)
        return (peak_pixel, island[peak_pixel])
    else:
        return None


def max_pixel_index(masked_image):
    """
    Returns max pixel index in np array ordering, i.e. (y_max, x_max)
    """
    return np.unravel_index(np.ma.argmax(masked_image),
                            masked_image.shape)
