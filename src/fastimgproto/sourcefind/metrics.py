import math

import numpy as np
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

from fastimgproto.sourcefind.fit import Gaussian2dParams


def mahalanobis_distance(gaussian2d_params, x, y):
    """
    Compute the Mahalanobis distance from Gaussian-centre to point (x,y).

    Mahalanobis distance is the Euclidean distance normalized by the
    standard-deviation of the Gaussian:
    https://en.wikipedia.org/wiki/Mahalanobis_distance

    Args:
        gaussian2d_params (Gaussian2dParams): Gaussian to compute distance for.
        x (float): X-coordinate of position
        y (float): Y-coordinate of position

    Returns:
        float: Mahalanobis distance
    """
    inv_covariance = np.linalg.inv(gaussian2d_params.covariance)
    g_centre = np.array(
        [gaussian2d_params.x_centre, gaussian2d_params.y_centre],
        dtype=np.float_
    )
    posn = np.array([x, y], dtype=np.float_)
    return scipy_mahalanobis(u=g_centre, v=posn, VI=inv_covariance)


def pixel_distance(gaussian2d_params, x, y):
    """
    Compute the distance from Gaussian-centre to point (x,y).

    (Regular distance measured in units of pixels)

    Args:
        gaussian2d_params (Gaussian2dParams): Gaussian to compute distance for.
        x (float): X-coordinate of position
        y (float): Y-coordinate of position

    Returns:
        float: Euclidean distance (units of pixels)
    """
    xdiff = gaussian2d_params.x_centre - x
    ydiff = gaussian2d_params.y_centre - y
    return np.sqrt(xdiff * xdiff + ydiff * ydiff)


def shape_difference(g1, g2):
    """
    A symmetric shape-difference metric for Gaussians

    Computes the Covariance-difference term of the Bhattacharyya_distance, cf
    https://en.wikipedia.org/wiki/Bhattacharyya_distance


    Args:
        g1 (Gaussian2dParams):
        g2 (Gaussian2dParams):

    Returns:
        float: Dimensionless semipositive value. 0 if g1 and g2 have same shape.
    """
    cov1 = g1.covariance
    cov2 = g2.covariance
    halfway_cov = (cov1 + cov2) / 2.0
    ratio = np.linalg.det(halfway_cov) / np.sqrt(
        np.linalg.det(cov1) * np.linalg.det(cov2)
    )
    return 0.5 * math.log(ratio)
