"""
Exact convolutional gridder routine
"""
import logging
import numpy as np
from fastimgproto.gridder.kernel_generation import Kernel

logger = logging.getLogger(__name__)


def exact_convolve_to_grid(kernel_func, support,
                           image_size,
                           uv, vis,
                           raise_bounds=True):
    """

    Args:
        kernel_func (func): Callable object that returns a convolution
            co-efficient for a given distance in pixel-widths.
        support (int): Defines the 'radius' of the bounding box within
            which convolution takes place. `Box width in pixels = 2*support+1`.
            (The central pixel is the one nearest to the UV co-ordinates.)
            (This is sometimes known as the 'half-support')
        image_size (int): Width of the image in pixels. NB we assume
            the pixel `[image_size//2,image_size//2]` corresponds to the origin
            in UV-space.
        uv (numpy.ndarray): UV-coordinates of visibilities.
            2d array of `float_`, shape: `(n_vis, 2)`.
            assumed ordering is u-then-v, i.e. `u, v = uv[idx]`
        vis (numpy.ndarray): Complex visibilities.
            1d array, shape: `(n_vis,)`.
        raise_bounds (bool): Raise an exception if any of the UV
            samples lie outside (or too close to the edge) of the grid.

    Returns:
        vis_grid (numpy.ndarray): The gridded visibilities.
            2d array of same dytpe as `vis`, shape `(image_size, image_size)`.
            Note numpy style index-order, i.e. access like `vis_grid[v,u]`.

    """
    assert len(uv) == len(vis)

    vis_grid = np.zeros((image_size, image_size), dtype=vis.dtype)

    # Calculate nearest integer pixel co-ords
    uv_rounded = np.around(uv)
    # sub-pixel vector from rounded-to-precise positions ('fractional coords'):
    uv_frac = uv - uv_rounded
    uv_round_int = uv_rounded.astype(np.int)
    # Now get the corresponding grid-pixels by adding the origin offset
    grid_centre_pixel_idx = uv_round_int + (image_size // 2, image_size // 2)

    good_vis_idx = _bounds_check_kernel_centre_locations(
        uv, grid_centre_pixel_idx,
        support=support, image_size=image_size,
        raise_if_bad=raise_bounds)

    for idx, vis_value in np.ndenumerate(vis[good_vis_idx]):
        gc_x, gc_y = grid_centre_pixel_idx[idx]

        # Generate a convolution kernel with the precise offset required:
        xrange = slice(gc_x - support, gc_x + support + 1)
        yrange = slice(gc_y - support, gc_y + support + 1)
        kernel = Kernel(kernel_func=kernel_func, support=support,
                        offset=uv_frac[idx],
                        oversampling=1)

        vis_grid[yrange, xrange] += vis_value * kernel.array / kernel.array.sum()
    return vis_grid


def _bounds_check_kernel_centre_locations(uv, kernel_centre_indices,
                                          support, image_size,
                                          raise_if_bad):
    """
    Vectorized bounds check, returns idx for good data.

    Check if kernel over-runs the image boundary for any of the chosen central
    pixels
    """

    out_of_bounds_bool = (
        (kernel_centre_indices[:, 0] - support < 0)
        | (kernel_centre_indices[:, 1] - support < 0)
        | (kernel_centre_indices[:, 0] + support >= image_size)
        | (kernel_centre_indices[:, 1] + support >= image_size)
    )
    out_of_bounds_idx = np.where(out_of_bounds_bool)
    good_vis_idx = np.where(np.invert(out_of_bounds_bool))

    if out_of_bounds_bool.any():
        bad_uv = uv[out_of_bounds_idx]
        msg = "{} UV locations are out-of-grid or too close to edge:{}".format(
            len(bad_uv), bad_uv)
        if raise_if_bad:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    return good_vis_idx
