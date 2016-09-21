"""
Exact convolutional gridder routine
"""
import numpy as np


def exact_convolve_to_grid(kernel_func, support,
                           image_size,
                           uv, vis, ):
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
            1d array of `complex_`, shape: `(n_vis,)`.

    Returns:
        vis_grid (numpy.ndarray): The gridded visibilities.
            2d array of `numpy.complex_`, shape `(image_size, image_size)`.
            Note numpy style index-order, i.e. access like `vis_grid[v,u]`.

    """
    assert len(uv) == len(vis)

    vis_grid = np.zeros((image_size, image_size), dtype=np.complex_)

    # Calculate nearest integer pixel co-ords
    uv_rounded = np.around(uv)
    # sub-pixel vector from rounded-to-precise positions ('fractional coords'):
    uv_frac = uv - uv_rounded
    uv_round_int = uv_rounded.astype(np.int)
    # Now get the corresponding grid-pixels by adding the origin offset
    grid_centre_pixel_idx = uv_round_int + (image_size//2, image_size//2)
    for vis, vis_idx in enumerate(vis):
        gc_x, gc_y = grid_centre_pixel_idx[vis_idx]
        #Bounds check, skip if kernel over-runs the image boundary
        if (gc_x - support < 0) or (gc_x + support >= image_size):
            continue
        if (gc_y - support < 0) or (gc_y + support >= image_size):
            continue



