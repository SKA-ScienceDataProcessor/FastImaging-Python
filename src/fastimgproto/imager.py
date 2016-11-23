import astropy.units as u
import numpy as np
from fastimgproto.gridder.gridder import convolve_to_grid


def fft_to_image_plane(uv_grid):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid)))


def image_visibilities(vis, uvw_lambda,
                       image_size, cell_size,
                       kernel_func, kernel_support,
                       kernel_oversampling,
                       normalize=True):

    image_size = image_size.to(u.pix)
    # Size of a UV-grid pixel, in multiples of wavelength (lambda):
    grid_pixel_width_lambda = 1.0 / (cell_size.to(u.rad) * image_size)
    uvw_in_pixels = (uvw_lambda / grid_pixel_width_lambda).value

    uv_in_pixels = uvw_in_pixels[:, :2]
    vis_grid, sample_grid = convolve_to_grid(kernel_func,
                                             support=kernel_support,
                                             image_size=int(image_size.value),
                                             uv=uv_in_pixels,
                                             vis=vis,
                                             oversampling=kernel_oversampling
                                             )
    image = fft_to_image_plane(vis_grid)
    beam = fft_to_image_plane(sample_grid)
    if normalize:
        beam_max = np.max(beam)
        beam /= beam_max
        image /= beam_max
    return (image, beam)
