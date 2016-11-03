import astropy.units as u
import fastimgproto.casa.io as casa_io
import fastimgproto.gridder.conv_funcs as kfuncs
import numpy as np
from drivecasa.utils import derive_out_path
from fastimgproto.gridder.gridder import convolve_to_grid


def make_image_map_fits(vis_path, output_dir,
                        image_size, cell_size):
    out_path = derive_out_path(vis_path, output_dir,
                               out_extension='.pyimg')

    uvw_in_wavelengths = casa_io.get_uvw_in_lambda(vis_path)
    stokes_i = casa_io.get_stokes_i_vis(vis_path)

    image_size = int(image_size.to(u.pix).value)
    grid_pixel_width_in_wavelengths = 1.0 / (cell_size.to(u.rad) * image_size)

    uvw_in_pixels = uvw_in_wavelengths / grid_pixel_width_in_wavelengths

    uv_in_pixels = uvw_in_pixels[:, :2]
    kernel = kfuncs.Triangle(1.5)
    uvgrid = convolve_to_grid(kernel, support=2,
                              image_size=image_size,
                              uv=uv_in_pixels,
                              vis=stokes_i
                              )
    ifft_data = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uvgrid)))
    image = np.real(ifft_data)
    return image
