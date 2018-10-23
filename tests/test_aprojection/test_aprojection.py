import astropy.units as u
import numpy as np
from pytest import approx

import fastimgproto.gridder.conv_funcs as conv_funcs
from fastimgproto.imager import image_visibilities


def test_aprojection():
    n_image = 24 * u.pix  # pixel co-ords -8 through 7.
    support = 3
    uvw_pixel_coords = np.array([
        (-4., 0, 1),
        (3., 0, 0.5),
        (0., 2.5, 0)
    ])
    lha = np.zeros(len(uvw_pixel_coords), dtype=np.float_)
    pbeam_coefs = np.array([4, 0, 1, 0, 0, 0])
    aproj_numtimesteps = 0
    obs_dec = 30
    obs_ra = 0

    # Real vis will be complex_, but we can substitute float_ for testing:
    vis_amplitude = 42.123
    vis = vis_amplitude * np.ones(len(uvw_pixel_coords), dtype=np.float_)
    vis_weights = np.array([1., 1.5, 3. / 4.])
    cell_size = 1000 * u.arcsec

    grid_pixel_width_lambda = 1.0 / (cell_size.to(u.rad) * n_image)
    uvw_lambda = uvw_pixel_coords * grid_pixel_width_lambda.value

    # Run imager without A-projection
    kernel_func = conv_funcs.PSWF(trunc=3.)
    image, beam = image_visibilities(vis,
                                     vis_weights=vis_weights,
                                     uvw_lambda=uvw_lambda,
                                     image_size=n_image,
                                     cell_size=cell_size,
                                     kernel_func=kernel_func,
                                     kernel_support=support,
                                     kernel_exact=False,
                                     kernel_oversampling=4,
                                     num_wplanes=4,
                                     max_wpconv_support=5,
                                     aproj_numtimesteps=aproj_numtimesteps,
                                     obs_dec=obs_dec,
                                     obs_ra=obs_ra,
                                     lha=lha,
                                     pbeam_coefs=pbeam_coefs
                                     )

    assert approx(beam.max()) == 1.0
    assert approx(image.max()) == vis_amplitude

    # Run imager with A-projection
    aproj_numtimesteps = 1
    image2, beam2 = image_visibilities(vis,
                                       vis_weights=vis_weights,
                                       uvw_lambda=uvw_lambda,
                                       image_size=n_image,
                                       cell_size=cell_size,
                                       kernel_func=kernel_func,
                                       kernel_support=support,
                                       kernel_exact=False,
                                       kernel_oversampling=4,
                                       num_wplanes=4,
                                       max_wpconv_support=5,
                                       aproj_numtimesteps=aproj_numtimesteps,
                                       obs_dec=obs_dec,
                                       obs_ra=obs_ra,
                                       lha=lha,
                                       pbeam_coefs=pbeam_coefs
                                       )

    assert approx(beam.max()) == 1.0
    assert approx(image.max()) == vis_amplitude

    # A-projection increases amplitude of surrounding sources (due to pbeam division).
    # Test it:
    sum_max1 = np.sum(image[np.where(image > 38)])
    sum_max2 = np.sum(image2[np.where(image2 > 38)])

    assert(sum_max2 > sum_max1)
