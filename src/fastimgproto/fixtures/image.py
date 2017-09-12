import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.modeling.models import Gaussian2D as AstropyGauss2d
from astropy.time import Time

import fastimgproto.visibility as visibility
from fastimgproto.skymodel.helpers import SkySource
from fastimgproto.sourcefind.fit import Gaussian2dParams
from fastimgproto.telescope.readymade import Meerkat


def uncorrelated_gaussian_noise_background(shape, mean=0, sigma=1.0):
    normal_noise = np.random.randn(*shape)
    return (sigma * normal_noise) + mean


def gaussian_point_source(x_centre,
                          y_centre,
                          amplitude=1.0,
                          semimajor=1.5,
                          semiminor=1.2,
                          theta=1,  # rad
                          ):
    """
    Wrapper around Gaussian2dFit providing some default values

    Args:
        x_centre:
        y_centre:
        amplitude:
        semimajor:
        semiminor:
        theta:

    Returns:
        Gaussian2dParams
    """
    return Gaussian2dParams(amplitude=amplitude,
                            x_centre=x_centre,
                            y_centre=y_centre,
                            semimajor=semimajor,
                            semiminor=semiminor,
                            theta=theta
                            )


def add_gaussian2d_to_image(gauss2d_pars, image):
    """
    Evaluate the Gaussian2dParams and add to the image-pixel values.
    Args:
        gauss2d_pars (Gaussian2dParams):
        image (numpy.ndarray):

    Returns:
        None

    """
    model = AstropyGauss2d(amplitude=gauss2d_pars.amplitude,
                           x_mean=gauss2d_pars.x_centre,
                           y_mean=gauss2d_pars.y_centre,
                           x_stddev=gauss2d_pars.semimajor,
                           y_stddev=gauss2d_pars.semiminor,
                           theta=gauss2d_pars.theta
                           )
    bb_width = 6. * gauss2d_pars.semimajor
    # model.bounding_box = (
    #     (gauss2d_pars.y_centre - bb_width, gauss2d_pars.y_centre + bb_width),
    #     (gauss2d_pars.x_centre - bb_width, gauss2d_pars.x_centre + bb_width),
    # )
    ydim, xdim = image.shape
    # ygrid, xgrid = np.mgrid[:ydim, :xdim]
    # image += model(xgrid, ygrid, with_bounding_box=True)
    eval_ymin = max(0, int(np.floor(gauss2d_pars.y_centre - bb_width)))
    eval_ymax = min(ydim, int(np.ceil(gauss2d_pars.y_centre + bb_width)))
    eval_xmin = max(0, int(np.floor(gauss2d_pars.x_centre - bb_width)))
    eval_xmax = min(xdim, int(np.ceil(gauss2d_pars.x_centre + bb_width)))
    yslice = slice(eval_ymin, eval_ymax)
    xslice = slice(eval_xmin, eval_xmax)
    ygrid, xgrid = np.mgrid[yslice, xslice]
    image[yslice, xslice] += model(xgrid, ygrid)


def sample_sim_radio_image(nstep):
    """
    Simulate a sample radio-image, to use as test-data in sourcefinding.
    """

    telescope = Meerkat()
    obs_central_frequency = 3. * u.GHz
    wavelength = const.c / obs_central_frequency
    pointing_centre = SkyCoord(0 * u.deg, -30 * u.deg)
    transit_time = telescope.next_transit(pointing_centre.ra,
                                          start_time=Time('2017-01-01'))
    obs_times = transit_time + np.linspace(-1, 1, nstep) * u.hr

    uvw_m = telescope.uvw_tracking_skycoord(
        pointing_centre, obs_times,
    )
    # From here on we use UVW as multiples of wavelength, lambda:
    uvw_lambda = (uvw_m / wavelength).to(u.dimensionless_unscaled).value
    vis_noise_level = 0.001 * u.Jy

    extra_src_position = SkyCoord(ra=pointing_centre.ra + 0.01 * u.deg,
                                  dec=pointing_centre.dec + 0.01 * u.deg, )

    sources = [
        SkySource(pointing_centre, flux=1 * u.Jy),
        SkySource(extra_src_position, flux=0.4 * u.Jy),
    ]

    # Simulate incoming data; includes transient sources, noise:
    data_vis = visibility.visibilities_for_source_list(
        pointing_centre, sources, uvw_lambda)
    data_vis = visibility.add_gaussian_noise(vis_noise_level, data_vis)
    return data_vis
