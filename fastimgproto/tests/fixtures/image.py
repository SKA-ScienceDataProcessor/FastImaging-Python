import numpy as np
import astropy.units as u
from astropy.modeling.models import Gaussian2D


def uncorrelated_gaussian_noise(shape, mean=0, sigma=1.0):
    normal_noise = np.random.randn(*shape)
    return (sigma * normal_noise) + mean


def gaussian_point_source(x_centre,
                          y_centre,
                          amplitude=1.0,
                          semimajor_gaussian_sigma=1.5,
                          semiminor_gaussian_sigma=1.2,
                          position_angle=1. * u.rad,
                          ):
    return Gaussian2D(amplitude=amplitude,
                      x_mean=x_centre,
                      y_mean=y_centre,
                      x_stddev=semimajor_gaussian_sigma,
                      y_stddev=semiminor_gaussian_sigma,
                      theta=position_angle.to(u.rad)
                      )


def add_models_to_background(bg_image, model_list):
    ydim, xdim = bg_image.shape
    ygrid, xgrid = np.mgrid[:ydim, :xdim]
    for model in model_list:
        bg_image += model(xgrid, ygrid)
