import astropy.io.fits as fits
import numpy as np


def load_fits_image_data(fits_path):
    hdulist = fits.open(fits_path)
    return hdulist[0].data.squeeze()


def save_as_fits(ndarray, fits_path):
    hdu = fits.PrimaryHDU(ndarray)
    hdu.writeto(fits_path)
