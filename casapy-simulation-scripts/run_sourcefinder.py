from astropy.io import fits
import fastimgproto.sourcefind as sf


fits_path = './vla.image.fits'

hdu0 = fits.open(fits_path)[0]
image_data = hdu0.data.squeeze()


# ...