
output_ms = 'vla-sim.MS'

# image properties
NPIX = 2048  # the image size
CELL = '0.4arcsec'  # the resolution of each pixel

# cleaning
GAIN = 0.1  # gain
NITER = 100  # cycles


# -----------------------------------------
#
# 3. We've generated the measurement set. Now let's create the images.
#
# -----------------------------------------

im = casac.imager()
im.open(output_ms)
im.defineimage(nx=NPIX, ny=NPIX, cellx=CELL, celly=CELL, stokes='I')
im.clean(algorithm='csclean', image='vla.image', model='vla.model',
         residual='vla.residual', mask='', niter=NITER, gain=GAIN)
im.close()
im.done()

exportfits('vla.image', 'vla.image.fits')