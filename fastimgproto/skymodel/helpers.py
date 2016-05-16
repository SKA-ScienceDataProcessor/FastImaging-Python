"""
Basic classes used to help structure data related to skymodel, skyregions, etc.
"""

from astropy.coordinates import Angle, SkyCoord
import astropy.units as u


class SkyRegion(object):
    """
    Defines a circular region of the sky.
    """

    def __init__(self, centre, radius):
        assert isinstance(centre, SkyCoord)
        assert isinstance(radius, Angle)
        self.centre = centre
        self.radius = radius


class PositionError(object):
    def __init__(self, ra_err, dec_err):
        assert isinstance(ra_err, Angle)
        assert isinstance(dec_err, Angle)
        self.ra = ra_err
        self.dec = dec_err

    def __str__(self):
        return "<PositionError: (ra, dec) in deg ({}, {})>".format(
            self.ra.deg, self.dec.deg)


class SkySource(object):
    """
    Basic point source w/ flux modelled at a single frequency

    Args:
        position (astropy.coordinates.SkyCoord): Sky-coordinates of source.
        flux (astropy.units.Quantity): Source flux at measured frequency.
        frequency (astropy.units.Quantity): Measurement frequency.
        variable (bool): 'Known variable' flag.
    """
    def __init__(self, position, flux,
                 frequency=2.5*u.GHz, variable=False):
        assert isinstance(position, SkyCoord)
        #This will raise if the flux has wrong units or no units:
        flux.to(u.Jy)
        frequency.to(u.Hz)
        self.position = position
        self.flux = flux
        self.variable = variable
        self.frequency = frequency
