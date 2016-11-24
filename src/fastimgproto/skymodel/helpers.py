"""
Basic classes used to help structure data related to skymodel, skyregions, etc.
"""

from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
import attr.validators
from attr import attrs, attrib


@attrs
class SkyRegion(object):
    """
    Defines a circular region of the sky.
    """
    centre = attrib(validator=attr.validators.instance_of(SkyCoord))
    radius = attrib(validator=attr.validators.instance_of(Angle))


@attrs
class PositionError(object):
    """
    Represent positional uncertainty.

    (Mainly used for representing entries in the SUMSS catalog.)
    """
    ra = attrib(validator=attr.validators.instance_of(Angle))
    dec = attrib(validator=attr.validators.instance_of(Angle))


@attrs
class SkySource(object):
    """
    Basic point source w/ flux modelled at a single frequency

    Attributes:
        position (astropy.coordinates.SkyCoord): Sky-coordinates of source.
        flux (astropy.units.Quantity): Source flux at measured frequency.
        frequency (astropy.units.Quantity): Measurement frequency.
        variable (bool): 'Known variable' flag.
    """
    position = attrib(validator=attr.validators.instance_of(SkyCoord))
    flux = attrib(convert=lambda x: x.to(u.Jy))
    frequency = attrib(default=2.5 * u.GHz, convert=lambda x: x.to(u.GHz))
    variable = attrib(default=False)
