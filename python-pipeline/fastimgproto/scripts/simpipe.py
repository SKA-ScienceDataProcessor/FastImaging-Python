"""
Simulated pipeline run
"""

import click
from astropy.coordinates import Angle,SkyCoord
import astropy.units as u
from fastimgproto.skymodel.helpers import SkyRegion, SkySource
from fastimgproto.pipeline import (get_lsm,
                                   get_spiral_source_test_pattern)


def main():
    field_of_view = SkyRegion(SkyCoord(180*u.deg, 34*u.deg),
                              radius=Angle(1*u.deg))


    # source_list = get_lsm(field_of_view)
    source_list = get_spiral_source_test_pattern(field_of_view)

    # Simulate 'incoming data' using casapy, this gives us UVW for free

    # Load up UVW and generate visibilities according to skymodel

    # Export to casa format

    # Subtract model-generated visibilities from incoming data
    # Image the difference

    # (also image incoming / model data separately for desk-checking).

    # Source find / verify output


@click.command()
def cli():
    main()
