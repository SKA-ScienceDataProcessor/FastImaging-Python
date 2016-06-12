
from astropy.utils.data import download_file
from astropy.coordinates import SkyCoord
import astropy.units as u
from fastimgproto.skymodel.helpers import (SkyRegion, SkySource, )
from fastimgproto.skymodel.extraction import (
    sumss_file_to_dataframe, lsm_extract
)


def get_lsm(field_of_view):
    catalog_file = download_file(
        'http://www.astrop.physics.usyd.edu.au/sumsscat/sumsscat.Mar-11-2008.gz',
        cache=True)
    full_cat = sumss_file_to_dataframe(catalog_file)
    lsm_cat = lsm_extract(field_of_view, full_cat)
    return lsm_cat


def get_spiral_source_test_pattern(field_of_view):
    assert isinstance(field_of_view, SkyRegion)

    offset = 0.05 * u.deg

    centre = field_of_view.centre
    north = SkyCoord(centre.ra, centre.dec + offset)
    east = SkyCoord(centre.ra + 2 * offset, centre.dec)
    south = SkyCoord(centre.ra, centre.dec - 3 * offset)
    west = SkyCoord(centre.ra - 4 * offset, centre.dec)

    srclist = []
    for posn in (
            centre,
            north,
            east,
            south,
            west):
        srclist.append(SkySource(
            position=posn,
            flux=1 * u.Jy
        ))
    return srclist