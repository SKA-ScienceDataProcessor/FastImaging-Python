import astropy.units as u
from astropy.time import Time
from astropy.coordinates import Longitude
from fastimgproto.coords import time_of_next_transit
import numpy as np


def test_basic_functionality():
    # Recall  ``LHA = LST - RA``

    # Pick an arbitrary date
    t0 = Time('2017-01-01 12:00:00', format='iso', scale='utc')
    # And location longitude. Let's go with Greenwich:
    lon0 = Longitude(0 * u.deg)
    # And target RA. Let's go with 0, so LHA = LST
    ra = Longitude(0 * u.deg)
    tol = 0.02 * u.arcsec

    # Next transit-time at lon=0
    tt_lon0 = time_of_next_transit(observer_longitude=lon0, target_ra=ra,
                                   start_time=t0, tolerance=tol)

    tt1_lst = tt_lon0.sidereal_time('apparent', longitude=lon0)
    assert np.fabs(tt1_lst) < tol

    # If we provide a close-enough-to-transit time, just return it, don't
    # endlessly cycle by a sidereal day:
    tt_lon0_tnt = time_of_next_transit(lon0, ra, tt_lon0, tol)
    assert tt_lon0_tnt == tt_lon0

    # Longitude is positive east. So a negative longitude observer (West) should
    # see a later transit:

    lon_m30 = Longitude(-30 * u.deg)
    tt_lon30 = time_of_next_transit(observer_longitude=lon_m30, target_ra=ra,
                                    start_time=t0, tolerance=tol)
    assert tt_lon30 > tt_lon0
    # Difference should be 30 / 360 = 2 sidereal hours
    shour = 1./24.*u.sday # Sidereal hour
    assert 2.0 == (tt_lon30 - tt_lon0).to(u.s)/shour.to(u.s)
