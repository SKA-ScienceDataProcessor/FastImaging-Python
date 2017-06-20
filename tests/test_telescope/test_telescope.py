from __future__ import print_function

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, Latitude, Longitude, SkyCoord
from astropy.time import Time

from fastimgproto.telescope import Telescope
from fastimgproto.telescope.base import parse_itrf_ascii_to_xyz_and_labels
from fastimgproto.telescope.data import meerkat_itrf_filepath
from fastimgproto.telescope.readymade import Meerkat


def test_telescope_from_itrf():
    meerkat = Telescope.from_itrf(
        *parse_itrf_ascii_to_xyz_and_labels(meerkat_itrf_filepath)
    )
    # for idx in range(5):
    #     print(meerkat.baseline_labels[idx], meerkat.baseline_local_xyz[idx])


def test_xyz_antennae_to_uvw_baselines():
    """
    We've already unit-tested the component parts of this, so we just include
    a basic test for sanity-checking.

    At Lat/Lon (0,0), Local XYZ maps trivally to UVW.

    X (local celestial-equator transit) -> W (source direction)
    Y (Local East) -> U (LHA - 6h)
    Z (Celestial North) -> V( declination of  (pi/2 - source_dec) )
    """
    centre = EarthLocation.from_geodetic(
        Longitude(0 * u.deg), Latitude(0 * u.deg))

    ant_local_xyz = np.array([
        [0, 0, 0],  # Origin
        [0, 1, 0],  # 1-East
        [0, 0, 1],  # 1-North
    ], dtype=np.float_) * u.m
    ant_labels = [
        'origin',
        '1-east',
        '1-north',
    ]

    expected_uvw_baselines = np.array([
        [1., 0., 0.],  # origin,1-east
        [0., 1., 0.],  # origin,1-north
        [-1., 1., 0.]]  # 1-east,1-north
    ) * u.m

    tel = Telescope(
        centre=centre,
        ant_labels=ant_labels,
        ant_local_xyz=ant_local_xyz
    )
    # print(tel.baseline_labels)
    # print(tel.baseline_local_xyz)

    lha = 0 * u.deg
    dec = 0 * u.deg
    uvw = tel.uvw_at_local_hour_angle(lha, dec)
    assert (uvw == expected_uvw_baselines).all()
    # print("UVW:")
    # print(repr(uvw))
    # print(tel.baseline_labels)


def test_uvw_tracking_skyposn():
    """
    Check UVW calculations for Time+RA+Dec vs LHA+Dec produce consistent results
    """
    telescope = Meerkat()
    azimuth_target = SkyCoord(ra=0 * u.deg, dec=telescope.lat)

    t0 = Time('2017-01-01 12:00:00', format='iso', scale='utc')
    tt = telescope.next_transit(azimuth_target.ra, t0)
    tt_lha = telescope.lha(azimuth_target.ra, tt)
    assert np.fabs(tt_lha) < 0.2 * u.arcsec

    obs_times = [tt - 10 * u.second, tt, tt + 10 * u.second]
    uvw_by_tracking = telescope._uvw_tracking_skycoord_by_lha(
        azimuth_target,
        obs_times=obs_times)
    assert len(uvw_by_tracking) == len(obs_times)

    uvw_by_lha = telescope.uvw_at_local_hour_angle(tt_lha, azimuth_target.dec)
    assert uvw_by_lha.to(u.m)
    assert list(uvw_by_tracking.keys())[1] == tt_lha
    assert (uvw_by_lha == list(uvw_by_tracking.values())[1]).all()

    n_total_baselines = len(telescope.baseline_local_xyz) * len(obs_times)
    flattened_uvw_array = telescope.uvw_tracking_skycoord(
        azimuth_target, obs_times)
    assert flattened_uvw_array.to(u.m)
    assert len(flattened_uvw_array) == n_total_baselines

    concat_array = np.concatenate(list(uvw_by_tracking.values()))
    # Work around np.concat returning uninitialsed units:
    # http://docs.astropy.org/en/stable/known_issues.html#id6
    concat_array = concat_array.value * uvw_by_lha.unit

    assert (flattened_uvw_array == concat_array).all()
