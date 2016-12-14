import numpy as np

from fastimgproto.telescope.base import parse_itrf_ascii_to_xyz_and_labels
from fastimgproto.telescope.data import meerkat_itrf_filepath


def test_itrf_parse():
    xyz, labels = parse_itrf_ascii_to_xyz_and_labels(meerkat_itrf_filepath)
    assert len(xyz) == len(labels)
    assert xyz.shape == (64,3)
    assert xyz.dtype == np.float_
    assert labels.dtype.type == np.str_

