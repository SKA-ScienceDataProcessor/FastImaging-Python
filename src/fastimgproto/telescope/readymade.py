from .base import (Telescope, parse_itrf_ascii_to_xyz_and_labels)
from .data import (meerkat_itrf_filepath,
                   vla_a_filepath, vla_b_filepath,
                   vla_c_filepath, vla_d_filepath
                   )


def Meerkat():
    return Telescope.from_itrf(
        *parse_itrf_ascii_to_xyz_and_labels(meerkat_itrf_filepath))


def VLA_A():
    return Telescope.from_itrf(
        *parse_itrf_ascii_to_xyz_and_labels(vla_a_filepath))

def VLA_B():
    return Telescope.from_itrf(
        *parse_itrf_ascii_to_xyz_and_labels(vla_b_filepath))

def VLA_C():
    return Telescope.from_itrf(
        *parse_itrf_ascii_to_xyz_and_labels(vla_c_filepath))

def VLA_D():
    return Telescope.from_itrf(
        *parse_itrf_ascii_to_xyz_and_labels(vla_d_filepath))
