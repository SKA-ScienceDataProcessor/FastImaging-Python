from .base import (Telescope, parse_itrf_ascii_to_xyz_and_labels)
from .data import (meerkat_itrf_filepath)

meerkat = Telescope.from_itrf(
    *parse_itrf_ascii_to_xyz_and_labels(meerkat_itrf_filepath))
