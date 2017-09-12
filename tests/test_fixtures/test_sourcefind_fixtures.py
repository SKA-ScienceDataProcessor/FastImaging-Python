import numpy as np
from pytest import approx

from fastimgproto.fixtures.sourcefind import (
    calculate_sourcegrid_base_positions,
    random_sources_on_grid,
)


def test_sourcegrid_base_positions():
    image_size = 100
    spacing_dist, grids = calculate_sourcegrid_base_positions(image_size=image_size,
                                                              n_sources=49)
    ygrid, xgrid = grids
    assert xgrid.shape == (7,7)
    row_posns = xgrid[0]

    assert row_posns[0] == approx(spacing_dist/2.)
    assert row_posns[-1] == approx(image_size - spacing_dist / 2.)

    _, grid_49 = calculate_sourcegrid_base_positions(image_size, n_sources=49)
    _, grid_37 = calculate_sourcegrid_base_positions(image_size, n_sources=37)
    assert (grid_37 == grid_49).all()

def test_random_sources_on_grid():
    """
    Check that the random-subpixel-offset source locating works as intended
    """
    image_size = 200
    n_sources = 25
    spacing_dist, grids = calculate_sourcegrid_base_positions(
        image_size=image_size,
        n_sources=n_sources)
    y_grid, x_grid = grids
    # This choice of image_size, n_sources results in integer-valued grids
    assert (x_grid == x_grid.astype(np.int_)).all()

    sources = random_sources_on_grid(image_size=image_size,
                                     n_sources=n_sources,
                                     amplitude_range=(1.,10,),
                                     semiminor_range=(1,1.5),
                                     axis_ratio_range=(1.,1.5),
                                     )
    y_posn = np.array([s.y_centre for s in sources])
    x_posn = np.array([s.x_centre for s in sources])

    # Check that offsets have been applied correctly
    assert (np.int_(y_posn) == y_grid.ravel()).all()
    assert (y_posn != y_grid.ravel()).all()

    assert (np.int_(x_posn) == x_grid.ravel()).all()
    assert (x_posn != x_grid.ravel()).all()
