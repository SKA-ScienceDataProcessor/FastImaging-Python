from pytest import approx

from fastimgproto.fixtures.sourcefits import generate_sourcegrid_base_positions


def test_sourcegrid_base_positions():
    image_size = 100
    spacing_dist, grids = generate_sourcegrid_base_positions(image_size=image_size,
                                                            n_sources=49)
    ygrid, xgrid = grids
    assert xgrid.shape == (7,7)
    row_posns = xgrid[0]

    assert row_posns[0] == approx(spacing_dist/2.)
    assert row_posns[-1] == approx(image_size - spacing_dist / 2.)

    _, grid_49 = generate_sourcegrid_base_positions(image_size, n_sources=49)
    _, grid_37 = generate_sourcegrid_base_positions(image_size, n_sources=37)
    assert (grid_37 == grid_49).all()
