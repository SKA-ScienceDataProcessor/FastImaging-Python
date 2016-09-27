import fastimgproto.gridder.conv_funcs as conv_funcs
import numpy as np


def test_triangle_func():
    triangle2 = conv_funcs.Triangle(half_base_width=2.0)

    # Test known simple inputs
    assert triangle2(0.0) == 1.0
    assert triangle2(1.0) == 0.5
    assert triangle2(2.0) == 0.0
    assert triangle2(2.000001) == 0.0
    assert triangle2(100) == 0.0
    assert triangle2(-0.1) == triangle2(0.1)

    # test with array input
    input = np.array([-2., 0.0, 0.5, 1.0, 2.0, 3.0])
    output = np.array([0, 1.0, 0.75, 0.5, 0.0, 0.0])
    assert (triangle2(input) == output).all()


def test_tophat_funct():
    tophat3 = conv_funcs.Pillbox(half_base_width=3.0)

    input = np.array([-4.2, -2.5, 0.0, 1.5, 2.999, 3.0, 3.1])
    output = np.array([0.0, 1.0, 1.0, 1., 1., 0.0, 0.0])

    # test with array input
    assert (tophat3(input) == output).all()

    # test with scalar input
    for idx, val in enumerate(input):
        assert tophat3(val) == output[idx]