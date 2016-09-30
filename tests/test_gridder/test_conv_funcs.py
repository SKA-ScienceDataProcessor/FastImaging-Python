import fastimgproto.gridder.conv_funcs as conv_funcs
import numpy as np


def check_function_results_close(func, input_output_pairs,
                                 symmetric=True,
                                 eps=1e-15):
    # test with scalar inputs
    for pair in input_output_pairs:
        assert np.fabs(func(pair[0]) - pair[1]) < eps

    # test with array input
    inputs = input_output_pairs[:, 0]
    outputs = input_output_pairs[:, 1]
    assert (np.fabs(func(inputs) - outputs) < eps).all()
    if symmetric:
        inputs *= -1.0
        assert (np.fabs(func(inputs) - outputs) < eps).all()


def test_triangle_func():
    triangle2 = conv_funcs.Triangle(half_base_width=2.0)

    io_pairs = np.array([
        [0.0, 1.0],
        [1.0, 0.5],
        [2.0, 0.0],
        [2.000001, 0.0],
        [100, 0.0],
        [0.1, 0.95],
        [0.5, 0.75],
    ])

    check_function_results_close(triangle2, io_pairs)


def test_tophat_func():
    tophat3 = conv_funcs.Pillbox(half_base_width=3.0)
    io_pairs = np.array([
        [0.0, 1.0],
        [2.5, 1.0],
        [2.999, 1.0],
        [3.0, 0.0],
        [4.2, 0.],
    ])
    check_function_results_close(tophat3, io_pairs)


def test_sinc():
    sinc = conv_funcs.Sinc(trunc=3.0)
    io_pairs = np.array([
        [0.0, 1.0],
        [0.5, 1. / (0.5 * np.pi)],
        [1.0, 0.0],
        [1.5, -1. / (1.5 * np.pi)],
        [2.0, 0.0],
        [2.5, 1. / (2.5 * np.pi)],
        [3.5, 0.],  # Truncated
    ])
    check_function_results_close(sinc, io_pairs)


def test_gaussian():
    trunc = 3.0
    sinc = conv_funcs.Gaussian(trunc=trunc, w=1.0)
    io_pairs = np.array([
        [0.0, 1.0],
        [1.0, 1./np.exp(1.)],
        [3.1, 0.],  # Truncated
    ])
    check_function_results_close(sinc, io_pairs)

def test_gaussian_sinc():
    # Use conventional scaling values:
    w1 = 2.52
    w2 = 1.55
    trunc = 5.0
    sinc = conv_funcs.GaussianSinc(trunc=trunc, w1=w1, w2=w2)
    io_pairs = np.array([
        [0.0, 1.0],
        [w2 * 0.5,
         np.exp(-1. * (0.5 * w2 / w1) ** 2) * 1. / (0.5 * np.pi)],
        [w2, 0.0],
        [w2 * 1.5,
         np.exp(-1. * (1.5 * w2 / w1) ** 2) * -1. / (1.5 * np.pi)],
        [w2 * 2., 0.0],
        [w2 * 2.5,
         np.exp(-1. * (2.5 * w2 / w1) ** 2) * 1. / (2.5 * np.pi)],
        [5.5, 0.],  # Truncated
    ])
    check_function_results_close(sinc, io_pairs)
