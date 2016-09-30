import fastimgproto.gridder.conv_funcs as conv_funcs
import numpy as np
from fastimgproto.gridder.kernel_generation import Kernel


def test_regular_sampling_pillbox():
    testfunc = conv_funcs.Pillbox(half_base_width=1.1)
    support = 2
    oversampling = 1

    # Map subpixel offset to expected results
    expected_results = {}

    # No offset - expect 1's for all central 3x3 pixels:
    expected_results[(0., 0.)] = np.array(
        [[0., 0., 0., 0., 0.],
         [0., 1., 1., 1., 0.],
         [0., 1., 1., 1., 0.],
         [0., 1., 1., 1., 0.],
         [0., 0., 0., 0., 0.]]
    )

    # Tiny offset (less than pillbox overlap) - expect same:
    expected_results[(0.05, 0.05)] = expected_results[(0., 0.)]

    ## Now shift the pillbox just enough to +ve x that we drop a column
    expected_results[(0.15, 0.0)] = np.array(
        [[0., 0., 0., 0., 0.],
         [0., 0., 1., 1., 0.],
         [0., 0., 1., 1., 0.],
         [0., 0., 1., 1., 0.],
         [0., 0., 0., 0., 0.]]
    )

    ## And shift to -ve x:
    expected_results[(-0.15, 0.0)] = np.array(
        [[0., 0., 0., 0., 0.],
         [0., 1., 1., 0., 0.],
         [0., 1., 1., 0., 0.],
         [0., 1., 1., 0., 0.],
         [0., 0., 0., 0., 0.]]
    )

    ## Shift to +ve y:
    expected_results[(0.0, 0.15)] = np.array(
        [[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 1., 1., 1., 0.],
         [0., 1., 1., 1., 0.],
         [0., 0., 0., 0., 0.]]
    )

    ## Shift to +ve y & +ve x:
    expected_results[(0.15, 0.15)] = np.array(
        [[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 1., 1., 0.],
         [0., 0., 1., 1., 0.],
         [0., 0., 0., 0., 0.]]
    )

    for offset, expected_array in expected_results.items():
        k = Kernel(kernel_func=testfunc,
                   support=support,
                   offset=offset,
                   oversampling=oversampling,
                   normalize=False,
                   )
        assert (k.array == expected_array).all()


def test_regular_sampling_triangle():
    testfunc = conv_funcs.Triangle(half_base_width=1.5)
    support = 2
    oversampling = 1

    kernel_value_at_1pix = 1. - 1. / 1.5
    kv1 = kernel_value_at_1pix
    kv1sq = kv1 * kv1

    # Map subpixel offset to expected results
    expected_results = {}

    # No offset
    expected_results[(0., 0.)] = np.array(
        [[0., 0., 0., 0., 0.],
         [0., kv1sq, kv1, kv1sq, 0.],
         [0., kv1, 1., kv1, 0.],
         [0., kv1sq, kv1, kv1sq, 0.],
         [0., 0., 0., 0., 0.]]
    )

    # .5 pix offset right:
    kv_half = 1. - 0.5 / 1.5
    expected_results[(0.5, 0.)] = np.array(
        [[0., 0., 0., 0., 0.],
         [0., 0., kv_half * kv1, kv_half * kv1, 0.],
         [0., 0., kv_half, kv_half, 0.],
         [0., 0., kv_half * kv1, kv_half * kv1, 0.],
         [0., 0., 0., 0., 0.]]
    )

    for offset, expected_array in expected_results.items():
        k = Kernel(kernel_func=testfunc,
                   support=support,
                   offset=offset,
                   oversampling=oversampling,
                   normalize=False,
                   )

        assert (k.array == expected_array).all()


def test_oversampled_pillbox():
    testfunc = conv_funcs.Pillbox(half_base_width=0.7)
    support = 1
    oversampling = 3

    # Map subpixel offset to expected results
    expected_results = {}

    # No offset - expect 1's for all central 5x5 pixels, since cut-off is
    # just above 2/3:
    expected_results[(0., 0.)] = np.array(
        [[0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 1., 1., 1., 1., 0.],
         [0., 1., 1., 1., 1., 1., 0.],
         [0., 1., 1., 1., 1., 1., 0.],
         [0., 1., 1., 1., 1., 1., 0.],
         [0., 1., 1., 1., 1., 1., 0.],
         [0., 0., 0., 0., 0., 0., 0.]]
    )
    # Same for tiny offset
    expected_results[(0.01, 0.01)] = expected_results[(0.00, 0.00)]

    # Displace towards -ve x a bit:
    expected_results[(-.05, 0.)] = np.array(
        [[0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 1., 1., 1., 0., 0.],
         [0., 1., 1., 1., 1., 0., 0.],
         [0., 1., 1., 1., 1., 0., 0.],
         [0., 1., 1., 1., 1., 0., 0.],
         [0., 1., 1., 1., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.]]
    )

    expected_results[(0.4, 0.)] = np.array(
        [[0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 1., 1., 1.],
         [0., 0., 0., 1., 1., 1., 1.],
         [0., 0., 0., 1., 1., 1., 1.],
         [0., 0., 0., 1., 1., 1., 1.],
         [0., 0., 0., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0., 0., 0.]]
    )
    for offset, expected_array in expected_results.items():
        k = Kernel(kernel_func=testfunc,
                   support=support,
                   offset=offset,
                   oversampling=oversampling,
                   normalize=False,
                   )

        assert (k.array == expected_array).all()


def test_oversampled_pillbox_small():
    testfunc = conv_funcs.Pillbox(half_base_width=0.25)
    support = 1
    oversampling = 5

    # Map subpixel offset to expected results
    expected_results = {}

    expected_results[(0., 0.)] = np.array(
        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         ]
    )

    expected_results[(0.4, 0.0)] = np.array(
        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         ]
    )

    for offset, expected_array in expected_results.items():
        k = Kernel(kernel_func=testfunc,
                   support=support,
                   offset=offset,
                   oversampling=oversampling,
                   normalize=False,
                   )

        assert (k.array == expected_array).all()
