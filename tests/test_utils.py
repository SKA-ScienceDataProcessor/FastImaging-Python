import numpy as np

from fastimgproto.utils import nonzero_bounding_slice_2d


def test_bounding_slice_2d():
    a = np.zeros((5, 5), dtype=np.float_)
    # Should return None if no non-zero elements:
    assert nonzero_bounding_slice_2d(a) == None
    val1 = 42
    val2 = 3.14

    a[1, 2] = val1
    bs = nonzero_bounding_slice_2d(a)
    y_slice, x_slice = bs
    assert y_slice == slice(1, 2)
    assert x_slice == slice(2, 3)
    assert len(a[bs].ravel()) == 1

    a[2, 3] = val2
    bs = nonzero_bounding_slice_2d(a)
    bb_subarray = a[bs]
    expected_subarray = np.array([[val1, 0],
                                  [0, val2], ])
    assert (bb_subarray == expected_subarray).all()
    bb_nz_vals = bb_subarray[np.nonzero(bb_subarray)].ravel()
    assert len(bb_nz_vals) == 2
    assert (bb_nz_vals == np.array([val1, val2])).all()

    # Add a non-connected value and check the bounding box encompasses the lot
    a[4, 4] = val1
    bs = nonzero_bounding_slice_2d(a)
    y_slice, x_slice = bs
    assert y_slice == slice(1, 5)
    assert x_slice == slice(2, 5)
