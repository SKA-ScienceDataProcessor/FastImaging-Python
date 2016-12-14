import astropy.units as u
import numpy as np

from fastimgproto.coord_transforms import xyz_to_uvw_rotation_matrix

# Define the basis vectors with component co-ordinates in their
# own systems:
e_u = e_x = np.array([1, 0, 0], dtype=np.float_)
e_v = e_y = np.array([0, 1, 0], dtype=np.float_)
e_w = e_z = np.array([0, 0, 1], dtype=np.float_)


def check_uvw_results_as_expected(io_pairs, lha, dec, tolerance=1e-10):
    for xyz_vec, expected_uvw_vec in io_pairs:
        rotation = xyz_to_uvw_rotation_matrix(lha, dec)
        calculated_uvw_vec = np.dot(rotation, xyz_vec)
        assert (
        np.fabs(expected_uvw_vec - calculated_uvw_vec) < tolerance).all()


# Some useful invariants:
# - A source lying in the celestial equatorial plane will always imply
#   e_z and e_v are aligned. (Z -> V)
# - A source at LHA=0 will always imply e_y and e_u are aligned (Y -> U)

def test_equatorial_at_zero_lha():
    """
    Verify results for equatorial source at transit.

    I.e. a source at LHA, Dec of (0,0).

    In this case, mapping from XYZ to UVW is trivial.
    X (local celestial-equator transit) -> W (source direction)
    Y (Local East) -> U (LHA - 6h)
    Z (Celestial North) -> V( declination of  (pi/2 - source_dec) )

    """

    io_pairs = [
        (e_x, e_w),
        (e_y, e_u),
        (e_z, e_v),
    ]
    check_uvw_results_as_expected(io_pairs,
                                  lha=0 * u.degree, dec=0 * u.degree)


def test_equatorial_at_lha_minus_6():
    """
    Verify results for equatorial source at LHA=-6h (about to rise in the East)

    I.e. a source at LHA, Dec of ``(-90*u.deg,0)``.

    In this case, mapping from XYZ to UVW is as follows.
    X (local celestial-equator transit / LHA) -> -U (LHA - 6h)
    Y (Local East) -> W (Source direction)
    Z (Celestial North) -> V( declination of  (pi/2 - source_dec) )

    """
    io_pairs = [
        (e_x, -e_u),
        (e_y, e_w),
        (e_z, e_v),
    ]
    check_uvw_results_as_expected(io_pairs,
                                  lha=-90 * u.degree, dec=0 * u.degree)


def test_equatorial_at_lha_plus_6():
    """
    Verify results for equatorial source at LHA=+6h (about to set in the West)

    I.e. a source at LHA, Dec of ``(90*u.deg,0)``.

    In this case, mapping from XYZ to UVW is as follows.
    X (local celestial-equator transit / LHA) -> U (LHA - 6h)
    Y (Local East) -> -W (Source direction)
    Z (Celestial North) -> V( declination of  (pi/2 - source_dec) )

    """
    io_pairs = [
        (e_x, e_u),
        (e_y, -e_w),
        (e_z, e_v),
    ]
    check_uvw_results_as_expected(io_pairs,
                                  lha=90 * u.degree, dec=0 * u.degree)


def test_ncp():
    """
    Test easily verified results for a UVW relevant to a source at NCP

    (North Celestial Pole).

    Formally we'll define this as a source at LHA, Dec of ``(0, 90*u.deg)``.

    Note that even though the Hour angle of a *source* at the pole is
    ill-defined, the hour-angle of the transformation is both well defined and
    critically important, as it determines the orientation of the U-V plane
    with regard to the orientation of the X-Y plane.

    The mapping is then as follows:
    X (local celestial-equator transit) -> -V( declination of  (pi/2 - source_dec) )
    Y (Local East) -> U (LHA - 6h)
    Z (Celestial North) -> W (source direction)
    """
    io_pairs = [
        (e_x, -e_v),
        (e_y, e_u),
        (e_z, e_w),
    ]
    check_uvw_results_as_expected(io_pairs,
                                  lha=0 * u.degree, dec=90 * u.degree)


def test_dec_45():
    """
    Verify results for a source at LHA=0, Dec=45deg.

    Easiest to think of this in terms of reverse transform:
    W (source) -> 1/sqrt(2)* (X+Z)
    U (LHA-6) -> Y
    V (dec pi/2 - source_dec) -> 1/sqrt(2)* (Z-X)

    The mapping is then as follows:
    X (local celestial-equator transit) -> 1/sqrt(2)* (W-V)
    Y (Local East) -> U (LHA - 6h)
    Z (Celestial North) -> 1/sqrt(2)* (W+V)
    """

    io_pairs = [
        (e_x, (e_w - e_v) / np.sqrt(2.)),
        (e_y, e_u),
        (e_z, (e_w + e_v) / np.sqrt(2.)),
    ]
    check_uvw_results_as_expected(io_pairs,
                                  lha=0 * u.degree, dec=45 * u.degree)


def test_vectorial_operation():
    """
    It's desirable to make use of Numpy's vector optimizations and transform
    many XYZ positions in one go, so let's see if that works correctly:
    """
    lha = 123 * u.degree
    dec = 42.3213 * u.degree
    rotation = xyz_to_uvw_rotation_matrix(lha, dec)

    xyz_vec_tuple = (e_x, e_x, e_y, e_z, e_y, e_z)

    # Construct a known-good answer from each basis vector in turn:
    uvw_expected_results = []
    for vec in xyz_vec_tuple:
        uvw_expected_results.append(np.dot(rotation, vec))

    # By default, our 'vectors' are numpy 1-d horizontal arrays.
    # So, we vstack them, then transpose to get the correct orientation.
    # This results in a 3xn matrix for n XYZ vectors.
    xyz_vec_array = np.vstack(xyz_vec_tuple).T
    assert xyz_vec_array.shape == (3, len(xyz_vec_tuple))

    # We then get a 3xn matrix of UVW vectors:
    uvw_vectorized_results = np.dot(rotation, xyz_vec_array)
    assert uvw_vectorized_results.shape == (3, len(xyz_vec_tuple))

    # Transpose back to get easy 'tuple-style' indexing into the collection
    # of vectors:
    uvw_vectorized_results = uvw_vectorized_results.T
    for idx in range(len(uvw_expected_results)):
        assert (uvw_expected_results[idx] == uvw_vectorized_results[idx]).all()
