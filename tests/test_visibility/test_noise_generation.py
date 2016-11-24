from __future__ import print_function
import numpy as np
import astropy.units as u
import fastimgproto.visibility as visibility
import pytest


def test_complex_gaussian_noise_generation():
    """
    Check that the noise added has the expected properties for both real
    and complex components, by asserting mean / std.dev. are close to expected
    values.
    """
    complex_zeroes = np.zeros(1e6, dtype=np.complex128)
    complex_noise_1jy = visibility.add_gaussian_noise(noise_level=1 * u.Jy,
                                                  vis=complex_zeroes,
                                                  seed=1234
                                                  )
    for component_array in np.real(complex_noise_1jy), np.imag(complex_noise_1jy):
        mean = np.mean(component_array)
        std_dev = np.std(component_array)
        print("Mean:", mean)
        print("S.D.:", std_dev)
        assert mean < 0.001
        assert np.fabs(std_dev - 1.0)< 0.02

    complex_noise_5jy = visibility.add_gaussian_noise(noise_level=5 * u.Jy,
                                                  vis=complex_zeroes,
                                                  seed=5678
                                                  )
    for component_array in np.real(complex_noise_5jy), np.imag(complex_noise_5jy):
        mean = np.mean(component_array)
        std_dev = np.std(component_array)
        print("Mean:", mean)
        print("S.D.:", std_dev)
        assert mean < 0.006
        assert np.fabs(std_dev - 5.0)< 0.1

