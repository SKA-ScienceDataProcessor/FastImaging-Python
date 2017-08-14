import numpy as np


def reset_progress_bar(progress_bar, total, description="Doing something"):
    """
    Short subroutine to reset a progress bar.

    Currently setup for tqdm progress bars, but we could imagine this
    function evolving for a different libary or even handling progress bars
    polymorphically.

    Mainly it just tidies 3 irrelevant lines into one clear function call,
    which is less distracting from key logic.

    Currently
    Args:
        progress_bar (tqdm.tqdm): Progress display class
        total (int): total number of iterations being stepped through
        description (str):  Description to display

    Returns:

    """
    progress_bar.n = 0
    progress_bar.total = total
    progress_bar.desc = description
    return



def positive_negative_sign_validator(instance, attribute, value):
    """
    Attrs validator for verifying ``value in (-1,1)``.
    """
    if value not in (-1, 1):
        raise ValueError("'sign' should be +1 or -1")


def nonzero_bounding_slice_2d(input):
    """
    Get slices defining the bounding box for any nonzero / True-valued subarray.

    ...of a 2-dimensional ndarray.

    NB Not a Number (NaN), positive infinity and negative infinity evaluate to
    True because these are not equal to zero (cf numpy.any).

    cf stackoverflow answer:
    https://stackoverflow.com/a/31402351/725650
    - contains extension to n-dimensions, but then you have to consider
    all (n-1)-choices from a set of n axes, so it's a fair bit trickier to
    follow. Probably YAGNI.

    Args:
        input (numpy.ndarray): Input array

    Returns:
        tuple or None: A tuple of slices, `(y_range, x_range)` which can be
        used to iterate over the bounding box of all non-zero values.
        If there are no nonzero values, returns `None`.
    """
    assert isinstance(input, np.ndarray)
    assert input.ndim == 2
    y_nz_idx = np.any(input, axis=1)
    if not y_nz_idx.any():
        return None
    x_nz_idx = np.any(input, axis=0)
    y_nz_min, y_nz_max = np.where(y_nz_idx)[0][[0, -1]]
    x_nz_min, x_nz_max = np.where(x_nz_idx)[0][[0, -1]]
    yslice = slice(y_nz_min, y_nz_max + 1)
    xslice = slice(x_nz_min, x_nz_max + 1)
    return (yslice, xslice)
