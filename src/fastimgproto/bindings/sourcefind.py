def _cpp_sourcefind(image_data,
                    detection_n_sigma,
                    analysis_n_sigma,
                    rms_est=0.0):
    pass


def _python_sourcefind(image_data,
                       detection_n_sigma,
                       analysis_n_sigma,
                       rms_est=0):
    """
    Equivalent Python code for validation of _cpp_sourcefind

    Args:
        image_data (numpy.ndarray): Real component of iFFT'd image
            (Array of np.float_).
        detection_n_sigma (float): Detection threshold as multiple of RMS
        analysis_n_sigma (float): Analysis threshold as multiple of RMS
        rms_est (float): RMS estimate (may be `0.0`, in which case RMS is
            estimated from the image data).

    Returns:
        list: List of tuples representing the source-detections. Tuple
        components are as follows:
        ``(sign, val, x_idx, y_idx, xbar, ybar)``
        Where:
        - ``sign`` is ``+1`` or ``-1`` (int), representing whether the source is
            positive or negative
        - ``val`` (float) is the 'extremum_val', i.e. max or min pixel value for
        the positive or negative source case.
        - ``x_idx,y_idx`` (int) are the pixel-index of the extremum value.
        - ``xbar, ybar`` (float) are 'centre-of-mass' locations for the
        source-detection island.
    """
    pass
