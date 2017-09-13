from fastimgproto.sourcefind.image import (
    Gaussian2dParams,
    IslandParams,
    Pixel,
    PixelIndex,
)

from .present import CPP_BINDINGS_PRESENT

if CPP_BINDINGS_PRESENT:
    import stp_python


def cpp_sourcefind_result_to_islandparams(result):
    r = result
    sign = r[0]
    extremum = Pixel(value=r[1],
                     index=PixelIndex(x=r[2], y=r[3]), )
    xbar = r[4]
    ybar = r[5]
    fitvals = r[6]
    report = r[7]
    gpars = Gaussian2dParams(x_centre=fitvals.x_centre,
                             y_centre=fitvals.y_centre,
                             amplitude=fitvals.amplitude,
                             semimajor=fitvals.semimajor,
                             semiminor=fitvals.semiminor, theta=fitvals.theta
                             )
    return IslandParams(sign=sign, extremum=extremum, leastsq_fit=gpars,
                        fitter_report=report)


def cpp_sourcefind(image_data,
                   detection_n_sigma,
                   analysis_n_sigma,
                   rms_est,
                   find_negative=True,
                   sigma_clip_iters=5,
                   binapprox_median=True,
                   compute_barycentre=True,
                   gaussian_fitting=True,
                   generate_labelmap=False,
                   # Other options: stp_python.CeresDiffMethod.AutoDiff, stp_python.CeresDiffMethod.AnalyticDiff_SingleResBlk, stp_python.CeresDiffMethod.AnalyticDiff
                   ceres_diffmethod=stp_python.CeresDiffMethod.AnalyticDiff,
                   # Other options: stp_python.CeresSolverType.LinearSearch_LBFGS, stp_python.CeresSolverType.TrustRegion_DenseQR
                   ceres_solvertype=stp_python.CeresSolverType.TrustRegion_DenseQR,
                   ):
    if not CPP_BINDINGS_PRESENT:
        raise OSError("Cannot import stp_python (C++ bindings module)")

    # Call source_find
    raw_results = stp_python.source_find_wrapper(
        image_data, detection_n_sigma, analysis_n_sigma, rms_est, find_negative,
        sigma_clip_iters,
        binapprox_median, compute_barycentre, gaussian_fitting,
        generate_labelmap, ceres_diffmethod, ceres_solvertype)

    return [cpp_sourcefind_result_to_islandparams(r) for r in raw_results]


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
