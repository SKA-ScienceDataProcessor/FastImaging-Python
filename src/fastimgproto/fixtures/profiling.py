from memory_profiler import profile

from fastimgproto.sourcefind.image import SourceFindImage


@profile
def memprof_sourcefindimage(data, detection_n_sigma, analysis_n_sigma):
    return SourceFindImage(data = data, detection_n_sigma=5, analysis_n_sigma=3)
