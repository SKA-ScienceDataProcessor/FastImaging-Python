from fastimgproto.sourcefind.image import SourceFindImage
from memory_profiler import profile


@profile
def memprof_sourcefindimage(data, detection_n_sigma, analysis_n_sigma):
    return SourceFindImage(data = data, detection_n_sigma=5, analysis_n_sigma=3)
