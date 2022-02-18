import numpy as np

from tssearch.distances.elastic_distances import dtw
from tssearch.distances.elastic_utils import traceback


def tam(x, y):
    """Calculates the Time Alignment Measurement (TAM) based on an optimal warping path
    between two time series.

    Reference: Folgado et. al, Time Alignment Measurement for Time Series, 2016.

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.

    Returns
    -------
    In case ``report=instants`` the number of indexes in advance, delay and phase
    will be returned.
    For ``report=ratios``, the ratio of advance, delay and phase.
    will be returned. In case ``report=distance``, only the TAM will be returned.

    """
    ac = dtw(x, y, report="cost_matrix")

    path = traceback(ac)

    # Delay and advance counting
    delay = len(np.where(np.diff(path[0]) == 0)[0])
    advance = len(np.where(np.diff(path[1]) == 0)[0])

    # Phase counting
    incumbent = np.where((np.diff(path[0]) == 1) * (np.diff(path[1]) == 1))[0]
    phase = len(incumbent)

    # Estimated and reference time series duration.
    len_estimation = path[1][-1]
    len_ref = path[0][-1]

    p_advance = advance * 1.0 / len_ref
    p_delay = delay * 1.0 / len_estimation
    p_phase = phase * 1.0 / np.min([len_ref, len_estimation])

    distance = p_advance + p_delay + (1 - p_phase)
    return distance
