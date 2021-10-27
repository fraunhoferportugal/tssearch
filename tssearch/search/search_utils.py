import numpy as np
from scipy.signal import find_peaks


def elastic_search(distance, query, sequence, tq=None, ts=None, weight=None):
    """

    Parameters
    ----------
    sequence
    query
    distance

    Returns
    -------

    """

    exec("from tssearch import *")

    # distance function
    func_total = distance["function"]

    # Check for parameters
    parameters_total = {}
    if distance["parameters"] != "":
        parameters_total = distance["parameters"]
    parameters_total["report"] = "search"

    if "dtw_type" in parameters_total:
        if parameters_total["dtw_type"] == "dtw":
            parameters_total["dtw_type"] = "sub-dtw"

    if "time" in parameters_total:
        parameters_total_copy = parameters_total.copy()
        del parameters_total_copy["time"]
        distance, ac = locals()[func_total](query, sequence, tq, ts, **parameters_total_copy)
    else:
        distance, ac = locals()[func_total](query, sequence, **parameters_total)

    return distance, ac


def lockstep_search(distance, query, sequence, weight):
    """
    :param ts1: (Array) Time series one
    :param ts2: (Array) Time series two
    :param dict_distances: Dictionary of distances
    :return: distances between time series
    """

    exec("from tssearch import *")

    # distance function
    func_total = distance["function"]

    # Check for parameters
    parameters_total = {}
    if distance["parameters"] != "":
        parameters_total = distance["parameters"]

    lw = len(query)
    res = np.zeros(len(sequence) - lw, "d")
    for i in range(len(sequence) - lw):
        seq_window = sequence[i : i + lw]

        eval_result = locals()[func_total](seq_window, query, weight, **parameters_total)

        res[i] = eval_result / lw  # default normalization

    return res


def start_sequences_index(res, output=("number", 1), overlap=1.0):

    # pks - min
    pks, _ = find_peaks(-res, distance=overlap)  # TODO if necessary add first and last sequence
    pks_val = res[pks]

    if output[0] == "number":
        num_events = output[1]
        pks_val_sort = np.argsort(pks_val)
        id_s = pks[pks_val_sort[:num_events]]
    elif output[0] == "percentile":
        perct = output[1]
        perct_val = np.percentile(res, 100 - perct)
        pks_perct = np.where(pks_val < perct_val)[0]
        id_s = pks[pks_perct]
    elif output[0] == "threshold":
        thres = output[1]
        pks_thres = np.where(pks_val < thres)[0]
        id_s = pks[pks_thres]
    else:
        id_s = pks[np.argmin(pks_val)]

    return id_s
