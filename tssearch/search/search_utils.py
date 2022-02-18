import numpy as np
from scipy.signal import find_peaks


def elastic_search(dict_distances, query, sequence, tq=None, ts=None, weight=None):
    """
    Query search for elastic measures

    Parameters
    ----------
    dict_distances: dict
        Configuration file with distances
    query: nd-array
        Query time series.
    sequence: nd-array
        Sequence time series.
    tq: nd-array
        Time stamp time series query.
    ts: nd-array
        Time stamp time series sequence.
    weight: nd-array (Default: None)
        query weight values

    Returns
    -------
    distance: nd-array
        distance value between query and sequence
    ac: nd-array
        accumulated cost matrix
    """

    exec("from tssearch import *")

    # distance function
    func_total = dict_distances["function"]

    # Check for parameters
    parameters_total = {}
    if dict_distances["parameters"] != "":
        parameters_total = dict_distances["parameters"]
    parameters_total["report"] = "search"

    if "dtw_type" in parameters_total:
        if parameters_total["dtw_type"] == "dtw":
            parameters_total["dtw_type"] = "sub-dtw"

    if "time" in parameters_total:
        parameters_total_copy = parameters_total.copy()
        del parameters_total_copy["time"]
        distances, ac = locals()[func_total](query, sequence, tq, ts, **parameters_total_copy)
    else:
        distances, ac = locals()[func_total](query, sequence, **parameters_total)

    return distances, ac


def lockstep_search(dict_distances, query, sequence, weight):
    """
    Query search for lockstep measures

    Parameters
    ----------
    dict_distances: dict
        Configuration file with distances
    query: nd-array
        Query time series.
    sequence: nd-array
        Sequence time series.
    weight: nd-array (Default: None)
        query weight values

    Returns
    -------
    res: nd-array
        distance value between query and sequence
    """

    exec("from tssearch import *")

    # distance function
    func_total = dict_distances["function"]

    # Check for parameters
    parameters_total = {}
    if dict_distances["parameters"] != "":
        parameters_total = dict_distances["parameters"]

    lw = len(query)
    res = np.zeros(len(sequence) - lw, "d")
    for i in range(len(sequence) - lw):
        seq_window = sequence[i : i + lw]

        eval_result = locals()[func_total](seq_window, query, weight, **parameters_total)

        res[i] = eval_result / lw  # default normalization

    return res


def start_sequences_index(distance, output=("number", 1), overlap=1.0):
    """
    Method to retrieve the k-best occurrences from a given vector distance

    Parameters
    ----------
    distance: nd-array
        distance values
    output: tuple
        number of occurrences
    overlap: float
        minimum distance between occurrences

    Returns
    -------
    id_s: nd-array
        indexes of k-best occurrences
    """

    # pks - min
    pks, _ = find_peaks(-distance, distance=overlap)  # TODO if necessary add first and last sequence
    pks_val = distance[pks]

    if output[0] == "number":
        num_events = output[1]
        pks_val_sort = np.argsort(pks_val)
        id_s = pks[pks_val_sort[:num_events]]
    elif output[0] == "percentile":
        perct = output[1]
        perct_val = np.percentile(distance, 100 - perct)
        pks_perct = np.where(pks_val < perct_val)[0]
        id_s = pks[pks_perct]
    elif output[0] == "threshold":
        thres = output[1]
        pks_thres = np.where(pks_val < thres)[0]
        id_s = pks[pks_thres]
    else:
        id_s = pks[np.argmin(pks_val)]

    return id_s
