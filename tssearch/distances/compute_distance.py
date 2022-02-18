import numpy as np
import pandas as pd
from tssearch.search.segmentation import time_series_segmentation


def time_series_distance(dict_distances, x, y, tx=None, ty=None):
    """

    Parameters
    ----------
    dict_distances: dict
        Dictionary of distances parameters.
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y.
    tx: nd-array
        Time stamp time series x.
    ty: nd-array
        Time stamp time series y.

    Returns
    -------
    distances: pandas DataFrame
        Distances values.

    """

    exec("from tssearch import *")

    distance_results = []
    distance_names = []

    multivariate = True if len(np.shape(x)) > 1 else False

    for d_type in dict_distances:
        for dist in dict_distances[d_type]:

            # Only returns used functions
            if "use" not in dict_distances[d_type][dist] or dict_distances[d_type][dist]["use"] == "yes":
                # remove unidimensional distances
                if multivariate and dict_distances[d_type][dist]["multivariate"] == "no":
                    continue

                func_total = dict_distances[d_type][dist]["function"]

                # Check for parameters
                parameters_total = {}
                if dict_distances[d_type][dist]["parameters"] != "":
                    parameters_total = dict_distances[d_type][dist]["parameters"]

                if "time" in parameters_total:
                    parameters_total_copy = parameters_total.copy()
                    del parameters_total_copy["time"]
                    eval_result = locals()[func_total](x, y, tx, ty, **parameters_total_copy)
                else:
                    eval_result = locals()[func_total](x, y, **parameters_total)

                distance_results += [eval_result]
                distance_names += [dist]

    distances = pd.DataFrame(data=np.array(distance_results), index=np.array(distance_names), columns=["Distance"])

    return distances


def time_series_distance_windows(dict_distances, x, y, tx=None, ty=None, segmentation=None):
    """

    Parameters
    ----------
    dict_distances: dict
        Dictionary of distances parameters.
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y (windows).
    tx: nd-array
        Time stamp time series x.
    ty: nd-array
        Time stamp time series y (windows).
    segmentation: dict
        Dictionary of distances parameters.

    Returns
    -------
    dist_windows: pandas DataFrame
        Distances values per window.

    """

    if segmentation is not None:
        results = time_series_segmentation(segmentation, x, y, tx, ty)
        func_name = list(segmentation[list(dict_distances.keys())[0]].keys())[0]

        ts_w = None if ty is None else []
        windows = []
        for i in range(len(results[func_name]) - 1):
            if ty is not None:
                ts_w += [ty[results[func_name][i] : results[func_name][i + 1]]]
            windows += [y[results[func_name][i] : results[func_name][i + 1]]]
    else:
        windows = y
        ts_w = ty

    multivariate = True if len(np.shape(x)) > 1 else False

    exec("from tssearch import *")

    dist_windows = pd.DataFrame()
    for d_type in dict_distances:
        for dist in dict_distances[d_type]:

            # Only returns used functions
            if "use" not in dict_distances[d_type][dist] or dict_distances[d_type][dist]["use"] == "yes":

                if multivariate and dict_distances[d_type][dist]["multivariate"] == "no":
                    continue

                func_total = dict_distances[d_type][dist]["function"]

                # Check for parameters
                parameters_total = {}
                if dict_distances[d_type][dist]["parameters"] != "":
                    parameters_total = dict_distances[d_type][dist]["parameters"]

                distance_results = []
                if "time" in parameters_total:
                    parameters_total_copy = parameters_total.copy()
                    del parameters_total_copy["time"]
                    for ty_window, window in zip(ts_w, windows):
                        eval_result = locals()[func_total](x, window, tx, ty_window, **parameters_total_copy)
                        distance_results += [eval_result]
                else:
                    for window in windows:
                        eval_result = locals()[func_total](x, window, **parameters_total)
                        distance_results += [eval_result]

                dist_windows[dist] = distance_results

    return dist_windows
