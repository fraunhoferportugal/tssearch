from scipy.signal import find_peaks
from tssearch.search.search_utils import lockstep_search, elastic_search


def time_series_segmentation(dict_distances, x, y, tx=None, ty=None, weight=None):
    """
    Time series segmentation locates the time instants between consecutive query repetitions on a more extended and
    repetitive sequence.

    Parameters
    ----------
    dict_distances: dict
        Configuration file with distances
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y.
    tx: nd-array
        Time stamp time series x.
    ty: nd-array
        Time stamp time series y.
    weight: nd-array (Default: None)
        query weight values
    Returns
    -------
    segment_results: dict
        Segmented time instants for each given distances
    """

    l_query = len(x)
    segment_results = {}

    for d_type in dict_distances:
        for dist in dict_distances[d_type]:

            if "use" not in dict_distances[d_type][dist] or dict_distances[d_type][dist]["use"] == "yes":
                segment_results[dist] = {}
                if d_type == "lockstep":
                    distance = lockstep_search(dict_distances[d_type][dist], x, y, weight)
                elif d_type == "elastic":
                    distance, ac = elastic_search(dict_distances[d_type][dist], x, y, tx, ty, weight)
                else:
                    print("WARNING")
                    continue

                pks, _ = find_peaks(-distance, distance=l_query / 2)
                segment_results[dist] = pks

    return segment_results
