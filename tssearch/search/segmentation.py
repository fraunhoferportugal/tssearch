from scipy.signal import find_peaks
from tssearch.search.search_utils import lockstep_search, elastic_search


def time_series_segmentation(dict_distances, query, sequence, tq=None, ts=None, weight=None):
    """
    Time series segmentation locates the time instants between consecutive query repetitions on a more extended and
    repetitive sequence.

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
    segment_results: dict
        Segmented time instants for each given distances
    """

    l_query = len(query)
    segment_results = {}

    for d_type in dict_distances:
        for dist in dict_distances[d_type]:

            if "use" not in dict_distances[d_type][dist] or dict_distances[d_type][dist]["use"] == "yes":
                segment_results[dist] = {}
                if d_type == "lockstep":
                    distance = lockstep_search(dict_distances[d_type][dist], query, sequence, weight)
                elif d_type == "elastic":
                    distance, ac = elastic_search(dict_distances[d_type][dist], query, sequence, tq, ts, weight)
                else:
                    print("WARNING")
                    continue

                pks, _ = find_peaks(-distance, distance=l_query / 2)
                segment_results[dist] = pks

    return segment_results
