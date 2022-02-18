import numpy as np
from tssearch.distances.elastic_utils import traceback_adj, lcss_path, lcss_score
from tssearch.search.search_utils import lockstep_search, elastic_search, start_sequences_index


def time_series_search(dict_distances, query, sequence, tq=None, ts=None, weight=None, output=("number", 1)):
    """
    Time series search method locates the k-best occurrences of a given query on a more extended sequence based on a
    distance measurement.

    Parameters
    ----------
    dict_distances: dict
        Configuration file with distances.
    query: nd-array
        Query time series.
    sequence: nd-array
        Sequence time series.
    tq: nd-array
        Time stamp time series query.
    ts: nd-array
        Time stamp time series sequence.
    weight: nd-array (Default: None)
        query weight values.
    output: tuple
        number of occurrences.

    Returns
    -------
    distance_results: dict
        time instants, optimal alignment path and distance for each occurrence per distance.
    """

    l_query = len(query)
    distance_results = {}

    for d_type in dict_distances:
        for dist in dict_distances[d_type]:

            if "use" not in dict_distances[d_type][dist] or dict_distances[d_type][dist]["use"] == "yes":
                distance_results[dist] = {}
                if d_type == "lockstep":
                    distance = lockstep_search(dict_distances[d_type][dist], query, sequence, weight)

                    start_index = start_sequences_index(distance, output=output, overlap=l_query)
                    end_index, path = [], []
                    for start in start_index:
                        end_index += [start + l_query]
                        path += [(np.arange(l_query), np.arange(start, end_index[-1]))]
                    distance_results[dist]["path_dist"] = distance[start_index]
                elif d_type == "elastic":
                    distance, ac = elastic_search(dict_distances[d_type][dist], query, sequence, tq, ts, weight)

                    if dist == "Longest Common Subsequence":
                        eps = dict_distances[d_type][dist]["parameters"]["eps"]
                        if len(np.shape(query)) == 1:
                            query_copy = query.reshape(-1, 1)
                            sequence_copy = sequence.reshape(-1, 1)
                            path = [lcss_path(query_copy, sequence_copy, ac, eps)]
                        else:
                            path = [lcss_path(query, sequence, ac, eps)]
                        distance_results[dist]["path_dist"] = [lcss_score(ac)]
                        end_index = [path_i[1][-1] for path_i in path]
                    else:
                        end_index = start_sequences_index(distance, output=output, overlap=l_query / 2)
                        # check if traceback_adj is equal to other elastic measures
                        path = [traceback_adj(ac[:, : int(pk) + 1]) for pk in end_index]
                        distance_results[dist]["path_dist"] = distance[end_index]
                    start_index = [path_i[1][0] for path_i in path]

                else:
                    print("WARNING")
                    continue

                distance_results[dist]["distance"] = distance
                distance_results[dist]["start"] = start_index
                distance_results[dist]["end"] = end_index
                distance_results[dist]["path"] = path

    return distance_results
