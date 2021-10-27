import json
import tssearch


def load_json(json_path):
    """Loads the json file given by filename.
    Parameters
    ----------
    json_path : string
        Json path
    Returns
    -------
    Dict
        Dictionary
    """

    return json.load(open(json_path))


def get_distances_by_type(domain=None, json_path=None):
    """Creates a dictionary with the features settings by domain.
    Parameters
    ----------
    domain : string
        Available domains: "statistical"; "spectral"; "temporal"
        If domain equals None, then the features settings from all domains are returned.
    json_path : string
        Directory of json file. Default: package features.json directory
    Returns
    -------
    Dict
        Dictionary with the features settings
    """

    if json_path is None:
        json_path = tssearch.__path__[0] + "/distances/distances.json"

        if domain not in ["elastic", "lockstep", "time", None]:
            raise SystemExit("No valid domain. Choose: lockstep, elastic, time or None (for all distances settings).")

    dict_features = load_json(json_path)
    if domain is None:
        return dict_features
    else:
        return {domain: dict_features[domain]}


def get_distance_dict(dist_list):

    json_path = tssearch.__path__[0] + "/distances/distances.json"

    dict_features = load_json(json_path)

    select_distances = {}
    for d in dist_list:
        if d in dict_features["elastic"]:
            d_type = "elastic"
        elif d in dict_features["lockstep"]:
            d_type = "lockstep"
        elif d in dict_features["time"]:
            d_type = "time"
        else:
            continue

        if d_type not in select_distances:
            select_distances[d_type] = {}
        select_distances[d_type][d] = dict_features[d_type][d]

    return select_distances
