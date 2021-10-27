from tssearch import *


if __name__ == "__main__":

    # Example of a sequence to search in
    t = np.arange(0, 20 * np.pi, 0.1)
    ts1 = np.sin(t)
    ts2 = np.sin(2 * t)

    dict_distances = load_json("distances_dist.json")

    distances = time_series_distance(dict_distances, ts1, ts2, t, t)
