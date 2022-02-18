from tssearch import *

import matplotlib.pyplot as plt


if __name__ == "__main__":
    # time
    t = np.arange(0, 20 * np.pi, 0.1)

    # 1. 1D, sample, euclidean distance
    sequence = np.sin(t)
    query = np.sin(t[:70])

    dict_distances = {
        "lockstep": {"Euclidean Distance": {"function": "euclidean_distance", "parameters": "", "use": "yes"}}
    }

    result1 = time_series_search(dict_distances, query, sequence, output=("number", 1))

    plt.figure(1)
    start = result1["Euclidean Distance"]["start"][0]
    end = result1["Euclidean Distance"]["end"][0]
    path = [np.arange(len(query)), np.arange(start, end)]
    plot_alignment(query, sequence, path, hoffset=start)

    # 2. 3-axis, reference, sdtw, equal weight 3 axis fw = [1,1,1]
    sequence = np.array([np.sin(t), np.sin(2 * t), np.cos(t)]).T
    query = sequence[70:140]

    dict_distances = {
        "elastic": {"Dynamic Time Warping": {"function": "dtw", "parameters": {"dtw_type": "sub-dtw"}, "use": "yes"}}
    }

    result2 = time_series_search(dict_distances, query, sequence, output=("number", 1))

    path = result2["Dynamic Time Warping"]["path"][0]
    plt.figure(2)
    plot_alignment(query[:, 1], sequence[:, 1], path, hoffset=path[1][0])

    # 3. 3-axis, reference, sdtw, different axes weights
    # derivate and abs with different weight fw = [.7,.7,.7,.3,.3,.3]
    sequence = np.array([np.sin(t), np.sin(2 * t), np.cos(t)]).T
    query = np.array([np.sin(t[:70]), np.sin(2 * t[10:80]), np.cos(t[30:100])]).T
    weight = np.ones_like(query)
    weight[:, 2] = 0.5
    weight[:, 1] = 0.8

    dict_distances = {
        "elastic": {"Dynamic Time Warping": {"function": "dtw", "parameters": {"dtw_type": "sub-dtw"}, "use": "yes"}}
    }

    result3 = time_series_search(dict_distances, query, sequence, weight=weight, output=("number", 1))

    path = result3["Dynamic Time Warping"]["path"][0]
    plt.figure(3)
    plot_alignment(query[:, 0], sequence[:, 0], path, hoffset=path[1][0])

    # 4. with 4 points to be forced in time and amplitude qw = [10000010000...0011]

    # 5. Emulate gaussian process

    plt.show()
