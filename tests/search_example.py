from tssearch import *


if __name__ == "__main__":

    # Example of a sequence to search in
    t = np.arange(0, 20 * np.pi, 0.1)
    sequence = np.sin(t)

    # Example of a sequence to search for
    tq = t[:70]
    query = np.sin(tq)

    dict_distances = get_distances_by_type()

    result = time_series_search(dict_distances, query, sequence, tq, t, output=("number", 1))

    plt.figure()
    plt.title("Dynamic Time Warping")
    plot_alignment(query, sequence, result["Dynamic Time Warping"]["path"][0])

    plt.figure()
    plt.title("Longest Common Subsequence")
    plot_alignment(query, sequence, result["Longest Common Subsequence"]["path"][0])

    plt.figure()
    plt.title("Time Warp Edit Distance")
    plot_alignment(query, sequence, result["Time Warp Edit Distance"]["path"][0])

    plt.figure()
    plt.title("Euclidean Distance")
    start = result["Euclidean Distance"]["start"][0]
    end = result["Euclidean Distance"]["end"][0]
    path = [np.arange(len(query)), np.arange(start, end)]
    plot_alignment(query, sequence, path, hoffset=start)

    plt.show()
