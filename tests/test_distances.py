import numpy as np

from tssearch import *


def test_distances(dist):
    np.testing.assert_almost_equal(dist["Time Warp Edit Distance"], 223.85651832411503)
    np.testing.assert_almost_equal(dist["Dynamic Time Warping"], 0.2509773694532439)
    np.testing.assert_almost_equal(dist["Longest Common Subsequence"], 0.7774244833068362)
    np.testing.assert_almost_equal(dist["Time Alignment Measurement"], 1.492823)
    np.testing.assert_almost_equal(dist["Euclidean Distance"], 25.066280)
    # np.testing.assert_almost_equal(dist['Minkowski Distance'], )
    np.testing.assert_almost_equal(dist["Chebyshev Distance"], 1.760120)
    np.testing.assert_almost_equal(dist["Cross Correlation Distance"], 1.0000008394102025)
    np.testing.assert_almost_equal(dist["Pearson Correlation Distance"], 2.000001678820405)
    np.testing.assert_almost_equal(dist["Short Time Series Distance"], 3.9573142706233573)


if __name__ == "__main__":

    # Example of a sequence to search in
    t = np.arange(0, 20 * np.pi, 0.1)
    ts1 = np.sin(t)
    ts2 = np.sin(2 * t)

    dict_distances = get_distances_by_type()
    dist = time_series_distance(dict_distances, ts1, ts2, t, t)

    test_distances(dist.to_dict()["Distance"])
