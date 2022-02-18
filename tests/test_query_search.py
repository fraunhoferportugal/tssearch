import numpy as np

from tssearch import load_ecg_example, get_distance_dict, time_series_search


def query_search():
    data = load_ecg_example()
    cfg = get_distance_dict(["Dynamic Time Warping", "Longest Common Subsequence", "Euclidean Distance"])
    out = time_series_search(cfg, data["query"], data["sequence"], tq=data["tq"], ts=data["ts"], weight=data["weight"])

    np.testing.assert_almost_equal(out["Dynamic Time Warping"]["path_dist"][0], 0.09974783)
    np.testing.assert_almost_equal(out["Dynamic Time Warping"]["start"][0], 445)
    np.testing.assert_almost_equal(out["Dynamic Time Warping"]["end"][0], 510)

    np.testing.assert_almost_equal(out["Longest Common Subsequence"]["path_dist"][0], 1.0)
    np.testing.assert_almost_equal(out["Longest Common Subsequence"]["start"][0], 844)
    np.testing.assert_almost_equal(out["Longest Common Subsequence"]["end"][0], 921)

    np.testing.assert_almost_equal(out["Euclidean Distance"]["path_dist"][0], 0.05480903)
    np.testing.assert_almost_equal(out["Euclidean Distance"]["start"][0], 596)
    np.testing.assert_almost_equal(out["Euclidean Distance"]["end"][0], 674)

    return out


if __name__ == "__main__":
    out = query_search()
