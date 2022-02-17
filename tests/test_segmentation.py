import numpy as np

from tssearch import load_ecg_example, get_distance_dict, time_series_segmentation, get_distances_by_type


def segmentation():
    data = load_ecg_example()
    cfg = get_distance_dict(["Dynamic Time Warping", "Euclidean Distance"])
    out = time_series_segmentation(
        cfg, data["query"], data["sequence"], tq=data["tq"], ts=data["ts"], weight=data["weight"]
    )

    np.testing.assert_almost_equal(
        out["Dynamic Time Warping"], [7, 51, 120, 161, 210, 263, 318, 394, 444, 510, 584, 666, 740, 804, 878]
    )

    np.testing.assert_almost_equal(
        out["Euclidean Distance"], [10, 58, 127, 193, 257, 320, 384, 452, 527, 596, 667, 736, 807]
    )

    return out


if __name__ == "__main__":
    out = segmentation()
