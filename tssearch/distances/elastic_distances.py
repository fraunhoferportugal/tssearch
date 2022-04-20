import numpy as np

from tssearch.distances.elastic_utils import (
    cost_matrix,
    accumulated_cost_matrix,
    acc_initialization,
    lcss_accumulated_matrix,
    lcss_path,
    lcss_score,
    traceback_adj,
    backtracking,
)


def dtw(x, y, weight=None, **kwargs):
    """Computes Dynamic Time Warping (DTW) of two time series.

    Parameters
    ----------
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y.
    dist: function
        The distance used as a local cost measure. None defaults to the squared euclidean distance.
    \**kwargs:
    See below:

    * *do_sign_norm* (``bool``) --
      If ``True`` the signals will be normalized before computing the DTW,
      (default: ``False``)

    * *do_dist_norm* (``bool``) --
      If ``True`` the DTW distance will be normalized by dividing the summation of the path dimension.
      (default: ``True``)

    * *window* (``String``) --
      Selects the global window constrains. Available options are ``None`` and ``sakoe-chiba``.
      (default: ``None``)

    * *factor* (``Float``) --
      Selects the global constrain factor.
      (default: ``min(xl, yl) * .50``)


    Returns
    -------
    d: float
        The DTW distance.
    ac: nd-array
        The accumulated cost matrix.
    path: nd-array
        The optimal warping path between the two sequences.
    """

    xl, yl = len(x), len(y)

    alpha = kwargs.get("alpha", 1)
    do_dist_norm = kwargs.get("dist_norm", True)
    window = kwargs.get("window", None)
    factor = kwargs.get("factor", np.min((xl, yl)) * 0.50)
    dtw_type = kwargs.get("dtw_type", "dtw")
    tolerance = kwargs.get("tolerance", 0)
    report = kwargs.get("report", "distance")

    # cost matrix
    c = cost_matrix(x, y, alpha, weight=weight)
    # Acc cost matrix
    ac = accumulated_cost_matrix(c, window=window, factor=factor, dtw_type=dtw_type, tolerance=tolerance)

    # Distance
    if report == "cost_matrix":
        return ac
    elif report == "search":
        d = ac[-1, :]
        return d, ac
    elif report == "path":
        path = traceback_adj(ac)
        return path
    else:  # report = "distance" default
        d = ac[-1, -1] / xl if do_dist_norm else ac[-1, -1]
        return d


def lcss(x, y, eps=1, **kwargs):
    """Computes the Longest Common Subsequence (LCSS) distance between two numeric time series.

    Parameters
    ----------
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y.
    eps : float
        Amplitude matching threshold.
    \**kwargs:
    See below:

    * *window* (``String``) --
      Selects the global window constrains. Available options are ``None`` and ``sakoe-chiba``.
      (default: ``None``)

    Returns
    -------
    d: float
        The LCSS distance.
    ac: nd-array
        The similarity matrix.
    path: nd-array
        The optimal path between the two sequences.
    """

    window = kwargs.get("window", None)
    report = kwargs.get("report", "distance")

    dim = len(np.shape(x))  # tem de dar erro se forem inseridas duas TS com dims diferentes
    if dim == 1:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    ac = lcss_accumulated_matrix(x, y, eps=eps)
    path = lcss_path(x, y, ac, eps=eps)
    sim_score = lcss_score(ac)

    if report == "cost_matrix":
        return ac
    elif report == "search":
        return sim_score, ac
    elif report == "path":
        return path
    else:
        return sim_score


def dlp(x, y, p=2):
    """Computes Lp norm distance between two time series.

    Parameters
    ----------
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y.
    p: int
        Lp norm distance degree for local cost computation.

    Returns
    -------
        The Lp distance.
    """

    cost = np.sum(np.power(np.abs(x - y), p))
    return np.power(cost, 1 / p)


def twed(x, y, tx, ty, nu=0.001, lmbda=1.0, p=2, report="distance"):
    """Computes Time Warp Edit Distance (TWED) of two time series.

    Reference :
       Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching".
       IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (2): 306–318. arXiv:cs/0703033
       http://people.irisa.fr/Pierre-Francois.Marteau/

    Parameters
    ----------
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y.
    tx: nd-array
        Time stamp time series x.
    ty: nd-array
        Time stamp time series y.
    nu: int
        Stiffness parameter (nu >= 0)
            nu = 0, TWED distance measure on amplitude.
            nu > 0, TWED distance measure on amplitude x time.
    lmbda: int
        Penalty for deletion operation (lmbda >= 0).
    p: int
        Lp norm distance degree for local cost computation.
    report: str
        distance, cost matrix, path.

    Returns
    -------
    d: float
        The TWED distance.
    ac: nd-array
        The accumulated cost matrix.
    path: nd-array
        The optimal warping path between the two sequences.
    """

    # Check if input arguments
    if len(x) != len(tx):
        print("The length of x is not equal length of tx")
        return None, None

    if len(y) != len(ty):
        print("The length of y is not equal length of ty")
        return None, None

    if nu < 0:
        print("nu is negative")
        return None, None

    # Dynamical programming
    ac = acc_initialization(len(x), len(y), report)

    # Add padding
    query = np.array([0] + list(x))
    tq = np.array([0] + list(tx))
    sequence = np.array([0] + list(y))
    ts = np.array([0] + list(ty))

    n = len(query)
    m = len(sequence)

    # Compute minimal cost
    for i in range(1, n):
        for j in range(1, m):
            # Calculate and save cost of various operations
            C = np.ones((3, 1)) * np.inf
            # Deletion in A
            C[0] = ac[i - 1, j] + dlp(query[i - 1], query[i], p) + nu * (tq[i] - tq[i - 1]) + lmbda
            # Deletion in B
            C[1] = ac[i, j - 1] + dlp(sequence[j - 1], sequence[j], p) + nu * (ts[j] - ts[j - 1]) + lmbda
            # Keep data points in both time series
            C[2] = (
                ac[i - 1, j - 1]
                + dlp(query[i], sequence[j], p)
                + dlp(query[i - 1], sequence[j - 1], p)
                + nu * (abs(tq[i] - ts[j]) + abs(tq[i - 1] - ts[j - 1]))
            )
            # Choose the operation with the minimal cost and update c Matrix
            ac[i, j] = np.min(C)

    if report == "cost_matrix":
        return ac
    elif report == "search":
        d = ac[n - 1, :]
        return d, ac
    elif report == "path":
        path = backtracking(ac)
        return path
    else:  # report = 'search'
        return ac[n - 1, m - 1]
