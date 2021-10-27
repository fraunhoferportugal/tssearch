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
    x: nd-array
        The reference signal.
    y: nd-array
        The estimated signal.
    dist: function
        The distance used as a local cost measure. None defaults to the squared euclidean distance

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
           C: nd-array
            The local cost matrix.
           ac: nd-array
            The accumulated cost matrix.
           path nd-array
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
    x: nd-array
        The reference signal.
    y: nd-array
        The estimated signal.
    eps : float
            Amplitude matching threshold.

    \**kwargs:
    See below:

    * *window* (``String``) --
      Selects the global window constrains. Available options are ``None`` and ``sakoe-chiba``.
      (default: ``None``)

    Returns
    -------
    Returns
    -------
           d: float
            The LCSS distance.
           C: nd-array
            The similarity matrix.
           path nd-array
            The optimal path between the two sequences.
    """

    window = kwargs.get("window", None)
    report = kwargs.get("report", "distance")

    dim = len(np.shape(x))  # tem de dar erro se forem inseridas duas TS com dims diferentes
    if dim == 1:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    sim_mat = lcss_accumulated_matrix(x, y, eps=eps)
    path = lcss_path(x, y, sim_mat, eps=eps)
    sim_score = lcss_score(sim_mat=sim_mat)

    if report == "cost_matrix":
        return sim_mat
    elif report == "search":
        return sim_score, sim_mat
    elif report == "path":
        return path
    else:
        return sim_score


def dlp(A, B, p=2):
    cost = np.sum(np.power(np.abs(A - B), p))
    return np.power(cost, 1 / p)


def twed(query, sequence, tq, ts, nu=0.001, lmbda=1.0, degree=2, report="distance"):
    """Computes Time Warp Edit Distance (TWED) of two time series.

    Reference :
       Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching".
       IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (2): 306–318. arXiv:cs/0703033
       http://people.irisa.fr/Pierre-Francois.Marteau/

    query: nd-array
        The reference signal.
    sequence: nd-array
        The estimated signal.
    tq: nd-array
        Time stamp reference signal.
    ts: nd-array
        Time stamp estimated signal.
    nu: int
        Stiffness parameter (nu >= 0)
            nu = 0, TWED distance measure on amplitude
            nu > 0, TWED distance measure on amplitude x time
    lmbda: int
        Penalty for deletion operation (lmbda >= 0).
    degree: int
        Lp norm distance degree for local cost computation.
    report: str
        distance, cost matrix, path

    Returns
    -------
           d: float
            The TWED distance.
           c: nd-array
            The local cost matrix.
           path: nd-array
            The optimal warping path between the two sequences.
    """

    # Check if input arguments
    if len(query) != len(tq):
        print("The length of A is not equal length of timeSA")
        return None, None

    if len(sequence) != len(ts):
        print("The length of B is not equal length of timeSB")
        return None, None

    if nu < 0:
        print("nu is negative")
        return None, None

    # Dynamical programming
    DP = acc_initialization(len(query), len(sequence), report)

    # Add padding
    query = np.array([0] + list(query))
    tq = np.array([0] + list(tq))
    sequence = np.array([0] + list(sequence))
    ts = np.array([0] + list(ts))

    n = len(query)
    m = len(sequence)

    # Compute minimal cost
    for i in range(1, n):
        for j in range(1, m):
            # Calculate and save cost of various operations
            C = np.ones((3, 1)) * np.inf
            # Deletion in A
            C[0] = DP[i - 1, j] + dlp(query[i - 1], query[i], degree) + nu * (tq[i] - tq[i - 1]) + lmbda
            # Deletion in B
            C[1] = DP[i, j - 1] + dlp(sequence[j - 1], sequence[j], degree) + nu * (ts[j] - ts[j - 1]) + lmbda
            # Keep data points in both time series
            C[2] = (
                DP[i - 1, j - 1]
                + dlp(query[i], sequence[j], degree)
                + dlp(query[i - 1], sequence[j - 1], degree)
                + nu * (abs(tq[i] - ts[j]) + abs(tq[i - 1] - ts[j - 1]))
            )
            # Choose the operation with the minimal cost and update DP Matrix
            DP[i, j] = np.min(C)

    if report == "cost_matrix":
        return DP
    elif report == 'search':
        return DP[n - 1, :], DP
    elif report == 'path':
        path = backtracking(DP)
        return path
    else:  # report = 'search'
        return DP[n - 1, m - 1]
