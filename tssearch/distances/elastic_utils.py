import numpy as np
from numba import njit, prange
from tssearch.utils.preprocessing import standardization


@njit(parallel=True, fastmath=True)
def _cost_matrix(x, y):
    """

    Parameters
    ----------
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y.

    Returns
    -------
    c: nd-array
        The cost matrix.
    """
    l1 = x.shape[0]
    l2 = y.shape[0]
    c = np.zeros((l1, l2), dtype=np.float32)

    for i in prange(l1):
        for j in prange(l2):
            c[i, j] = (x[i] - y[j]) ** 2

    return c


@njit(parallel=True, fastmath=True)
def _multidimensional_cost_matrix(x, y, weight):
    """Helper function for fast computation of cost matrix in cost_matrix_diff_vec.
    Defined outside to prevent recompilation from numba

    Parameters
    ----------
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y.

    Returns
    -------
    c: nd-array
        The cost matrix.
    """
    l1 = x.shape[0]
    l2 = y.shape[0]
    l3 = x.shape[1]
    c = np.zeros((l1, l2), dtype=np.float32)

    for i in prange(l1):
        for j in prange(l2):
            dist = 0.0
            for di in range(l3):
                diff = x[i, di] - y[j, di]
                dist += weight[i, di] * (diff * diff)
            c[i, j] = dist ** 0.5

    return c


@njit(nogil=True, fastmath=True)
def _accumulated_cost_matrix(ac):
    """Fast computation of accumulated cost matrix using cost matrix.

    Parameters
    ----------
    ac: nd-array
        Given cost matrix c, ac = acc_initialization(...), ac[1:, 1:] = c.

    Returns
    -------
        The accumulated cost matrix.
    """
    for i in range(ac.shape[0] - 1):
        for j in range(ac.shape[1] - 1):
            ac[i + 1, j + 1] += min(ac[i, j + 1], ac[i + 1, j], ac[i, j])
    return ac


def acc_initialization(x, y, _type, tolerance=0):
    """Initializes the cost matrix according to the dtw type.

    Parameters
    ----------
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y.
    _type: string
        Name of dtw type.
    tolerance: int
        Tolerance value.

    Returns
    -------
    ac: nd-array
        The accumulated cost matrix.
    """
    ac = np.zeros((x + 1, y + 1))
    if _type == "dtw":
        ac[0, 1:] = np.inf
        ac[1:, 0] = np.inf
    elif _type == "oe-dtw":
        ac[0, 1:] = np.inf
        ac[1:, 0] = np.inf
    elif _type == "obe-dtw" or _type == "sub-dtw" or _type == "search":
        ac[1:, 0] = np.inf
    elif _type == "psi-dtw":
        ac[0, tolerance + 1 :] = np.inf
        ac[tolerance + 1 :, 0] = np.inf
    else:
        ac[0, 1:] = np.inf
        ac[1:, 0] = np.inf

    return ac


def cost_matrix(x, y, alpha=1, weight=None):
    """Computes cost matrix using a specified distance (dist) between two time series.

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
    c: nd-array
        The cost matrix.
    """
    if weight is None:
        weight = np.ones_like(x)

    if len(np.shape(weight)) == 1:
        weight = weight.reshape(-1, 1)

    if alpha == 1:
        C_d = 1
        if len(np.shape(x)) == 1:
            C_n = (_cost_matrix(x, y) * weight) / np.max(weight)
        else:
            C_n = _multidimensional_cost_matrix(x, y, weight)
    else:
        # standardization parameters
        abs_norm = np.mean(x, axis=0), np.std(x, axis=0)
        diff_norm = np.mean(np.diff(x, axis=0), axis=0), np.std(np.diff(x, axis=0), axis=0)

        # Derivative calculation and standardization
        _x = standardization(np.diff(x, axis=0), param=diff_norm)
        _y = standardization(np.diff(y, axis=0), param=diff_norm)
        # same length of derivative
        x = standardization(x[:-1], param=abs_norm)
        y = standardization(y[:-1], param=abs_norm)

        weight = weight[:-1]

        if len(np.shape(x)) == 1:
            C_d = _cost_matrix(_x, _y) * weight
            C_n = _cost_matrix(x, y) * weight
        else:
            C_d = _multidimensional_cost_matrix(_x, _y, weight)
            C_n = _multidimensional_cost_matrix(x, y, weight)

    c = alpha * C_n + (1 - alpha) * C_d

    return c


def accumulated_cost_matrix(c, **kwargs):
    """

    Parameters
    ----------
    c: nd-array
        The cost matrix.

    \**kwargs:

    Returns
    -------
    ac: nd-array
        The accumulated cost matrix.
    """
    xl, yl = np.shape(c)

    window = kwargs.get("window", None)
    factor = kwargs.get("factor", np.min((xl, yl)) * 0.50)
    dtw_type = kwargs.get("dtw_type", "dtw")
    tolerance = kwargs.get("tolerance", 0)

    if window == "sakoe-chiba":
        c[np.abs(np.diff(np.indices(c.shape), axis=0))[0] > factor] = np.inf

    ac = acc_initialization(xl, yl, dtw_type, tolerance)
    ac[1:, 1:] = c.copy()
    ac = _accumulated_cost_matrix(ac)[1:, 1:]

    return ac


@njit(nogil=True, fastmath=True)
def traceback(ac):
    """Computes the traceback path of the matrix c.

    Parameters
    ----------
    ac: nd-array
        The accumulated cost matrix.

    Returns
    -------
        Coordinates p and q of the minimum path.

    """

    i, j = np.array(ac.shape) - 2
    p, q = [i], [j]
    while (i > 0) and (j > 0):
        tb = 0
        if ac[i, j + 1] < ac[i, j]:
            tb = 1
        if ac[i + 1, j] < ac[i, j + tb]:
            tb = 2
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    while j > 0:
        j -= 1
        p.insert(0, i)
        q.insert(0, j)
    while i > 0:
        i -= 1
        p.insert(0, i)
        q.insert(0, j)

    return np.array(p), np.array(q)


@njit(nogil=True, fastmath=True)
def traceback_adj(ac):
    """Computes the adjusted traceback path of the matrix c.

    Parameters
    ----------
    ac: nd-array
        The accumulated cost matrix.

    Returns
    -------
        Coordinates p and q of the minimum path adjusted.

    """
    i, j = np.array(ac.shape) - 2
    p, q = [i], [j]
    while (i > 0) and (j > 0):
        tb = 0
        if ac[i, j + 1] < ac[i, j]:
            tb = 1
        if ac[i + 1, j] < ac[i, j + tb]:
            tb = 2
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    while i > 0:
        i -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def backtracking(ac):
    """Compute the most cost-efficient path.

    Parameters
    ----------
    ac: nd-array
        The accumulated cost matrix.

    Returns
    -------
         Coordinates of the most cost-efficient path.
    """
    x = np.shape(ac)
    i = x[0] - 1
    j = x[1] - 1

    # The indices of the paths are save in opposite direction
    # path = np.ones((i + j, 2 )) * np.inf;
    best_path = []

    steps = 0
    while i != 0 or j != 0:

        best_path.append((i - 1, j - 1))

        C = np.ones((3, 1)) * np.inf

        # Keep data points in both time series
        C[0] = ac[i - 1, j - 1]
        # Deletion in A
        C[1] = ac[i - 1, j]
        # Deletion in B
        C[2] = ac[i, j - 1]

        # Find the index for the lowest cost
        idx = np.argmin(C)

        if idx == 0:
            # Keep data points in both time series
            i = i - 1
            j = j - 1
        elif idx == 1:
            # Deletion in A
            i = i - 1
            j = j
        else:
            # Deletion in B
            i = i
            j = j - 1
        steps = steps + 1

    best_path.append((i - 1, j - 1))

    best_path.reverse()
    best_path = np.array(best_path[1:])

    return best_path[:, 0], best_path[:, 1]


# DTW SW
def dtw_sw(x, y, winlen, alpha=0.5, **kwargs):
    """Computes Dynamic Time Warping (DTW) of two time series using a sliding window.
    TODO: Check if this needs to be speed up.

    Parameters
    ----------
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y.
    winlen: int
        The sliding window length.
    alpha: float
        A factor between 0 and 1 which weights the amplitude and derivative contributions.
        A higher value will favor amplitude and a lower value will favor the first derivative.

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
        The SW-DTW distance.
    c: nd-array
        The local cost matrix.
    ac: nd-array
        The accumulated cost matrix.
    path: nd-array
        The optimal warping path between the two sequences.

    """
    xl, yl = len(x), len(y)

    do_sign_norm = kwargs.get("normalize", False)
    do_dist_norm = kwargs.get("dist_norm", True)
    window = kwargs.get("window", None)
    factor = kwargs.get("factor", np.min((xl, yl)) * 0.50)

    if do_sign_norm:
        x, y = standardization(x), standardization(y)

    ac = np.zeros((xl + 1, yl + 1))
    ac[0, 1:] = np.inf
    ac[1:, 0] = np.inf
    tmp_ac = ac[1:, 1:]

    nx = get_mirror(x, winlen)
    ny = get_mirror(y, winlen)

    dnx = np.diff(nx)
    dny = np.diff(ny)

    nx = nx[:-1]
    ny = ny[:-1]

    # Workaround to deal with even window sizes
    if winlen % 2 == 0:
        winlen -= 1

    swindow = np.hamming(winlen)
    swindow = swindow / np.sum(swindow)

    for i in range(xl):
        for j in range(yl):
            pad_i, pad_j = i + winlen, j + winlen
            # No window selected
            if window is None:
                tmp_ac[i, j] = sliding_dist(
                    nx[pad_i - (winlen // 2) : pad_i + (winlen // 2) + 1],
                    ny[pad_j - (winlen // 2) : pad_j + (winlen // 2) + 1],
                    dnx[pad_i - (winlen // 2) : pad_i + (winlen // 2) + 1],
                    dny[pad_j - (winlen // 2) : pad_j + (winlen // 2) + 1],
                    alpha,
                    swindow,
                )

            # Sakoe-Chiba band
            elif window == "sakoe-chiba":
                if abs(i - j) < factor:
                    tmp_ac[i, j] = sliding_dist(
                        nx[pad_i - (winlen // 2) : pad_i + (winlen // 2) + 1],
                        ny[pad_j - (winlen // 2) : pad_j + (winlen // 2) + 1],
                        dnx[pad_i - (winlen // 2) : pad_i + (winlen // 2) + 1],
                        dny[pad_j - (winlen // 2) : pad_j + (winlen // 2) + 1],
                        alpha,
                        swindow,
                    )
                else:
                    tmp_ac[i, j] = np.inf

            # As last resource, the complete window is calculated
            else:
                tmp_ac[i, j] = sliding_dist(
                    nx[pad_i - (winlen / 2) : pad_i + (winlen / 2) + 1],
                    ny[pad_j - (winlen / 2) : pad_j + (winlen / 2) + 1],
                    dnx[pad_i - (winlen / 2) : pad_i + (winlen / 2) + 1],
                    dny[pad_j - (winlen / 2) : pad_j + (winlen / 2) + 1],
                    alpha,
                    swindow,
                )

    c = tmp_ac.copy()

    for i in range(xl):
        for j in range(yl):
            tmp_ac[i, j] += min([ac[i, j], ac[i, j + 1], ac[i + 1, j]])

    path = traceback(ac)

    if do_dist_norm:
        d = ac[-1, -1] / np.sum(np.shape(path))
    else:
        d = ac[-1, -1]

    return d, c, ac, path


def sliding_dist(xw, yw, dxw, dyw, alpha, win):
    """Computes the sliding distance.

    Parameters
    ----------
    xw: nd-array
        x coords window.
    yw: nd-array
        y coords window.
    dxw: nd-array
        x coords diff window.
    dyw: nd-array
        y coords diff window.
    alpha: float
        Rely more on absolute or difference values 1- abs, 0 - diff.
    win: nd-array
        Signal window used for sliding distance.

    Returns
    -------
        Sliding distance
    """
    return (1 - alpha) * np.sqrt(np.sum((((dxw - dyw) * win) ** 2.0))) + alpha * np.sqrt(
        np.sum((((xw - yw) * win) ** 2.0))
    )


def get_mirror(s, ws):
    """Performs a signal windowing based on a double inversion from the start and end segments.

    Parameters
    ----------
    s: nd-array
            the input-signal.
    ws: int
            window size.

    Returns
    -------
        Signal windowed
    """

    return np.r_[2 * s[0] - s[ws:0:-1], s, 2 * s[-1] - s[-2 : -ws - 2 : -1]]


@njit()
def _lcss_point_dist(x, y):
    """

    Parameters
    ----------
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y.

    Returns
    -------
        The LCSS distance.
    """
    dist = 0.0
    for di in range(x.shape[0]):
        diff = x[di] - y[di]
        dist += diff * diff

    return dist ** 0.5


def lcss_accumulated_matrix(x, y, eps):
    """Computes the LCSS cost matrix using the euclidean distance (dist) between two time series.

    Parameters
    ----------
    x: nd-array
        Time series x (query).
    y: nd-array
        Time series y.
    eps : float
        Amplitude matching threshold.

    Returns
    -------
    ac : nd-array
            The accumulated cost matrix.
    """

    xl, yl = len(x), len(y)

    ac = np.zeros((xl + 1, yl + 1))

    for i in range(1, xl + 1):
        for j in range(1, yl + 1):
            if _lcss_point_dist(x[i - 1, :], y[j - 1, :]) <= eps:
                ac[i, j] = 1 + ac[i - 1, j - 1]
            else:
                ac[i, j] = max(ac[i, j - 1], ac[i - 1, j])

    return ac


def lcss_path(x, y, c, eps):
    """Computes the LCSS path between two time series.

    Parameters
    ----------
    x: nd-array
        The reference signal.
    y: nd-array
        The estimated signal.
    c : nd-array
        The cost matrix.
    eps : float
        Matching threshold.

    Returns
    -------
        Coordinates of the minimum LCSS path.
    """
    i, j = len(x), len(y)
    path = []

    while i > 0 and j > 0:
        if _lcss_point_dist(x[i - 1, :], y[j - 1, :]) <= eps:
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif c[i - 1, j] > c[i, j - 1]:
            i -= 1
        else:
            j -= 1

    path = np.array(path[::-1])
    return path[1:, 0], path[1:, 1]


def lcss_score(c):
    """Computes the LCSS similarity score between two time series.

    Parameters
    ----------
    c : nd-array
        The cost matrix.

    Returns
    -------
        The LCSS score.
    """

    xl = c.shape[0] - 1
    yl = c.shape[1] - 1

    return float(c[-1, -1]) / min([xl, yl])
