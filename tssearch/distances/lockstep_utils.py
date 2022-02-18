import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _lnorm_multidimensional(x, y, weight, p=2):
    """

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    weight: nd-array (Default: None)
        query weight values.
    p: int
        Lp norm distance degree.

    Returns
    -------
        The Lp norm distance.
    """
    l1 = x.shape[0]
    l3 = x.shape[1]

    distance = np.zeros_like(x, dtype=float)
    for i in prange(l1):
        dist = 0.0
        for di in range(l3):
            diff = x[i, di] - y[i, di]
            dist += weight[i, di] * (diff ** p)
        distance[i] = dist ** (1 / p)

    return distance


def _lnorm_unidimensional(x, y, weight, p=2):
    """

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    weight: nd-array (Default: None)
        query weight values.
    p: int
        Lp norm distance degree.

    Returns
    -------
        The Lp norm distance.
    """
    distance = weight * np.power(np.power(np.abs(x - y), p), (1 / p))

    return distance
