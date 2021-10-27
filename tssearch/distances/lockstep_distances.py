import numpy as np
from scipy import stats
from scipy.spatial import distance
from tssearch.utils.preprocessing import interpolation
from tssearch.distances.lockstep_utils import _lnorm_multidimensional, _lnorm_unidimensional


def euclidean_distance(x, y, weight=None):
    """Computes the euclidean distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x
    y : nd-array
        Time series y

    Returns
    -------
    float
        Euclidean distance value

    """

    p = 2

    if len(x) != len(y):
        x, y = interpolation(x, y)

    if weight is None:
        ed = np.linalg.norm(x - y, p)
    else:
        if len(np.shape(x)) > 1:
            distance = _lnorm_multidimensional(x, y, weight, p=p)
        else:
            distance = _lnorm_unidimensional(x, y, weight, p=p)
        ed = np.sum(distance)
    return ed


def minkowski_distance(x, y, weight=None, p=3):
    """Computes the minkowski distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x
    y : nd-array
        Time series y

    Returns
    -------
    float
        Minkowski distance value

    """

    if len(x) != len(y):
        x, y = interpolation(x, y)

    if weight is None and (p < 3 or p == np.inf):
        distance = np.linalg.norm(x - y, p)
    else:
        if weight is None:
            weight = np.ones_like(x)
        if len(np.shape(x)) > 1:
            distance = _lnorm_multidimensional(x, y, weight, p=p)
        else:
            distance = _lnorm_unidimensional(x, y, weight, p=p)
        distance = np.sum(distance)

    return distance


def manhattan_distance(x, y, weight=None):
    """Computes the manhattan distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x
    y : nd-array
        Time series y

    Returns
    -------
    float
        Manhattan distance value

    """
    p = 1

    if len(x) != len(y):
        x, y = interpolation(x, y)

    if weight is None:
        distance = np.linalg.norm(x - y, p)
    else:
        if len(np.shape(x)) > 1:
            distance = _lnorm_multidimensional(x, y, weight, p=p)
        else:
            distance = _lnorm_unidimensional(x, y, weight, p=p)
        distance = np.sum(distance)

    return distance


def chebyshev_distance(x, y, weight=None):
    """Computes the chebyshev distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x
    y : nd-array
        Time series y

    Returns
    -------
    float
        Chebyshev distance value

    """

    p = np.inf

    if len(x) != len(y):
        x, y = interpolation(x, y)

    if weight is None:
        d = np.linalg.norm(x - y, p)
    else:
        if len(np.shape(x)) > 1:
            distance = _lnorm_multidimensional(x, y, weight, p=p)
        else:
            distance = _lnorm_unidimensional(x, y, weight, p=p)
        d = np.sum(distance)
    return d


def correlation_distance(x, y, weight=None):
    """Computes the correlation between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x
    y : nd-array
        Time series y

    Returns
    -------
    float
        Correlation distance value

    """

    if len(x) != len(y):
        x, y = interpolation(x, y)

    correlation_d = distance.correlation(x, y, weight)

    return correlation_d


def pearson_correlation(x, y, beta=None):

    if len(x) != len(y):
        x, y = interpolation(x, y)

    r, p = stats.pearsonr(x, y)

    if beta is None:
        d = 2 * (1 - r)
    else:
        d = ((1 - r) / (1 + r)) ** beta
    return d


def short_time_series_distance(x, y, tx=None, ty=None):
    """The Short Time Series distance (STS) introduced by M ̈oller-Levet, Klawonn, Cho, and
    Wolkenhauer (2003)

    Parameters
    ----------
    x
    y
    tx
    ty

    Returns
    -------
    """
    if len(x) != len(y):
        x, y = interpolation(x, y)

    if tx is None:
        tx = np.arange(len(x))
    if ty is None:
        ty = np.arange(len(y))

    sts = np.sqrt(np.sum((np.diff(y) / np.diff(tx) - np.diff(x) / np.diff(ty)) ** 2))

    return sts


def braycurtis_distance(x, y, weight=None):
    """Computes the braycurtis distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x
    y : nd-array
        Time series y

    Returns
    -------
    float
        Braycurtis distance value

    """

    if len(x) != len(y):
        x, y = interpolation(x, y)

    braycurtis_d = distance.braycurtis(x, y, weight)

    return braycurtis_d


def canberra_distance(x, y, weight=None):
    """Computes the canberra distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x
    y : nd-array
        Time series y

    Returns
    -------
    float
        Canberra distance value

    """
    if len(x) != len(y):
        x, y = interpolation(x, y)

    canberra_d = distance.canberra(x, y, weight)

    return canberra_d


def cosine_distance(x, y, weight=None):
    """Computes the correlation between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x
    y : nd-array
        Time series y

    Returns
    -------
    float
        Correlation distance value

    """

    if len(x) != len(y):
        x, y = interpolation(x, y)

    cosine_d = distance.cosine(x, y, weight)

    return cosine_d


def mahalanobis_distance(x, y, weight=None):
    """Computes the mahalanobis between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x
    y : nd-array
        Time series y
    """

    if len(x) != len(y):
        x, y = interpolation(x, y)

    mahalanobis_d = distance.mahalanobis(x, y, weight)

    return mahalanobis_d


def sqeuclidean_distance(x, y, weight=None):
    """Computes the sqeuclidean between two time series.

    If the time series do not have the same length, an interpolation is performed.

    """

    if len(x) != len(y):
        x, y = interpolation(x, y)

    sqeuclidean_d = distance.sqeuclidean(x, y, weight)

    return sqeuclidean_d


def hamming_distance(x, y, weight=None):
    """Computes the hamming between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x
    y : nd-array
        Time series y

    Returns
    -------
    float
        hamming distance value

    """

    if len(x) != len(y):
        x, y = interpolation(x, y)

    hamming_d = distance.hamming(x, y, weight)

    return hamming_d
