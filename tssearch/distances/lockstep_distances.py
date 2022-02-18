import numpy as np
from scipy import stats
from scipy.spatial import distance
from tssearch.utils.preprocessing import interpolation
from tssearch.distances.lockstep_utils import _lnorm_multidimensional, _lnorm_unidimensional


def euclidean_distance(x, y, weight=None):
    """Computes the Euclidean distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    weight: nd-array (Default: None)
        query weight values.

    Returns
    -------
    float
        Euclidean distance value.

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
    """Computes the Minkowski distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

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
    float
        Minkowski distance value.

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
    """Computes the Manhattan distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    weight: nd-array (Default: None)
        query weight values.

    Returns
    -------
    float
        Manhattan distance value.

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
    """Computes the Chebyshev distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    weight: nd-array (Default: None)
        query weight values.

    Returns
    -------
    float
        Chebyshev distance value.

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
    """Computes the correlation distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    weight: nd-array (Default: None)
        query weight values.

    Returns
    -------
    float
        Correlation distance value.

    """
    if len(x) != len(y):
        x, y = interpolation(x, y)

    correlation_d = distance.correlation(x, y, weight)

    return correlation_d


def pearson_correlation(x, y, beta=None):
    """Computes the Pearson correlation between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    beta: float
        Beta coefficient.

    Returns
    -------
    float
        Pearson correlation value.

    """
    if len(x) != len(y):
        x, y = interpolation(x, y)

    r, p = stats.pearsonr(x, y)

    if beta is None:
        d = 2 * (1 - r)
    else:
        d = ((1 - r) / (1 + r)) ** beta
    return d


def short_time_series_distance(x, y, tx=None, ty=None):
    """Computes the short time series distance (STS) between two time series.

    Reference: Möller-Levet, C. S., Klawonn, F., Cho, K., and Wolkenhauer, O. (2003).

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    tx : nd-array
        Sampling index of time series x.
    ty : nd-array
        Sampling index of time series y.

    Returns
    -------
    float
        Short time series distance value.

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
    """Computes the Braycurtis distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    weight: nd-array (Default: None)
        query weight values.

    Returns
    -------
    float
        Braycurtis distance value.

    """
    if len(x) != len(y):
        x, y = interpolation(x, y)

    braycurtis_d = distance.braycurtis(x, y, weight)

    return braycurtis_d


def canberra_distance(x, y, weight=None):
    """Computes the Canberra distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    weight: nd-array (Default: None)
        query weight values.

    Returns
    -------
    float
        Canberra distance value.

    """
    if len(x) != len(y):
        x, y = interpolation(x, y)

    canberra_d = distance.canberra(x, y, weight)

    return canberra_d


def cosine_distance(x, y, weight=None):
    """Computes the cosine distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    weight: nd-array (Default: None)
        query weight values.

    Returns
    -------
    float
        Cosine distance value.

    """
    if len(x) != len(y):
        x, y = interpolation(x, y)

    cosine_d = distance.cosine(x, y, weight)

    return cosine_d


def mahalanobis_distance(x, y, weight=None):
    """Computes the Mahalanobis distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    weight: nd-array (Default: None)
        query weight values.

    Returns
    -------
    float
        Mahalanobis distance value.

    """
    if len(x) != len(y):
        x, y = interpolation(x, y)

    mahalanobis_d = distance.mahalanobis(x, y, weight)

    return mahalanobis_d


def sqeuclidean_distance(x, y, weight=None):
    """Computes the squared Euclidean distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    weight: nd-array (Default: None)
        query weight values.

    Returns
    -------
    float
        Squared Euclidean distance value.

    """
    if len(x) != len(y):
        x, y = interpolation(x, y)

    sqeuclidean_d = distance.sqeuclidean(x, y, weight)

    return sqeuclidean_d


def hamming_distance(x, y, weight=None):
    """Computes the Hamming distance between two time series.

    If the time series do not have the same length, an interpolation is performed.

    Parameters
    ----------
    x : nd-array
        Time series x.
    y : nd-array
        Time series y.
    weight: nd-array (Default: None)
        query weight values.

    Returns
    -------
    float
        Hamming distance value.

    """
    if len(x) != len(y):
        x, y = interpolation(x, y)

    hamming_d = distance.hamming(x, y, weight)

    return hamming_d
