import numpy as np


def standardization(signal, fit=False, param=None):
    """Normalizes a given signal by subtracting the mean and dividing by the standard deviation.

    Parameters
    ----------
    signal : nd-array
        input signal

    Returns
    -------
    nd-array
        standardized signal

    """
    if param is not None:
        s_mean = param[0]
        s_std = param[1]
    else:
        s_mean = np.mean(signal, axis=0)
        s_std = np.std(signal, axis=0)

    if fit:
        d_mean = np.mean(np.diff(signal, axis=0), axis=0)
        d_std = np.std(np.diff(signal, axis=0), axis=0)
        return (signal - s_mean) / s_std, np.array([s_mean, s_std, d_mean, d_std])
    else:
        return (signal - s_mean) / s_std


def interpolation(x, y):
    """Computes the interpolation given two time series of different length.


    Parameters
    ----------
    x : nd-array
        Time series x
    y : nd-array
        Time series y

    Returns
    -------
    interp_signal (nd-array)
        Interpolated signal
    nd-array
        Time series

    """

    lx = len(x)
    ly = len(y)
    if lx > ly:
        t_old = np.linspace(0, lx, ly)
        t_new = np.linspace(0, lx, lx)
        if len(np.shape(x)) == 1:
            y_new = np.interp(t_new, t_old, y)
        else:
            y_new = np.array([np.interp(t_new, t_old, y[:, ax]) for ax in range(np.shape(x)[1])]).T
        return x, y_new
    else:
        t_old = np.linspace(0, ly, lx)
        t_new = np.linspace(0, ly, ly)

        if len(np.shape(x)) == 1:
            x_new = np.interp(t_new, t_old, x)
        else:
            x_new = np.array([np.interp(t_new, t_old, x[:, ax]) for ax in range(np.shape(x)[1])]).T
        return x_new, y
