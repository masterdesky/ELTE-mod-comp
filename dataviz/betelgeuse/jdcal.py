import numpy as np
import pandas as pd


def jd2g(jd):
    '''
    Convert Julian Date to Gregorian Datetime in a NumPy and Pandas
    compatible format.

    Algorithm is the same as in `jdcal.jd2gcal()` seen in
    https://github.com/phn/jdcal/blob/master/jdcal.py#L193.

    Parameters
    ----------
    jd : scalar or array_like
        Julian Date or array of Julian Dates.
    '''
    f, jd_i = np.modf(jd)

    # Calculate years, months and days
    ell = jd_i + 68569
    n = np.int64((4 * ell) / 146097.0)
    ell -= np.int64(((146097 * n) + 3) / 4.0)
    i = np.int64((4000 * (ell + 1)) / 1461001)
    ell -= np.int64((1461 * i) / 4.0) - 31
    j = np.int64((80 * ell) / 2447.0)
    D = ell - np.int64((2447 * j) / 80.0)
    ell = np.int64(j / 11.0)
    M = j + 2 - (12 * ell)
    Y = 100 * (n - 49) + i + ell

    # Calculate hours, minutes and seconds
    r, HH = np.modf(f*24)
    r, MM = np.modf(r*60)
    r, SS = np.modf(r*60)

    T = np.stack((Y, M, D, HH, MM, SS), axis=-1)
    T = np.array(T, copy=False, dtype=np.int64)

    # Convert values to integers and then to `numpy.datetime64` arrays
    fmt = '{:04}-{:02}-{:02}T{:02}:{:02}:{:02}'
    gd = pd.DatetimeIndex([
        fmt.format(y, m, d, hh, mm, ss) for y, m, d, hh, mm, ss in T
    ])

    return gd


def gen_date_ticks(dates, N=6):
    """
    Create a list of string date ticks to use them on a figure.
    """
    date_ticks = pd.date_range(min(dates), max(dates), periods=N).to_numpy()
    
    return date_ticks