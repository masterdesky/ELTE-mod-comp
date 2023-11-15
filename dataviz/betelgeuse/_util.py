import scipy
import numpy as np


def set_colors(arr, cmap,
               *,
               vmin=0.0, vmax=1.0, imin=0.0, imax=1.0):
    '''
    Create and scale a colormap for an interval of the values in an
    array, using SciPy's 1D interpolation.

    Parameters
    ----------
    arr : array-like
        Input values to scale the colormap to.
    cmap : 
        Any matplotlib colormap the 
    '''

    arr_c = np.array(arr, copy=True)
    amin, amax = np.min(arr_c), np.max(arr_c)

    mmin = amin + imin*(amax - amin)
    mmax = amin + imax*(amax - amin)
    arr_c[arr_c < mmin] = mmin
    arr_c[arr_c > mmax] = mmax

    # Scale the colorscale with 
    m = scipy.interpolate.interp1d(
        x=[mmax, mmin], y=[vmin, vmax], kind='linear'
    )

    c = cmap(m(arr_c))

    return c