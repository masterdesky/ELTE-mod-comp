import noise
import numpy as np

from sklearn.datasets import make_regression, make_blobs
from sklearn.preprocessing import MinMaxScaler


def regression(n_samples=1000, noise=20):

    X, y, coef = make_regression(n_samples=n_samples, n_features=1,
                                 n_informative=1, noise=noise,
                                 coef=True, random_state=0)
    X = X.reshape((-1, 1))
    y = y.reshape((-1, 1))
    X_s = MinMaxScaler().fit_transform(X)
    y_s = MinMaxScaler().fit_transform(y)
    
    return np.c_[X_s, y_s]


def clustering(n_samples=1000, n_centers=4):
    X, y = make_blobs(
        n_samples=n_samples, centers=n_centers,
        center_box=(-3, 3), cluster_std=1.0
    )
    return X, y


def generate_terrain(width, height,
                     *,
                     scale=0.1, octaves=6, persistence=0.5, lacunarity=2.0):
    terrain = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            terrain[i][j] = noise.pnoise2(i * scale, 
                                          j * scale, 
                                          octaves=octaves, 
                                          persistence=persistence, 
                                          lacunarity=lacunarity)
    return terrain


def laguerre(X, alpha, n):
    '''
    Calculates the generalized Laguerre polynomial of `k`th order.
    '''
    if n == 0:
        return np.ones_like(X)
    elif n == 1:
        return 1 + alpha - X
    else:
        # Here we calculate the n-th Laguerre polynomial
        # The 'n' input parameter thus becomes 'k+1' in the equation above
        # This means, that:
        #   - k+1 := n
        #   - k   := n-1
        #   - k-1 := n-2
        return (
            (2 * (n-1) + 1 + alpha - X) * laguerre(X, alpha, (n-1)) \
            - \
            ((n-1) + alpha) * laguerre(X, alpha, (n-2))
        )/n