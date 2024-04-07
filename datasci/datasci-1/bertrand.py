import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def circle(num=100) -> list:
    """
    Parameters
    ----------
    num : int
      Sampling frequency of the circle's circumference

    Returns
    -------
    cc : list
      Points on the circumference of a circle of R=1.
    """
    phi = np.linspace(0, 2*np.pi, num=num)
    cc = [[np.sin(r), np.cos(r)] for r in phi]
    return cc


def bertrand_1() -> list:
    """
    The "random endpoints" method in Bertrand's paradox.

    Returns
    -------
    cc : list
      The coordinates of the two endpoints of the chord.
    """
    phi = 2*np.pi*np.random.random(size=2)
    cc = [[np.sin(r), np.cos(r)] for r in phi]
    return cc


def bertrand_2() -> list:
    """
    The "random radial point" method in Bertrand's paradox.

    Returns
    -------
    cc : numpy.ndarray
      The coordinates of the two endpoints of the chord.
    """
    phi = 2*np.pi*np.random.random(size=1)
    mid = np.random.random(size=1)
    t = [phi+np.arccos(mid), phi-np.arccos(mid)]
    cc = [[np.sin(r), np.cos(r)] for r in t]
    return cc


def bertrand_3() -> list:
    """
    The "random midpoint" method in Bertrand's paradox.

    Returns
    -------
    cc : list
      The coordinates of the two endpoints of the chord.
    """
    phi = 2 * np.pi * np.random.random(size=1)
    mid = np.sqrt(np.random.random(size=1))
    t = [phi + np.arccos(mid), phi - np.arccos(mid)]
    cc = [[np.sin(r), np.cos(r)] for r in t]
    return cc


def calculate_p(cc):
    """
    Calculates what percentage of the chords are longer than sqrt(3).

    Parameters
    ----------
    cc : numpy.ndarray of shape (-1, 2, 2)
      List of coordinates of the endpoints of chords.

    Returns
    -------
    P : float
      Percentage of chords longer than sqrt(3).
    """
    d = np.linalg.norm(cc[:, 1, :] - cc[:, 0, :], axis=1)
    P = np.sum(d > np.sqrt(3)) / len(cc)

    return P


def plot_circle(ax) -> matplotlib.axes.Axes:

    cc = np.array(circle(num=100))
    ax.plot(cc[:, 0], cc[:, 1],
            color='black', alpha=0.7,
            lw=2, zorder=2)
    return ax


def plot_midpoints(f, cc, c, P) -> None:

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    ax = plot_circle(ax)

    ax.scatter(*np.mean(cc, axis=1).transpose((1, 0)),
               color=c, alpha=0.8,
               s=3**2, zorder=1)

    ax.set_title(f'Method #{f.__name__.split("_")[1]} P = {P:.3f}',
                 fontsize=20, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])

    if not os.path.exists('./out/'):
        os.makedirs('./out/')
    plt.savefig(f'./out/{f.__name__}-midpoints.png')
    plt.close(fig)

    return


def plot_chords(f, cc, c, P) -> None:

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    ax = plot_circle(ax)

    # Transform for chord plotting (transpose 2. and 3. axes)
    cc = cc.transpose(0, 2, 1)
    ax.plot(*(cc.reshape((-1, 2))),
            color=c, alpha=0.4,
            lw=1, zorder=1)

    ax.set_title(f'Method #{f.__name__.split("_")[1]} | P = {P:.3f}',
                 fontsize=20, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])

    if not os.path.exists('./out/'):
        os.makedirs('./out/')
    plt.savefig(f'./out/{f.__name__}-chords.png')
    plt.close(fig)

    return


def main() -> None:
    n = int(sys.argv[1])

    fs = [bertrand_1, bertrand_2, bertrand_3]
    cs = ['#99004D', '#003CB3', '#226600']
    for f, c in zip(fs, cs):
        cc = np.array([f() for _ in range(n)]).reshape((-1, 2, 2))
        P = calculate_p(cc)
        plot_chords(f, cc, c, P)
        plot_midpoints(f, cc, c, P)

    return


if __name__ == '__main__':
    main()
