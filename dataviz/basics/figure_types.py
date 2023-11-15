import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

from basics.data import generate_terrain


def fig_parts_ax1(ax, Z):
    ax.imshow(Z)
    ax.set_title('plt.imshow(Z)',
                 color='white', fontsize=18, fontweight='bold')
    return ax

def fig_parts_ax2(ax, X, Y, Z):
    ax.contour(X, Y, Z[::-1])
    ax.set_title('ax.contour(X, Y, Z[::-1])',
                 color='white', fontsize=18, fontweight='bold')
    return ax

def fig_parts_ax3(ax, X, Y, Z):
    ax.contourf(X, Y, Z[::-1])
    ax.set_title('ax.contourf(X, Y, Z[::-1])',
                 color='white', fontsize=18, fontweight='bold')
    return ax

def fig_parts_ax4(ax, X, Y, Z):

    # Ripped from here:
    # https://matplotlib.org/stable/gallery/mplot3d/custom_shaded_3d_surface.html
    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)
    ax.set_title('Custom 3D hillshading',
                 color='white', fontsize=18, fontweight='bold')
    return ax

def figure_types(N):
    nr, nc = 1, 4
    fig, axes = plt.subplots(nr, nc, figsize=(7*nc, 7*nr), facecolor='black')
    fig.subplots_adjust(wspace=0.2)
    axes = axes.flatten()
    
    for ax in axes:
        ax.set(adjustable='box', aspect='equal')
        ax.tick_params(axis='both', which='both', colors='black')

    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    Z = generate_terrain(N, N)
    _ = fig_parts_ax1(axes[0], Z)
    _ = fig_parts_ax2(axes[1], X, Y, Z)
    _ = fig_parts_ax3(axes[2], X, Y, Z)

    axes[3] = fig.add_subplot(1, 4, 4, projection='3d')
    axes[3].view_init(elev=50, azim=190)
    _ = fig_parts_ax4(axes[3], X, Y, Z)
    
    plt.show()