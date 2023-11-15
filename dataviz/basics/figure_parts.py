import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

from .data import regression, clustering, laguerre


def fig_parts_ax1(ax):

    ax.set_xlabel('Extremely long label that is\nnecessary to cut in half',
                  color='white', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel('Some quantity $\\left[ \dfrac{kg \cdot m}{s^{2}} \\right]$',
                  color='white', fontsize=18, fontweight='bold', labelpad=2)

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_minor_locator(AutoMinorLocator(n=2))
        axis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.set_minor_formatter(FormatStrFormatter("%.2f"))

    ax.set_title('1. Axis labels',
                 color='white', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelcolor='white',
                   labelsize=16, pad=8)
    ax.tick_params(axis='both', which='minor', labelcolor='white',
                   labelsize=10, pad=5)
    ax.tick_params(axis='x', which='both', labelcolor='white',
                   rotation=50)
    return ax

def fig_parts_ax2(ax):

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_minor_locator(AutoMinorLocator(n=3))
    
    ax.grid(True, which='major', ls='--', lw=2, alpha=0.6)
    ax.grid(True, which='minor', ls='-.', lw=1, alpha=0.6)
    
    ax.set_title('2. Grid lines',
                 color='white', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='both', labelcolor='black')
    return ax

def fig_parts_ax3(ax):

    # custom legends
    custom_lines = [Line2D([0], [0], color='tab:orange', lw=3),
                    Line2D([0], [0], color='tab:blue', lw=3, ls=(0, (2, 0.6))),
                    Patch(facecolor='red', alpha=0.2),
                    Patch(facecolor='tab:green', alpha=0.8, hatch=r'\\\\')]
    ax.legend(handles=custom_lines,
              labels=['Series 1', 'Series 2', 'Markup 1', 'Markup 2'],
              loc='upper center', ncols=2, fontsize=17,
              facecolor='white', edgecolor='none', framealpha=0.6)
    
    ax.set_title('3. Legends',
                 color='white', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='both', colors='black')
    return ax

def fig_parts_ax4(ax):
    
    ax.set_title('$\Leftarrow\Downarrow$ 4. Arrangement of figures',
                 color='white', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='both', colors='black')
    return ax

def fig_parts_ax5(ax, n_samples, n_centers):

    X, y = clustering(n_samples, n_centers)
    colors = cm.tab10(y)
    ax.scatter(*X.T,
               c=colors, ec='none', s=7**2, alpha=0.6, zorder=2)

    ax.set_title('5. Colors',
                 color='white', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='both', labelcolor='black')
    return ax

def fig_parts_ax6(ax, k_max):

    X = np.linspace(-10, 10, 100)
    colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']
    for k in range(0, k_max+1):
        L = laguerre(X, alpha=0.0, n=k)
        ax.plot(X, L, label=f'$L_{k}^{{(0)}}$', lw=3, c=colors[k])
    for k in range(0, k_max+1):
        L = laguerre(X, alpha=1.0, n=k)
        ax.plot(X, L, label=f'$L_{k}^{{(1)}}$', lw=3, ls='--', c=colors[k])
    ax.set_xlim(-2, 8)
    ax.set_ylim(-4, 6)

    ax.set_title('6. Line styles',
                 color='white', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='both', colors='white',
                   labelsize=16)
    return ax

def fig_parts_ax7(ax, n_samples, n_centers):

    X, y = clustering(n_samples, n_centers)

    markers = ['o', 's', '^', '*', 'x', '+', 'D', 'p']
    colors = cm.viridis(np.linspace(0, 1, len(markers)))

    for i, marker in enumerate(markers):
        indices = y == i
        ax.scatter(X[indices, 0], X[indices, 1], marker=marker,
                   alpha=0.5, s=10**2)
    
    ax.set_title('7. Marker styles',
                 color='white', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='both', colors='black')
    return ax

def fig_parts_ax8(ax):

    ax.plot(np.random.normal(size=100), c='red')
    for i in range(8):
        ax.plot(np.random.normal(size=100), c='grey', alpha=0.2)
    
    ax.set_title('8. Transparency',
                 color='white', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='both', colors='black')
    return ax

def figure_parts():
    nr, nc = 2, 4
    fig, axes = plt.subplots(nr, nc, figsize=(7*nc, 7*nr), facecolor='black')
    fig.subplots_adjust(wspace=0.35, hspace=0.42)
    axes = axes.flatten()
    
    for ax in axes:
        ax.set(adjustable='box', aspect='auto')
        ax.tick_params(axis='both', which='both', colors='white')
        ax.set_facecolor('whitesmoke')

    _ = fig_parts_ax1(axes[0])
    _ = fig_parts_ax2(axes[1])
    _ = fig_parts_ax3(axes[2])
    _ = fig_parts_ax4(axes[3])
    _ = fig_parts_ax5(axes[4], n_samples=1000, n_centers=4)
    _ = fig_parts_ax6(axes[5], k_max=4)
    _ = fig_parts_ax7(axes[6], n_samples=100, n_centers=3)
    _ = fig_parts_ax8(axes[7])
    
    plt.show()