import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ._util import *
from ._style import RC_PARAMS
from .jdcal import gen_date_ticks

def plot_1(data):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.plot(data['mag'])
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Magnitude', fontsize=20)
    plt.show()


def plot_2(data):
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(12, 4), dpi=100,
                               facecolor='black', subplot_kw=dict(facecolor='black'))

        colors = set_colors(data['mag'], cmap=cm.magma,
                            vmin=0.5, vmax=1.0, imin=0.0, imax=1.0)
        ax.scatter(data['gd'], data['mag'],
                   c=colors, s=9, alpha=0.8, zorder=1)
        ax.errorbar(data['gd'], data['mag'], yerr=data['err'],
                    ecolor=colors, elinewidth=0.5, linestyle='None',
                    alpha=0.4, zorder=0)

        ax.set_xlim(data['gd'].values[0] - pd.Timedelta(1, 'W'),
                    data['gd'].values[-1] + pd.Timedelta(1, 'W'))
        d = max(data['mag']) - min(data['mag'])
        ax.set_ylim(min(data['mag'])-0.2*d, max(data['mag'])+0.2*d)

        ax.set_title('\\textbf{Fig. 2.} AAVSO light curve of Betelgeuse', y=-0.35,
                     color='white', fontsize=18)
        ax.set_ylabel('V-band magnitude',
                      color='white', fontsize=20, linespacing=0.8)

        ax.tick_params(axis='x', which='both', labelcolor='white',
                       labelsize=14)
        ax.tick_params(axis='y', which='both', labelcolor='white',
                       labelsize=16)

        # DATE FORMATTING SOURCE:
        #   - https://matplotlib.org/gallery/text_labels_and_annotations/date.html
        # Format the ticks by calling the locator instances of matplotlib.dates
        date_ticks = gen_date_ticks(data['gd'], N=12)
        ax.set_xticks(date_ticks)
        ax.set_xticklabels(date_ticks, rotation=30, ha='center')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))#-%d

        # Invert y-axis to show magnitudes correctly
        ax.invert_yaxis()

        # Source text
        ax.text(x=0.99, y=0.975, s='\\texttt{Source of data: https://www.aavso.org/}',
                color='white', fontsize=8, alpha=0.8,
                ha='right', va='top', transform=ax.transAxes,
                bbox=dict(facecolor='black', alpha=0.2, lw=0))
        plt.show()