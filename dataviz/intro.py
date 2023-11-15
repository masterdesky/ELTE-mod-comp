import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


#######
#
#    INTRODUCTION PLOTS
#
##########################################################################
def generate_random(S=1.0):
    """
    S : float, default: 1.0
    Scale parameter
    """
    sign = np.random.choice([-1,1])
    return np.random.random() * S * sign


def plot_curve(ax):

    signal = np.sin(np.linspace(0,8*np.pi,1000))
    slope = -0.17 * np.linspace(0,8*np.pi,1000) + 3
    noise = np.random.normal(loc=1.0, scale=1.8, size=1000)
    
    ax.plot(slope + signal + noise,
            color='tab:red')
    ax.set_ylim(-5,9)
    ax.set_title('a) Some kind of signal\nprocessing homework',
                 fontsize=18, y=-0.2)
    return ax

def plot_scatter(ax):

    # Generate sample data
    n_samples = 4000
    n_blobs = 4
    #centers = [[generate_random(S=7) for i in range(2)] for i in range(n_blobs)]
    centers = np.array((
        [5.1, -7.2],
        [-6.3, -2.2],
        [-3.2, 6.3],
        [3.2, 0.4],
    ))
  
    X, y_true = make_blobs(n_samples=n_samples,
                           centers=centers,
                           cluster_std=2.0,
                           center_box=(-10.0, 10.0),
                           random_state=0)
    X = X[:, ::-1]

    colors = ['#4EACC5', '#FF9C34', '#4E9A06', 'm']
    for k, col in enumerate(colors):
        cluster_data = y_true == k
        ax.scatter(X[cluster_data, 0], X[cluster_data, 1],
                   c=col, marker='.', s=10)
    ax.set_xlim(-12,12)
    ax.set_ylim(-12,12)
    ax.set_title('b) Data that can be clustered',
                 fontsize=18, y=-0.2)
    return ax

def complex_plot(ax):
    """
    Ripped from:
    https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html#sphx-glr-auto-examples-calibration-plot-calibration-multiclass-py
    """
    X, y = make_blobs(n_samples=2000, n_features=2, centers=3, random_state=42,
                  cluster_std=5.0)
    X_train, y_train = X[:600], y[:600]
    X_valid, y_valid = X[600:1000], y[600:1000]
    X_train_valid, y_train_valid = X[:1000], y[:1000]
    X_test, y_test = X[1000:], y[1000:]
    
    
    clf = RandomForestClassifier(n_estimators=25)
    clf.fit(X_train, y_train)
    cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    cal_clf.fit(X_valid, y_valid)
    
    colors = ["r", "g", "b"]
    
    clf_probs = clf.predict_proba(X_test)
    cal_clf_probs = cal_clf.predict_proba(X_test)
    # Plot arrows
    for i in range(clf_probs.shape[0]):
        ax.arrow(clf_probs[i, 0], clf_probs[i, 1],
                 cal_clf_probs[i, 0] - clf_probs[i, 0],
                 cal_clf_probs[i, 1] - clf_probs[i, 1],
                 color=colors[y_test[i]], head_width=1e-2)

    # Plot perfect predictions, at each vertex
    ax.plot([1.0], [0.0], 'ro', ms=20, label="Class 1")
    ax.plot([0.0], [1.0], 'go', ms=20, label="Class 2")
    ax.plot([0.0], [0.0], 'bo', ms=20, label="Class 3")
    
    # Plot boundaries of unit simplex
    ax.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")
    
    # Annotate points 6 points around the simplex, and mid point inside simplex
    ax.annotate(r'($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)',
                xy=(1.0/3, 1.0/3), xytext=(1.0/3, .23), xycoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='center', verticalalignment='center')
    ax.plot([1.0/3], [1.0/3], 'ko', ms=5)
    ax.annotate(r'($\frac{1}{2}$, $0$, $\frac{1}{2}$)',
               xy=(.5, .0), xytext=(.5, .1), xycoords='data',
               arrowprops=dict(facecolor='black', shrink=0.05),
               horizontalalignment='center', verticalalignment='center')
    ax.annotate(r'($0$, $\frac{1}{2}$, $\frac{1}{2}$)',
               xy=(.0, .5), xytext=(.1, .5), xycoords='data',
               arrowprops=dict(facecolor='black', shrink=0.05),
               horizontalalignment='center', verticalalignment='center')
    ax.annotate(r'($\frac{1}{2}$, $\frac{1}{2}$, $0$)',
               xy=(.5, .5), xytext=(.6, .6), xycoords='data',
               arrowprops=dict(facecolor='black', shrink=0.05),
               horizontalalignment='center', verticalalignment='center')
    ax.annotate(r'($0$, $0$, $1$)',
               xy=(0, 0), xytext=(.1, .1), xycoords='data',
               arrowprops=dict(facecolor='black', shrink=0.05),
               horizontalalignment='center', verticalalignment='center')
    ax.annotate(r'($1$, $0$, $0$)',
               xy=(1, 0), xytext=(1, .1), xycoords='data',
               arrowprops=dict(facecolor='black', shrink=0.05),
               horizontalalignment='center', verticalalignment='center')
    ax.annotate(r'($0$, $1$, $0$)',
               xy=(0, 1), xytext=(.1, 1), xycoords='data',
               arrowprops=dict(facecolor='black', shrink=0.05),
               horizontalalignment='center', verticalalignment='center')
    # Add grid
    ax.grid(False)
    for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ax.plot([0, x], [x, 0], 'k', alpha=0.2)
        ax.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
        ax.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)
    
    ax.set_xlabel('Probability class 1', fontsize=15, fontweight='bold')
    ax.set_ylabel('Probability class 2', fontsize=15, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    _ = ax.legend(loc="best")
    
    ax.set_title('c) Something complicated thing wow complicated yes',
                 fontsize=18, y=-0.2)
    return ax

def intro_figures():
    nr, nc = 1, 3
    fig, axes = plt.subplots(nr, nc, figsize=(9*nc, 7*nr), facecolor='0.95')
    fig.subplots_adjust(wspace=0.2)
    axes = axes.flatten()
    
    functions = (plot_curve, plot_scatter, complex_plot)
    for f, ax in zip(functions, axes):
        ax.set_aspect('auto')
        f(ax)
    
    plt.suptitle(
        'Fig. 1. Different types of data visualizations that you may ' +
        'find in Physics assignments/homeworks.',
        fontsize=20, fontweight='bold', y=-0.1)
    plt.show()