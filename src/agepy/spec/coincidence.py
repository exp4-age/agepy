import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def plot_coinc_map(coincmap, xedges, yedges, figsize=None, cmap='YlOrRd',
                   title=None, norm=None, vmin=1, vmax=None, num=None,
                   xlabel="early electron kinetic energy",
                   ylabel="late electron kinetic energy"):
    """Plot a coincedence map and its projections on the x and y axes.

    Parameters
    ----------
    coincmap : numpy.ndarray
        2d array of shape (m,n) containing the coincidence map. In most
        cases this will be the output of ``numpy.histogram2d()``.
    xedges : numpy.ndarray
        1d array of shape (m+1) containing the bin edges of the x-axis.
    yedges : numpy.ndarray
        1d array of shape (n+1) containing the bin edges of the y-axis.
    figsize : tuple, optional
        Figure size in inches. Default: None
    cmap : matplotlib.colors.Colormap or str or None, optional
        Colormap passed to ``matplotlib.pyplot.pcolormesh()``.
        Default: 'YlOrRd'
    title : str, optional
        Title of the figure. Default: None
    norm : str or matplotlib.colors.Normalize or None, optional
        Normalization passed to ``matplotlib.pyplot.pcolormesh()``.
        Default: None
    vmin, vmax : float, optional
        Minimum and maximum value for the colormap passed to
        ``matplotlib.pyplot.pcolormesh()``. Default: 1, None
    num: int or str or matplotlib.figure.Figure, optional
        Figure identifier passed to ``matplotlib.pyplot.figure()``.
    xlabel, ylabel : str, optional
        Labels of the x and y axes. Default:
        "early electron kinetic energy", "late electron kinetic energy"

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object.
    ax: tuple of matplotlib.axes.Axes
        Tuple of matplotlib Axes objects containing the coincidence map,
        the projection on the x-axis and the projection on the y-axis.

    """
    fig = plt.figure(num=num, figsize=figsize, clear=True)

    # grid with columns=2, row=2
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                           wspace=0.05, hspace=0.05)
    # coinc matrix is subplot 2: lower left
    ax_coinc = plt.subplot(gs[2])
    # spectrum of E0 is subplot 0: upper left
    ax_x = plt.subplot(gs[0], sharex=ax_coinc)
    # spectrum of E1 is subplot 3: lower right
    ax_y = plt.subplot(gs[3], sharey=ax_coinc)
    # colorbar is subplot 1: upper right
    ax_cb = plt.subplot(gs[1])

    # sum spectrum of E0 (top)
    hist_x = np.sum(coincmap, axis=1)
    ax_x.set_xlim(xedges[0], xedges[-1])
    ax_x.stairs(hist_x, xedges, color='k')

    # sum spectrum of E1 (right)
    hist_y = np.sum(coincmap, axis=0)
    ax_y.set_ylim(yedges[0], yedges[-1])
    ax_y.stairs(hist_y, yedges, color='k', orientation="horizontal")

    # coinc matrix
    X, Y = np.meshgrid(xedges, yedges)
    pcm = ax_coinc.pcolormesh(X, Y, coincmap.T, cmap=cmap, norm=norm,
                              vmin=vmin, vmax=vmax, rasterized=True)

    # Generate a colorbar for the histogram in the upper right panel
    ax_cb.axis("off")
    ax_cb_inset = ax_cb.inset_axes([0.0, 0.0, 0.25, 1.0])
    fig.colorbar(pcm, cax=ax_cb_inset)

    # Set the labels
    ax_coinc.set_xlabel(xlabel)
    ax_coinc.set_ylabel(ylabel)

    # Remove x and y tick labels
    ax_x.tick_params(axis='both', labelleft=False, labelbottom=False)
    ax_y.tick_params(axis='both', labelleft=False, labelbottom=False)

    # Set the title
    if title is not None:
        fig.suptitle(title)

    return fig, (ax_coinc, ax_x, ax_y)
