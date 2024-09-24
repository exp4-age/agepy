"""Functions for working with coincidence maps.

"""
from __future__ import annotations
from typing import Union, Tuple
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from agepy.interactive import AGEpp
from agepy.interactive.coincedence import AGECoincViewer


class CoincMap:
    """Plot a coincedence map and its projections on the x and y axes.

    Parameters
    ----------
    data: numpy.ndarray
        2d array of shape (m,n) containing the coincidence map. In most
        cases this will be the output of ``numpy.histogram2d()``.
    xedges: numpy.ndarray
        1d array of shape (m+1) containing the bin edges of the x-axis.
    yedges: numpy.ndarray
        1d array of shape (n+1) containing the bin edges of the y-axis.

    Attributes
    ----------
    data: numpy.ndarray 
        Access the 2D histogram data.
    xedges: numpy.ndarray 
        Access the bin edges of the x-axis.
    yedges: numpy.ndarray 
        Access the bin edges of the y-axis.
    roi: tuple of tuples
        Region of interest. Should be set using ``set_roi()``.
    fig: matplotlib.figure.Figure
        Matplotlib Figure object. Created by the ``plot()`` method.
    ax: Sequence of matplotlib.axes.Axes
        Sequence of matplotlib Axes objects containing the coincidence
        map, the projection on the x-axis and the projection on the
        y-axis and the colorbar.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from agepy import ageplot
    >>> ageplot.use("age")

    Create some simple example data:

    >>> xyrange = ((0, 2), (0, 2))
    >>> n = 10000
    >>> rng = np.random.default_rng(42)
    >>> x = rng.normal(0.8, 0.2, size=n)
    >>> y = rng.normal(1.2, 0.4, size=n)
    >>> H, xedges, yedges = np.histogram2d(x, y, bins=50, range=xyrange)

    Create a CoincMap object and plot the data:

    >>> from agepy.spec.coincidence import CoincMap
    >>> coinc = CoincMap(H, xedges, yedges)
    >>> coinc.plot(figsize=(4.8, 4.8))
    >>> # Start an interactive session with coinc.interactive()
    
    """

    def __init__(self,
        data: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray,
    ) -> None:
        # Save the data
        self.data = data
        self.xedges = xedges
        self.yedges = yedges
        self._X, self._Y = np.meshgrid(xedges, yedges)
        # Set some default values
        self.roi = None
        self.fig = None
        self.ax = None
        self.cmap = 'YlOrRd'
        self.norm = None
        self.vmin = 1
        self.vmax = None

    def set_roi(self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float
    ) -> None:
        """Set the region of interest (ROI) for the plot.

        Parameters
        ----------
        xmin, xmax : float
            Minimum and maximum value for the x-axis.
        ymin, ymax : float
            Minimum and maximum value for the y-axis.

        """
        if (xmin >= xmax) or (ymin >= ymax):
            raise ValueError("Given min value is greater than max value.")
        self.roi = ((xmin, ymin), (xmax, ymax))

    def plot(self,
        xlabel: str = "early electron kinetic energy",
        ylabel: str = "late electron kinetic energy",
        title: str = None,
        figsize: Tuple[float, float] = None,
        cmap: Union[str, plt.colors.Colormap] = "YlOrRd",
        norm: Union[str, plt.colors.Normalize] = None,
        vmin: float = 1,
        vmax: float = None,
        num: Union[str, int] = None
    ) -> None:
        """Create a matplotlib figure with the coincidence map and its
        projections on the x and y axes. The figure is stored in the
        attribute ``fig`` and the axes in ``ax``.

        Parameters
        ----------
        xlabel, ylabel : str, optional
            Labels of the x and y axes. Default:
            "early electron kinetic energy", "late electron kinetic energy"
        title : str, optional
            Title of the figure. Default: None
        figsize : tuple, optional
            Figure size in inches. Default: None
        cmap : matplotlib.colors.Colormap or str, optional
            Colormap passed to ``matplotlib.pyplot.pcolormesh()``.
            Default: 'YlOrRd'
        norm : str or matplotlib.colors.Normalize or None, optional
            Normalization passed to ``matplotlib.pyplot.pcolormesh()``.
            Default: None
        vmin, vmax : float, optional
        Minimum and maximum value for the colormap passed to
            ``matplotlib.pyplot.pcolormesh()``. Default: 1, None
        num: int or str or matplotlib.figure.Figure, optional
            Figure identifier passed to ``matplotlib.pyplot.figure()``.

        """
        # Set options
        self.cmap = cmap
        self.norm = norm
        self.vmin = vmin
        self.vmax = vmax
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        # Create the figure
        self.fig = plt.figure(num=num, figsize=figsize, clear=True)
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
        ax_cb.axis("off")
        ax_cb_inset = ax_cb.inset_axes([0.0, 0.0, 0.25, 1.0])
        # Set the labels
        ax_coinc.set_xlabel(xlabel)
        ax_coinc.set_ylabel(ylabel)
        # Remove x and y tick labels
        ax_x.tick_params(axis='both', labelleft=False, labelbottom=False)
        ax_y.tick_params(axis='both', labelleft=False, labelbottom=False)
        # Remove grid from the coinc map and colorbar
        ax_coinc.grid(False)
        ax_cb_inset.grid(False)
        # Set the title
        if title is not None:
            self.fig.suptitle(title)
        # Save the axes for later use
        self.ax = (ax_coinc, ax_x, ax_y, ax_cb_inset)
        # Plot the data
        self.update()

    def interactive(self):
        """Start an interactive PyQt session to view the data and add
        an ROI.
        
        """
        if self.fig is None:
            self.plot()
        app = AGEpp(AGECoincViewer, self)
        app.run()

    def update(self):
        """Update the plot with the current data and ROI settings.
        
        """
        if self.fig is None:
            self.plot()
            return
        # Get the axes
        ax_coinc, ax_x, ax_y, cax = self.ax
        # Plot the data
        # Clear the axis before plotting
        ax_coinc.clear()
        ax_coinc.grid(False)
        # Plot the coincidence map
        pcm = ax_coinc.pcolormesh(
            self._X, self._Y, self.data.T, cmap=self.cmap, norm=self.norm,
            vmin=self.vmin, vmax=self.vmax, rasterized=True)
        self.fig.colorbar(pcm, cax=cax)
        cax.grid(False)
        # Plot the projections
        # Clear the axis before plotting
        ax_x.clear()
        ax_y.clear()
        # sum spectrum of E0 (top)
        hist_x = np.sum(self.data, axis=1)
        ax_x.set_xlim(self.xedges[0], self.xedges[-1])
        ax_x.stairs(hist_x, self.xedges, color='k')
        # sum spectrum of E1 (right)
        hist_y = np.sum(self.data, axis=0)
        ax_y.set_ylim(self.yedges[0], self.yedges[-1])
        ax_y.stairs(hist_y, self.yedges, color='k', orientation="horizontal")
        # Draw ROI projections if an ROI is defined
        if self.roi is not None:
            # Get the indices of the bins that correspond to the roi boundaries
            xl_idx = np.searchsorted(self.xedges, self.roi[0][0], side='left')
            xr_idx = np.searchsorted(self.xedges, self.roi[1][0], side='right')
            yl_idx = np.searchsorted(self.yedges, self.roi[0][1], side='left')
            yr_idx = np.searchsorted(self.yedges, self.roi[1][1], side='right')
            # Slice the 2D histogram to get the roi
            data_roi = self.data[xl_idx:xr_idx, yl_idx:yr_idx]
            # Extract the corresponding x and y edges
            xedges_roi = self.xedges[xl_idx:xr_idx+1]
            yedges_roi = self.yedges[yl_idx:yr_idx+1]
            # sum spectrum of E0 (top)
            data_roi_x = np.sum(data_roi, axis=1)
            ax_x.stairs(data_roi_x, xedges_roi)
            # sum spectrum of E1 (right)
            data_roi_y = np.sum(data_roi, axis=0)
            ax_y.stairs(data_roi_y, yedges_roi, orientation="horizontal")
            # Draw the ROI on the coinc map
            ax_coinc.add_patch(plt.Rectangle(
                (xedges_roi[0], yedges_roi[0]), xedges_roi[-1] - xedges_roi[0],
                yedges_roi[-1] - yedges_roi[0], edgecolor='k',
                facecolor='none', linestyle='--'))


def plot_coinc_map(coincmap, xedges, yedges, figsize=None, cmap='YlOrRd',
                   title=None, norm=None, vmin=1, vmax=None, num=None,
                   xlabel="early electron kinetic energy",
                   ylabel="late electron kinetic energy",
                   interactive=False):
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

    if interactive:
        app = AGEpp(AGECoincViewer, fig, (ax_coinc, ax_x, ax_y, ax_cb_inset))
        app.run()
    else:
        return fig, (ax_coinc, ax_x, ax_y, ax_cb_inset)
