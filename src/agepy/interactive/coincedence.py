from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import numpy as np

from agepy.interactive import AGEDataViewer
from agepy import ageplot

if TYPE_CHECKING:
    from agepy.spec.coincidence import CoincMap


class AGECoincViewer(AGEDataViewer):
    """Show all spectra in a scan.

    """

    def __init__(self, coinc_map: CoincMap):
        super().__init__(width=1000, height=1000)
        # Save reference to the CoincMap object
        self.coinc = coinc_map
        # Add plot to canvas
        with ageplot.context(["age", "dataviewer"]):
            self.coinc.plot(
                xlabel=self.coinc.xlabel, ylabel=self.coinc.ylabel,
                title=self.coinc.title, cmap=self.coinc.cmap,
                norm=self.coinc.norm, vmin=self.coinc.vmin,
                vmax=self.coinc.vmax)
        self.add_plot(fig=coinc_map.fig, ax=coinc_map.ax)
        # Add the coinc toolbar
        self.add_toolbar()
        # Add ROI button to toolbar
        self.add_rect_selector(self.ax[0], self.on_select, interactive=False)

    def on_select(self, eclick: MouseEvent, erelease: MouseEvent):
        self.coinc.set_roi(eclick.xdata, erelease.xdata,
                           eclick.ydata, erelease.ydata)
        with ageplot.context(["age", "dataviewer"]):
            self.coinc.update()
            self.canvas.draw()
