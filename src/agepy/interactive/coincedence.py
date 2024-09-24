from __future__ import annotations
from typing import TYPE_CHECKING
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
        # Add the toolbar
        self.add_toolbar()
        # Add ROI button to toolbar
        self.add_roi_action(self.toggle_selector)
        # Add ROI selector
        self.selector = RectangleSelector(
            self.ax[0], self.on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords="pixels",
            #interactive=True,
            props={"linewidth": 0.83, "linestyle": "--", "fill": False},
            handle_props={"markersize": 0})
        self.selector.set_active(False)

    def toggle_selector(self):
        self.selector.set_active(not self.selector.active)

    def on_select(self, eclick: MouseEvent, erelease: MouseEvent):
        self.coinc.set_roi(eclick.xdata, erelease.xdata,
                           eclick.ydata, erelease.ydata)
        with ageplot.context(["age", "dataviewer"]):
            self.coinc.update()
            self.canvas.draw()


