from __future__ import annotations
from PyQt5.QtWidgets import QPushButton, QStyle
import matplotlib.pyplot as plt
import numpy as np

from agepy.interactive import AGEDataViewer
from agepy import ageplot


class AGEScanViewer(AGEDataViewer):
    """Show all spectra in a scan.

    """

    def __init__(self, scan, bins: int = 512) -> None:
        super().__init__()
        # Add plot to canvas
        self.add_plot()
        # Add the toolbar
        self.add_toolbar()
        # Add forward and backward buttons
        self.add_forward_backward_action(self.plot_previous, self.plot_next)
        # Get the data
        self.y = []
        self.err = []
        _, self.x = np.histogram([], bins=bins, range=(0, 1))
        for step in scan.steps:
            y, err = scan.spectrum_at(step, self.x)
            self.y.append(y)
            self.err.append(err)
        # Remember current step
        self.step = 0
        # Plot the first step
        self.plot(self.step)
    
    def plot(self, step: int) -> None:
        with ageplot.context(["age", "dataviewer"]):
            self.ax.clear()
            self.ax.stairs(self.y[step], self.x)
            self.canvas.draw()

    def plot_previous(self) -> None:
        self.step -= 1
        if self.step < 0:
            self.step = 0
        self.plot(self.step)

    def plot_next(self) -> None:
        self.step += 1
        if self.step >= len(self.y):
            self.step = len(self.y) - 1
        self.plot(self.step)
