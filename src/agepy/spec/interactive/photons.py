from __future__ import annotations
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QStyle
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np

from agepy.interactive import AGEDataViewer



class AGEScanViewer(AGEDataViewer):
    """Show all spectra in a scan.

    """

    def __init__(self, scan, bins: int = 512) -> None:
        super().__init__()
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
        # Add previous and next buttons
        self.prev = QPushButton()
        self.prev.clicked.connect(self.plot_previous)
        icon_prev = self.style().standardIcon(QStyle.SP_ArrowBack)
        self.prev.setIcon(icon_prev)
        self.toolbar.addWidget(self.prev)
        self.next = QPushButton()
        self.next.clicked.connect(self.plot_next)
        icon_next = self.style().standardIcon(QStyle.SP_ArrowForward)
        self.next.setIcon(icon_next)
        self.toolbar.addWidget(self.next)
        # Plot the first step
        self.plot(self.step)
    
    def plot(self, step: int) -> None:
        with plt.style.context(["agepy.ageplot.age",
                                "agepy.ageplot.dataviewer"]):
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