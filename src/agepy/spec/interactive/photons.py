from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget,QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np

from agepy.interactive import AGEDataViewer
#from agepy.spec.photons import Scan



class AGEScanViewer(AGEDataViewer):
    """Show all spectra in a scan.
    
    """
    def __init__(self, scan) -> None:
        super().__init__()
        # Get the data
        self.y = []
        _, self.x = np.histogram([], bins=512, range=(0, 1))
        for step in scan.steps:
            y, err = scan.spectrum_at(step, self.x)
            self.y.append(y)
        # Remember current step
        self.step = 0
        # Add previos and next buttons
        self.prev = QPushButton("Previous")
        self.prev.clicked.connect(self.plot_previous)
        self.layout.addWidget(self.prev)
        self.next = QPushButton("Next")
        self.next.clicked.connect(self.plot_next)
        self.layout.addWidget(self.next)
        # Plot the first step
        self.plot(self.step)
    
    def plot(self, step: int) -> None:
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