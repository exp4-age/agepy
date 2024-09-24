from __future__ import annotations
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np

from agepy.interactive import AGEDataViewer
from agepy import ageplot


class AGECoincViewer(QMainWindow):
    """Show all spectra in a scan.

    """

    def __init__(self, fig, axes):
        super().__init__()
        # Set up the PyQt window
        self.setWindowTitle("AGE Data Viewer")
        self.setGeometry(100, 100, 800, 800)
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        # Matplotlib
        # Draw with the agepy plotting style, but don't overwrite the
        # users rcParams
        with ageplot.context(["age", "dataviewer"]):
            # Create and add the canvas
            self.canvas = FigureCanvas(fig)
            self.layout.addWidget(self.canvas)
            # Create the axis
            self.ax = axes
            # Draw the empty plot
            self.canvas.draw()
        # Add the toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
