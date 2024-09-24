from __future__ import annotations
from typing import Union, Sequence
import importlib.resources as imrsrc
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QAction
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from agepy import ageplot


class AGEDataViewer(QMainWindow):
    """Minimal implementation of the AGE Data Viewer.
    Should be used as a base class for more complex viewers.

    """
    def __init__(self, width: int = 1200, height: int = 800):
        super().__init__()
        # Set up the PyQt window
        self.setWindowTitle("AGE Data Viewer")
        self.setGeometry(100, 100, width, height)
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        # Initialize attributes
        self.canvas = None
        self.toolbar = None

    def add_plot(self,
        fig: Figure = None,
        ax: Union[Axes, Sequence[Axes]] = None
    ) -> None:
        # Draw with the agepy plotting style, but don't overwrite the
        # users rcParams
        with ageplot.context(["age", "dataviewer"]):
            # Create and add the canvas
            if fig is not None:
                self.canvas = FigureCanvas(fig)
            else:
                self.canvas = FigureCanvas(Figure())
            self.layout.addWidget(self.canvas)
            # Create the axis
            if ax is not None:
                self.ax = ax
            else:
                self.ax = self.canvas.figure.add_subplot(111)
            # Draw the empty plot
            self.canvas.draw()

    def add_toolbar(self):
        # Add the toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)

    def add_roi_action(self, callback: callable):
        # Add ROI button to toolbar
        with imrsrc.path("agepy.interactive.icons", "roi.svg") as icon_path:
            roi = QAction(QIcon(str(icon_path)), "Add ROI", self)
        roi.setCheckable(True)
        roi.triggered.connect(callback)
        actions = self.toolbar.actions()
        self.roi_button = self.toolbar.insertAction(actions[-1], roi)

class AGEpp:
    def __init__(self, viewer: QMainWindow, *args, **kwargs):
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        self.viewer = viewer(*args, **kwargs)

    def run(self):
        self.viewer.show()
        return self.app.exec_()
