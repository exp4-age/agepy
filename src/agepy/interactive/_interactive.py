from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from agepy import ageplot


class AGEDataViewer(QMainWindow):
    """Minimal implementation of the AGE Data Viewer.
    Should be used as a base class for more complex viewers.

    """
    def __init__(self):
        super().__init__()
        # Set up the PyQt window
        self.setWindowTitle("AGE Data Viewer")
        self.setGeometry(100, 100, 1200, 800)
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        # Matplotlib
        # Draw with the agepy plotting style, but don't overwrite the
        # users rcParams
        with ageplot.context(["agepy.ageplot.age",
                              "agepy.ageplot.dataviewer"]):
            # Create and add the canvas
            self.canvas = FigureCanvas(Figure())
            self.layout.addWidget(self.canvas)
            # Create the axis
            self.ax = self.canvas.figure.add_subplot(111)
            # Draw the empty plot
            self.canvas.draw()
        # Add the toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)


class AGEpp:
    def __init__(self, viewer: QMainWindow, *args, **kwargs):
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        self.viewer = viewer(*args, **kwargs)

    def run(self):
        self.viewer.show()
        return self.app.exec_()
