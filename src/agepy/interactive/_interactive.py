from __future__ import annotations
from typing import TYPE_CHECKING
# Import importlib.resources for getting the icon paths
import importlib.resources as imrsrc
# Import PyQt6 modules
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLayout,
    QVBoxLayout,
    QWidget,
    QSlider
)
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import pyqtSignal
# Import matplotlib modules
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT
)
from matplotlib.widgets import RectangleSelector
from matplotlib.figure import Figure
# Internal agepy imports
from agepy import ageplot
# Import modules for type hinting
if TYPE_CHECKING:
    from typing import Union, Sequence
    from matplotlib.axes import Axes

__all__ = []


class FloatSlider(QSlider):
    floatValueChanged = pyqtSignal(float)

    def __init__(self, orientation, parent=None, line_edit=None):
        super().__init__(orientation, parent)
        super().setMinimum(0)
        super().setMaximum(1000)
        super().setValue(500)
        self._min = 0.0
        self._max = 1.0
        self._line_edit = line_edit
        self.valueChanged.connect(self._emit_float_value_changed)

    def _emit_float_value_changed(self, value):
        float_value = self._int_to_float(value)
        self._line_edit.setText(str(float_value))
        self.floatValueChanged.emit(float_value)

    def _int_to_float(self, value):
        return self._min + (value / 1000) * (self._max - self._min)

    def _float_to_int(self, value):
        return int((value - self._min) / (self._max - self._min) * 1000)

    def setMinimum(self, min_value):
        self._min = min_value

    def setMaximum(self, max_value):
        self._max = max_value

    def setValue(self, value):
        super().setValue(self._float_to_int(value))

    def value(self):
        return self._int_to_float(super().value())

    def setSliderPosition(self, value):
        super().setSliderPosition(self._float_to_int(value))


class AGEDataViewer(QMainWindow):
    """Minimal implementation of the AGE Data Viewer.
    Should be used as a base class for more complex viewers.

    """
    def __init__(self, width: int = 1200, height: int = 800, parent=None):
        super().__init__(parent)
        # Set up the PyQt window
        self.setWindowTitle("AGE Data Viewer")
        self.setGeometry(100, 100, width, height)
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        # Initialize attributes
        self.canvas = None
        self.toolbar = None

    def add_plot(
        self,
        fig: Figure = None,
        ax: Union[Axes, Sequence[Axes]] = None,
        layout: QLayout = None,
        width: int = 1200,
        height: int = 800
    ) -> None:
        # Draw with the agepy plotting style, but don't overwrite the
        # users rcParams
        with ageplot.context(["age", "dataviewer"]):
            # Create and add the canvas
            if fig is not None:
                self.canvas = FigureCanvasQTAgg(fig)
            else:
                self.canvas = FigureCanvasQTAgg(Figure())
            # Set fixed size for the canvas
            self.canvas.setFixedSize(width, height)
            if layout is None:
                self.layout.addWidget(self.canvas)
            else:
                layout.addWidget(self.canvas)
            # Create the axis
            if ax is not None:
                self.ax = ax
            else:
                self.ax = self.canvas.figure.add_subplot(111)
            # Draw the empty plot
            self.canvas.draw()

    def add_toolbar(self):
        # Add the toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.toolbar)

    def add_roi_action(self, callback: callable):
        # Add ROI button to toolbar
        with imrsrc.path("agepy.interactive.icons", "roi.svg") as ipath:
            roi = QAction(QIcon(str(ipath)), "Add ROI", self)
        roi.setCheckable(True)
        roi.triggered.connect(callback)
        actions = self.toolbar.actions()
        self.roi_button = self.toolbar.insertAction(actions[-1], roi)

    def add_rect_selector(
        self,
        ax: Axes,
        on_select: callable,
        interactive: bool = True
    ) -> None:
        # Add the action
        self.add_roi_action(self.toggle_selector)
        # Add ROI selector
        self.selector = RectangleSelector(
            ax, on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords="pixels",
            interactive=interactive,
            props={"linewidth": 0.83, "linestyle": "--", "fill": False},
            handle_props={"markersize": 0})
        self.selector.set_active(False)

    def toggle_selector(self):
        self.selector.set_active(not self.selector.active)

    def add_forward_backward_action(
        self,
        bw_callback: callable,
        fw_callback: callable
    ) -> None:
        actions = self.toolbar.actions()
        # Add backward step to toolbar
        with imrsrc.path("agepy.interactive.icons", "bw-step.svg") as ipath:
            bw = QAction(QIcon(str(ipath)), "Step Backward", self)
        bw.triggered.connect(bw_callback)
        self.bw = self.toolbar.insertAction(actions[-1], bw)
        # Add forward step to toolbar
        with imrsrc.path("agepy.interactive.icons", "fw-step.svg") as ipath:
            fw = QAction(QIcon(str(ipath)), "Step Forward", self)
        fw.triggered.connect(fw_callback)
        self.fw = self.toolbar.insertAction(actions[-1], fw)

    def add_lookup_action(self, callback: callable) -> None:
        actions = self.toolbar.actions()
        with imrsrc.path("agepy.interactive.icons", "search.svg") as ipath:
            lu = QAction(QIcon(str(ipath)), "Look Up", self)
        lu.triggered.connect(callback)
        self.lu = self.toolbar.insertAction(actions[-1], lu)


class AGEpp:
    def __init__(self, viewer: QMainWindow, *args, **kwargs):
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        self.viewer = viewer(*args, **kwargs)

    def run(self):
        self.viewer.show()
        return self.app.exec()
