from __future__ import annotations
from importlib.resources import path
import inspect
from functools import wraps, partial
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.axes import Axes
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QLayout, QHBoxLayout, QGridLayout, QVBoxLayout, QGroupBox, QComboBox, QTextEdit, QSlider, QLineEdit, QPushButton
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
import numpy as np
from jacobi import propagate

from agepy import ageplot
from agepy.interactive import AGEDataViewer, AGEpp


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


class FitResultWindow(AGEDataViewer):
    """

    """

    def __init__(self, fit_result, parent):
        super().__init__(parent=parent)
        self.setWindowTitle("Fit Result")
        # Set a bigger font size
        font = QFont("Courier")
        font.setPointSize(14)
        self.setFont(font)
        # Create a QTextEdit widget
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        # Set the fit result text
        self.text_edit.setPlainText(fit_result)
        # Calculate the size of the text
        font_metrics = QFontMetrics(font)
        text_size = font_metrics.size(0, fit_result)
        text_width = text_size.width() + 50
        text_height = text_size.height() + 50
        # Set the window size based on the text size
        self.resize(text_width, text_height)
        # Add to the layout
        self.layout.addWidget(self.text_edit)
        # Move the window
        parent_rect = parent.geometry()
        self.move(parent_rect.topLeft() + QPoint(100, 100))


class AGEFitViewer(QMainWindow):
    """
    
    """

    def __init__(self,
        xdata: np.ndarray = None,
        ydata: np.ndarray = None,
        yerr: np.ndarray = None,
        data: np.ndarray = None,
        backend: str = None,
        cost: str = "LeastSquares",
        sig: str = "Gaussian",
        bkg: str = "Exponential",
        start: dict = {},
        parent: QMainWindow = None
    ) -> None:
        # Setup the QMainWindow
        super().__init__(parent=parent)
        with path("agepy.interactive.fitting", "agefit.ui") as ui_path:
            uic.loadUi(str(ui_path), self)
        # Save the data
        self.xdata = xdata
        self.ydata = ydata
        self.yerr = yerr
        self.data = data
        # Initialize attributes
        self.cost = cost
        self.sig = sig
        self.bkg = bkg
        self.params = start
        # Add matplotlib canvas
        with ageplot.context(["age", "dataviewer"]):
            # Create and add the canvas
            self.canvas = FigureCanvas(Figure())
            # Set fixed size for the canvas
            #self.canvas.setFixedSize(width, height)
            self.layoutFitView.addWidget(self.canvas)
            # Create the axis
            self.ax = self.canvas.figure.add_subplot(111)
            # Draw the plot
            _agefit = AGEFit(xdata, ydata, yerr, data)
            _agefit.plot_data(self.ax)
            self.ax.set_title("AGE Fit")
            self.ax.set_xlim(*self.ax.get_xlim())
            self.ax.set_ylim(*self.ax.get_ylim())
            self.canvas.draw()
        # Remember the last Line2D objects in order to remove them when
        # updating the prediction 
        self.pred_line = None
        self.pred_errband = None
        # Check for available backends
        self.backends = {}
        if backend is None or backend == "iminuit":
            try:
                from agepy.interactive.fitting.iminuit import AGEIminuit
                self.backends["iminuit"] = AGEIminuit
            except ImportError:
                pass
        # Add the available backends to the selection
        self.selectBackend.addItems(list(self.backends.keys()))
        # Set the selected backend
        if backend is not None and backend in self.backends:
            self.selectBackend.setCurrentText(backend)
        # Initialize the selected backend
        self.init_backend()
        # Connect the backend selection to the initialization
        self.selectBackend.currentIndexChanged.connect(self.init_backend)
        # Connect signals
        self.selectCost.currentIndexChanged.connect(self.change_cost)
        self.selectSignal.currentIndexChanged.connect(self.change_model)
        self.selectBackground.currentIndexChanged.connect(self.change_model)
        # Add Fit Button
        self.buttonFit.clicked.connect(self.fit)
        # Draw the initial model
        self.update_params(None)

    def init_backend(self):
        if len(self.backends) == 0:
            raise ImportError("No fitting backends available.")
        backend = self.selectBackend.currentText()
        self.backend = self.backends[backend](
            self.xdata, self.ydata, self.yerr, self.data)
        # Update the cost function selection
        available_costs = self.backend.costs
        self.selectCost.clear()
        self.selectCost.addItems(available_costs)
        if self.cost in available_costs:
            self.selectCost.setCurrentText(self.cost)
        #else:
        #    self.cost = self.selectCost.currentText()
        self.change_cost()
        # Update the model selection
        available_sig = self.backend.signals
        self.selectSignal.clear()
        self.selectSignal.addItems(available_sig)
        if self.sig in available_sig:
            self.selectSignal.setCurrentText(self.sig)
        #else:
        #    self.sig = self.selectSignal.currentText()
        available_bkg = self.backend.backgrounds
        self.selectBackground.clear()
        self.selectBackground.addItems(available_bkg)
        if self.bkg in available_bkg:
            self.selectBackground.setCurrentText(self.bkg)
        #else:
        #    self.bkg = self.selectBackground.currentText()
        self.change_model()

    def change_cost(self):
        self.cost = self.selectCost.currentText()

    def change_model(self):
        # Get the selected signal and background
        self.sig = self.selectSignal.currentText()
        self.bkg = self.selectBackground.currentText()
        # Initialize the model
        self.backend.init_model(self.sig, self.bkg)
        self.model = self.backend.model
        params = self.backend.params
        # Clear the current parameters
        self.clear_params(self.layoutParams)
        # Get x limits
        xlim = self.ax.get_xlim()
        xlim = (round(xlim[0], 1), round(xlim[1], 1))
        for par in params:
            # Add group box
            group = QGroupBox(par)
            self.layoutParams.addWidget(group)
            layoutGroup = QGridLayout()
            group.setLayout(layoutGroup)
            # Add line edit to display slider value
            editValue = QLineEdit()
            # Add value slider
            slider = FloatSlider(Qt.Horizontal, line_edit=editValue)
            # Add line edit for changing the limits
            editLLimit = QLineEdit()
            editULimit = QLineEdit()
            # Set the values
            if par == "loc":
                val = 0.5 * (xlim[0] + xlim[1])
                llimit, ulimit = xlim
            elif par == "scale":
                val = 0.5 * (xlim[1] - xlim[0])
                llimit = 0.0001 * val
                ulimit = 2 * val
            else:
                val, llimit, ulimit = (0, -1000, 1000)
            if par not in self.params:
                self.params[par] = {"val": val, "limits": (llimit, ulimit)}
            elif "val" in self.params[par] and "limits" in self.params[par]:
                val = self.params[par]["val"]
                llimit, ulimit = self.params[par]["limits"]
            slider.setMinimum(llimit)
            slider.setMaximum(ulimit)
            slider.setValue(val)
            editValue.setText(str(val))
            editLLimit.setText(str(llimit))
            editULimit.setText(str(ulimit))
            # Add widgets to the layout
            layoutGroup.addWidget(slider, 0, 0)
            layoutGroup.addWidget(editValue, 0, 1)
            layoutGroup.addWidget(editLLimit, 1, 0)
            layoutGroup.addWidget(editULimit, 1, 1)
            # Connect slider value change to update the label
            slider.valueChanged.connect(self.update_params)
            editValue.returnPressed.connect(self.update_params)
            #update_limits = partial(self.update_limits)
            editLLimit.returnPressed.connect(self.update_limits)
            editULimit.returnPressed.connect(self.update_limits)
            # Save the parameter
            self.params[par]["qt"] = [slider, editValue, editLLimit, editULimit]
        self.layoutParams.addStretch()

    def clear_params(self, layout: QLayout):
        self.params = {}
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                child_layout = item.layout()
                if child_layout is not None:
                    self.clear_params(child_layout)

    def get_params(self):
        return [self.params[par]["val"] for par in self.params]

    def get_limits(self):
        limits = {}
        for par in self.params:
            limits[par] = self.params[par]["limits"]
        return limits

    def update_limits(self):
        for par in self.params:
            slider, editValue, editLLimit, editULimit = self.params[par]["qt"]
            # Get the limits
            llimit = float(editLLimit.text())
            ulimit = float(editULimit.text())
            # Update the limits
            self.params[par]["limits"] = (llimit, ulimit)
            # Adjust the slider limits
            current_value = slider.value()
            slider.setMinimum(llimit)
            slider.setMaximum(ulimit)
            # Ensure the slider's value is within the new range
            slider.blockSignals(True)
            slider.setValue(llimit)
            if current_value < llimit:
                slider.setValue(llimit)
                editValue.setText(str(llimit))
            elif current_value > ulimit:
                slider.setValue(ulimit)
                editValue.setText(str(ulimit))
            else:
                slider.setValue(current_value)
            slider.blockSignals(False)

    def update_params(self, slider_value=None, params=None, cov=None):
        if self.pred_line is not None:
            self.pred_line.remove()
            self.pred_line = None
        if self.pred_errband is not None:
            self.pred_errband.remove()
            self.pred_errband = None
        if params is None:
            params = self.get_params()
        else:
            par = list(self.params.keys())
            for i, val in enumerate(params):
                # Update QT widgets
                self.params[par[i]]["qt"][0].blockSignals(True)
                self.params[par[i]]["qt"][0].setValue(val)
                self.params[par[i]]["qt"][0].blockSignals(False)
                self.params[par[i]]["qt"][1].setText(str(val))
                # Update the stored value
                self.params[par[i]]["val"] = val
        xlim = self.ax.get_xlim()
        x = np.linspace(xlim[0], xlim[1], 1000)
        if cov is None:
            y = self.model(x, *params)
        else:
            y, ycov = propagate(lambda p: self.model(x, *p), params, cov)
        with ageplot.context(["age", "dataviewer"]):
            # Draw the prediction
            self.pred_line, = self.ax.plot(x, y, color=ageplot.colors[1])
            # Draw 1 sigma error band
            if cov is not None:
                yerr = np.sqrt(np.diag(ycov))
                self.pred_errband = self.ax.fill_between(x, y - yerr, y + yerr,
                    facecolor=ageplot.colors[1], alpha=0.5)
            self.canvas.draw()

    def fit(self):
        start = self.get_params()
        limits = self.get_limits()
        params, cov, res = self.agefit.fit(start, limits)
        self.update_params(params=params, cov=cov)
        res_window = FitResultWindow(res, self)
        res_window.show()


class AGEFit:
    """
    
    """

    def __init__(self, xdata=None, ydata=None, yerr=None, data=None):
        self.data = data
        self.y = ydata
        self.yerr = yerr
        if xdata is None:
            self.xe = None
            self.x = None
        elif len(xdata) == len(ydata):
            self.x = xdata
            self.xe = 0.5 * (xdata[1:] + xdata[:-1])
            self.xe = np.insert(
                self.xe, 0, xdata[0] - 0.5 * (xdata[1] - xdata[0]))
        elif len(xdata) == len(ydata) + 1:
            self.x = 0.5 * (xdata[1:] + xdata[:-1])
            self.xe = xdata
        self.result = None
        self.model = None
        self.cost = None

    def plot_data(self, ax: Axes):
        if self.x is None:
            return
        if self.y is not None:
            ax.errorbar(self.x, self.y, yerr=self.yerr, fmt="s",
                        color=ageplot.colors[0])
        elif self.data is not None:
            n, = np.histogram(self.data, bins=self.xe)
            ax.errorbar(self.x, n, yerr=np.sqrt(n), fmt="s",
                        color=ageplot.colors[0])


def agefit(
    xdata: np.ndarray = None,
    ydata: np.ndarray = None,
    yerr: np.ndarray = None,
    data: np.ndarray = None,
    backend: str = None,
    cost: str = "LeastSquares",
    sig: str = "Gaussian",
    bkg: str = "Exponential",
    start: dict = {},
    parent: QMainWindow = None
) -> None:
    """
    
    """
    kwargs = {
        "xdata": xdata,
        "ydata": ydata,
        "yerr": yerr,
        "data": data,
        "backend": backend,
        "cost": cost,
        "sig": sig,
        "bkg": bkg,
        "start": start,
        "parent": parent
    }
    if parent is None:
        app = AGEpp(AGEFitViewer, **kwargs)
        app.run()
    else:
        fitviewer = AGEFitViewer(**kwargs)
        fitviewer.show()

