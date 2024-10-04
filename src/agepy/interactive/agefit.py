from __future__ import annotations
import inspect
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PyQt5.QtWidgets import QLayout, QHBoxLayout, QGridLayout, QVBoxLayout, QGroupBox, QComboBox, QLabel, QSlider, QLineEdit, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from numba_stats import norm
from jacobi import propagate

from agepy import ageplot
from agepy.interactive import AGEDataViewer, AGEpp


def norm_pdf(x, s, loc, scale):
    return s * norm.pdf(x, loc, scale)

def norm_cdf(x, s, loc, scale):
    return s * norm.cdf(x, loc, scale)


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


class AGEFitViewer(AGEDataViewer):
    """
    
    """
    models = {
        "gaussian": {"density": norm_pdf, "integral": norm_cdf},
    }

    def __init__(self, agefit: AGEFit, fig: Figure, ax: Axes):
        super().__init__(width=1600, height=800)
        self.agefit = agefit
        # Set a bigger font size
        font = QFont()
        font.setPointSize(14)
        self.setFont(font)
        # Add Fit View
        self.fit_view = QHBoxLayout()
        self.layout.addLayout(self.fit_view)
        # Add plot to canvas
        self.add_plot(fig=fig, ax=ax, layout=self.fit_view)
        self.ax.set_title("AGE Fit")
        self.ax.set_xlim(*self.ax.get_xlim())
        self.ax.set_ylim(*self.ax.get_ylim())
        self.pred_line = None
        self.pred_errband = None
        # Add Fit Parameter Layout
        self.fit_params = QVBoxLayout()
        self.fit_view.addLayout(self.fit_params)
        self.params = {}
        # Add the toolbar
        self.add_toolbar()
        # Add Fitting Setup Layout
        self.fit_setup = QHBoxLayout()
        self.layout.addLayout(self.fit_setup)
        # Add Setup options
        self.select_cost = QComboBox(self)
        self.select_cost.addItems(agefit.costs)
        self.fit_setup.addWidget(self.select_cost)
        self.select_cost.currentIndexChanged.connect(self.update_cost)
        self.update_cost()
        self.select_model = QComboBox(self)
        self.select_model.addItems(list(self.models.keys()))
        self.fit_setup.addWidget(self.select_model)
        self.select_model.currentIndexChanged.connect(self.update_model)
        self.update_model()
        # Add Fit Button
        self.fit_button = QPushButton("Minimize")
        self.fit_setup.addWidget(self.fit_button)
        self.fit_button.clicked.connect(self.fit)

    def update_cost(self):
        self.agefit.cost = self.select_cost.currentText()

    def update_model(self):
        self.init_params()
        self.agefit.model = self.models[self.select_model.currentText()]

    def init_params(self):
        # Clear the current parameters
        self.clear_params(self.fit_params)
        # Get the selected model
        name = self.select_model.currentText()
        model = self.models[name]["density"]
        params = inspect.signature(model).parameters
        # Get x limits
        xlim = self.ax.get_xlim()
        # Round xlim to the first nonzero decimal
        xlim = (round(xlim[0], 1), round(xlim[1], 1))
        skip_x = True
        for param_name, param in params.items():
            if skip_x:
                skip_x = False
                continue
            # Add group box
            group = QGroupBox(param_name)
            self.fit_params.addWidget(group)
            group_layout = QGridLayout()
            group.setLayout(group_layout)
            # Add line edit to display slider value
            edit_value = QLineEdit()
            # Add value slider
            slider = FloatSlider(Qt.Horizontal, line_edit=edit_value)
            # Add line edit for changing the limits
            edit_llimit = QLineEdit()
            edit_ulimit = QLineEdit()
            if param_name == "loc":
                slider.setMinimum(xlim[0])
                slider.setMaximum(xlim[1])
                edit_llimit.setText(str(xlim[0]))
                edit_ulimit.setText(str(xlim[1]))
                slider.setValue(xlim[0] + (xlim[1] - xlim[0]) * 0.5)
            elif param_name == "scale":
                slider.setMinimum((xlim[1] - xlim[0]) * 0.001)
                slider.setMaximum(xlim[1] - xlim[0] * 0.75)
                edit_llimit.setText(str((xlim[1] - xlim[0]) * 0.001))
                edit_ulimit.setText(str(xlim[1] - xlim[0]))
                slider.setValue((xlim[1] - xlim[0]) * 0.1)
            elif param_name in ["s", "b"]:
                slider.setMinimum(0)
                slider.setMaximum(10)
                edit_llimit.setText("0")
                edit_ulimit.setText("10")
                slider.setValue(5)
            else:
                slider.setMinimum(-10)
                slider.setMaximum(10)
                edit_llimit.setText("-10")
                edit_ulimit.setText("10")
                slider.setValue(0)
            edit_value.setText(str(slider.value()))
            group_layout.addWidget(slider, 0, 0)
            group_layout.addWidget(edit_value, 0, 1)
            group_layout.addWidget(edit_llimit, 1, 0)
            group_layout.addWidget(edit_ulimit, 1, 1)
            # Connect slider value change to update the label
            slider.valueChanged.connect(self.update_params)
            edit_value.returnPressed.connect(self.update_params)
            edit_llimit.returnPressed.connect(self.update_limits)
            edit_ulimit.returnPressed.connect(self.update_limits)
            # Save the parameter
            self.params[param_name] = [slider, edit_value, edit_llimit,
                                       edit_ulimit]
        self.fit_params.addStretch()

    def clear_params(self, layout: QLayout):
        self.params = {}
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                self.clear_params(item.layout())

    def get_params(self):
        params = []
        for param_name, widgets in self.params.items():
            slider, edit_value, edit_llimit, edit_ulimit = widgets
            if slider.value() != float(edit_value.text()):
                slider.setValue(float(edit_value.text()))
                params.append(float(edit_value.text()))
            params.append(slider.value())
        return params

    def get_limits(self):
        limits = []
        for param_name, widgets in self.params.items():
            slider, value_label, edit_llimit, edit_ulimit = widgets
            limits.append((float(edit_llimit.text()), float(edit_ulimit.text())))
        return limits

    def update_limits(self):
        for param_name, widgets in self.params.items():
            slider, edit_value, edit_llimit, edit_ulimit = widgets
            llimit = float(edit_llimit.text())
            ulimit = float(edit_ulimit.text())
            current_value = slider.value()
            slider.setMinimum(llimit)
            slider.setMaximum(ulimit)
            # Ensure the slider's value is within the new range
            slider.setValue(llimit)
            if current_value < llimit:
                slider.setValue(llimit)
                edit_value.setText(str(llimit))
            elif current_value > ulimit:
                slider.setValue(ulimit)
                edit_value.setText(str(ulimit))
            else:
                slider.setValue(current_value)

    def update_params(self, slider_value, params=None, cov=None):
        if self.pred_line is not None:
            self.pred_line.remove()
        if self.pred_errband is not None:
            self.pred_errband.remove()
        if params is None:
            params = self.get_params()
        else:
            par_names = list(self.params.keys())
            for i, par in enumerate(params):
                self.params[par_names[i]][0].setValue(par)
                self.params[par_names[i]][1].setText(str(par))
        model = self.agefit.model["density"]
        xlim = self.ax.get_xlim()
        x = np.linspace(xlim[0], xlim[1], 1000)
        if cov is None:
            y = model(x, *params)
        else:
            y, ycov = propagate(lambda p: model(x, *p), params, cov)
        with ageplot.context(["age", "dataviewer"]):
            # Draw the prediction
            self.pred_line,  = self.ax.plot(x, y, color=ageplot.colors[1])
            # Draw 1 sigma error band
            if cov is not None:
                yerr = np.sqrt(np.diag(ycov))
                self.ax.fill_between(x, y - yerr, y + yerr,
                    facecolor=ageplot.colors[1], alpha=0.5)
            self.canvas.draw()

    def fit(self):
        start = self.get_params()
        limits = self.get_limits()
        params, cov = self.agefit.fit(start, limits)
        self.update_params(params=params, cov=cov)

class AGEFit:
    def __init__(self, xdata, ydata, yerr=None):
        self.y = ydata
        self.yerr = yerr
        if len(xdata) == len(ydata):
            self.x = xdata
            self.xe = 0.5 * (xdata[1:] + xdata[:-1])
            self.xe = np.insert(
                self.xe, 0, xdata[0] - 0.5 * (xdata[1] - xdata[0]))
        elif len(xdata) == len(ydata) + 1:
            self.x = 0.5 * (xdata[1:] + xdata[:-1])
            self.xe = xdata
        self.model = None
        self.cost = None

    def interactive(self):
        with ageplot.context(["age", "dataviewer"]):
            fig, ax = plt.subplots()
            ax.errorbar(self.x, self.y, yerr=self.yerr, fmt="s",
                        color=ageplot.colors[0])
        app = AGEpp(AGEFitViewer, self, fig, ax)
        app.run()


class AGEIminuit(AGEFit):
    """

    """
    costs = ["LeastSquares"]

    def fit(self, start, limits):
        # Create the cost function
        if self.cost == "LeastSquares":
            cost = LeastSquares(self.model["density"], self.x, self.y,
                                self.yerr)
        # Create the minimizer
        m = Minuit(cost, *start)
        params = inspect.signature(self.model["density"]).parameters
        for i, param_name in enumerate(list(params.keys())):
            m.limits[param_name] = limits[i]
        # Perform the minimization
        m.migrad()
        # Get the fitted parameters
        params = np.array(m.values)
        cov = np.array(m.covariance)
        return params, cov
