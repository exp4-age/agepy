from __future__ import annotations
import inspect
from functools import wraps
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PyQt5.QtWidgets import QLayout, QHBoxLayout, QGridLayout, QVBoxLayout, QGroupBox, QComboBox, QTextEdit, QSlider, QLineEdit, QPushButton
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from numba_stats import truncnorm, truncexpon, bernstein
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
        super().__init__(width=1600, height=800, parent=parent)
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


class AGEFitViewer(AGEDataViewer):
    """
    
    """

    def __init__(self, agefit: AGEFit, fig: Figure, ax: Axes):
        super().__init__(width=1650, height=800)
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
        cost_group = QGroupBox("Cost Function")
        self.fit_setup.addWidget(cost_group)
        cost_layout = QHBoxLayout()
        cost_group.setLayout(cost_layout)
        self.select_cost = QComboBox(self)
        self.select_cost.addItems(agefit.costs)
        cost_layout.addWidget(self.select_cost)
        self.select_cost.currentIndexChanged.connect(self.update_cost)
        self.update_cost()
        model_group = QGroupBox("Model (Signal + Background)")
        self.fit_setup.addWidget(model_group)
        model_layout = QHBoxLayout()
        model_group.setLayout(model_layout)
        self.select_signal = QComboBox(self)
        self.select_signal.addItems(self.agefit.signals)
        model_layout.addWidget(self.select_signal)
        self.select_signal.currentIndexChanged.connect(self.update_model)
        self.select_bkg = QComboBox(self)
        self.select_bkg.addItems(self.agefit.backgrounds)
        model_layout.addWidget(self.select_bkg)
        self.select_bkg.currentIndexChanged.connect(self.update_model)
        self.update_model()
        # Add Fit Button
        fit_group = QGroupBox("Fit")
        self.fit_setup.addWidget(fit_group)
        fit_layout = QHBoxLayout()
        fit_group.setLayout(fit_layout)
        self.fit_button = QPushButton("Minimize")
        fit_layout.addWidget(self.fit_button)
        self.fit_button.clicked.connect(self.fit)
        # Draw the initial model
        self.update_params(None)

    def update_cost(self):
        self.agefit.cost = self.select_cost.currentText()

    def update_model(self):
        self.agefit.init_model(self.select_signal.currentText(),
                               self.select_bkg.currentText())
        self.init_params()

    def init_params(self):
        # Clear the current parameters
        self.clear_params(self.fit_params)
        # Get the selected model
        model = self.agefit.model
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
            edit_value.setAlignment(Qt.AlignLeft)
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
                slider.setMaximum(1000)
                edit_llimit.setText("0")
                edit_ulimit.setText("1000")
                slider.setValue(10)
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
                child_layout = item.layout()
                if child_layout is not None:
                    self.clear_params(child_layout)

    def get_params(self):
        params = []
        for param_name, widgets in self.params.items():
            slider, edit_value, edit_llimit, edit_ulimit = widgets
            params.append(float(edit_value.text()))
        return params

    def get_limits(self):
        limits = {}
        for param_name, widgets in self.params.items():
            slider, value_label, edit_llimit, edit_ulimit = widgets
            limits[param_name] = (float(edit_llimit.text()),
                                  float(edit_ulimit.text()))
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
            slider.blockSignals(True)
            slider.setValue(llimit)
            if current_value < llimit:
                slider.setValue(llimit)
                edit_value.setText(str(llimit))
            elif current_value > ulimit:
                slider.setValue(ulimit)
                edit_value.setText(str(ulimit))
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
            par_names = list(self.params.keys())
            for i, par in enumerate(params):
                self.params[par_names[i]][0].blockSignals(True)
                self.params[par_names[i]][0].setValue(par)
                self.params[par_names[i]][0].blockSignals(False)
                self.params[par_names[i]][1].setText(str(par))
        model = self.agefit.model
        xlim = self.ax.get_xlim()
        x = np.linspace(xlim[0], xlim[1], 1000)
        if cov is None:
            y = model(x, *params)
        else:
            y, ycov = propagate(lambda p: model(x, *p), params, cov)
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
    backgrounds = ["Bernstein1d", "Bernstein2d", "Bernstein3d", "Bernstein4d",
                   "Exponential"]
    signals = ["Gaussian"]

    def init_model(self, sig, bkg):
        sig, par_sig = self.init_sig(sig)
        bkg, par_bkg = self.init_bkg(bkg)
        combined_params = ["x"]
        combined_params.extend(par_sig)
        combined_params.extend(par_bkg)
        # Create the parameters for the function signature
        parameters = [
            inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for arg in combined_params]
        # Create the signature object
        func_signature = inspect.Signature(parameters)
        # Define the model function
        def model(x, *args):
            return sig(x, *args[:len(par_sig)]) + bkg(x, *args[len(par_sig):])
        # Set the new signature to the model function
        model.__signature__ = func_signature
        self.model = model

    def init_sig(self, sig):
        if sig == "Gaussian":
            par_names = ["s", "loc", "scale"]
            def sig_model(x, s, loc, scale):
                return s * truncnorm.pdf(
                    x, self.xe[0], self.xe[-1], loc, scale)
            return sig_model, par_names

    def init_bkg(self, bkg):
        if "Bernstein" in bkg:
            deg = int(bkg[-2])
            par_names = ["a{}".format(i) for i in range(deg+1)]
            def bkg_model(x, *args):
                return bernstein.density(x, args, self.xe[0], self.xe[-1])
            return bkg_model, par_names
        elif bkg == "Exponential":
            par_names = ["b", "loc_expon", "scale_expon"]
            def bkg_model(x, b, loc, scale):
                return b * truncexpon.pdf(
                    x, self.xe[0], self.xe[-1], loc, scale)
            return bkg_model, par_names

    def fit(self, start, limits):
        # Create the cost function
        if self.cost == "LeastSquares":
            cost = LeastSquares(self.x, self.y, self.yerr, self.model)
        # Create the minimizer
        m = Minuit(cost, *start)
        for par, limit in limits.items():
            m.limits[par] = limit
        # Perform the minimization
        m.migrad()
        # Get the fitted parameters
        params = np.array(m.values)
        cov = np.array(m.covariance)
        return params, cov, m.__str__()
