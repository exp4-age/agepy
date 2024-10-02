from __future__ import annotations
import inspect
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PyQt5.QtWidgets import QLayout, QHBoxLayout, QGridLayout, QVBoxLayout, QComboBox, QLabel, QSlider
from PyQt5.QtCore import Qt
from numba_stats import norm


from agepy.interactive import AGEDataViewer


class AGEFitViewer(AGEDataViewer):
    """
    
    """
    models = {
        "gaussian": {"density": norm.pdf, "integral": norm.cdf},
    }

    def __init__(self, fig: Figure, ax: Axes):
        super().__init__()
        # Add Fit Parameter Layout
        self.fit_view = QHBoxLayout()
        self.layout.addLayout(self.fit_view)
        self.fit_params = QGridLayout()
        self.fit_view.addLayout(self.fit_params)
        self.params = {}
        # Add plot to canvas
        self.add_plot(fig=fig, ax=ax, layout=self.fit_view)
        self.pred_line = None
        self.pred_errband = None
        # Add the toolbar
        self.add_toolbar()
        # Add Fitting Setup Layout
        self.fit_setup = QHBoxLayout()
        self.layout.addLayout(self.fit_setup)
        # Add Setup options
        self.select_lib = QComboBox(self)
        self.select_lib.addItems(["iminuit", "lmfit"])
        self.fit_setup.addWidget(self.select_lib)
        self.select_cost = QComboBox(self)
        self.select_cost.addItems(["No Fitting Library Selected"])
        self.fit_setup.addWidget(self.select_cost)
        self.select_model = QComboBox(self)
        self.select_model.addItems(list(self.models.keys()))
        self.fit_setup.addWidget(self.select_model)
        # 

    def update_params(self):
        # Clear the current parameters
        self.clear_params(self.fit_params)
        # Get the selected model
        name = self.select_model.currentText()
        model = self.models[name]["density"]
        params = inspect.signature(model).parameters
        row = 0
        for param_name, param in params.items():
            # Display the parameter name
            label = QLabel(param_name)
            self.fit_params.addWidget(label, row, 0)
            # Add value slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-10)
            slider.setMaximum(10)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(1)
            self.fit_params.addWidget(slider, row, 1)
            # Add label to display slider value
            value_label = QLabel("0")  # Default value, you can adjust this
            self.fit_params.addWidget(value_label, row, 2)
            # Connect slider value change to update the label
            slider.valueChanged.connect(lambda value, lbl=value_label: lbl.setText(str(value)))
            # Save the parameter
            self.params[param_name] = slider
            # Increment row
            row += 1

    def clear_params(self, layout: QLayout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                self.clear_params(item.layout())

    def update_prediction(self):
        if self.pred_line is not None:
            self.pred_line.remove()
        if self.pred_errband is not None:
            self.pred_errband.remove()


class AGEFit:
    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata
