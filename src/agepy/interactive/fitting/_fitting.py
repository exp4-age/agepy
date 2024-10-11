from __future__ import annotations
from typing import Sequence, Tuple, Union, Dict
from importlib.resources import path
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QLayout, QGridLayout, QGroupBox, QComboBox, QLineEdit
from PyQt6.QtCore import Qt

from agepy import ageplot
from agepy.interactive import FloatSlider


class ParamBox(QGroupBox):
    """
    
    """

    def __init__(self, parameter: str, parent=None) -> None:
        super().__init__(parameter, parent=parent)
        # Add layout
        layout = QGridLayout()
        self.setLayout(layout)
        # Add line edit to display slider value
        self.editValue = QLineEdit()
        # Add value slider
        self.slider = FloatSlider(Qt.Orientation.Horizontal,
                                  line_edit=self.editValue)
        # Add line edit for changing the limits
        self.editLLimit = QLineEdit()
        self.editULimit = QLineEdit()
        # Add widgets to the layout
        layout.addWidget(self.slider, 0, 0)
        layout.addWidget(self.editValue, 0, 1)
        layout.addWidget(self.editLLimit, 1, 0)
        layout.addWidget(self.editULimit, 1, 1)

    def connect_limits(self, callback):
        self.editLLimit.returnPressed.connect(callback)
        self.editULimit.returnPressed.connect(callback)

    def connect_value(self, callback):
        self.slider.floatValueChanged.connect(callback)
        self.editValue.returnPressed.connect(callback)

    def set_limits(self, llimit: float, ulimit: float):
        self.slider.setMinimum(llimit)
        self.slider.setMaximum(ulimit)
        self.editLLimit.setText(str(llimit))
        self.editULimit.setText(str(ulimit))
        # Ensure the slider's value is within the new range
        current_value = self.get_value()
        if current_value < llimit:
            self.slider.setValue(llimit)
            self.editValue.setText(str(llimit))
        elif current_value > ulimit:
            self.slider.setValue(ulimit)
            self.editValue.setText(str(ulimit))
        else:
            self.slider.blockSignals(True)
            self.slider.setValue(llimit)
            self.slider.setValue(current_value)
            self.slider.blockSignals(False)

    def set_value(self, value: float):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.editValue.setText(str(value))
        self.slider.blockSignals(False)

    def get_limits(self) -> tuple[float, float]:
        return float(self.editLLimit.text()), float(self.editULimit.text())

    def get_value(self) -> float:
        return float(self.editValue.text())


class AGEFitViewer(QMainWindow):
    """
    
    """

    def __init__(self,
        backend,
        parent: QMainWindow = None
    ) -> None:
        # Setup the QMainWindow
        super().__init__(parent=parent)
        with path("agepy.interactive.fitting", "fitting.ui") as ui_path:
            uic.loadUi(str(ui_path), self)
        # Set the fitting backend
        self.backend = backend
        # Add matplotlib canvas
        #matplotlib.use("Qt5Agg")
        with ageplot.context(["age", "dataviewer"]):
            # Create and add the canvas
            self.canvas = FigureCanvas(Figure())
            self.layoutFitView.addWidget(self.canvas)
            # Create the axis
            self.ax = self.canvas.figure.add_subplot(111)
            # Draw the plot
            self.backend.plot_data(self.ax)
            self.ax.set_xlim(*self.ax.get_xlim())
            self.ax.set_ylim(*self.ax.get_ylim())
            self.canvas.draw()
        # Remember the last Line2D objects in order to remove them when
        # updating the prediction 
        self.mpl_lines = []
        # Initialize the cost function selection
        available_costs = self.backend.list_costs()
        self.selectCost.addItems(available_costs)
        self.selectCost.setCurrentText(self.backend.which_cost())
        self.selectCost.currentTextChanged.connect(self.change_cost)
        # Initialize the model selection
        available_models = self.backend.list_models()
        if isinstance(available_models, list):
            available_models = {"model": available_models}
        # Get the selected model
        selected_model = self.backend.which_model()
        if isinstance(selected_model, str):
            selected_model = {"model": selected_model}
        # Add the model selection
        self.selectModel = {}
        for model_type in available_models:
            selectModel = QComboBox()
            self.layoutModel.addWidget(selectModel)
            selectModel.addItems(available_models[model_type])
            selectModel.setCurrentText(selected_model[model_type])
            selectModel.currentTextChanged.connect(self.change_model)
            self.selectModel[model_type] = selectModel
        self.change_model()
        # Connect Fit Button
        self.buttonFit.clicked.connect(self.fit)
        # Add the result text box
        self.textResults.setPlainText(self.backend.print_result())
        # Draw the initial model
        self.update_prediction()

    def change_cost(self):
        self.backend.select_cost(self.selectCost.currentText())

    def change_model(self):
        # Get the selected models
        selected_model = {}
        for model_type in self.selectModel:
            selected_model[model_type] = self.selectModel[
                model_type].currentText()
        # Pass the selected models to the backend
        self.backend.select_model(**selected_model)
        # Get the new parameters
        params = self.backend.list_params()
        limits = self.backend.list_limits()
        # Clear the current parameters
        self.clear_params(self.layoutParams)
        # Add the new parameters
        self.params = {}
        for par in params:
            # Add group box
            parambox = ParamBox(par)
            self.layoutParams.addWidget(parambox)
            # Set the values
            val = params[par]
            llimit, ulimit = limits[par]
            parambox.set_value(val)
            parambox.set_limits(llimit, ulimit)
            # Connect signals
            parambox.connect_limits(self.update_limits)
            parambox.connect_value(self.update_backend_params)
            # Save the parameter
            self.params[par] = parambox
        self.layoutParams.addStretch()
        self.update_prediction()

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

    def update_limits(self):
        for par in self.params:
            parambox = self.params[par]
            # Get the limits
            llimit, ulimit = parambox.get_limits()
            # Update the limits
            self.backend.set_limits(par, (llimit, ulimit))
            # Adjust the slider limits
            parambox.set_limits(llimit, ulimit)

    def update_backend_params(self, slider_value=None):
        for par in self.params:
            self.backend.set_value(par, self.params[par].get_value())
        self.update_prediction()

    def update_gui_params(self):
        params = self.backend.list_params()
        for par in params:
            self.params[par].set_value(params[par]) 

    def update_prediction(self):
        # Remove the previous prediction
        for line in self.mpl_lines:
            line.remove()
        # Plot the new prediction
        with ageplot.context(["age", "dataviewer"]):
            self.mpl_lines = self.backend.plot_prediction(self.ax)
            self.canvas.draw_idle()

    def fit(self):
        # Perform the fit
        self.backend.fit()
        # Show the fit result
        self.textResults.clear()
        self.textResults.setPlainText(self.backend.print_result())
        # Update the displayed parameters
        self.update_gui_params()
        # Update the prediction
        self.update_prediction()


class AGEFit:
    """

    """

    def __init__(self) -> None:
        self.select_model("Not implemented!")
        self.select_cost("Not implemented!")

    def list_costs(self) -> Sequence[str]:
        return ["Not implemented!"]

    def which_cost(self) -> str:
        return self._cost_name

    def select_cost(self, name: str) -> None:
        self._cost_name = name

    def list_models(self) -> Union[Sequence[str], Dict[str, Sequence[str]]]:
        return ["Not implemented!"]

    def which_model(self) -> Union[str, Dict[str, str]]:
        return self._model_name

    def select_model(self, model: str) -> None:
        self._model_name = model
        self._params = {"Not implemented!": 0}
        self._limits = {"Not implemented!": (-1, 1)}

    def list_params(self) -> Dict[str, float]:
        return self._params

    def list_limits(self) -> Dict[str, Tuple[float, float]]:
        return self._limits

    def value(self, name: str) -> float:
        if name not in self.list_params():
            raise ValueError(f"Parameter {name} not available.")
        return self._params[name]

    def set_value(self, name: str, value: float) -> None:
        if name not in self.list_params():
            raise ValueError(f"Parameter {name} not available.")
        self._params[name] = value
        self._cov = None

    def set_limits(self, name: str, limits: Tuple[float, float]) -> None:
        if name not in self.list_params():
            raise ValueError(f"Parameter {name} not available.")
        self._limits[name] = limits

    def fit(self) -> None:
        pass

    def plot_data(self, ax: Axes) -> None:
        ax.set_title("AGE Fit")

    def plot_prediction(self, ax: Axes) -> Sequence[Line2D]:
        return []

    def print_result(self) -> str:
        return "Not implemented!"
