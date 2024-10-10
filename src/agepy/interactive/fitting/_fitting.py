from __future__ import annotations
from importlib.resources import path
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QLayout, QHBoxLayout, QGridLayout, QGroupBox, QComboBox, QSlider, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal

from agepy import ageplot


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


class AGEFitViewer(QMainWindow):
    """
    
    """

    def __init__(self,
        backend,
        parent: QMainWindow = None
    ) -> None:
        # Setup the QMainWindow
        super().__init__(parent=parent)
        with path("agepy.interactive.fitting", "agefit.ui") as ui_path:
            uic.loadUi(str(ui_path), self)
        # Set the fitting backend
        self.backend = backend
        # Add matplotlib canvas
        with ageplot.context(["age", "dataviewer"]):
            # Create and add the canvas
            self.canvas = FigureCanvas(Figure())
            self.layoutFitView.addWidget(self.canvas)
            # Create the axis
            self.ax = self.canvas.figure.add_subplot(111)
            # Draw the plot
            self.backend.plot_data(self.ax)
            self.ax.set_title("AGE Fit")
            self.ax.set_xlim(*self.backend.xr)
            self.ax.set_ylim(*self.ax.get_ylim())
            self.canvas.draw()
        # Remember the last Line2D objects in order to remove them when
        # updating the prediction 
        self.mpl_lines = []
        # Initialize the cost function selection
        self._init_cost_selection()
        # Initialize the model selection
        self._init_model_selection()
        # Add Fit Button
        group = QGroupBox("Fit")
        self.layoutFitSetup.addWidget(group)
        layoutGroup = QHBoxLayout()
        group.setLayout(layoutGroup)
        buttonFit = QPushButton("Minimize", self)
        layoutGroup.addWidget(buttonFit)
        buttonFit.clicked.connect(self.fit)
        # Add the result text box
        self.textResults.setPlainText(self.backend.print_result())
        # Draw the initial model
        self.update_prediction()

    def _init_cost_selection(self):
        group = QGroupBox("Cost Function")
        self.layoutFitSetup.addWidget(group)
        layoutGroup = QHBoxLayout()
        group.setLayout(layoutGroup)
        self.selectCost = QComboBox()
        layoutGroup.addWidget(self.selectCost)
        # Get the available cost functions
        available_costs = self.backend.list_costs()
        self.selectCost.addItems(available_costs)
        self.selectCost.setCurrentText(self.backend.which_cost())
        self.selectCost.currentTextChanged.connect(self.change_cost)

    def _init_model_selection(self):
        group = QGroupBox("Model")
        self.layoutFitSetup.addWidget(group)
        layoutGroup = QHBoxLayout()
        group.setLayout(layoutGroup)
        # Get the available fit models
        available_models = self.backend.list_models()
        if isinstance(available_models, list):
            available_models = {"models": available_models}
        # Get the selected model
        selected_model = self.backend.which_model()
        if isinstance(selected_model, str):
            selected_model = {"models": selected_model}
        # Add the model selection
        self.selectModel = {}
        for model_type in available_models:
            selectModel = QComboBox()
            layoutGroup.addWidget(selectModel)
            selectModel.addItems(available_models[model_type])
            selectModel.setCurrentText(selected_model[model_type])
            selectModel.currentTextChanged.connect(self.change_model)
            self.selectModel[model_type] = selectModel
        self.change_model()

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
            val = params[par]
            llimit, ulimit = limits[par]
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
            # Connect signals
            slider.valueChanged.connect(self.update_backend_params)
            editValue.returnPressed.connect(self.update_backend_params)
            editLLimit.returnPressed.connect(self.update_limits)
            editULimit.returnPressed.connect(self.update_limits)
            # Save the parameter
            self.params[par] = [slider, editValue, editLLimit, editULimit]
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
            slider, editValue, editLLimit, editULimit = self.params[par]
            # Get the limits
            llimit = float(editLLimit.text())
            ulimit = float(editULimit.text())
            # Update the limits
            self.backend.change_limit(par, (llimit, ulimit))
            # Adjust the slider limits
            current_value = self.backend.value(par)
            slider.setMinimum(llimit)
            slider.setMaximum(ulimit)
            # Ensure the slider's value is within the new range
            slider.blockSignals(True)
            slider.setValue(llimit)
            if current_value < llimit:
                slider.setValue(llimit)
                editValue.setText(str(llimit))
                self.update_backend_params()
            elif current_value > ulimit:
                slider.setValue(ulimit)
                editValue.setText(str(ulimit))
                self.update_backend_params()
            else:
                slider.setValue(current_value)
            slider.blockSignals(False)

    def update_backend_params(self, slider_value=None):
        for par in self.params:
            editValue = self.params[par][1]
            self.backend.change_value(par, float(editValue.text()))
        self.update_prediction()

    def update_gui_params(self):
        params = self.backend.list_params()
        for par in params:
            slider, editValue, _, _ = self.params[par]
            slider.blockSignals(True)
            slider.setValue(params[par])
            editValue.setText(str(params[par]))
            slider.blockSignals(False)

    def update_prediction(self):
        # Remove the previous prediction
        for line in self.mpl_lines:
            line.remove()
        # Plot the new prediction
        with ageplot.context(["age", "dataviewer"]):
            self.mpl_lines = self.backend.plot_prediction(self.ax)
            self.canvas.draw()

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
