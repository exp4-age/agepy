from PyQt6.QtWidgets import QSlider
from PyQt6.QtCore import pyqtSignal


class Ui_FloatSlider(QSlider):
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
