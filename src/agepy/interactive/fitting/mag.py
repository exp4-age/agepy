from __future__ import annotations
from typing import List, Sequence
from functools import partial
from PyQt6.QtWidgets import QMainWindow
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

import numpy as np
from lmfit.models import LinearModel, QuadraticModel, PolynomialModel, GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel, ExponentialModel, PowerLawModel, StepModel, RectangleModel, ExpressionModel
from lmfit import Model, Parameter

from agepy import ageplot
from agepy.interactive import AGEpp
from agepy.interactive.fitting import AGEFitViewer, AGEFit


class MagFit(AGEFit):
    """

    """
    _models = {
        "linear": LinearModel,
        "quadratic": QuadraticModel,
        "polynomial": PolynomialModel,
        "gaussian": GaussianModel,
        "lorentzian": LorentzianModel,
        "voigt": VoigtModel,
        "pseudo_voigt": PseudoVoigtModel,
        "exponential": ExponentialModel,
        "power_law": PowerLawModel,
        "step": StepModel,
        "rectangle": RectangleModel,
        "expression": ExpressionModel
    }

    def __init__(self,
        x: np.ndarray,
        y: np.ndarray,
        model: str = "linear"
    ) -> None:
        # Handle the data
        self.x = x
        self.y = y
        # Initialize the fit
        self._cost_name = "LeastSquares"
        self.select_model(model)
        self._err = None
        self._result = None

    def list_costs(self) -> list[str]:
        return ["LeastSquares"]

    def list_models(self) -> Sequence[str]:
        return list(self._models.keys())

    def select_model(self, model) -> None:
        self._model_name = model
        self._model = self._models[model]()
        self._params, self._limits = {}, {}
        for name, param in self._model.make_params().items():
            self._params[name] = param.value
            self._limits[name] = (-1000, 1000)
        self._err = None
        self._result = None

    def fit(self) -> None:
        params = self._model.guess(self.y, x=self.x)
        self._result = self._model.fit(self.y, params, x=self.x)
        self._err = {}
        for name, param in self._result.params.items():
            self._params[name] = param.value
            self._err[name] = param.stderr

    def plot_data(self, ax: Axes) -> None:
        ax.errorbar(self.x, self.y, yerr=None, fmt="s",
                    color=ageplot.colors[0])
        ax.set_title("Mag Fit (lmfit)")

    def plot_prediction(self, ax: Axes) -> Sequence[Line2D]:
        x = np.linspace(self.x[0], self.x[-1], 1000)
        params = {name: Parameter(name, value=val)
                  for name, val in self._params.items()}
        pred_line, = ax.plot(x, self._model.eval(params, x=x),
                             color=ageplot.colors[1])
        return [pred_line]

    def print_result(self) -> str:
        if self._result is None:
            return "No fit performed."
        return self._result.fit_report()
        

def fit_mag(
    x: np.ndarray,
    y: np.ndarray,
    parent: QMainWindow = None
) -> None:
    """
    
    """
    _magfit = MagFit(x, y)
    if parent is None:
        app = AGEpp(AGEFitViewer, _magfit)
        app.run()
    else:
        fitviewer = AGEFitViewer(_magfit, parent)
        fitviewer.show()
