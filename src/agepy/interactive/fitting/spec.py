from __future__ import annotations
from typing import List, TypedDict, Sequence, Tuple
from functools import partial
from PyQt6.QtWidgets import QMainWindow
import inspect
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

import numpy as np
from iminuit import Minuit, cost
from numba_stats import bernstein, truncnorm, truncexpon
from jacobi import propagate

from agepy import ageplot
from agepy.interactive import AGEpp
from agepy.interactive.fitting import AGEFitViewer, AGEFit


class ModelDict(TypedDict):
    signal: List[str]
    background: List[str]


class SpecFit(AGEFit):
    """

    """

    def __init__(self,
        xedges: np.ndarray,
        binned_data: np.ndarray,
        unbinned_data: np.ndarray,
        cost: str = "LeastSquares",
        signal: str = "Gaussian",
        background: str = "Exponential",
        start: dict = {},
        limits: dict = {}
    ) -> None:
        # Handle the data
        self.xe = xedges
        self.x = 0.5 * (xedges[1:] + xedges[:-1])
        self.xr = (xedges[0], xedges[-1])
        self.binned_data = binned_data
        self.unbinned_data = unbinned_data
        # Create model dictionary
        self._models: ModelDict = {
            "signal": {
                "Gaussian": self.gaussian
            },
            "background": {
                "None": None,
                "Constant": self.constant, 
                "Exponential": self.exponential,
                "Bernstein1d": partial(self.bernstein, 1),
                "Bernstein2d": partial(self.bernstein, 2),
                "Bernstein3d": partial(self.bernstein, 3),
                "Bernstein4d": partial(self.bernstein, 4),
            }
        }
        # Initialize starting values and limits
        self._params = start
        self._limits = limits
        # Initialize the fit
        self._cost_name = cost  # select_cost will be called in select_model
        self.select_model(signal, background)
        self._minuit = None

    def list_costs(self) -> list[str]:
        costs = ["LeastSquares", "ExtendedBinnedNLL"]
        if self.unbinned_data is not None:
            costs.append("ExtendedUnbinnedNLL")
        return costs

    def select_cost(self, name: str) -> None:
        if name not in self.list_costs():
            raise ValueError(f"Cost function {name} not available.")
        self._cost_name = name
        if name == "LeastSquares":
            self._cost = cost.LeastSquares(
                self.x, self.binned_data[:, 0],
                np.sqrt(self.binned_data[:, 1]), self._model)
        elif name == "ExtendedBinnedNLL":
            self._cost = cost.ExtendedBinnedNLL(
                self.binned_data, self.xe, self._integral)
        elif name == "ExtendedUnbinnedNLL":
            def model(x, *args):
                yields = np.diff(self._integral(self.xr, *args))[0]
                return yields, self._model(x, *args)
            model.__signature__ = self._model.__signature__
            self._cost = cost.ExtendedUnbinnedNLL(
                self.unbinned_data, model)
        self._cov = None

    def list_models(self) -> ModelDict:
        models = {
            "signal": list(self._models["signal"].keys()),
            "background": list(self._models["background"].keys())
        }
        return models

    def select_model(self, signal, background) -> None:
        if signal not in self._models["signal"]:
            raise ValueError(f"Model {signal} not available.")
        if background not in self._models["background"]:
            raise ValueError(f"Model {background} not available.")
        self._model_name = {"signal": signal, "background": background}
        # Remember current parameters and limits
        _params = self._params
        _limits = self._limits
        # Initialize the signal and background models
        if self._models["background"][background] is None:
            sig = self._models["signal"][signal]()
            self._model = sig[0]
            self._integral = sig[1]
            self._params = sig[2]
            self._limits = sig[3]
        else:
            sig = self._models["signal"][signal]()
            bkg = self._models["background"][background]()
            # Combine the parameters and limits
            self._params = dict(sig[2], **bkg[2])
            self._limits = dict(sig[3], **bkg[3])
            # Create the function signature
            func_signature = [
                inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for arg in ["x"] + list(self._params)]
            func_signature = inspect.Signature(func_signature)
            # Define the combined model function
            idx = len(sig[2])
            def model(x, *args):
                return sig[0](x, *args[:idx]) + bkg[0](x, *args[idx:])
            # Set the new signature
            model.__signature__ = func_signature
            self._model = model
            def integral(x, *args):
                return sig[1](x, *args[:idx]) + bkg[1](x, *args[idx:])
            # Set the new signature
            integral.__signature__ = func_signature
            self._integral = integral
        # Keep previous parameters and limits if possible
        for par in _params:
            if par in self._params:
                self._params[par] = _params[par]
        for par in _limits:
            if par in self._limits:
                self._limits[par] = _limits[par]
        # Update the cost function
        self.select_cost(self._cost_name)

    def gaussian(self) -> Tuple[callable, callable, dict, dict]:
        def model(x, s, loc, scale):
            return s * truncnorm.pdf(x, self.xr[0], self.xr[1], loc, scale)
        def integral(x, s, loc, scale):
            return s * truncnorm.cdf(x, self.xr[0], self.xr[1], loc, scale)
        dx = self.xr[1] - self.xr[0]
        params = {"s": 10, "loc": 0.5 * (self.xr[0] + self.xr[1]),
                  "scale": 0.1 * dx}
        limits = {"s": (0, 1000), "loc": self.xr,
                  "scale": (0.0001 * dx, 0.5 * dx)}
        return model, integral, params, limits

    def constant(self) -> Tuple[callable, callable, dict, dict]:
        def model(x, b):
            return np.full_like(x, b)
        def integral(x, b):
            return b * (x - self.xr[0])
        params = {"b": 10}
        limits = {"b": (0, 1000)}
        return model, integral, params, limits

    def exponential(self) -> Tuple[callable, callable, dict, dict]:
        def model(x, b, loc, scale):
            return b * truncexpon.pdf(x, self.xr[0], self.xr[1], loc, scale)
        def integral(x, b, loc, scale):
            return b * truncexpon.cdf(x, self.xr[0], self.xr[1], loc, scale)
        params = {"b": 10, "loc_expon": 0, "scale_expon": 1}
        limits = {"b": (0, 1000), "loc_expon": (-1, 0),
                  "scale_expon": (-1000, 1000)}
        fixed = {"b": False, "loc_expon": True, "scale_expon": False}
        return model, integral, params, limits

    def bernstein(self, deg: int) -> Tuple[callable, callable, dict, dict]:
        def model(x, *args):
            return bernstein.density(x, args, self.xr[0], self.xr[1])
        def integral(x, b, *args):
            return bernstein.integral(x, args, self.xr[0], self.xr[1])
        params = {f"a{i}": 1 for i in range(deg+1)}
        limits = {f"a{i}": (-1000, 1000) for i in range(deg+1)}
        return model, integral, params, limits

    def fit(self) -> None:
        self._minuit = Minuit(self._cost, **self._params)
        for name, limits in self._limits.items():
            self._minuit.limits[name] = limits
        self._minuit.migrad()
        if not self._minuit.valid:
            self._minuit.migrad()
        for par in self._params:
            self._params[par] = self._minuit.values[par]
        if self._minuit.accurate:
            self._cov = np.array(self._minuit.covariance)
        else:
            self._cov = None

    def plot_data(self, ax: Axes) -> None:
        ax.errorbar(self.x, self.binned_data[:, 0],
                    yerr=np.sqrt(self.binned_data[:, 1]), fmt="s",
                    color=ageplot.colors[0])
        ax.set_xlim(*self.xr)
        ax.set_title("Spectrum Fit (iminuit)")

    def plot_prediction(self, ax: Axes) -> Sequence[Line2D]:
        dx = 1
        if self._cost_name != "LeastSquares":
            dx = self.xe[1] - self.xe[0]
        x = np.linspace(self.xr[0], self.xr[1], 1000)
        params = np.array([self._params[par] for par in self._params])
        cov = self._cov
        if cov is None:
            y = self._model(x, *params)
        else:
            y, ycov = propagate(lambda p: self._model(x, *p), params, cov)
        # Normalize the prediction
        y *= dx
        # Draw the prediction
        pred_line, = ax.plot(x, y, color=ageplot.colors[1])
        mpl_lines = [pred_line]
        # Draw 1 sigma error band
        if cov is not None:
            yerr = np.sqrt(np.diag(ycov)) * dx
            pred_errband = ax.fill_between(x, y - yerr, y + yerr,
                facecolor=ageplot.colors[1], alpha=0.5)
            mpl_lines.append(pred_errband)
        return mpl_lines

    def print_result(self) -> str:
        if self._minuit is None:
            return "No fit performed."
        return self._minuit.__str__()
        

def fit_spectrum(
    xdata: np.ndarray = None,
    ydata: np.ndarray = None,
    yerr: np.ndarray = None,
    data: np.ndarray = None,
    cost: str = "LeastSquares",
    sig: str = "Gaussian",
    bkg: str = "Exponential",
    start: dict = {},
    parent: QMainWindow = None
) -> None:
    """
    
    """
    binned_data = np.stack([ydata, yerr**2], axis=-1)
    #_specfit = SpecFit(xdata, binned_data, data, cost, sig, bkg)
    _specfit = AGEFit()
    if parent is None:
        app = AGEpp(AGEFitViewer, _specfit)
        app.run()
    else:
        fitviewer = AGEFitViewer(_specfit, parent)
        fitviewer.show()
