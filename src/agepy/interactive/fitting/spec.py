from __future__ import annotations
from typing import List, TypedDict, Sequence, Tuple
from PyQt5.QtWidgets import QMainWindow
import inspect
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

import numpy as np
from iminuit import Minuit, cost
from numba_stats import bernstein, truncnorm, truncexpon
from jacobi import propagate

from agepy import ageplot
from agepy.interactive import AGEpp
from agepy.interactive.fitting import AGEFitViewer

class ModelDict(TypedDict):
    signal: List[str]
    background: List[str]


class SpecFit:
    """

    """
    _models: ModelDict = {
        "signal": ["Gaussian"],
        "background": ["None", "Constant", "Exponential", "Bernstein1d"]
    }

    def __init__(self,
        xedges: np.ndarray,
        binned_data: np.ndarray,
        unbinned_data: np.ndarray,
        cost: str = "LeastSquares",
        signal: str = "Gaussian",
        background: str = "Exponential"
    ) -> None:
        self.xe = xedges
        self.x = 0.5 * (xedges[1:] + xedges[:-1])
        self.xr = (xedges[0], xedges[-1])
        self.binned_data = binned_data
        self.unbinned_data = unbinned_data
        # Initialize the fit
        self._cost_name = cost  # select_cost will be called in select_model
        self.select_model(signal, background)
        self._minuit = None

    def list_costs(self) -> list[str]:
        costs = ["LeastSquares", "BinnedNLL", "ExtendedBinnedNLL"]
        if self.unbinned_data is not None:
            costs.extend(["UnbinnedNLL", "ExtendedUnbinnedNLL"])
        return costs

    def which_cost(self) -> str:
        return self._cost_name

    def select_cost(self, name: str) -> None:
        if name not in self.list_costs():
            raise ValueError(f"Cost function {name} not available.")
        self._cost_name = name
        if name == "LeastSquares":
            self._cost = cost.LeastSquares(
                self.x, self.binned_data[:, 0],
                np.sqrt(self.binned_data[:, 1]), self._model)
        self._cov = None

    def list_models(self) -> ModelDict:
        return self._models

    def which_model(self) -> dict:
        return self._model_name

    def select_model(self, signal, background) -> None:
        if signal not in self._models["signal"]:
            raise ValueError(f"Model {signal} not available.")
        if background not in self._models["background"]:
            raise ValueError(f"Model {background} not available.")
        self._model_name = {"signal": signal, "background": background}
        # Initialize the signal and background models
        self.init_model(signal, background)
        # Update the cost function
        self.select_cost(self._cost_name)

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
        self._model = model
        # Set the parameter names
        self._params = {}
        self._limits = {}
        for par in combined_params[1:]:
            self._params[par] = 1
            self._limits[par] = (-1000, 1000)

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

    def list_params(self) -> dict:
        return self._params

    def list_limits(self) -> dict:
        return self._limits

    def change_value(self, name: str, value: float) -> None:
        if name not in self.list_params():
            raise ValueError(f"Parameter {name} not available.")
        self._params[name] = value
        self._cov = None

    def change_limit(self, name: str, limits: Tuple[float, float]) -> None:
        if name not in self.list_params():
            raise ValueError(f"Parameter {name} not available.")
        self._limits[name] = limits

    def fit(self) -> None:
        self._minuit = Minuit(self._cost, **self._params)
        for name, limits in self._limits.items():
            self._minuit.limits[name] = limits
        self._minuit.migrad()
        if not self._minuit.valid:
            self._minuit.migrad()
        for par in self._params:
            self._params[par] = self._minuit.values[par]
        self._cov = np.array(self._minuit.covariance)

    def plot_data(self, ax: Axes) -> None:
        ax.errorbar(self.x, self.binned_data[:, 0],
                    yerr=np.sqrt(self.binned_data[:, 1]), fmt="s",
                    color=ageplot.colors[0])

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
        # Draw the prediction
        y *= dx
        pred_line, = ax.plot(x, y, color=ageplot.colors[1])
        mpl_lines = [pred_line]
        # Draw 1 sigma error band
        if cov is not None:
            yerr = np.sqrt(np.diag(ycov))
            pred_errband = ax.fill_between(x, y - yerr, y + yerr,
                facecolor=ageplot.colors[1], alpha=0.5)
            mpl_lines.append(pred_errband)
        return mpl_lines

    def print_result(self) -> str:
        if self._minuit is None:
            return "No fit performed."
        return self._minuit.__str__()
        

def specfit(
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
    binned_data = np.stack([ydata, yerr**2], axis=-1)
    _specfit = SpecFit(xdata, binned_data, None, cost, sig, bkg)
    if parent is None:
        app = AGEpp(AGEFitViewer, _specfit)
        app.run()
    else:
        fitviewer = AGEFitViewer(_specfit, parent)
        fitviewer.show()
