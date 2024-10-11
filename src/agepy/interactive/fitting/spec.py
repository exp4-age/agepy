from __future__ import annotations
from typing import TYPE_CHECKING, List, Sequence, Tuple, Dict
import warnings
# Import the necessary modules
from functools import partial
import inspect
import numpy as np
from iminuit import Minuit, cost
from numba_stats import bernstein, truncnorm, truncexpon, uniform, voigt, cruijff, crystalball, crystalball_ex
import numba as nb
from jacobi import propagate
# Import the internal modules
from agepy import ageplot
from agepy.interactive import AGEpp
from agepy.interactive.fitting import AGEFitViewer, AGEFit
# Import modules for type hints
if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from numpy.typing import NDArray


class SpecFit(AGEFit):
    """Spectrum fitting class for the AGEFitViewer using iminuit.

    """

    def __init__(self,
        xedges: NDArray,
        binned_data: NDArray,
        unbinned_data: NDArray,
        cost: str = "LeastSquares",
        signal: str = "Gaussian",
        background: str = "Exponential",
        start: Dict[str, float] = {},
        limits: Dict[str, Tuple[float, float]] = {}
    ) -> None:
        # Handle the data
        self.xe = xedges
        self.x = 0.5 * (xedges[1:] + xedges[:-1])
        self.xr = (xedges[0], xedges[-1])
        self.binned_data = binned_data
        self.unbinned_data = unbinned_data
        # Create model dictionary
        self._models = {
            "signal": {
                "Gaussian": self.gaussian,
                "Voigt": self.voigt,
                "Cruijff": self.cruijff,
                "CrystalBall": self.crystalball,
                "2SidedCrystalBall": self.crystalball_ex,
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
        # Initialize the iminuit object
        self._minuit = None
        # Initialize the placeholder for numerical integration
        self._numint_cdf = None

    def list_costs(self) -> Sequence[str]:
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

    def list_models(self) -> Dict[str, List[str]]:
        models = {
            "signal": list(self._models["signal"].keys()),
            "background": list(self._models["background"].keys())
        }
        return models

    def select_model(self, signal: str, background: str) -> None:
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

    def _jit_numint_cdf(self) -> None:
        if self._numint_cdf is not None:
            return
        @nb.njit(parallel=True, fastmath={"reassoc", "contract", "arcp"})
        def numint_cdf(_x, _pdf):
            y = np.empty_like(_x)
            for i in nb.prange(len(_x)):
                y[i] = np.trapz(_pdf[:i+1], x=_x[:i+1])
            return y
        self._numint_cdf = numint_cdf

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

    def voigt(self) -> Tuple[callable, callable, dict, dict]:
        def model(x, s, gamma, loc, scale):
            return s * voigt.pdf(x, gamma, loc, scale)
        self._jit_numint_cdf()
        _x = np.linspace(self.xr[0], self.xr[1], 1000)
        def integral(x, s, gamma, loc, scale):
            _cdf = self._numint_cdf(_x, voigt.pdf(_x, gamma, loc, scale))
            return s * np.interp(x, _x, _cdf)
        dx = self.xr[1] - self.xr[0]
        params = {"s": 10, "gamma": 0.1 * dx,
                  "loc": 0.5 * (self.xr[0] + self.xr[1]), "scale": 0.1 * dx}
        limits = {"s": (0, 1000), "gamma": (0.0001 * dx, 0.5 * dx),
                  "loc": self.xr, "scale": (0.0001 * dx, 0.5 * dx)}
        return model, integral, params, limits

    def cruijff(self) -> Tuple[callable, callable, dict, dict]:
        _x = np.linspace(self.xr[0], self.xr[1], 1000)
        self._jit_numint_cdf()
        def model(x, s, beta_left, beta_right, loc, scale_left, scale_right):
            _cdf = self._numint_cdf(_x, cruijff.density(_x, beta_left,
                beta_right, loc, scale_left, scale_right))
            return s / _cdf[-1] * cruijff.density(x, beta_left, beta_right, loc,
                scale_left, scale_right)
        def integral(x, s, beta_left, beta_right, loc, scale_left,
                     scale_right):
            _cdf = self._numint_cdf(_x, cruijff.density(_x, beta_left,
                beta_right, loc, scale_left, scale_right))
            return s  / _cdf[-1] * np.interp(x, _x, _cdf)
        dx = self.xr[1] - self.xr[0]
        params = {"s": 10, "beta_left": 0.1, "beta_right": 0.1,
                  "loc": 0.5 * (self.xr[0] + self.xr[1]),
                  "scale_left": 0.1 * dx, "scale_right": 0.1 * dx}
        limits = {"s": (0, 1000), "beta_left": (0, 1), "beta_right": (0, 1),
                    "loc": self.xr, "scale_left": (0.0001 * dx, 0.5 * dx),
                    "scale_right": (0.0001 * dx, 0.5 * dx)}
        return model, integral, params, limits

    def crystalball(self) -> Tuple[callable, callable, dict, dict]:
        def model(x, s, beta, m, loc, scale):
            return s * crystalball.pdf(x, beta, m, loc, scale)
        def integral(x, s, beta, m, loc, scale):
            return s * crystalball.cdf(x, beta, m, loc, scale)
        dx = self.xr[1] - self.xr[0]
        params = {"s": 10, "beta": 1, "m": 2,
                  "loc": 0.5 * (self.xr[0] + self.xr[1]), "scale": 0.1 * dx}
        limits = {"s": (0, 1000), "beta": (0, 5), "m": (1, 10),
                  "loc": self.xr, "scale": (0.0001 * dx, 0.5 * dx)}
        return model, integral, params, limits

    def crystalball_ex(self) -> Tuple[callable, callable, dict, dict]:
        def model(x, s, beta_left, m_left, scale_left, beta_right, m_right,
                  scale_right, loc):
            return s * crystalball_ex.pdf(x, beta_left, m_left, scale_left,
                beta_right, m_right, scale_right, loc)
        def integral(x, s, beta_left, m_left, scale_left, beta_right, m_right,
                     scale_right, loc):
            return s * crystalball_ex.cdf(x, beta_left, m_left, scale_left,
                beta_right, m_right, scale_right, loc)
        dx = self.xr[1] - self.xr[0]
        params = {"s": 10, "beta_left": 1, "m_left": 2, "scale_left": 0.1 * dx,
                  "beta_right": 1, "m_right": 2, "scale_right": 0.1 * dx,
                  "loc": 0.5 * (self.xr[0] + self.xr[1])}
        limits = {"s": (0, 1000), "beta_left": (0, 5), "m_left": (1, 10),
                  "scale_left": (0.0001 * dx, 0.5 * dx), "beta_right": (0, 5),
                  "m_right": (1, 10), "scale_right": (0.0001 * dx, 0.5 * dx),
                  "loc": self.xr}
        return model, integral, params, limits

    def constant(self) -> Tuple[callable, callable, dict, dict]:
        def model(x, b):
            return b * uniform.pdf(x, self.xr[0], self.xr[1] - self.xr[0])
        def integral(x, b):
            return b * uniform.cdf(x, self.xr[0], self.xr[1] - self.xr[0])
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
                  "scale_expon": (-100, 100)}
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
        

def fit_spectrum(
    x: NDArray,
    binned_data: NDArray = None,
    uncertainties: NDArray = None,
    unbinned_data: NDArray = None,
    cost: str = "ExtendedBinnedNLL",
    sig: str = "Gaussian",
    bkg: str = "Exponential",
    start: Dict[str, float] = {},
    limits: Dict[str, Tuple[float, float]] = {},
    parent: QMainWindow = None
) -> None:
    """Interactively fit a spectrum with a signal and background
    component using iminuit.

    iminuit provides accurate error estimates for the fit parameters
    and when using the "Extended" cost functions the yield parameters
    ``s`` and ``b`` are estimates of the number of signal and background
    events in the spectrum.  

    Parameters
    ----------
    x : NDArray
        The x values of the spectrum. Can be either the bin edges or
        bin centers.
    binned_data : NDArray, optional
        Binned 1d (histogram) data. Must be of length ``len(x)`` or
        ``len(x) - 1``. At least one of ``binned_data`` or
        ``unbinned_data`` must be provided.
    uncertainties : NDArray, optional
        Uncertainties of the binned data. Must be of length
        ``len(binned_data)``. If not provided, Poisson uncertainties
        are assumed.
    unbinned_data : NDArray, optional
        Unbinned 1d data. If provided, the ExtendedUnbinnedNLL cost
        function becomes available.
    cost : str, optional
        The cost function to use. Default is "ExtendedBinnedNLL".
    sig : str, optional
        The signal model to use. Default is "Gaussian".
    bkg : str, optional
        The background model to use. Default is "Exponential".
    start : Dict[str, float], optional
        Starting values for the fit parameters. The keys must match
        the parameter names of the selected signal and background
        models. Parameter names can be found in ``SpecFit``. Default is
        an empty dictionary.
    limits : Dict[str, Tuple[float, float]], optional
        Limits for the fit parameters. The keys must match the parameter
        names of the selected signal and background models. Parameter
        names can be found in ``SpecFit``. Default is an empty dictionary.
    parent : QMainWindow, optional
        The parent window for the fit viewer. If not provided, a new
        application is created and run. Default is ``None``.

    """
    # Check if data is provided
    if binned_data is None and unbinned_data is None:
        raise ValueError("Either binned or unbinned data must be provided.")
    # Check if x values are equally spaced
    if not np.allclose(np.diff(x), np.diff(x)[0]):
        raise ValueError("x values must be equally spaced.")
    if binned_data is not None:
        # Check if x values are edges or centers
        if len(x) == len(binned_data):
            x = 0.5 * (x[1:] + x[:-1])
        elif len(x) != len(binned_data) + 1:
            raise ValueError("x values must be of length len(binned_data) or "
                             "len(binned_data) + 1, if binned data is "
                             "provided.")
        # Check if uncertainties have the correct shape
        if uncertainties is not None:
            if len(uncertainties) != len(binned_data):
                raise ValueError("Uncertainties must be of same length as "
                                 "the binned data.")
            else:
                binned_data = np.stack([binned_data, uncertainties**2],
                                       axis=-1)
        else:
            warnings.warn("Per bin Poisson uncertainties are assumed.")
    else:
        if uncertainties is not None:
            warnings.warn("Uncertainties are ignored for unbinned data.")
    if unbinned_data is not None:
        # Histogram the data
        n = np.histogram(unbinned_data, bins=x)[0]
        if binned_data is None:
            binned_data = np.stack([n, n], axis=-1)
        elif not np.allclose(n, binned_data[:, 0]):
            raise ValueError("Unbinned data is not consistent with binned "
                             "data.")
    _specfit = SpecFit(x, binned_data, unbinned_data, cost=cost, signal=sig,
                       background=bkg, start=start, limits=limits)
    if parent is None:
        app = AGEpp(AGEFitViewer, _specfit)
        app.run()
    else:
        fitviewer = AGEFitViewer(_specfit, parent)
        fitviewer.show()
