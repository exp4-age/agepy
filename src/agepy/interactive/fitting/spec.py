from __future__ import annotations
from typing import TYPE_CHECKING
import warnings
from functools import partial
# Import the modules for the fitting
import numpy as np
from iminuit import Minuit, cost
from iminuit._repr_text import format_row, format_line, pdg_format
from numba_stats import (
    bernstein,
    truncnorm,
    truncexpon,
    uniform,
    voigt,
    cruijff,
    crystalball,
    crystalball_ex,
    qgaussian
)
import numba as nb
import jax
from jax import numpy as jnp
from jax.scipy.stats import norm as jnorm
from jax.scipy.stats import expon as jexpon
from resample.bootstrap import variance as bvar
from jacobi import propagate
# Import the internal modules
from agepy import ageplot
from agepy.interactive import AGEpp
from agepy.interactive.fitting import AGEFitViewer, AGEFitBackend
# Import modules for type hints
if TYPE_CHECKING:
    from typing import Sequence, Tuple, Dict, Union
    from PyQt6.QtWidgets import QMainWindow
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "fit_spectrum",
    "fit_calibration"
]


def fit_spectrum(
    x: ArrayLike,
    y: ArrayLike,
    xerr: ArrayLike = None,
    yerr: ArrayLike = None,
    cost: str = "LeastSquares",
    sig: str = "Gaussian",
    bkg: str = "Exponential",
    start: Dict[str, float] = {},
    limits: Dict[str, Tuple[float, float]] = {},
    parent: QMainWindow = None
) -> SpectrumFit:
    """Interactively fit a spectrum with a signal and background
    component using iminuit.

    iminuit provides accurate error estimates for the fit parameters
    and when using the "Extended" cost functions the yield parameters
    ``s`` and ``b`` are estimates of the number of signal and background
    events in the spectrum.

    Parameters
    ----------
    x : ArrayLike
        The x values of the spectrum. Can be either the bin edges, bin
        centers or unevenly spaced values. The NLL cost functions
        require bin edges or bin centers.
    y : ArrayLike
        Either the bin contents, unbinned data points or the y values.
        The unbinned NLL cost function requires unbinned data points.
    yerr : ArrayLike, optional
        The uncertainties of the y values. Should not be provided for
        unbinned data points, but needs to be provided for unevenly
        spaced x values. If ``y`` is the bin contents and ``yerr`` is
        ``None``, per bin Poisson uncertainties are assumed.
    cost : str, optional
        The cost function to use. Default is "LeastSquares".
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
    fit = SpectrumFit(x, y, xerr, yerr, cost, (sig, bkg), start=start,
                      limits=limits)
    if parent is None:
        app = AGEpp(AGEFitViewer, fit)
        app.run()
    else:
        fitviewer = AGEFitViewer(fit, parent)
        fitviewer.show()
    return fit


def fit_calibration(
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike,
    cost: str = "LeastSquares",
    model: str = "Linear",
    xerr: ArrayLike = None,
    start: Dict[str, float] = {},
    limits: Dict[str, Tuple[float, float]] = {},
    parent: QMainWindow = None
) -> CalibrationFit:
    """Interactively fit a calibration curve using iminuit.

    Parameters
    ----------
    x : ArrayLike
        The x values of the calibration curve.
    y : ArrayLike
        The y values of the calibration curve.
    yerr : ArrayLike
        The uncertainties of the y values.
    cost : str, optional
        The cost function to use. Default is "LeastSquares".
    model : str, optional
        The model to use. Default is "Constant".
    start : Dict[str, float], optional
        Starting values for the fit parameters. The keys must match
        the parameter names of the selected model Default is
        an empty dictionary.
    limits : Dict[str, Tuple[float, float]], optional
        Limits for the fit parameters. The keys must match the parameter
        names of the selected model. Default is an empty dictionary.
    parent : QMainWindow, optional
        The parent window for the fit viewer. If not provided, a new
        application is created and run. Default is ``None``.

    """
    fit = CalibrationFit(x, y, xerr, yerr, cost, model, start=start,
                         limits=limits)
    if parent is None:
        app = AGEpp(AGEFitViewer, fit)
        app.run()
    else:
        fitviewer = AGEFitViewer(fit, parent)
        fitviewer.show()
    return fit


class IminuitBackend(AGEFitBackend):
    """Base class for fitting backends using iminuit.

    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        xerr: ArrayLike,
        yerr: ArrayLike,
        cost: str,
        model: Union[str, Tuple[str, str]],
        start: Dict[str, float] = {},
        limits: Dict[str, Tuple[float, float]] = {}
    ) -> None:
        # Handle the data
        self._parse_data(x, y, xerr, yerr)
        # Create model dictionary
        self._init_model_list()
        # Initialize starting values and limits
        self._params = start
        self._limits = limits
        # Initialize the fit
        self._cost_name = cost  # select_cost will be called in select_model
        if isinstance(model, str):
            self.select_model(model)
        elif isinstance(model, tuple):
            self.select_model(*model)
        # Initialize the iminuit object
        self._minuit = None
        # Initialize the placeholder for numerical integration
        self._numint_cdf = None

    def list_costs(self) -> Sequence[str]:
        costs = ["LeastSquares", "RobustLeastSquares"]
        if self.xerr is not None:
            costs.append("LeastSquaresWithXUncertainties")
        if self.xe is not None:
            costs.append("ExtendedBinnedNLL")
            if self.data is not None:
                costs.append("ExtendedUnbinnedNLL")
        return costs

    def select_cost(self, name: str) -> None:
        if name not in self.list_costs():
            raise ValueError(f"Cost function {name} not available.")
        self._cost_name = name
        if name == "LeastSquares":
            self._cost = cost.LeastSquares(
                self.x, self.y, self.yerr, self._model)
        elif name == "RobustLeastSquares":
            self._cost = self._robust_leastsquares(self.x, self.y, self.yerr)
        elif name == "LeastSquaresWithXUncertainties":
            if self._jax_model is None:
                warnings.warn("Derivative not available for the selected "
                              "model.")
                self.select_cost("LeastSquares")
                return
            jax.config.update("jax_enable_x64", True)
            f = self._jax_model
            fp = jax.jit(jax.grad(self._jax_model))

            @jax.jit
            def _cost(par):
                result = 0.0
                for xi, yi, dxi, dyi in zip(self.x, self.y, self.xerr,
                                            self.yerr):
                    y_var = dyi**2 + (fp(xi, par) * dxi) ** 2
                    result += (yi - f(xi, par)) ** 2 / y_var
                return result

            _cost.errordef = Minuit.LEAST_SQUARES
            self._cost = _cost
        elif name == "ExtendedBinnedNLL":
            n = np.stack([self.y, self.yerr**2], axis=-1)
            self._cost = cost.ExtendedBinnedNLL(n, self.xe, self._integral)
        elif name == "ExtendedUnbinnedNLL":
            def model(x, args):
                yields = np.diff(self._integral(self.xr, args))[0]
                return yields, self._model(x, args)
            self._cost = cost.ExtendedUnbinnedNLL(self.data, model)
        self._cov = None

    def _robust_leastsquares(
        self,
        x: NDArray,
        y: NDArray,
        yerr: NDArray
    ) -> cost.LeastSquares:
        return cost.LeastSquares(x, y, yerr, self._model, loss="soft_l1")

    def fit(self) -> None:
        self._minuit = Minuit(
            self._cost, self.values(), name=list(self._params))
        for name, limits in self._limits.items():
            self._minuit.limits[name] = limits
        self._minuit.migrad()
        if not self._minuit.valid:
            self._minuit.migrad()
        if self._cost_name == "RobustLeastSquares":
            berr = bvar(self._bootstrap_robust_leastsquares, self.x, self.y,
                        self.yerr, size=1000, random_state=1)
            self._cov = np.array(berr)
        elif self._minuit.accurate:
            self._cov = np.array(self._minuit.covariance)
        else:
            self._cov = None
        for par in self._params:
            self._params[par] = self._minuit.values[par]

    def _bootstrap_robust_leastsquares(self, x, y, yerr):
        _cost = self._robust_leastsquares(x, y, yerr)
        _m = Minuit(_cost, self.values(), name=list(self._params))
        for name, limits in self._limits.items():
            _m.limits[name] = limits
        _m.migrad()
        if not _m.valid:
            _m.migrad()
        return _m.values

    def plot_data(self, ax: Axes) -> None:
        ax.errorbar(self.x, self.y, yerr=self.yerr, xerr=self.xerr, fmt="s",
                    color=ageplot.colors[0])
        ax.set_xlim(*self.xr)
        ax.set_title("iminuit")

    def plot_prediction(self, ax: Axes) -> Sequence[Line2D]:
        dx = 1
        if "LeastSquares" not in self._cost_name:
            dx = self.xe[1] - self.xe[0]
        x = np.linspace(self.xr[0], self.xr[1], 1000)
        params = np.array([self._params[par] for par in self._params])
        cov = self._cov
        if cov is None:
            y = self._model(x, params)
        else:
            y, ycov = propagate(lambda p: self._model(x, p), params, cov)
            yerr = np.sqrt(np.diag(ycov)) * dx
        # Normalize the prediction
        y = np.copy(y) * dx
        # Draw the prediction
        pred_line, = ax.plot(x, y, color=ageplot.colors[1])
        mpl_lines = [pred_line]
        # Draw 1 sigma error band
        if cov is not None:
            pred_errband = ax.fill_between(
                x, y - yerr, y + yerr, facecolor=ageplot.colors[1], alpha=0.5)
            mpl_lines.append(pred_errband)
        return mpl_lines

    def values(self) -> NDArray:
        return np.array([self._params[par] for par in self._params])

    def get_result(self) -> Minuit:
        return self._minuit

    def print_result(self) -> str:
        res = ""
        if self._minuit is None:
            l1 = format_line([19], "┌┐")
            header = format_row([19], "No fit performed!")
            l2 = format_line([19], "└┘")
            return "\n".join((l1, header, l2)) + "\n"
        if self._cost_name == "RobustLeastSquares":
            berr = np.sqrt(self._cov)
            n = len(self._params)
            ws = [15] + [11] * n
            l1 = format_line(ws, ["┌"] + ["┬"] * n + ["┐"])
            header = format_row(ws, "", *list(self._params))
            l2 = format_line(ws, "├" + "┼" * n + "┤")
            l3 = format_line(ws, "└" + "┴" * n + "┘")
            x = []
            for err in berr:
                x.append(pdg_format(None, err)[0])
            errors = format_row(ws, "Bootstrap Err", *x)
            res += "\n".join((l1, header, l2, errors, l3)) + "\n"
        res += self._minuit.__str__()
        return res

    def _jit_numint_cdf(self) -> None:
        if self._numint_cdf is not None:
            return
        # jit compile the numerical integration of the pdf

        @nb.njit(parallel=True, fastmath={"reassoc", "contract", "arcp"})
        def numint_cdf(_x, _pdf):
            y = np.empty_like(_x)
            for i in nb.prange(len(_x)):
                y[i] = np.trapz(_pdf[:i+1], x=_x[:i+1])
            return y

        self._numint_cdf = numint_cdf

    def constant(self) -> Tuple[callable, callable, dict, dict]:
        params = {"b": 10}
        limits = {"b": (0, 1000)}

        def model(x, par):
            return par[0] * uniform.pdf(x, self.xr[0], self.xr[1] - self.xr[0])

        def integral(x, par):
            return par[0] * uniform.cdf(x, self.xr[0], self.xr[1] - self.xr[0])

        return model, integral, None, params, limits

    def exponential(self) -> Tuple[callable, callable, dict, dict]:
        params = {"b": 10, "loc_expon": 0, "scale_expon": 1}
        limits = {"b": (0, 1000), "loc_expon": (-1, 0),
                  "scale_expon": (-100, 100)}

        def model(x, par):
            return par[0] * truncexpon.pdf(x, self.xr[0], self.xr[1], *par[1:])

        def integral(x, par):
            return par[0] * truncexpon.cdf(x, self.xr[0], self.xr[1], *par[1:])

        def jax_model(x, par):
            return par[0] * jexpon.pdf(x, *par[1:])

        return model, integral, jax_model, params, limits

    def bernstein(self, deg: int) -> Tuple[callable, callable, dict, dict]:
        params = {f"b_{i}{deg}": 1 for i in range(deg+1)}
        limits = {f"b_{i}{deg}": (0, 1000) for i in range(deg+1)}

        def model(x, args):
            return bernstein.density(x, args, self.xr[0], self.xr[1])

        def integral(x, args):
            return bernstein.integral(x, args, self.xr[0], self.xr[1])

        return model, integral, None, params, limits


class SpectrumFit(IminuitBackend):
    """Backend for interactive fitting of a spectrum with signal and
    background components using iminuit.

    """

    def _init_model_list(self) -> None:
        self._models = {
            "signal": {
                "Gaussian": self.gaussian,
                "Q-Gaussian": self.qgaussian,
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

    def _parse_data(
        self,
        x: ArrayLike,
        y: ArrayLike,
        xerr: ArrayLike,
        yerr: ArrayLike,
    ) -> None:
        x = np.asarray(x)
        y = np.asarray(y)
        if yerr is not None:
            yerr = np.asarray(yerr)
            if yerr.ndim != 1:
                raise ValueError("Only 1d data is supported.")
            if not np.issubdtype(yerr.dtype, np.number):
                raise ValueError("yerr values must be of type float or int.")
        if xerr is not None:
            xerr = np.asarray(xerr)
            if xerr.ndim != 1:
                raise ValueError("Only 1d data is supported.")
            if not np.issubdtype(xerr.dtype, np.number):
                raise ValueError("xerr values must be of type float or int.")
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Only 1d data is supported.")
        if not np.issubdtype(x.dtype, np.number) or \
                not np.issubdtype(y.dtype, np.number):
            raise ValueError("x and y values must be of type float or int.")
        # Check if x values are equally spaced
        dx = np.diff(x)
        if np.all(dx > 0) and np.allclose(dx, dx[0], rtol=1e-5):
            if len(x) == len(y):
                xe = 0.5 * (x[1:] + x[:-1])
                self.xe = np.insert(xe, 0, xe[0] - dx[0])
                self.x = x
            else:
                self.xe = x
                self.x = 0.5 * (self.xe[1:] + self.xe[:-1])
            self.xr = (self.xe[0], self.xe[-1])
            if len(self.x) != len(y):
                self.data = y
                self.y = np.histogram(y, bins=self.xe)[0]
                yerr = np.sqrt(self.y)
            else:
                self.y = y
                self.data = None
            if yerr is None:
                warnings.warn("Per bin Poisson uncertainties are assumed.")
                self.yerr = np.sqrt(self.y)
            elif len(yerr) == len(self.y):
                self.yerr = yerr
            else:
                raise ValueError("yerr must have the same length as y.")
            if xerr is not None:
                warnings.warn("xerr values are ignored for equally spaced x.")
            self.xerr = 0.5 * dx[0] * np.ones_like(self.x)
        else:
            self.x = x
            lenx = (x.max() - x.min())
            self.xr = (x.min() - lenx * 0.025, x.max() + lenx * 0.025)
            self.xe = None
            if len(y) == len(x) and yerr is not None and len(yerr) == len(x):
                self.y = y
                self.yerr = yerr
                self.data = None
            else:
                raise ValueError("For non-equally spaced x values, y and yerr "
                                 "must have the same length as x.")
            if xerr is not None and len(xerr) == len(x):
                self.xerr = xerr
            else:
                self.xerr = None

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
            self._jax_model = sig[2]
            self._params = sig[3]
            self._limits = sig[4]
        else:
            sig = self._models["signal"][signal]()
            bkg = self._models["background"][background]()
            # Combine the parameters and limits
            self._params = dict(sig[3], **bkg[3])
            self._limits = dict(sig[4], **bkg[4])
            # Define the combined model function
            idx = len(sig[3])

            def model(x, args):
                return sig[0](x, args[:idx]) + bkg[0](x, args[idx:])

            self._model = model
            # Define the combined integral function

            def integral(x, args):
                return sig[1](x, args[:idx]) + bkg[1](x, args[idx:])

            self._integral = integral
            # Define the combined derivative function
            if sig[2] is None or bkg[2] is None:
                self._jax_model = None
            else:

                def jax_model(x, args):
                    return sig[2](x, args[:idx]) + bkg[2](x, args[idx:])

                self._jax_model = jax_model
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
        dx = self.xr[1] - self.xr[0]
        params = {"s": 10, "loc": 0.5 * (self.xr[0] + self.xr[1]),
                  "scale": 0.1 * dx}
        limits = {"s": (0, 1000), "loc": self.xr,
                  "scale": (0.0001 * dx, 0.5 * dx)}

        def model(x, par):
            return par[0] * truncnorm.pdf(x, self.xr[0], self.xr[1], *par[1:])

        def integral(x, par):
            return par[0] * truncnorm.cdf(x, self.xr[0], self.xr[1], *par[1:])

        def jax_model(x, par):
            return par[0] * jnorm.pdf(x, *par[1:])

        return model, integral, jax_model, params, limits

    def qgaussian(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {"s": 10, "q": 2, "loc": 0.5 * (self.xr[0] + self.xr[1]),
                  "scale": 0.1 * dx}
        limits = {"s": (0, 1000), "q": (1, 3), "loc": self.xr,
                  "scale": (0.0001 * dx, 0.5 * dx)}

        def model(x, par):
            if par[1] < 1:
                par[1] = 1
                warnings.warn("q cannot be smaller than 1. Setting q=1.")
            if par[1] > 3:
                par[1] = 3
                warnings.warn("q cannot be larger than 3. Setting q=3.")
            return par[0] * qgaussian.pdf(x, *par[1:])

        def integral(x, par):
            if par[1] < 1:
                par[1] = 1
                warnings.warn("q cannot be smaller than 1. Setting q=1.")
            if par[1] > 3:
                par[1] = 3
                warnings.warn("q cannot be larger than 3. Setting q=3.")
            return par[0] * qgaussian.cdf(x, *par[1:])

        return model, integral, None, params, limits

    def voigt(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {"s": 10, "gamma": 0.1 * dx,
                  "loc": 0.5 * (self.xr[0] + self.xr[1]), "scale": 0.1 * dx}
        limits = {"s": (0, 1000), "gamma": (0.0001 * dx, 0.5 * dx),
                  "loc": self.xr, "scale": (0.0001 * dx, 0.5 * dx)}
        self._jit_numint_cdf()
        _x = np.linspace(self.xr[0], self.xr[1], 1000)

        def model(x, par):
            return par[0] * voigt.pdf(x, *par[1:])

        def integral(x, par):
            _cdf = self._numint_cdf(_x, voigt.pdf(_x, *par[1:]))
            return par[0] * np.interp(x, _x, _cdf)

        return model, integral, None, params, limits

    def cruijff(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {"s": 10, "beta_left": 0.1, "beta_right": 0.1,
                  "loc": 0.5 * (self.xr[0] + self.xr[1]),
                  "scale_left": 0.1 * dx, "scale_right": 0.1 * dx}
        limits = {"s": (0, 1000), "beta_left": (0, 1), "beta_right": (0, 1),
                  "loc": self.xr, "scale_left": (0.0001 * dx, 0.5 * dx),
                  "scale_right": (0.0001 * dx, 0.5 * dx)}
        _x = np.linspace(self.xr[0], self.xr[1], 1000)
        self._jit_numint_cdf()

        def model(x, par):
            _cdf = self._numint_cdf(_x, cruijff.density(_x, *par[1:]))
            return par[0] / _cdf[-1] * cruijff.density(x, *par[1:])

        def integral(x, par):
            _cdf = self._numint_cdf(_x, cruijff.density(_x, *par[1:]))
            return par[0] / _cdf[-1] * np.interp(x, _x, _cdf)

        return model, integral, None, params, limits

    def crystalball(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {"s": 10, "beta": 1, "m": 2,
                  "loc": 0.5 * (self.xr[0] + self.xr[1]), "scale": 0.1 * dx}
        limits = {"s": (0, 1000), "beta": (0, 5), "m": (1, 10),
                  "loc": self.xr, "scale": (0.0001 * dx, 0.5 * dx)}

        def model(x, par):
            return par[0] * crystalball.pdf(x, *par[1:])

        def integral(x, par):
            return par[0] * crystalball.cdf(x, *par[1:])

        return model, integral, None, params, limits

    def crystalball_ex(self) -> Tuple[callable, callable, dict, dict]:
        dx = self.xr[1] - self.xr[0]
        params = {"s": 10, "beta_left": 1, "m_left": 2, "scale_left": 0.1 * dx,
                  "beta_right": 1, "m_right": 2, "scale_right": 0.1 * dx,
                  "loc": 0.5 * (self.xr[0] + self.xr[1])}
        limits = {"s": (0, 1000), "beta_left": (0, 5), "m_left": (1, 10),
                  "scale_left": (0.0001 * dx, 0.5 * dx), "beta_right": (0, 5),
                  "m_right": (1, 10), "scale_right": (0.0001 * dx, 0.5 * dx),
                  "loc": self.xr}

        def model(x, par):
            return par[0] * crystalball_ex.pdf(x, *par[1:])

        def integral(x, par):
            return par[0] * crystalball_ex.cdf(x, *par[1:])

        return model, integral, None, params, limits


class CalibrationFit(IminuitBackend):
    """Backend for interactive fitting of a calibration curve using
    iminuit.

    """

    def _init_model_list(self) -> None:
        self._models = {
            "model": {
                "Constant": self.constant,
                "Linear": partial(self.polynomial, 1),
                "Quadratic": partial(self.polynomial, 2),
                "Cubic": partial(self.polynomial, 3),
                "Exponential": self.exponential,
                "Bernstein1d": partial(self.bernstein, 1),
                "Bernstein2d": partial(self.bernstein, 2),
                "Bernstein3d": partial(self.bernstein, 3),
                "Bernstein4d": partial(self.bernstein, 4),
                "Bernstein5d": partial(self.bernstein, 5),
            }
        }

    def _parse_data(
        self,
        x: ArrayLike,
        y: ArrayLike,
        xerr: ArrayLike,
        yerr: ArrayLike,
    ) -> None:
        x = np.asarray(x)
        y = np.asarray(y)
        yerr = np.asarray(yerr)
        if xerr is not None:
            xerr = np.asarray(xerr)
            if xerr.ndim != 1:
                raise ValueError("Only 1d data is supported.")
            if len(xerr) != len(x):
                raise ValueError("x and xerr must have the same length.")
            if not np.issubdtype(xerr.dtype, np.number):
                raise ValueError("xerr values must be of type float or int.")
            self.xerr = xerr
        else:
            self.xerr = None
        if x.ndim != 1 or y.ndim != 1 or yerr.ndim != 1:
            raise ValueError("Only 1d data is supported.")
        if len(x) != len(y) or len(x) != len(yerr):
            raise ValueError("x, y and yerr must have the same length.")
        if not np.issubdtype(x.dtype, np.number) \
            or not np.issubdtype(y.dtype, np.number) \
                or not np.issubdtype(yerr.dtype, np.number):
            raise ValueError("x, y and yerr values must be of type "
                             "float or int.")
        self.x = x
        lenx = (x.max() - x.min())
        self.xr = (x.min() - lenx * 0.025, x.max() + lenx * 0.025)
        self.xe = None
        self.y = y
        self.yerr = yerr
        self.data = None

    def select_model(self, model: str) -> None:
        if model not in self._models["model"]:
            raise ValueError(f"Model {model} not available.")
        self._model_name = model
        # Remember current parameters and limits
        _params = self._params
        _limits = self._limits
        # Initialize the model
        model = self._models["model"][model]()
        self._model, _, self._jax_model, self._params, self._limits = model
        # Keep previous parameters and limits if possible
        for par in _params:
            if par in self._params:
                self._params[par] = _params[par]
        for par in _limits:
            if par in self._limits:
                self._limits[par] = _limits[par]
        # Update the cost function
        self.select_cost(self._cost_name)

    def polynomial(self, deg: int) -> Tuple[callable, callable, dict, dict]:
        params = {f"a{i}": 1 for i in range(deg+1, 0, -1)}
        limits = {f"a{i}": (-1000, 1000) for i in range(deg+1, 0, -1)}

        def model(x, args):
            return np.polyval(args, x)

        def jax_model(x, args):
            return jnp.polyval(args, x)

        return model, None, jax_model, params, limits
