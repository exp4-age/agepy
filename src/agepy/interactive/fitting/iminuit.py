from __future__ import annotations
import inspect
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares, ExtendedBinnedNLL
from numba_stats import truncnorm, truncexpon, bernstein

from .agefit import AGEFit


class AGEIminuit(AGEFit):
    """

    """

    @property
    def costs(self):
        costs = []
        if self.x is not None:
            if self.y is not None:
                costs.extend(["BinnedNLL", "ExtendedBinnedNLL"])
                if self.yerr is not None:
                    costs.append("LeastSquares")
            elif self.data is not None:
                costs.extend(["UnbinnedNLL", "ExtendedUnbinnedNLL"])
        return costs

    def init_model(self, sig, bkg):
        # Combine the signal and background models
        sig_func, sig_int = (sig.model(), sig.integral())
        bkg_func, bkg_int = (bkg.model(), bkg.integral())
        if sig_func is None and bkg_func is None:
            self.model = lambda x: np.zeros_like(x)
            self.integral = lambda x: np.zeros_like(x)
            self.params = []
        elif sig_func is None:
            self.model = bkg_func
            self.integral = bkg_int
            self.params = bkg.params
        elif bkg_func is None:
            self.model = sig_func
            self.integral = sig_int
            self.params = sig.params
        else:
            combined_params = ["x"]
            combined_params.extend(sig.params)
            combined_params.extend(bkg.params)
            # Create the parameters for the function signature
            parameters = [
                inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for arg in combined_params]
            # Create the signature object
            func_signature = inspect.Signature(parameters)
            # Define the model function
            n = len(sig.params)
            def model(x, *args):
                return sig_func(x, *args[:n]) + bkg_func(x, *args[n:])
            # Set the new signature to the model function
            model.__signature__ = func_signature
            self.model = model
            # Define the integral function
            if sig_int is None or bkg_int is None:
                self.integral = None
            else:
                def integral(x, *args):
                    return sig_func(x, *args[:n]) + bkg_func(x, *args[n:])
                integral.__signature__ = func_signature
                self.integral = integral
            # Set the parameter names
            self.params = combined_params[1:]

    def fit(self, start, limits):
        # Create the cost function
        if self.cost == "LeastSquares":
            cost = LeastSquares(self.x, self.y, self.yerr, self.model)
        elif self.cost == "ExtendedBinnedNLL":
            if self.yerr is None:
                n = self.y
            else:
                n = np.stack([self.y, self.yerr**2], axis=-1)
            cost = ExtendedBinnedNLL(n, self.xe, self.integral)
        # Create the minimizer
        m = Minuit(cost, *start)
        for par, limit in limits.items():
            m.limits[par] = limit
        # Perform the minimization
        m.migrad()
        if not m.valid:
            m.migrad()
        # Get the fitted parameters
        params = np.array(m.values)
        cov = np.array(m.covariance)
        if not m.valid:
            cov = None
        return params, cov, m.__str__()
