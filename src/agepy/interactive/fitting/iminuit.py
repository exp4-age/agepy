from __future__ import annotations
import inspect
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from numba_stats import truncnorm, truncexpon, bernstein

from .agefit import AGEFit


class AGEIminuit(AGEFit):
    """

    """
    backgrounds = ["Bernstein1d", "Bernstein2d", "Bernstein3d", "Bernstein4d",
                   "Exponential"]
    signals = ["Gaussian"]

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
        self.model = model
        # Set the parameter names
        self.params = combined_params[1:]

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

    def fit(self, start, limits):
        # Create the cost function
        if self.cost == "LeastSquares":
            cost = LeastSquares(self.x, self.y, self.yerr, self.model)
        # Create the minimizer
        m = Minuit(cost, *start)
        for par, limit in limits.items():
            m.limits[par] = limit
        # Perform the minimization
        m.migrad()
        # Get the fitted parameters
        params = np.array(m.values)
        cov = np.array(m.covariance)
        return params, cov, m.__str__()
