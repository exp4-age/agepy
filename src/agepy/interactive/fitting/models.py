from __future__ import annotations
from typing import Union, Sequence, Tuple
import inspect
from numba_stats import bernstein, truncnorm, truncexpon


def list_signal_models():
    return [cls.__name__ for cls in SignalModel.__subclasses__()]

def list_background_models():
    return [cls.__name__ for cls in BackgroundModel.__subclasses__()]

def get_model(name: str):
    for cls in SignalModel.__subclasses__():
        if cls.__name__ == name:
            return cls
    for cls in BackgroundModel.__subclasses__():
        if cls.__name__ == name:
            return cls
    return None


class CustomModel:
    """

    """

    def __init__(self,
        func_model: callable,
        name: str = "Custom"
    ) -> None:
        self.name = name
        self.params = list(inspect.signature(func_model).parameters.keys())[1:]
        self.start = [1] * len(self.params)
        self.limits = [(-1000, 1000)] * len(self.params)
        self.func_model = func_model

    def model(self):
        return self.func_model

    def integral(self):
        raise NotImplementedError("Integral not implemented for CustomModel.")


class FitModel:
    """

    """

    def __init__(self, xr: Tuple[float, float]) -> None:
        self.xr = xr
        self.setup()

    def setup(self):
        raise NotImplementedError("Setup not implemented.")

    def model(self):
        return None

    def integral(self):
        return None


class SignalModel(FitModel):
    pass

class BackgroundModel(FitModel):
    pass


class NoSignal(SignalModel):
    pass


class Gaussian(SignalModel):
    """

    """

    def setup(self) -> None:
        self.params = ["s", "loc", "scale"]
        dx = self.xr[1] - self.xr[0]
        self.start = [10, 0.5 * (self.xr[0] + self.xr[1]), 0.1 * dx]
        self.limits = [(0, 1000), self.xr, (0.0001 * dx, 0.5 * dx)]

    def model(self):
        def model(x, s, loc, scale):
            return s * truncnorm.pdf(x, self.xr[0], self.xr[1], loc, scale)
        return model

    def integral(self):
        def integral(x, s, loc, scale):
            return s * truncnorm.cdf(x, self.xr[0], self.xr[1], loc, scale)
        return integral


class NoBackground(BackgroundModel):
    pass


class Exponential(BackgroundModel):
    """

    """

    def setup(self) -> None:
        self.params = ["b", "loc_expon", "scale_expon"]
        self.start = [10, self.xr[0], 1]
        dx = self.xr[1] - self.xr[0]
        self.limits = [(0, 1000), (self.xr[0] - dx, self.xr[1] + dx),
                       (-1000, 1000)]

    def model(self):
        def model(x, b, loc, scale):
            return b * truncexpon.pdf(x, self.xr[0], self.xr[1], loc, scale)
        return model

    def integral(self):
        def integral(x, b, loc, scale):
            return b * truncexpon.cdf(x, self.xr[0], self.xr[1], loc, scale)
        return integral


class Constant(BackgroundModel):
    """

    """

    def setup(self) -> None:
        self.params = ["b"]
        self.start = [10]
        self.limits = [(0, 1000)]

    def model(self):
        def model(x, b):
            return b
        return model

    def integral(self):
        def integral(x, b):
            return b * (x - self.xr[0])
        return integral


class Bernstein1d(BackgroundModel):
    """

    """

    def setup(self) -> None:
        self.params = ["a0", "a1"]
        self.start = [1] * len(self.params)
        self.limits = [(-1000, 1000)] * len(self.params)

    def model(self):
        def model(x, a0, a1):
            return bernstein.density(x, (a0, a1), self.xr[0], self.xr[1])
        return model

    def integral(self):
        def integral(x, a0, a1):
            return bernstein.integral(x, (a0, a1), self.xr[0], self.xr[1])
        return integral


class Bernstein2d(BackgroundModel):
    """

    """

    def setup(self) -> None:
        self.params = ["a0", "a1", "a2"]
        self.start = [1] * len(self.params)
        self.limits = [(-1000, 1000)] * len(self.params)

    def model(self):
        def model(x, a0, a1, a2):
            return bernstein.density(x, (a0, a1, a2), self.xr[0], self.xr[1])
        return model

    def integral(self):
        def integral(x, a0, a1, a2):
            return bernstein.integral(x, (a0, a1, a2), self.xr[0], self.xr[1])
        return integral


class Bernstein3d(BackgroundModel):
    """

    """

    def setup(self) -> None:
        self.params = ["a0", "a1", "a2", "a3"]
        self.start = [1] * len(self.params)
        self.limits = [(-1000, 1000)] * len(self.params)

    def model(self):
        def model(x, a0, a1, a2, a3):
            return bernstein.density(
                x, (a0, a1, a2, a3), self.xr[0], self.xr[1])
        return model

    def integral(self):
        def integral(x, a0, a1, a2, a3):
            return bernstein.integral(
                x, (a0, a1, a2, 3), self.xr[0], self.xr[1])
        return integral
