"""Interactive fitting.

"""
from ._fitting import AGEFitViewer, AGEFitBackend
from ._spec import (
    fit_spectrum,
    fit_calibration,
    SpectrumFit,
    CalibrationFit
)
from ._mag import fit_mag

__all__ = [
    "fit_spectrum",
    "SpectrumFit",
    "fit_calibration",
    "CalibrationFit",
    "fit_mag",
]
