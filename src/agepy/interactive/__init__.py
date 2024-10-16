"""QT6 based interactive tools for agepy.

"""
from ._interactive import AGEDataViewer, AGEpp, FloatSlider
from .fitting.spec import fit_spectrum, fit_calibration
from .fitting.mag import fit_mag

__all__ = [
    "fit_spectrum",
    "fit_calibration",
    "fit_mag",
]
