"""Functions for working with hysteresis measurement data.

Currently only includes Faraday correction via analytic optimization.
Faraday correction based on measured data can be added on request.
"""

from typing import Sequence
from numpy.typing import NDArray, ArrayLike
import numpy as np
from scipy.optimize import curve_fit, minimize, minimize_scalar
import logging

logger = logging.getLogger(__name__)


def normalize(*magnetization: Sequence,
              saturation_mean_percentage: float = 10) -> NDArray:
    """Normalizes hysteresis to the saturation region.

    Magnetization is normalized to the averaged saturation over a given
    range. The range over which to average is computed from a
    percentage relative to the number of data points in one branch.
    The magnetization needs to be supplied either as one continous
    sequence containing both hysteresis branches or both branches as
    separate sequences of uniform length. In the latter case, the
    second branch needs to be sorted by time recorded, not by magnetic
    field value.

    Parameters
    ----------
    *magnetization: Sequence
        One or two sequences containing the hysteresis branches of the
        magnetization.
    saturation_mean_percentage: float, default 10
        Percentage of half the length of ``magnetization`` over which
        to average the saturation value.

    Returns
    -------
    NDArray
        The normalized magnetization values. If ``magnetization`` is
        one continous array, then a 1-D array with the same length is
        returned. If ``magnetization`` is two arrays, a 2-D array is
        returned containing the two branches in the first axis.
    """

    if len(magnetization) == 1:
        branch_1, branch_2 = np.split(np.array(magnetization[0]), 2)

    if len(magnetization) == 2:
        branch_1, branch_2 = np.array(magnetization)

    if len(magnetization) > 2:
        raise ValueError("Too many input sequences.")

    # calculate number of points over wich to average saturation
    # ceil ensures the presence of at least one data point
    mean_range = np.ceil(len(branch_1) * saturation_mean_percentage
                         / 100).astype(int)

    # normalize
    max_mag = np.mean(np.concatenate((branch_1[:mean_range],
                                      branch_2[len(branch_2) - mean_range:])))
    min_mag = np.mean(np.concatenate((branch_2[:mean_range],
                                      branch_1[len(branch_2) - mean_range:])))

    branch_1 = 2 * (branch_1 - min_mag) / (max_mag - min_mag) - 1
    branch_2 = 2 * (branch_2 - min_mag) / (max_mag - min_mag) - 1

    if len(magnetization) == 1:
        return np.concatenate((branch_1, branch_2))

    if len(magnetization) == 2:
        return np.array((branch_1, branch_2))


def shift_branch(*magnetization: Sequence, shift_value: float) -> NDArray:
    """Shifts hysteresis branches relative to each other.

    The shift is applied to the second branch.

    Parameters
    ----------
    *magnetization: Sequence
        One or two sequences containing the hysteresis branches of the
        magnetization.
    shift_value: float
        Value added to the second hysteresis branch.

    Returns
    -------
    NDArray
        The shifted magnetization values. If ``magnetization`` is
        one continous array, then a 1-D array with the same length is
        returned. If ``magnetization`` is two arrays, a 2-D array is
        returned containing the two branches in the first axis.
    """

    if len(magnetization) == 1:
        branch_1, branch_2 = np.split(np.array(magnetization[0]), 2)

    if len(magnetization) == 2:
        branch_1, branch_2 = np.array(magnetization)

    if len(magnetization) > 2:
        raise ValueError("Too many input sequences.")

    # python built-in lists don't support += operator for maths
    branch_2 = np.array(branch_2)
    branch_2 = branch_2 + shift_value

    if len(magnetization) == 1:
        return np.concatenate((branch_1, branch_2))

    if len(magnetization) == 2:
        return np.array((branch_1, branch_2))


def drift_correction(*hysteresis: ArrayLike, slope: float) -> NDArray:
    """Corrects thermal drift.

    Thermal correction is applied by subtracting a linear function from
    the time ordered hysteresis loop.
    Hysteresis data can be supplied either as one tuple containing the
    magntic field and magnetization repectively or as multiple tuples
    containing the two hysteresis branches.
    This function is purely for subtracting a linear function in time
    domain and does not determine the optical slope for the drift
    correction itself.

    Parameters
    ----------
    *hysteresis: ArrayLike
        One or two 2-D arrays of shape (2, n) containing the
        magnetic field and magnetization of length n in the first axis.
    slope: float
        Slope of the linear function ``M(H) = slope * H`` to subtract.

    Returns
    -------
    NDArray
        The corrected magnetization values. If ``hysteresis`` is
        supplied as whole sequence , then a 1-D array with the same
        length as the magnetization is returned. If ``hysteresis`` is
        two branches, a 2-D array is returned containing the
        two magnetization branches in the first axis.


    Examples
    --------
    Viable example inputs.

    >>> drift_correction((H, M), slope=10)
    array([<M-values>])

    >>> drift_correction((H1, M1), (H2, M2), slope=10)
    array([[<M1-values>], [<M2-values>]])

    >>> hyst = array([H, M])
    >>> drift_correction(hyst, slope=10)
    array([<M-values>])

    >>> hyst = array([[H1, M1], [H2, M2]])
    >>> drift_correction(*hyst, slope=10)
    array([[<M1-values>], [<M2-values>]])
    """

    if len(hysteresis) == 1:
        H, M = np.array(hysteresis[0])
        H_1, H_2 = np.split(H, 2)
        M_1, M_2 = np.split(M, 2)

    if len(hysteresis) == 2:
        (H_1, M_1), (H_2, M_2) = np.array(hysteresis)
        # to support math operations for built-in lists

    if len(hysteresis) > 2:
        raise ValueError("Too many input sequences.")

    M_1 = M_1 + H_1 * slope
    offset = M_1[-1] - M_2[0] + H_2[0] * slope  # preserves line continuity
    M_2 = M_2 - H_2 * slope + offset

    if len(hysteresis) == 1:
        return np.concatenate((M_1, M_2))

    if len(hysteresis) == 2:
        return np.array((M_1, M_2))


def linear_correction(*hysteresis: ArrayLike,
                      saturation_percentage: float = 10) -> NDArray:
    """Subtracts a linear fit to the saturation regions.

    Linear fit is performed on the saturation region with respect to the
    magnetic field. The saturation region is defined by a percentage
    value relative to the number of data points of a single branch.
    A function ``f(x) = fit_slope * x`` is then subtracted from the
    magnetization.
    Hysteresis data can be supplied either as one tuple containing the
    magntic field and magnetization repectively or as multiple tuples
    containing the two hysteresis branches.

    Parameters
    ----------
    *hysteresis: ArrayLike
        One or two 2-D arrays of shape (2, n) containing the
        magnetic field and magnetization of length n in the first axis.
    saturation_percentage: float, default 10
        Percentage value of points in at the end of the hysteresis
        branches belonging to the saturation region.

    Returns
    -------
    NDArray
        The corrected magnetization values. If ``hysteresis`` is
        supplied as whole sequence , then a 1-D array with the same
        length as the magnetization is returned. If ``hysteresis`` is
        two branches, a 2-D array is returned containing the
        two magnetization branches in the first axis.

    Examples
    --------
    Viable example inputs.

    >>> linear_correction((H, M), saturation_percentage=10)
    array([<M-values>])

    >>> linear_correction((H1, M1), (H2, M2), saturation_percentage=10)
    array([[<M1-values>], [<M2-values>]])

    >>> hyst = array([H, M])
    >>> linear_correction(hyst, saturation_percentage=10)
    array([<M-values>])

    >>> hyst = array([[H1, M1], [H2, M2]])
    >>> linear_correction(*hyst, saturation_percentage=10)
    array([[<M1-values>], [<M2-values>]])
    """

    if len(hysteresis) == 1:
        H, M = np.array(hysteresis[0])
        H_1, H_2 = np.split(H, 2)
        M_1, M_2 = np.split(M, 2)

    if len(hysteresis) == 2:
        (H_1, M_1), (H_2, M_2) = np.array(hysteresis)
        # to support math operations for built-in lists
        H_1, M_1 = np.array(H_1), np.array(M_1)
        H_2, M_2 = np.array(H_2), np.array(M_2)

    if len(hysteresis) > 2:
        raise ValueError("Too many input sequences.")

    # ceil ensures at least one data point
    sat_range = np.ceil(len(H_1) * saturation_percentage / 100).astype(int)
    H1_sat = np.concatenate((H_1[:sat_range], H_2[len(H_1) - sat_range:]))
    H2_sat = np.concatenate((H_2[:sat_range], H_1[len(H_1) - sat_range:]))
    M1_sat = np.concatenate((M_1[:sat_range], M_2[len(H_1) - sat_range:]))
    M2_sat = np.concatenate((M_2[:sat_range], M_1[len(H_1) - sat_range:]))

    # fit linear function to saturation region only
    fit_para1, _ = curve_fit(lambda x, m, t: m*x+t, H1_sat, M1_sat)
    fit_para2, _ = curve_fit(lambda x, m, t: m*x+t, H2_sat, M2_sat)
    fit_para = np.mean((fit_para1, fit_para2), axis=0)

    # subtract slope from entire magnetization
    M_1 = M_1 - fit_para[0] * H_1
    M_2 = M_2 - fit_para[0] * H_2

    if len(hysteresis) == 1:
        return np.concatenate((M_1, M_2))

    if len(hysteresis) == 2:
        return np.array((M_1, M_2))


def faraday_correction(*hysteresis: ArrayLike,
                       initial_guess: Sequence[float],
                       saturation_percentage: float = 10,
                       log_level: int = logging.INFO) -> NDArray:
    """Performs a correction of the Faraday-effect via analytic
    optimization method.

    Note: It is recommended to use ``optimzed_verdet_correction``
    instead as it removes the additional guess of the cos amplitude.

    Performs optimization of a functional describing the difference
    between saturation region and analytic expression of the Faraday-
    effect. The correction might need to be combined with other
    correction methods (thermal drift, relative shifting, ...) to obtain
    proper looking hysteresis curves.
    Note, that the optimized functional considers the linearly
    corrected data, but the function itself does not subtract the linear
    part.
    Hysteresis data can be supplied either as one tuple containing the
    magntic field and magnetization repectively or as multiple tuples
    containing the two hysteresis branches.

    Parameters
    ----------
    *hysteresis: ArrayLike
        One or two 2-D arrays of shape (2, n) containing the
        magnetic field and magnetization of length n in the first axis.
    initial_guess: Sequence[float, float, float]
        Initial guess for verdet constant of the objective lens,
        amplitude and left shift of cos^2 function.
    saturation_percentage: float, default 10
        Percentage value of points in at the end of the hysteresis
        branches belonging to the saturation region.
    log_level: int, default 10
        log behavior of ``python.logging`` module for the optimization
        results.

    Returns
    -------
    NDArray
        The corrected magnetization values. If ``hysteresis`` is
        supplied as whole sequence , then a 1-D array with the same
        length as the magnetization is returned. If ``hysteresis`` is
        two branches, a 2-D array is returned containing the
        two magnetization branches in the first axis.

    Examples
    --------
    Viable example inputs.

    >>> faraday_correction((H, M), initial_guess=(0.01, 5, 100),
                            saturation_percentage=10)
    array([<M-values>])

    >>> faraday_correction((H1, M1), (H2, M2), initial_guess=(0.01, 5, 100),
                            saturation_percentage=10)
    array([[<M1-values>], [<M2-values>]])

    >>> hyst = array([H, M])
    >>> faraday_correction(hyst, initial_guess=(0.01, 5, 100),
                            saturation_percentage=10)
    array([<M-values>])

    >>> hyst = array([[H1, M1], [H2, M2]])
    >>> faraday_correction(*hyst, initial_guess=(0.01, 5, 100),
                            saturation_percentage=10)
    array([[<M1-values>], [<M2-values>]])
    """

    if len(hysteresis) == 1:
        H, M = np.array(hysteresis[0])

    if len(hysteresis) == 2:
        (H_1, M_1), (H_2, M_2) = np.array(hysteresis)
        H = np.concatenate((H_1, H_2))
        M = np.concatenate((M_1, M_2))

    if len(hysteresis) > 2:
        raise ValueError("Too many input sequences.")

    # ceil ensures at least one data point
    sat_range = np.ceil(len(H) / 2 * saturation_percentage / 100).astype(int)

    def cos_sq(x, A, v, s):
        """Parameterized cos^2 function.

        Parameters
        ----------
        x: float
            function input.
        A: float
            amplitude
        v: float
            verdet constant
        s: float
            left shift
        """

        return A * np.cos(v * (x + s) + np.pi/2)**2

    def functional(x):
        """Functional to optimize.

        Assumes the ideal correction is reached, when the saturation
        flattens out.
        """

        cos_squared = cos_sq(H, *x)
        corrected = linear_correction(
            (H, M - cos_squared), saturation_percentage=saturation_percentage)
        derivative = np.gradient(corrected)
        branch_1, branch_2 = np.split(derivative, 2)
        saturation = np.concatenate((
            branch_1[:sat_range], branch_1[len(branch_1) - sat_range:],
            branch_2[:sat_range], branch_2[len(branch_2) - sat_range:]))

        return np.mean(np.abs(saturation))

    # optimize the functional
    minimum = minimize(functional, x0=initial_guess)
    logger.log(log_level,
               f"Computed minimum for faraday correction:\n{minimum}")
    M_corrected = M - cos_sq(H, *(minimum.x))

    if len(hysteresis) == 1:
        return M_corrected

    if len(hysteresis) == 2:
        return np.array(np.split(M_corrected, 2))


def optimized_faraday_correction(*hysteresis: ArrayLike,
                                 verdet_guess: float,
                                 left_shift_guess: float,
                                 bracket: Sequence[float] = None,
                                 bounds: Sequence[float] = None,
                                 saturation_percentage: float = 10,
                                 log_level: int = logging.INFO) -> NDArray:
    """Performs a correction of the Faraday-effect via analytic
    optimization method.

    Performs optimization of a functional describing the difference
    between saturation region and analytic expression of the Faraday-
    effect. The correction might need to be combined with other
    correction methods (thermal drift, relative shifting, ...) to obtain
    proper looking hysteresis curves.
    Note, that the optimized functional considers the linearly
    corrected data, but the function itself does not subtract the linear
    part.
    Hysteresis data can be supplied either as one tuple containing the
    magntic field and magnetization repectively or as multiple tuples
    containing the two hysteresis branches.

    Parameters
    ----------
    *hysteresis: ArrayLike
        One or two 2-D arrays of shape (2, n) containing the
        magnetic field and magnetization of length n in the first axis.
    verdet_guess: float
        Initial guess for verdet constant of the objective lens.
    left_shift_guess: float
        Initial guess for the extremum position of the cos^2 function.
    bracket: Sequence[float, float(, float)], default None
        Bracket parameter of ``scipy.optimize.minimize_scalar``.
        Defines the bracketing interval.
        Must be supplied if ``bounds`` is ``None``.
    bounds: Sequence[float, float]
        Bounds parameter of ``scipy.optimizle.minimize_scalar``.
        Defines the upper and lower bounds of the guess values.
        Must be supplied if ``bracket`` is ``None``.
    saturation_percentage: float, default 10
        Percentage value of points in at the end of the hysteresis
        branches belonging to the saturation region.
    log_level: int, default 10
        log behavior of ``python.logging`` module for the optimization
        results.

    Returns
    -------
    NDArray
        The corrected magnetization values. If ``hysteresis`` is
        supplied as whole sequence , then a 1-D array with the same
        length as the magnetization is returned. If ``hysteresis`` is
        two branches, a 2-D array is returned containing the
        two magnetization branches in the first axis.

    Examples
    --------
    The optimization parameters for this function try to describe the
    curvature in a hysteresis curve with a cos^2 function. The
    verdet constant is a material constant, that in this cas scales the
    width of the cos^2 function. For usual field ranges (10 - 100 mT)
    it is quite small and is in the proximity of 0.01 or lower.
    The left shift describes were the extremum of the of the cos^2 is
    and is in field units. This parameter can usually be estimated by
    plotting the hysteresis beforehand. The ``bounds`` or ``bracket``
    parameters describe the amplitude of the cos^2 and can take values
    from the range of the distance between saturation branches up until
    several times of that.

    The following example demonstrates a typical correction code.
    >>> M, H = ...
    >>> M = drift_correction((H, M), slope=0.2)
    >>> M = shift_branch(M, shift_value=5)
    >>> M = optimized_faraday_correction((H, M),
                                         verdet_guess=0.01,
                                         left_shift_guess=5,
                                         bounds=(20, 500),
                                         saturation_percentage=30)
    >>> M = linear_correction((H, M), saturation_percentage=30)

    The first two functions clean the raw data from potential thermal
    drift and tries to align the two hysteresis branches as good as
    possible. The better the alignment beforehand, the better the
    faraday correction works. It is suggested to plot the intermediate
    states and play with the parameters until it aligns best.

    The faraday correction is then applied with the recommended guess
    parameters. Don't expect these to work right out of the box though.
    Sometimes, one has to play around with the guess parameters to find
    a good solution. For some measurement data it might not work at all
    due to particular strong distortions or other effects. Generally, a
    longer saturation region in the input data yields better results.

    Finally, a linear correction is applied to remove any remaining
    hysteresis tilt. The linear correction is considered in the
    optimized functional of the faraday correction. However, the
    correction itself only removes the faraday contribution itself. For
    a clean hysteresis this last step is therefore necessary.

    Viable example inputs.
    >>> faraday_correction((H, M), verdet_guess=0.01, left_shift_guess=5,
                            saturation_percentage=10)
    array([<M-values>])

    >>> faraday_correction((H1, M1), (H2, M2), verdet_guess=0.01,
                            left_shift_guess=5,
                            saturation_percentage=10)
    array([[<M1-values>], [<M2-values>]])

    >>> hyst = array([H, M])
    >>> faraday_correction(hyst, verdet_guess=0.01, left_shift_guess=5,
                            saturation_percentage=10)
    array([<M-values>])

    >>> hyst = array([[H1, M1], [H2, M2]])
    >>> faraday_correction(*hyst, verdet_guess=0.01, left_shift_guess=5,
                            saturation_percentage=10)
    array([[<M1-values>], [<M2-values>]])
    """

    if len(hysteresis) == 1:
        H, M = np.array(hysteresis[0])

    if len(hysteresis) == 2:
        (H_1, M_1), (H_2, M_2) = np.array(hysteresis)
        H = np.concatenate((H_1, H_2))
        M = np.concatenate((M_1, M_2))

    if len(hysteresis) > 2:
        raise ValueError("Too many input sequences.")

    # ceil ensures at least one data point
    sat_range = np.ceil(len(H)/2 * saturation_percentage / 100).astype(int)

    def functional(scale):
        """Wrapping functional for determining the correct amplitude.

        Assumes the ideal correction is reached, when the saturation
        flattens out.
        """

        corrected = faraday_correction(
            (H, M), initial_guess=(scale, verdet_guess, left_shift_guess),
            saturation_percentage=saturation_percentage,
            log_level=logging.DEBUG
        )
        derivative = np.gradient(corrected)
        branch_1, branch_2 = np.split(derivative, 2)
        saturation = np.concatenate((
            branch_1[:sat_range], branch_1[len(branch_1) - sat_range:],
            branch_2[:sat_range], branch_2[len(branch_2) - sat_range:]))

        return np.mean(np.abs(saturation))

    # optimize the functional
    if bracket is not None:
        minimum = minimize_scalar(functional, bracket=bracket)
    if bounds is not None:
        minimum = minimize_scalar(functional, bounds=bounds, method="bounded")

    logger.log(log_level, f"Computed minimum for amplitude guess:\n{minimum}")
    M_corrected = faraday_correction(
        (H, M), initial_guess=(minimum.x, verdet_guess, left_shift_guess),
        saturation_percentage=saturation_percentage, log_level=log_level
    )

    if len(hysteresis) == 1:
        return M_corrected

    if len(hysteresis) == 2:
        return np.array(np.split(M_corrected, 2))
