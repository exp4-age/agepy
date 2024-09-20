"""Functions for working with trajectories.

"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.typing import DataFrameGroupBy
import warnings


def euclid_dist(p: np.ndarray, q: np.ndarray) -> np.float64:
    """Get euclidean distance between two points.

    """
    return np.linalg.norm(p - q)


def filter_short(df: pd.DataFrame, min_len: float, printing: bool = False) -> DataFrameGroupBy:
    """Filter trajectories by euclidean distance of x_end - x_start.

    """
    if printing:
        print(f"before: {df.particle.nunique()}")
    filtered = df.groupby("particle").filter(
        lambda g: (euclid_dist(g.x.iloc[0], g.x.iloc[-1]) > min_len)
    )
    if printing:
        print(f"after filtering short: {filtered.particle.nunique()}")
    return filtered


def drop_x_y(df: pd.DataFrame, xmin: int, ymin: int) -> pd.DataFrame:
    """Return dataframe without particles located below xmin or ymin.

    This gets rid of particles too close to the border.
    Do this before linking.

    """
    return df.drop(df[(df.x < xmin) | (df.y < ymin)].index).reset_index(drop=True)


def add_diff(df: pd.DataFrame, periods: int = 1, dim_col: str = "x") -> pd.DataFrame:
    """Adds a "diff" column for differentiation of a coordinate.

    This function adds "diff_x" for the differentiation of x the column.
    This function adds the column to the original DataFrame, so its impure :(

    Parameters
    ----------
    df: DataFrame
        with particle data (from trackpy)
    periods: int, optional
        Window of frames to get diff/velocity from. Default is 1.
    dim_col: string, optional
        Calculate velocity on column x or y. Default is "x".

    Returns
    -------
    df: DataFrame
        The original dataframe with the new "diff" column

    """
    if dim_col == "x":
        df["diff_x"] = df.groupby("particle")["x"].diff(periods=periods)
    elif dim_col == "y":
        df["diff_y"] = df.groupby("particle")["y"].diff(periods=periods)
    elif dim_col == "z":
        df["diff_z"] = df.groupby("particle")["z"].diff(periods=periods)
    else:
        warnings.warn(
            f"The column {dim_col} is not x/y/z. Trying to differentiate anyways. Check your results!"
        )
        df[f"diff_{dim_col}"] = df.groupby("particle")[f"{dim_col}"].diff(
            periods=periods
        )
    return df


def get_vmax(df: pd.DataFrame, periods: int = 1, dim_col: str = "x") -> float:
    """Calculates the absolute mean maximum velocity of the particles.

    Takes the dataframe of particles, calculates diff for given dimension,
    then maximum absolute value for each particle in this df.
    Finally returns the mean of all these particles.

    Parameters
    ----------
    df: DataFrame
        with particle data (from trackpy)
    periods: int, optional
        Window of frames to get diff/velocity from. Default is 1.
    dim_col: string, optional
        Calculate velocity on column x or y. Default is "x".

    Returns
    -------
    vmax:
        The mean absolute maximum velocity of each particle.

    """
    df = df.copy()
    df_diff = add_diff(df, periods, dim_col)
    df_diff[f"diff_{dim_col}_abs"] = df_diff[f"diff_{dim_col}"].abs()
    vmax = df_diff.groupby("particle")[f"diff_{dim_col}_abs"].transform("max").mean()
    return vmax


def get_vavg(
    df: pd.DataFrame, periods: int = 1, dim_col: str = "x", cutoff: float = 0.5
) -> pd.Series:
    """Calculates the mean velocity of the particles.

    Takes the dataframe of particles, calculates diff for given dimension, then mean of absolute value for each particle in this df. Finally returns the mean of all these particles.

    Parameters
    ----------
    df: DataFrame
        with particle data (from trackpy)
    periods: int, optional
        Window of frames to get diff/velocity from. Default is 1.
    dim_col: string, optional
        Calculate velocity on column x or y. Default is "x".
    cutoff: float, optional
        Will only evaluate velocities which are x% of max. v.

    Returns
    -------
    vavg: pd.Series
        The mean absolute maximum velocity of each particle.

    """
    df = df.copy()
    diff_max = get_vmax(df)
    df_diff = add_diff(df, periods, dim_col)
    df_diff[f"diff_{dim_col}_abs"] = df_diff[f"diff_{dim_col}"].abs()
    df_grouped = df_diff[df_diff[f"diff_{dim_col}_abs"] > cutoff * diff_max].groupby(
        "particle"
    )
    vavg = df_grouped[f"diff_{dim_col}_abs"].transform("mean").mean()
    return vavg


def get_mobile(
    df: pd.DataFrame, thresh: float, periods: int = 1, dim_col: str = "x"
) -> np.ndarray:
    """Get mobile particles.

    Particles are mobile if their velocity between two frames is greather than the given threshold value.

    Parameters
    ----------
    df: DataFrame
        with particle data (from trackpy).
    thresh: float
        threshold velocity.
    periods: int, optional
        Window of frames to get diff/velocity from. Default is 1.
    dim_col: string, optional
        Calculate velocity on column x or y. Default is "x".

    Returns
    -------
    mobile: ndarray
        Sorted array with names/numbers of mobile particles.

    """
    df_copy = df.copy()
    if dim_col == "x":
        df_copy["diff_x"] = df_copy.groupby("particle")["x"].transform(
            "diff", periods=periods
        )
        mobile = df_copy[abs(df_copy.diff_x) > thresh].particle.unique()
    if dim_col == "y":
        df_copy["diff_y"] = df_copy.groupby("particle")["y"].transform(
            "diff", periods=periods
        )
        mobile = df.copy[abs(df.copy.diff_y) > thresh].particle.unique()
    return np.sort(mobile)


def get_mobility(
    df: pd.DataFrame, thresh: float, periods: int = 1, dim_col: str = "x"
) -> float:
    """Get mobility percentage of particles.

    Particles are mobile if their velocity between two frames is greather than the given threshold value.

    Parameters
    ----------
    df: DataFrame
        with particle data (from trackpy).
    thresh: float
        threshold velocity.
    periods: int, optional
        Window of frames to get diff/velocity from. Default is 1.
    dim_col: string, optional
        Calculate velocity on column x or y. Default is "x".

    Returns
    -------
    mobility: float
        percentage of mobile particles.

    """
    mobile = get_mobile(df, thresh, periods=periods, dim_col=dim_col)
    return len(mobile) / df.particle.nunique()
