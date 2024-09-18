#!/usr/bin/env python3
import pytest
import numpy as np
import pandas as pd
from agepy.mag import trajectory

@pytest.fixture
def sample_df():
    '''Return a sample DataFrame for testing.'''
    data = {
        'particle': [1, 1, 1, 2, 2, 2, 3, 3],
        'x': [0, 1, 2, 0, 2, 4, 10, 10],
        'y': [0, 1, 2, 0, 1, 2, 10, 10],
        'frame': [0, 1, 2, 0, 1, 2, 0, 1]
    }
    return pd.DataFrame(data)

def test_euclid_dist():
    p1 = np.array([0, 0])
    q1 = np.array([0, 1])
    q2 = np.array([0, -1])
    q3 = np.array([3, 4])
    assert trajectory.euclid_dist(p1, q1) == 1
    assert trajectory.euclid_dist(p1, q2) == 1
    assert trajectory.euclid_dist(p1, q3) == pytest.approx(5.0)


def test_filter_short(sample_df):
    filtered = trajectory.filter_short(sample_df, min_len=2.5)
    assert filtered['particle'].nunique() == 1
    assert 1  not in filtered['particle'].unique()
    assert 2  in filtered['particle'].unique()
    assert 3  not in filtered['particle'].unique()


def test_drop_x_y(sample_df):
    result = trajectory.drop_x_y(sample_df, xmin=1, ymin=1)
    assert result['x'].min() >= 1
    assert result['y'].min() >= 1


def test_add_diff(sample_df):
    result = trajectory.add_diff(sample_df, periods=1, dim_col='x')
    result = trajectory.add_diff(sample_df, periods=1, dim_col='y')
    assert 'diff_x' in result.columns
    assert 'diff_y' in result.columns
    assert result['diff_x'].iloc[1] == 1.0
    assert result['diff_x'].iloc[4] == 2.0

def test_get_vmax(sample_df):
    vmax = trajectory.get_vmax(sample_df, periods=1, dim_col='x')
    vmax_1 = trajectory.get_vmax(sample_df[sample_df.particle == 1], periods=1, dim_col='x')
    vmax_2 = trajectory.get_vmax(sample_df[sample_df.particle == 2], periods=1, dim_col='x')
    vmax_3 = trajectory.get_vmax(sample_df[sample_df.particle == 3], periods=1, dim_col='x')
    assert vmax_1 == 1.0
    assert vmax_2 == 2.0
    assert vmax_3 == 0.0
    assert vmax == 1.125

def test_get_vavg(sample_df):
    vavg = trajectory.get_vavg(sample_df, periods=1, dim_col='x', cutoff=0.1)
    assert np.isclose(vavg, 1.5)

def test_get_mobile(sample_df):
    mobile = trajectory.get_mobile(sample_df, thresh=1.5, periods=1, dim_col='x')
    assert np.array_equal(mobile, np.array([2]))

def test_get_mobility(sample_df):
    mobility = trajectory.get_mobility(sample_df, thresh=1.5, periods=1, dim_col='x')
    assert np.isclose(mobility, 1/3)


# Additional tests for edge cases and error handling

def test_filter_short_empty_df():
    empty_df = pd.DataFrame(columns=['particle', 'x', 'y', 'frame'])
    result = trajectory.filter_short(empty_df, min_len=1)
    assert result.empty

def test_add_diff_invalid_column():
    df = pd.DataFrame({'particle': [1, 1], 'invalid': [0, 1]})
    with pytest.warns(UserWarning):
        result = trajectory.add_diff(df, dim_col='invalid')
    assert 'diff_invalid' in result.columns

def test_get_vmax_single_frame():
    df = pd.DataFrame({'particle': [1, 2], 'x': [0, 1], 'frame': [0, 0]})
    vmax = trajectory.get_vmax(df)
    assert np.isnan(vmax)

def test_get_mobility_all_static():
    df = pd.DataFrame({'particle': [1, 1, 2, 2], 'x': [0, 0, 1, 1], 'frame': [0, 1, 0, 1]})
    mobility = trajectory.get_mobility(df, thresh=0.1)
    assert mobility == 0.0

def test_get_mobility_all_mobile():
    df = pd.DataFrame({'particle': [1, 1, 2, 2], 'x': [0, 2, 1, 3], 'frame': [0, 1, 0, 1]})
    mobility = trajectory.get_mobility(df, thresh=1.5)
    assert mobility == 1.0
