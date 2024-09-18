#!/usr/bin/env python3
import numpy as np
from src.agepy.mag import trajectory

def test_euclid_dist():
    p = np.array([0, 0])
    q = np.array([0, 1])
    q1 = np.array([0, -1])
    assert trajectory.euclid_dist(p, q) == 1
    assert trajectory.euclid_dist(p, q1) == 1


def test_filter_short():
    pass


def test_drop_x_y():
    pass


def test_add_diff():
    pass


def test_get_vmax():
    pass


def test_get_vavg():
    pass


def test_get_mobile():
    pass


def test_get_mobility():
    pass
