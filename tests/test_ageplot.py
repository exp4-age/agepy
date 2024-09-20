import pytest
import matplotlib.pyplot as plt
from agepy import ageplot


def test_use_with_age_styles():
    """Test the validity of all age styles.

    """
    for style in ageplot.age_styles:
        ageplot.use(style)

def test_use_with_mpl_styles():
    """Test the compatibility with all mpl styles.

    """
    for style in ageplot.mpl_styles:
        ageplot.use(style)

def test_use_with_multiple_styles():
    """Test the compatibility with the base age style.

    """
    for style in ageplot.age_styles:
        if style != "age":
            ageplot.use(["age", style])

def test_context_manager():
    """Test the context manager with all age styles.

    """
    # Save the original rcParams
    original_rc_params = plt.rcParams.copy()
    for style in ageplot.age_styles:
        with ageplot.context(style):
            pass
        # Check if rcParams are reset
        assert plt.rcParams.copy() == original_rc_params
