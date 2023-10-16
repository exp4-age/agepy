import pytest
from agepy import ageplot

def test_use_with_age_styles():
    for style in ageplot.age_styles:
        ageplot.use(style)

def test_use_with_mpl_styles():
    for style in ageplot.mpl_styles:
        ageplot.use(style)

def test_use_with_invalid_style():
    with pytest.raises(OSError):
        ageplot.use("non_existent_style")

def test_use_with_multiple_styles():
    ageplot.use(["age", "pccp"])

def test_use_with_invalid_input():
    with pytest.raises(TypeError):
        ageplot.use(123)