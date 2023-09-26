# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from agepy import __version__ as version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "agepy"
release = f"{version}"
copyright = "2023, Adrian Peter Krone and AGE"
author = "Adrian Peter Krone"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx" # https://jupyter-tutorial.readthedocs.io/de/latest/sphinx/nbsphinx.html
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_google_docstring = False
napoleon_include_init_with_doc = True

copybutton_prompt_text = ">>> "



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo_age.png"
