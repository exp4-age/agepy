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

rst_epilog = """

.. |versionshield| image:: https://img.shields.io/badge/version-{}-blue
   :target: https://img.shields.io/badge/version-{}-blue
.. |licenseshield| image:: https://img.shields.io/badge/License-MIT-blue
   :target: https://github.com/exp4-age/agepy/blob/main/LICENSE
.. |testshield| image:: https://github.com/exp4-age/agepy/actions/workflows/test.yml/badge.svg?branch=develop
   :target: https://github.com/exp4-age/agepy/tree/develop
.. |docsshield| image:: https://github.com/exp4-age/agepy/actions/workflows/docs.yml/badge.svg?branch=develop
   :target: http://141.51.197.64:9001

""".format(version, version)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",  # https://jupyter-tutorial.readthedocs.io/de/latest/sphinx/nbsphinx.html
]

autosummary_imported_members = True
autosummary_generate = True
autoclass_content = "class"
html_show_sourcelink = False
set_type_checking_flag = True
nbsphinx_allow_errors = True
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = True
numpydoc_xref_param_type = True
numpydoc_use_plots = True

plot_include_source = True
plot_html_show_source_link = False
plot_formats = [("hires.png", 300)]
plot_html_show_formats = False

autosummary_generate = ["reference"]

copybutton_prompt_text = ">>> "

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo_age.png"
