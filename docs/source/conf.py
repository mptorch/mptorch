# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import mptorch

project = "mptorch"
copyright = "2024-2026, The MPTorch developers"
author = "The MPTorch developers"
release = mptorch.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = []


# Include the documentation of classes' constructor
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}
intersphinx_disabled_domains = ["std"]

# Every code block in these pages is a file under docs/snippets, included
# together with the output docs/run_snippets.py recorded for it.
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = []
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mptorch/mptorch",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        }
    ],
    "show_nav_level": 2,
    "navigation_with_keys": False,
}
html_title = "MPTorch"

# -- Options for EPUB output
epub_show_urls = "footnote"
