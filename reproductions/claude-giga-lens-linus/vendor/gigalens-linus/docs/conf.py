# -- gigalens documentation build configuration ------------------------------
#
# Sphinx + MyST: author everything in Markdown, generate the API reference from
# the (RST-flavoured) docstrings in src/gigalens via autodoc + autosummary.
#
# Design choices (see docs/README.md for the rationale):
#   * MyST-Parser  -> Markdown authoring, admonitions, tabs, math.
#   * myst-nb      -> render demo notebooks as tutorial pages (NOT executed in
#                     CI; they need a GPU -- see nb_execution_mode below).
#   * autodoc      -> renders the existing ``:class:``-style RST docstrings
#                     natively, so no docstring rewrites are needed.
#   * intersphinx  -> auto-links jax / numpy / tfp / lenstronomy symbols.
#
# The build must NOT require a GPU (or even a working JAX/TF install): heavy
# leaf dependencies are mocked (autodoc_mock_imports) so autodoc can introspect
# gigalens' own classes/signatures from source without importing the real
# numerical stack. If you hit import-time errors from a class *body* that runs
# JAX at definition time, switch the reference pages to the static-analysis
# backend (sphinx-autodoc2) documented in docs/README.md.

import os
import sys
from datetime import date

# Make the package importable for autodoc (src/ layout).
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "gigalens"
author = "GIGALens developers"
copyright = f"2021–{date.today().year}, {author}"

# Single-source the version from the installed package if available.
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("gigalens")
except Exception:
    release = "1.2.1"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    # NB: do NOT also list "myst_parser" here — myst_nb loads it and registers the
    # ".md" parser itself; listing both triggers a duplicate-parser warning.
    "myst_nb",                  # Markdown (MyST) + Jupyter notebooks as pages
    "sphinx.ext.autodoc",       # pull docstrings from source
    "sphinx.ext.autosummary",   # generate per-object summary tables/stubs
    "sphinx.ext.napoleon",      # tolerate Google/NumPy "Args:"/"Attributes:" docstrings
    "sphinx.ext.intersphinx",   # cross-link to jax/numpy/tfp docs
    "sphinx.ext.viewcode",      # [source] links
    "sphinx.ext.mathjax",       # lensing math
    "sphinx_copybutton",        # copy button on code blocks
    "sphinx_design",            # grids/cards/tabs (pairs with MyST)
]

# myst_nb registers the ".md" and ".ipynb" parsers; do not also register
# myst_parser's ".md" handler or Sphinx warns about a duplicate.
source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".rst": "restructuredtext",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# Render a dataclass/class ``Attributes:`` section as :ivar: fields inside the
# class body instead of as separate object descriptions — otherwise autodoc also
# documents the dataclass fields and every attribute is described twice.
napoleon_use_ivar = True

# The library's RST docstrings use a few conventions docutils flags as warnings —
# bare magnitude bars (``|alpha|``, ``|gamma|``) read as RST substitutions, and a
# couple of inline-literal/definition-list quirks. These are upstream-docstring
# cosmetics, not doc-site defects; suppress only the ``docutils`` category so the
# build can still run under -W. Cross-reference / toctree / MyST warnings (the
# ones that catch real doc breakage) are separate categories and stay fatal.
suppress_warnings = ["docutils"]

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",     # ::: fenced admonitions/directives
    "deflist",         # definition lists
    "dollarmath",      # $...$ / $$...$$ math
    "amsmath",         # AMS environments
    "fieldlist",       # :field: lists
    "attrs_inline",    # inline attributes
    "linkify",         # bare URLs -> links
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3  # auto-slug h1-h3 so cross-page #anchors resolve

# -- Notebook (myst-nb) configuration ----------------------------------------
# Demo notebooks are GPU/JAX-heavy; embed them PRE-EXECUTED and never run them
# during the docs build. Re-run notebooks locally on a GPU and commit outputs.
nb_execution_mode = "off"

# -- autodoc / autosummary ---------------------------------------------------

# The autosummary directives are used only for the summary TABLES at the top of
# each reference page; the objects themselves are documented by automodule on the
# same page. Generating stub pages too would double-document every member.
autosummary_generate = False
autodoc_typehints = "description"        # move type hints into the body
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Autodoc imports each module to read its docstrings. We install the *light*
# CPU deps that run at import time (numpy, scipy, a CPU-only jax — see
# requirements.txt) and mock only the genuinely heavy or optional ones. jax must
# NOT be mocked: several profiles evaluate constants like ``jnp.pi`` at import.
autodoc_mock_imports = [
    "tensorflow",
    "tensorflow_probability",
    "optax",
    "objax",
    "lenstronomy",
    "tqdm",
    "blackjax",
    "numpyro",
    "fastprogress",
]

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    # NB: TFP does not publish a standard objects.inv, so it is intentionally
    # omitted — adding it 404s and (under -W) fails the build.
}
# Cross-refs to un-inventoried externals shouldn't nitpick-fail the build.
intersphinx_disabled_reftypes = ["*"]

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "gigalens"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/seanxuseanxu/gigalens",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_prev_next": True,
    "navigation_with_keys": True,
    "icon_links": [],
}
# Fail the build on broken cross-references so docs can't silently rot.
nitpicky = False  # flip to True once the reference pages are complete
