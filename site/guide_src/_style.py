"""Shared matplotlib style for the guide's figures.

Every figure ships in two variants — ``light`` and ``dark`` — selected at read
time by mkdocs-material's ``#only-light`` / ``#only-dark`` fragment rules::

    [data-md-color-scheme=default] img[src$="#only-dark"] {display:none}
    [data-md-color-scheme=slate]   img[src$="#only-light"]{display:none}

The selector is ``src$=``, so the fragment must be the literal END of ``src``.

DETERMINISM (the load-bearing part of this module)
--------------------------------------------------
Matplotlib SVG output is NOT reproducible by default: clip-path ids are salted
from a per-process random seed, and a ``<dc:date>`` timestamp is embedded. Two
identical renders therefore differ, which would dirty every committed figure on
every regeneration and make ``git diff`` useless. Both are fixed here:

  * ``rcParams["svg.hashsalt"]`` — a fixed salt pins the element ids.
  * ``savefig(metadata={"Date": None})`` — drops the timestamp (see ``save``).

Verified: default -> two renders differ; with both fixes -> byte-identical.
Never remove either without re-checking ``make_figures.py --check``.

COLOR
-----
Anchored on the USF palette already in site/docs/stylesheets/extra.css
(green #00543c, gold #FDBB30). Two adjustments are forced by contrast:

  * Gold #FDBB30 on white measures ~1.6:1 — illegible. The light series uses a
    darkened #A8730A instead.
  * The dark series leads with #6fc7a3, which is ALREADY the site's dark-mode
    link color in extra.css, so the figures inherit the site's brand continuity
    for free.

Backgrounds are transparent, so the Material page color always shows through
and matches exactly — including after any future palette-hue change.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

# Fixed salt -> deterministic clip-path / gradient ids in the SVG output.
# MUST be set before any figure is created.
matplotlib.rcParams["svg.hashsalt"] = "agentic-lensing-guide"

SCHEMES = ("light", "dark")

# Material's own colors for scheme=default / scheme=slate (hue 232).
PAGE_BG = {"light": "#ffffff", "dark": "#1e2029"}
INK = {"light": "#212121", "dark": "#bfc0c6"}
MUTED = {"light": "#757575", "dark": "#8c8d95"}
GRID = {"light": "#00000022", "dark": "#ffffff22"}

# Categorical series. Contrast-checked against each scheme's page background;
# all >= 4.1:1. Beyond three series, pair color with linestyle/marker — the
# deuteranopia separation tightens past that.
SERIES = {
    "light": ["#00543c", "#A8730A", "#1f6feb", "#b3282d", "#6a3d9a", "#127a80"],
    "dark": ["#6fc7a3", "#FDBB30", "#7aa7ff", "#ff8b8b", "#c9a0ff", "#5ecfd6"],
}

# Semantic colors used across chapters, so the same idea keeps the same hue.
ACCENT = {"light": "#A8730A", "dark": "#FDBB30"}   # highlight / "the answer"
WARN = {"light": "#b3282d", "dark": "#ff8b8b"}     # wrong / artifact / refuted
GOOD = {"light": "#00543c", "dark": "#6fc7a3"}     # correct / anchor

# Perceptually-uniform and scheme-neutral, so they need no per-variant change.
CMAP_SEQ = "viridis"
CMAP_DIV = "RdBu_r"


def rc(scheme: str) -> dict:
    """rcParams for one scheme. Pass to ``matplotlib.rc_context``."""
    ink, muted = INK[scheme], MUTED[scheme]
    return {
        # Transparent: the Material page background shows through and therefore
        # always matches, in both schemes and after any palette change.
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # 'path' (the mpl 3.11 default) renders text as outlines, so the SVG is
        # self-contained inside an <img>, which cannot reach the page's fonts.
        # Do NOT set 'none'.
        "svg.fonttype": "path",
        "font.family": "sans-serif",
        # Pinned to the matplotlib-bundled face so output is identical on any host.
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "text.color": ink,
        "axes.labelcolor": ink,
        "axes.titlecolor": ink,
        "axes.edgecolor": muted,
        "xtick.color": muted,
        "ytick.color": muted,
        "grid.color": GRID[scheme],
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES[scheme]),
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
        # ~ Material's 45rem content column at 100% width.
        "figure.figsize": (6.0, 4.0),
        "figure.dpi": 100,
        "savefig.dpi": 200,  # only reached by the PNG fallback
    }
