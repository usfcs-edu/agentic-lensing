#!/usr/bin/env python
"""Generate the two figures for the strong-lens discovery opportunity report.

Data are the verified, priority-scored outputs of the deep-research workflow
(see ../workflow.js and ../OPPORTUNITIES.md). Run with a matplotlib-enabled
Python, e.g.:

    /home/benson/.venvs/lens/bin/python figures/make_figures.py

Outputs (next to this script):
    priority_ranking.png   top opportunities ranked by priority, coloured by tier
    coverage_map.png       footprint vs fresh/unsearched fraction, grouped by modality
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm, colors

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Figure 1 — priority ranking (top opportunities), coloured by tier
# (name, priority_score, tier)
# ---------------------------------------------------------------------------
TIER_COLOR = {
    "dropin-now":      "#2c7fb8",   # blue  — in-house method drops in
    "newtooling-now":  "#7b3294",   # purple — public but needs a build
    "future-watch":    "#e6a000",   # amber — high ceiling, not public yet
    "low":             "#9e9e9e",   # grey  — exhausted / low
}
TIER_LABEL = {
    "dropin-now":     "Drop-in now (in-house method)",
    "newtooling-now": "New-tooling now (build required)",
    "future-watch":   "Future watch-list",
    "low":            "Low / exhausted / refuted on deep-dive",
}

RANKED = [
    ("VLASS (radio)",                       78, "newtooling-now"),
    ("ALMA Science Archive",                78, "newtooling-now"),
    ("ALMACAL",                             75, "newtooling-now"),
    ("ATLAS (↓ glSN sliver)",          55, "dropin-now"),
    ("Roman HLWAS+HLTDS",                   70, "future-watch"),
    ("Rubin / LSST",                        69, "future-watch"),
    ("Euclid DR1 Wide",                     68, "future-watch"),
    ("4MOST (south)",                       68, "future-watch"),
    ("HST general archive (↓ deep-dive)", 55, "dropin-now"),
    ("Herschel-ATLAS",                      66, "newtooling-now"),
    ("HerMES+HeLMS+HerS",                   65, "newtooling-now"),
    ("GAMA DR4 (✗ refuted: FoF≠G-G)",  28, "low"),
    ("VHS (✗ refuted: too shallow)",   30, "low"),
    ("SPT-3G emissive catalog",             63, "newtooling-now"),
    ("JWST COSMOS-Web",                     62, "dropin-now"),
    ("ZTF (↓ saturated)",              45, "dropin-now"),
    ("DELVE DR3 (↓ downgraded: niche)", 45, "dropin-now"),
    ("UNIONS",                              60, "future-watch"),
    ("J-PLUS / S-PLUS",                     60, "dropin-now"),
    ("WEAVE",                               60, "future-watch"),
    ("Subaru PFS",                          60, "future-watch"),
    ("DESI-II / Spec-S5",                   59, "future-watch"),
    ("DESI DR2 spectra",                    58, "future-watch"),
    ("ACT DR6 mm catalog",                  58, "newtooling-now"),
    ("LAMOST DR11/12",                      57, "dropin-now"),
    ("DECam multi-epoch (new)",             57, "dropin-now"),
]

def fig_priority():
    RANKED.sort(key=lambda r: r[1])  # ascending so highest is at top of barh
    names = [r[0] for r in RANKED]
    scores = [r[1] for r in RANKED]
    bar_colors = [TIER_COLOR[r[2]] for r in RANKED]

    fig, ax = plt.subplots(figsize=(9.2, 9.0))
    y = range(len(names))
    ax.barh(list(y), scores, color=bar_colors, edgecolor="white", height=0.78)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Priority score  (0.75·opportunity + 0.25·actionability)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_title("Strong-lens discovery opportunities, ranked\n"
                 "(verified by adversarial red-team; baseline DESI footprints excluded)",
                 fontsize=11.5, pad=10)
    for yi, s in zip(y, scores):
        ax.text(s + 1, yi, str(s), va="center", fontsize=7.5, color="#333333")
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=TIER_COLOR[t], label=TIER_LABEL[t])
               for t in ["dropin-now", "newtooling-now", "future-watch", "low"]]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    out = os.path.join(HERE, "priority_ranking.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure 2 — footprint vs fresh/unsearched fraction, grouped by modality
# (survey, footprint_deg2, fresh_fraction, modality)
# fresh_fraction: 0 = fully searched, 1 = virgin/unsearched (incl. modality novelty)
# ---------------------------------------------------------------------------
COVERAGE = [
    # optical imaging
    ("DESI Legacy DR10 (baseline)", 19000, 0.00, "Optical imaging"),
    ("DES Y6",                       5000, 0.00, "Optical imaging"),
    ("Pan-STARRS 3pi",              30000, 0.02, "Optical imaging"),
    ("HSC-SSP PDR3",                  670, 0.15, "Optical imaging"),
    ("KiDS DR5",                     1347, 0.05, "Optical imaging"),
    ("DELVE DR2/3 (↓ ~80-88% searched)", 20000, 0.12, "Optical imaging"),
    ("UNIONS",                       3730, 0.25, "Optical imaging"),
    # NIR imaging
    ("VHS (✗ refuted)",             18000, 0.05, "NIR imaging"),
    ("VIKING",                       1350, 0.15, "NIR imaging"),
    ("UKIDSS LAS",                   4000, 0.12, "NIR imaging"),
    # spectroscopy
    ("DESI DR1 (baseline)",         14000, 0.00, "Spectroscopy"),
    ("SDSS/BOSS/eBOSS",             10000, 0.10, "Spectroscopy"),
    ("GAMA DR4",                      286, 1.00, "Spectroscopy"),
    ("4MOST (future)",              17000, 0.80, "Spectroscopy"),
    # radio
    ("VLASS",                       33885, 1.00, "Radio"),
    ("ASKAP EMU/RACS",              20000, 0.95, "Radio"),
    ("LOFAR LoTSS DR2",              5700, 1.00, "Radio"),
    # submm/mm
    ("ACT DR6",                     19000, 0.85, "Submm / mm"),
    ("SPT-3G",                       1500, 0.85, "Submm / mm"),
    ("Herschel (H-ATLAS+HerMES)",    1420, 0.90, "Submm / mm"),
    # space high-res / future
    ("Euclid DR1 (future)",          1900, 0.60, "Space / future"),
    ("Roman HLWAS (future)",         2000, 0.60, "Space / future"),
    ("HST archive (↓ re-mine)",       400, 0.20, "Space / future"),
]

MODALITY_ORDER = ["Optical imaging", "NIR imaging", "Spectroscopy",
                  "Radio", "Submm / mm", "Space / future"]

def fig_coverage():
    # group by modality, keep order, insert a blank gap between groups
    rows, labels, fres, foot = [], [], [], []
    yticks, yticklabels = [], []
    group_spans = []  # (modality, y_start, y_end)
    y = 0
    for mod in MODALITY_ORDER:
        members = [c for c in COVERAGE if c[3] == mod]
        members.sort(key=lambda c: c[1])
        y_start = y
        for (name, area, fresh, _m) in members:
            labels.append(name); foot.append(area); fres.append(fresh)
            yticks.append(y); yticklabels.append(name)
            y += 1
        group_spans.append((mod, y_start, y - 1))
        y += 1  # gap row between groups

    cmap = matplotlib.colormaps["RdYlGn"]
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    bar_colors = [cmap(norm(f)) for f in fres]

    fig, ax = plt.subplots(figsize=(9.4, 9.6))
    ax.barh(yticks, foot, color=bar_colors, edgecolor="#444444", lw=0.4, height=0.78)
    ax.set_xscale("log")
    ax.set_xlim(80, 60000)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xlabel("Footprint  (deg², log scale)", fontsize=10)
    ax.set_title("Sky-coverage landscape: footprint vs. fresh / unsearched fraction\n"
                 "(bar colour = how much is virgin; baseline DESI surveys shown at fresh≈0)",
                 fontsize=11.5, pad=10)
    # annotate fresh fraction at bar end
    for yi, a, f in zip(yticks, foot, fres):
        ax.text(a * 1.10, yi, f"{int(round(f*100))}% fresh", va="center",
                fontsize=7.0, color="#333333")
    # modality group labels in the left margin (vertical) + faint separators
    for mod, ys, ye in group_spans:
        ax.text(-0.255, (ys + ye) / 2.0, mod, transform=ax.get_yaxis_transform(),
                rotation=90, ha="center", va="center", fontsize=8.8,
                color="#444444", fontweight="bold", clip_on=False)
        if ye + 1 <= max(yticks):
            ax.axhline(ye + 1, color="#cfcfcf", lw=0.7, zorder=0)
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.04, fraction=0.035)
    cbar.set_label("fresh / unsearched fraction (incl. modality novelty)", fontsize=8.5)
    fig.tight_layout()
    fig.subplots_adjust(left=0.30)
    out = os.path.join(HERE, "coverage_map.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_priority()
    fig_coverage()
