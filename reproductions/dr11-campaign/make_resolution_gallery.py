#!/usr/bin/env python3
"""Appendix B assets: the 78 DESI-resolution grade-A/B candidates that have NO high-resolution
coverage (not in HSC-SSP, not in Euclid Q1) and so cannot be validated beyond DECaLS 1" --- a ranked
longtable + DESI grz contact sheets. Run with the claudenet venv (matplotlib + lensjudge.render):
  /home2/benson/.venvs/claudenet/bin/python dr11-campaign/make_resolution_gallery.py
"""
import sys
from pathlib import Path
D = Path("/home2/benson/git/agentic-lensing/reproductions")
sys.path.insert(0, str(D))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lensjudge.common import render, fetch

PAP = D / "dr11-campaign" / "papers"
FIG = PAP / "figures"
res = pd.read_csv(D / "dr11-campaign" / "data" / "dr11s_desi_resolution_AB.csv").astype({"name": str})
print(f"{len(res)} DESI-resolution A/B candidates")

# literature crosscheck (crosscheck_candidates.py) -> per-candidate NEW/KNOWN + survey,
# so each thumbnail and the longtable `known` column carry the identification.
SVC = {"DES": "DES", "Huang+2020": "H20", "AGEL": "AGEL", "CASSOWARY": "CSWA"}
XM = {}
_xmp = D / "dr11-campaign" / "data" / "dr11s_resolution_xmatch.csv"
if _xmp.exists():
    for _, _x in pd.read_csv(_xmp).astype({"name": str}).iterrows():
        XM[_x["name"]] = (_x["status"], SVC.get(str(_x.get("survey", "")), str(_x.get("survey", ""))))
else:
    print("  WARNING: dr11s_resolution_xmatch.csv missing -> NEW/KNOWN tags blank")

def xtag(nm):    # contact-sheet thumbnail label
    st, sv = XM.get(nm, ("new", ""))
    return f"KNOWN {sv}" if st == "known" else "NEW"

def xcode(nm):   # longtable `known` column
    st, sv = XM.get(nm, ("new", ""))
    return sv if st == "known" else "\\,---"

def xcolor(nm):  # green = new, grey = known
    return "#157f15" if XM.get(nm, ("new", ""))[0] != "known" else "#555555"

# --- contact sheets: DESI grz Lupton-RGB thumbnails, 6 cols, <=42/sheet ---
NCOL, PERSHEET = 6, 42
sheets = []
for s0 in range(0, len(res), PERSHEET):
    chunk = res.iloc[s0:s0 + PERSHEET].reset_index(drop=True)
    nrow = int(np.ceil(len(chunk) / NCOL))
    fig, axes = plt.subplots(nrow, NCOL, figsize=(NCOL * 1.15, nrow * 1.32))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for i, r in chunk.iterrows():
        ax = axes[i // NCOL, i % NCOL]
        p = fetch.on_disk_path(str(r["name"]), "dr11")
        cube = render.load_cube(p) if p is not None else None
        if cube is not None:
            ax.imshow(np.asarray(render.lupton(cube)), origin="upper")
        ax.set_title(f"{int(r['rank'])}. {r['name']}\n{r['grade_pred']}  p={r['p_lens']:.2f}\n{xtag(str(r['name']))}",
                     fontsize=4.4, pad=1.2, color=xcolor(str(r['name'])))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005, wspace=0.06, hspace=0.46)
    out = FIG / f"res_gallery_{len(sheets)+1}.png"
    fig.savefig(out, dpi=200); plt.close(fig); sheets.append(out.name)
    print(f"  wrote {out.name} ({len(chunk)} thumbnails, {nrow}x{NCOL})")

# --- ranked longtable ---
rows = []
for _, r in res.iterrows():
    nm = str(r["name"]).replace("_", "\\_")
    rows.append(f"{int(r['rank'])} & \\texttt{{{nm}}} & {r['ra']:.4f} & {r['dec']:+.4f} & "
                f"{r['grade_pred']} & {r['p_lens']:.2f} & {r['p_meta']:.2f} & {xcode(str(r['name']))} \\\\")
tbl = (
    "\\begin{longtable}{rlcccccc}\n"
    "\\caption{The 78 DESI-resolution grade-A/B candidates (LensJudge \\texttt{direct/escalate} grade at "
    "DECaLS $0.262\\arcsec$) with \\emph{no} HSC-SSP or Euclid~Q1 coverage, ranked by tier-1 $p_{\\rm lens}$. "
    "A literature crosscheck (\\texttt{crosscheck\\_candidates.py}: the Huang/Storfer/Inchausti catalogues "
    "$+$ a NED/SIMBAD cone search) finds \\textbf{53 are previously-published lenses} --- 38 DES, 8 "
    "Huang+2020, 6 AGEL, 1 CASSOWARY (\\emph{known} column: survey code; per-object matches in "
    "\\texttt{data/dr11s\\_resolution\\_xmatch.csv}) --- and \\textbf{25 are genuinely new} (\\,---\\,), all "
    "grade~B. The high known fraction is because this DECaLS-south footprint heavily overlaps the Dark "
    "Energy Survey lens searches, which the campaign crossmatch omitted. The 25 new ones cannot be "
    "validated beyond DECaLS resolution; at $1\\arcsec$ a genuine lens and an lrg+companion mimic are not "
    "separable, so an unknown fraction are mimics. Images in Figs.~\\ref{fig:resgal1}--\\ref{fig:resgal2}.}\\\\\n"
    "\\label{tab:resolution}\\\\\n\\toprule\n"
    "\\# & name & RA & Dec & grade & $p_{\\rm lens}$ & \\vb{} & known \\\\\n\\midrule\n\\endfirsthead\n"
    "\\multicolumn{8}{l}{\\footnotesize Table~\\ref{tab:resolution} continued}\\\\\n\\toprule\n"
    "\\# & name & RA & Dec & grade & $p_{\\rm lens}$ & \\vb{} & known \\\\\n\\midrule\n\\endhead\n"
    "\\midrule\\multicolumn{8}{r}{\\footnotesize continued\\ldots}\\\\\n\\endfoot\n\\bottomrule\n\\endlastfoot\n"
    + "\n".join(rows) + "\n\\end{longtable}\n")
(PAP / "appendix_resolution_table.tex").write_text(tbl)
print(f"  wrote appendix_resolution_table.tex ({len(res)} rows)")

# --- the appendix figure blocks (contact sheets) ---
figtex = []
for i, nm in enumerate(sheets, 1):
    figtex.append(
        f"\\begin{{figure}}[p]\\centering\n\\includegraphics[width=\\linewidth]{{figures/{nm}}}\n"
        f"\\caption{{DESI \\emph{{grz}} cutouts ($0.262\\arcsec$/px, Lupton RGB, $101$\\,px) of the "
        f"DESI-resolution A/B candidates, ranks {1+(i-1)*PERSHEET}--{min(i*PERSHEET,len(res))} of "
        f"Table~\\ref{{tab:resolution}}. Each thumbnail is labelled \\textsc{{new}} (green) or "
        f"\\textsc{{known}} with its survey (grey: H20/DES/AGEL/CSWA), from the literature crosscheck. "
        f"No high-resolution view exists for these positions.}}\n"
        f"\\label{{fig:resgal{i}}}\n\\end{{figure}}\n")
(PAP / "appendix_resolution_figs.tex").write_text("".join(figtex))
print(f"  wrote appendix_resolution_figs.tex ({len(sheets)} sheets)")
print("done")
