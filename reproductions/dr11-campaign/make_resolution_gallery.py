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
        ax.set_title(f"{int(r['rank'])}. {r['name']}\n{r['grade_pred']}  p={r['p_lens']:.2f}",
                     fontsize=4.4, pad=1.2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.005, wspace=0.06, hspace=0.32)
    out = FIG / f"res_gallery_{len(sheets)+1}.png"
    fig.savefig(out, dpi=200); plt.close(fig); sheets.append(out.name)
    print(f"  wrote {out.name} ({len(chunk)} thumbnails, {nrow}x{NCOL})")

# --- ranked longtable ---
rows = []
for _, r in res.iterrows():
    nm = str(r["name"]).replace("_", "\\_")
    rows.append(f"{int(r['rank'])} & \\texttt{{{nm}}} & {r['ra']:.4f} & {r['dec']:+.4f} & "
                f"{r['grade_pred']} & {r['p_lens']:.2f} & {r['p_meta']:.2f} \\\\")
tbl = (
    "\\begin{longtable}{rlccccc}\n"
    "\\caption{The 78 DESI-resolution grade-A/B candidates (LensJudge \\texttt{direct/escalate} grade at "
    "DECaLS $0.262\\arcsec$) with \\emph{no} HSC-SSP or Euclid~Q1 coverage --- ranked by tier-1 "
    "$p_{\\rm lens}$. These cannot be validated beyond DECaLS resolution with current high-res data; "
    "at $1\\arcsec$ a genuine lens and an lrg+companion mimic are not separable, so an unknown (likely "
    "large) fraction are mimics. Images in Figs.~\\ref{fig:resgal1}--\\ref{fig:resgal2}.}\\\\\n"
    "\\label{tab:resolution}\\\\\n\\toprule\n"
    "\\# & name & RA & Dec & grade & $p_{\\rm lens}$ & \\vb{} \\\\\n\\midrule\n\\endfirsthead\n"
    "\\multicolumn{7}{l}{\\footnotesize Table~\\ref{tab:resolution} continued}\\\\\n\\toprule\n"
    "\\# & name & RA & Dec & grade & $p_{\\rm lens}$ & \\vb{} \\\\\n\\midrule\n\\endhead\n"
    "\\midrule\\multicolumn{7}{r}{\\footnotesize continued\\ldots}\\\\\n\\endfoot\n\\bottomrule\n\\endlastfoot\n"
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
        f"Table~\\ref{{tab:resolution}}. No high-resolution view exists for these positions.}}\n"
        f"\\label{{fig:resgal{i}}}\n\\end{{figure}}\n")
(PAP / "appendix_resolution_figs.tex").write_text("".join(figtex))
print(f"  wrote appendix_resolution_figs.tex ({len(sheets)} sheets)")
print("done")
