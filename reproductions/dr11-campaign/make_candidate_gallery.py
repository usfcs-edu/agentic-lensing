#!/usr/bin/env python3
"""Render per-candidate triptychs for the report appendix: DESI grz (the CNN-seen cutout) /
HSC grizy RGB (the tier-2 confirmation) / HSC lens-light-subtracted (arc structure). Uses the
cascade's cached HSC PDR3 cutouts + the staged DESI grz FITS, with the SAME render functions
LensJudge showed the grader. Run with the lensjudge venv:
  PYTHONPATH=<reproductions> /home2/benson/.venvs/lensjudge/bin/python dr11-campaign/make_candidate_gallery.py
"""
import sys
from pathlib import Path
D = Path("/home2/benson/git/agentic-lensing/reproductions")
sys.path.insert(0, str(D))
import pandas as pd
from lensjudge.common import hsc, render, fetch

FIG = D / "dr11-campaign" / "papers" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

conf = pd.read_csv(D / "dr11-campaign" / "data" / "dr11s_confirmed_AB.csv")
new = conf[conf["class"] == "NEW"].sort_values("p_lens", ascending=False)
# s_345385_1514/1515 are the same system (~1"); keep the higher-tier-2 detection (1515)
new = new[new["name"] != "s_345385_1514"].reset_index(drop=True)
print(f"rendering {len(new)} distinct NEW systems (DESI grz / HSC RGB / HSC lens-sub)")
tex = []
for _, r in new.iterrows():
    nm = str(r["name"])
    # DESI grz cube -> Lupton RGB (the CNN-seen 101px cutout)
    p = fetch.on_disk_path(nm, "dr11")
    if p is not None:
        cube = render.load_cube(p)
        if cube is not None:
            render.lupton(cube).save(FIG / f"cand_{nm}_desi.png")
    # HSC grizy -> full RGB + lens-light-subtracted
    bands = hsc.load_hsc(float(r.ra), float(r.dec))
    if bands:
        hsc.rgb_view(bands).save(FIG / f"cand_{nm}_hsc.png")
        hsc.lum_sub_view(bands).save(FIG / f"cand_{nm}_hscsub.png")
        print(f"  {nm}: DESI + HSC(bands={sorted(bands)})")
    else:
        print(f"  WARN no HSC cache for {nm}")
    extra = " (= 8th detection s\\_345385\\_1514, the adjacent $\\sim$1\\arcsec\\ pair, omitted)" \
        if nm == "s_345385_1515" else ""
    nm_tex = nm.replace("_", "\\_")
    tex.append(
        f"\\begin{{figure}}[p]\\centering\n"
        f"\\begin{{minipage}}[b]{{0.32\\linewidth}}\\centering\\includegraphics[width=\\linewidth]"
        f"{{figures/cand_{nm}_desi.png}}\\\\[2pt]{{\\scriptsize DESI \\emph{{grz}} (0.262\\arcsec/px)}}\\end{{minipage}}\\hfill\n"
        f"\\begin{{minipage}}[b]{{0.32\\linewidth}}\\centering\\includegraphics[width=\\linewidth]"
        f"{{figures/cand_{nm}_hsc.png}}\\\\[2pt]{{\\scriptsize HSC \\emph{{grizy}} (0.168\\arcsec/px)}}\\end{{minipage}}\\hfill\n"
        f"\\begin{{minipage}}[b]{{0.32\\linewidth}}\\centering\\includegraphics[width=\\linewidth]"
        f"{{figures/cand_{nm}_hscsub.png}}\\\\[2pt]{{\\scriptsize HSC lens-light subtracted}}\\end{{minipage}}\n"
        f"\\caption{{\\texttt{{{nm_tex}}} (RA {r.ra:.4f}, Dec {r.dec:+.4f}) --- HSC tier-2 grade "
        f"\\textbf{{{r.grade_pred}}}, $p_{{\\rm lens}}$ {r.p_lens_tier1:.2f}$\\to${r.p_lens_tier2:.2f}, "
        f"\\vb{{}} {r.p_meta:.2f}.{extra}}}\n\\end{{figure}}\n")
out = D / "dr11-campaign" / "papers" / "appendix_candidates.tex"
out.write_text("".join(tex))
print(f"wrote {out} ({len(new)} candidate figures)")
print("done")
