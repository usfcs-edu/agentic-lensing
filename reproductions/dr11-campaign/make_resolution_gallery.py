#!/usr/bin/env python3
"""Appendix B assets: the 78 DESI-resolution grade-A/B candidates that have NO high-resolution
coverage (not in HSC-SSP, not in Euclid Q1) and so cannot be validated beyond DECaLS 1".

Emits a ranked longtable index (kept) PLUS an *expanded per-candidate gallery*: one
full | zoom | residual DESI grz triptych per candidate (the all-DESI three-view layout of
claudenet/papers/new_candidates.tex -- the right model here precisely because these positions
have no high-resolution data), with full metadata and a colored KNOWN/NEW + source tag.

The DESI grz cubes come from the staged-on-disk FITS in config.CUTOUT_DIRS["dr11"]
(unpack dr11s_cand500.npz with claudenet/120b_unpack_npz_to_fits.py first). Run with the
Mac lensjudge venv (astropy + scipy + PIL + numpy + pandas):

  ~/.venvs/lensjudge/bin/python dr11-campaign/make_resolution_gallery.py
"""
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent          # reproductions/
sys.path.insert(0, str(REPRO))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from lensjudge import config  # noqa: E402  (CUTOUT_DIRS, RENDER_PX, Lupton params)
from lensjudge.common import render  # noqa: E402  (load_cube, render_views, residual sigma=3)

PAP = REPRO / "dr11-campaign" / "papers"
FIG = PAP / "figures"
CDIR = config.CUTOUT_DIRS["dr11"]                        # data/cutouts_dr11/{name}.fits
VIEWS = ("full", "zoom", "residual")
TILE = config.RENDER_PX                                  # 400 px upsample per panel

res = pd.read_csv(REPRO / "dr11-campaign" / "data" / "dr11s_desi_resolution_AB.csv").astype({"name": str})
print(f"{len(res)} DESI-resolution A/B candidates")

# literature crosscheck (crosscheck_candidates.py) -> per-candidate NEW/KNOWN + survey +
# counterpart + separation + source, so the longtable `known` column AND the per-candidate
# caption both carry the identification.
SVC = {"DES": "DES", "Huang+2020": "H20", "AGEL": "AGEL", "CASSOWARY": "CSWA"}
XM = {}
_xmp = REPRO / "dr11-campaign" / "data" / "dr11s_resolution_xmatch.csv"
if _xmp.exists():
    for _, x in pd.read_csv(_xmp).astype({"name": str}).iterrows():
        XM[x["name"]] = {
            "status": str(x.get("status", "new")),
            "survey": str(x.get("survey", "")),
            "counterpart": str(x.get("counterpart", "")),
            "sep": x.get("sep_arcsec"),
            "source": str(x.get("source", "")),
        }
else:
    print("  WARNING: dr11s_resolution_xmatch.csv missing -> NEW/KNOWN tags blank")


def _tex(s):
    return str(s).replace("_", "\\_")


def xcode(nm):   # longtable `known` column: survey code or em-dash
    r = XM.get(nm)
    if not r or r["status"] != "known":
        return "\\,---"
    return SVC.get(r["survey"], r["survey"])


def captag(nm):  # per-candidate caption KNOWN/NEW tag (xcolor: grey known, green new)
    r = XM.get(nm, {"status": "new"})
    if r["status"] == "known":
        sep = r.get("sep")
        sep = f"{float(sep):.2f}" if sep is not None and sep == sep else "?"
        return (f"\\textcolor{{gray}}{{\\textbf{{KNOWN}}}} --- {r['survey']}: "
                f"\\texttt{{{_tex(r['counterpart'])}}}, ${sep}\\arcsec$ ({r['source']}).")
    return ("\\textcolor{forestgreen}{\\textbf{NEW}} --- new to the DESI lens-finders, the "
            "DES/AGEL/CASSOWARY searches, and the NED/SIMBAD literature.")


def _font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def composite(views: dict, out_path: Path) -> None:
    """Stitch full|zoom|residual side-by-side (1200x400) with a small view label baked in;
    id/coords/scores live in the LaTeX caption. Mirrors claudenet/93_make_gallery.composite."""
    imgs = []
    for v in VIEWS:
        im = views[v].convert("RGB").resize((TILE, TILE), Image.NEAREST)
        ImageDraw.Draw(im).text((8, 8), v, fill=(255, 255, 0), font=_font(22))
        imgs.append(im)
    canvas = Image.new("RGB", (TILE * len(imgs), TILE), (0, 0, 0))
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * TILE, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# --- per-candidate DESI grz triptychs (full | zoom | residual) ---------------
n_ok, missing = 0, []
for _, r in res.iterrows():
    nm = str(r["name"])
    p = CDIR / f"{nm}.fits"
    cube = render.load_cube(p) if p.exists() else None
    if cube is None:
        missing.append(nm)
        print(f"  WARNING: no/invalid cube for {nm} ({p})")
        continue
    # honest residual: signed chi=(data - elliptical model)/noise of the r+z luminance
    # (single square tile; red=unmodelled excess/arc, blue=over-subtraction). The full g|r|z
    # chi montage is render.residual(); the luminance tile keeps the full|zoom|residual triptych.
    views = render.render_views(cube, views=("full", "zoom"))
    views["residual"] = render.residual_chi_luminance(cube)
    if any(v not in views for v in VIEWS):
        missing.append(nm)
        print(f"  WARNING: incomplete render for {nm}: {sorted(views)}")
        continue
    composite(views, FIG / f"res_{nm}.png")
    n_ok += 1
print(f"  wrote {n_ok}/{len(res)} triptychs -> {FIG}/res_<name>.png")
if missing:
    raise SystemExit(f"ABORT: {len(missing)} candidates lack a renderable DESI cube "
                     f"(stage them from Perlmutter first): {missing}")

# --- ranked longtable index (kept) -------------------------------------------
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
    "separable, so an unknown fraction are mimics. Per-candidate full/zoom/residual panels follow "
    "(App.~\\ref{app:resolution}).}\\\\\n"
    "\\label{tab:resolution}\\\\\n\\toprule\n"
    "\\# & name & RA & Dec & grade & $p_{\\rm lens}$ & \\vb{} & known \\\\\n\\midrule\n\\endfirsthead\n"
    "\\multicolumn{8}{l}{\\footnotesize Table~\\ref{tab:resolution} continued}\\\\\n\\toprule\n"
    "\\# & name & RA & Dec & grade & $p_{\\rm lens}$ & \\vb{} & known \\\\\n\\midrule\n\\endhead\n"
    "\\midrule\\multicolumn{8}{r}{\\footnotesize continued\\ldots}\\\\\n\\endfoot\n\\bottomrule\n\\endlastfoot\n"
    + "\n".join(rows) + "\n\\end{longtable}\n")
(PAP / "appendix_resolution_table.tex").write_text(tbl)
print(f"  wrote appendix_resolution_table.tex ({len(res)} rows)")

# --- expanded per-candidate gallery (replaces the contact sheets) ------------
# Grouped KNOWN-first then NEW under their own sub-headings, ranked by tier-1 p_lens within
# each group (`res` is already rank-sorted, so filtering preserves the order). See panel().
def is_known(nm):
    return XM.get(str(nm), {"status": "new"})["status"] == "known"


def panel(r):
    # figure[H]: placed exactly here (float package), so 78 of them never overflow LaTeX's
    # float queue -- AND pandoc renders \caption as a real <figcaption> for the site (it drops
    # a bare \captionof). One triptych per row, full text width; view labels baked into the PNG.
    nm = str(r["name"])
    return (
        "\\begin{figure}[H]\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{figures/res_{nm}.png}}\n"
        f"\\caption{{\\textbf{{\\#{int(r['rank'])}}}~\\texttt{{{_tex(nm)}}} --- "
        f"RA {r['ra']:.4f}, Dec {r['dec']:+.4f}; grade \\textbf{{{r['grade_pred']}}}, "
        f"tier-1 $p_{{\\rm lens}}$ {r['p_lens']:.2f}, \\vb{{}} {r['p_meta']:.2f}. {captag(nm)}}}\n"
        "\\end{figure}\n")


known = res[res["name"].map(is_known)]
new = res[~res["name"].map(is_known)]
# forestgreen: a valid CSS named color, so pandoc's `color: forestgreen` works on the site;
# \definecolor sets the PDF shade to the darker #157f15 (gray already matches CSS exactly).
blocks = ["\\definecolor{forestgreen}{HTML}{157f15}\n"]
blocks.append(
    f"\\subsection*{{\\textcolor{{gray}}{{\\textsc{{Known}}}} --- previously-published lenses "
    f"recovered ({len(known)} of {len(res)})}}\n"
    "These coincide ($\\le$few\\,\\arcsec) with a catalogued strong lens --- 38 DES, 8 Huang+2020, "
    "6 AGEL, 1 CASSOWARY --- an external validation of the finder (the campaign crossmatch had "
    "omitted these catalogues). Ranked by tier-1 $p_{\\rm lens}$.\\par\\smallskip\n")
for _, r in known.iterrows():
    blocks.append(panel(r))
blocks.append(
    f"\\clearpage\n\\subsection*{{\\textcolor{{forestgreen}}{{\\textsc{{New}}}} --- candidates new to "
    f"the literature ({len(new)} of {len(res)})}}\n"
    "These are absent from the DESI lens-finders, the DES/AGEL/CASSOWARY searches, and the "
    "NED/SIMBAD literature; all are grade~B, and at DECaLS $1\\arcsec$ an unknown (likely large) "
    "fraction are lrg+companion mimics. Ranked by tier-1 $p_{\\rm lens}$.\\par\\smallskip\n")
for _, r in new.iterrows():
    blocks.append(panel(r))
(PAP / "appendix_resolution_figs.tex").write_text("".join(blocks))
print(f"  wrote appendix_resolution_figs.tex ({len(known)} known + {len(new)} new panels)")
print("done")
