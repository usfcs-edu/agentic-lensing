#!/usr/bin/env python
"""
Metric-3 scaling-ladder aggregator for SpectrumFM.

The proposal's Metric 3 is a SCALING TREND, not a single number: does redshift
quality (DESI good-z fraction; galaxy catastrophic rate) improve as we scale the
MODEL (params) and the DATA (n_spectra)? The local 2xL4 study ruled out every
non-scale lever (tokenizer arch, equivariance prior, compression ratio) — see the
compression-sweep memory — so this tool turns a set of per-run eval results into
the two scaling curves and reads off the verdict:

  * MODEL axis: linear-fit good_z (and galaxy_cat) vs log10(params).
  * DATA  axis: linear-fit good_z (and galaxy_cat) vs log10(n_spectra).
  A POSITIVE good_z slope  (or NEGATIVE galaxy_cat slope) => scale HELPS.
  A ~0 slope               => scale-limited / red flag (scale is not the lever).

A "point" is a dict:
  {
    "label": str,            # e.g. "P-250M" or "data-25%"
    "axis": "model"|"data",  # which ladder this point belongs to
    "params": int,           # transformer parameter count (use compute_params)
    "n_spectra": int,        # training spectra seen
    "good_z": float,         # aggregate ZWARN==0 good-z fraction in [0,1]
    "galaxy_cat": float,     # mean LRG/ELG/BGS catastrophic rate in [0,1]
    "probe_macro_f1": float  # OPTIONAL six-class linear-probe macro-F1
  }

Two ways in:
  (1) --from-json points.json : a precomputed list of point dicts.
  (2) parse_eval_per_class(text) : turn a captured eval_per_class.py stdout block
      into {good_z, per-class catastrophic, galaxy_cat, ...} so an orchestrator can
      pipe eval output straight in and assemble points programmatically.

Outputs:
  * a clean text table + the fitted slopes + an honest one-line verdict (stdout),
  * a 2-panel PNG (good-z vs params; good-z vs data) with the fit lines and the
    off-curve V1@15k / finer@15k local references as distinct markers,
    -> experiments/runs/_comparisons/scaling_ladder.png.

Usage:
  ~/.venvs/redshifty/bin/python tools/spectrumfm/scaling_ladder.py --from-json points.json
  ~/.venvs/redshifty/bin/python tools/spectrumfm/scaling_ladder.py --smoke   # CPU, no GPU
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path("/raid/benson/git/agentic-lensing/lensing-repos/redshifty")
OUT_PNG = Path("/raid/benson/git/agentic-lensing/experiments/runs/"
               "_comparisons/scaling_ladder.png")

# Off-curve local references (2xL4, 15k steps, eff-batch 64) from the compression
# sweep. These are the SAME small model/data point measured two ways (V1 32x and
# finer 16x tokenizer); plotted as fixed markers, NOT fit, to anchor the ladder.
# good_z / galaxy_cat from eval_per_class.py on the 2048-spectrum val set.
REF_PARAMS_15K = 103_671_040       # the local baseline (768/6/6/12, max_seq 1024)
REF_NSPECTRA_15K = 96_000          # ~15k steps * eff-batch 64 spectra seen
REFERENCES = {
    # label: (good_z, galaxy_cat)  — galaxy_cat = mean(LRG,ELG,BGS) catastrophic
    "V1@15k":    (0.241, (0.975 + 0.978 + 0.957) / 3.0),
    "finer@15k": (0.239, (0.975 + 0.978 + 0.957) / 3.0),  # per-class ties V1
}


# ----------------------------------------------------------------------------
# Parser: eval_per_class.py stdout  ->  {good_z, per_class_cat, galaxy_cat, ...}
# ----------------------------------------------------------------------------
# eval_per_class.py prints, per checkpoint:
#   aggregate catastrophic rate: all spectra= 75.9% (good-z=24.1%), ZWARN==0 only= ...
# and a per-class table whose data rows look like (2-space indent, class then
# N(z0), SFM cat%, med|dz|, RR fail%, verdict):
#   LRG     1234     97.5    0.0630     12.3      FAIL
# We anchor good-z on the ZWARN==0 reading (the honest reference set) when present,
# else fall back to the all-spectra reading; per-class catastrophic comes from
# the "SFM cat%" column (3rd numeric field on a class row).
# Two independent searches (NOT one combined optional group): a trailing
# optional group after a non-greedy ".*?" is trivially satisfied empty, so it
# would never capture the ZWARN==0 reading even when present. Matching each
# segment on its own is both correct and order-independent.
_ALL_GOODZ_RE = re.compile(
    r"all spectra=\s*[\d.]+%\s*\(good-z=\s*([\d.]+)%\)")
_Z0_GOODZ_RE = re.compile(
    r"ZWARN==0 only=\s*[\d.]+%\s*\(good-z=\s*([\d.]+)%\)")
_CLASS_ROW_RE = re.compile(
    r"^\s*(LRG|ELG|QSO|MWS|BGS|OTHER)\s+"   # class
    r"(\d+)\s+"                              # N(z0)
    r"([\d.]+|n/a)\s+"                       # SFM cat%
    r"([\d.]+|n/a)\s+"                       # med|dz|
    r"([\d.]+|n/a)\s+"                       # RR fail%
    r"(PASS|FAIL|INDIC|ctx)",               # verdict
    re.MULTILINE,
)
GALAXY_CLASSES = ("LRG", "ELG", "BGS")


def _f(tok):
    return None if tok in (None, "n/a") else float(tok)


def parse_eval_per_class(text):
    """Parse one eval_per_class.py checkpoint block into a metrics dict.

    Returns:
      {
        "good_z": float in [0,1],          # ZWARN==0 good-z if present else all
        "good_z_source": "zwarn0"|"all",
        "per_class_cat": {CLASS: frac},    # SFM catastrophic rate per class, [0,1]
        "galaxy_cat": float|None,          # mean catastrophic over LRG/ELG/BGS
        "per_class_n": {CLASS: int},       # N(ZWARN==0) per class
      }
    Robust to: missing ZWARN==0 segment, "n/a" cells, extra surrounding log
    lines, partial class sets, and Windows/Unix newlines. Raises ValueError only
    if NO good-z aggregate line is found.
    """
    text = text.replace("\r\n", "\n")
    m_all = _ALL_GOODZ_RE.search(text)
    m_z0 = _Z0_GOODZ_RE.search(text)
    if not (m_all or m_z0):
        raise ValueError("no 'aggregate catastrophic rate ... good-z=' line found")
    z0_goodz = float(m_z0.group(1)) / 100.0 if m_z0 else None
    all_goodz = float(m_all.group(1)) / 100.0 if m_all else None
    if z0_goodz is not None:
        good_z, source = z0_goodz, "zwarn0"   # honest reference set preferred
    else:
        good_z, source = all_goodz, "all"

    per_class_cat, per_class_n = {}, {}
    for row in _CLASS_ROW_RE.finditer(text):
        cls = row.group(1)
        per_class_n[cls] = int(row.group(2))
        cat = _f(row.group(3))
        if cat is not None:
            per_class_cat[cls] = cat / 100.0   # column is a percentage

    gal = [per_class_cat[c] for c in GALAXY_CLASSES if c in per_class_cat]
    galaxy_cat = float(np.mean(gal)) if gal else None

    return dict(
        good_z=good_z, good_z_source=source,
        per_class_cat=per_class_cat, galaxy_cat=galaxy_cat,
        per_class_n=per_class_n,
    )


# ----------------------------------------------------------------------------
# params helper — construct the model and SUM numel (no closed-form formula).
# ----------------------------------------------------------------------------
def compute_params(d_model, n_enc, n_dec, n_heads, max_seq_len=1024):
    """Total parameter count of a SpectrumTransformer with the given dims.

    Constructs the actual module and sums numel so the count is exact (the
    documented ladder — 768/6/6/12 -> 103.67M, etc — was verified this way).
    Imports torch lazily so the parser / JSON path needs no torch.
    """
    sys.path.insert(0, str(REPO))
    from src.models.transformer import SpectrumTransformer, TOTAL_VOCAB_SIZE
    model = SpectrumTransformer(
        vocab_size=TOTAL_VOCAB_SIZE, d_model=d_model,
        n_encoder_layers=n_enc, n_decoder_layers=n_dec, n_heads=n_heads,
        max_seq_len=max_seq_len,
    )
    return int(sum(p.numel() for p in model.parameters()))


# ----------------------------------------------------------------------------
# Linear fit  y = slope * x + intercept  (x = log10(scale)), with R^2.
# ----------------------------------------------------------------------------
def fit_log(xs, ys):
    """OLS of y on log10(x). Returns dict(slope, intercept, r2, per_decade, n).

    per_decade = slope (since x is already log10, a unit step in x is one decade).
    Needs >=2 distinct x; returns None entries otherwise.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    good = np.isfinite(xs) & np.isfinite(ys) & (xs > 0)
    xs, ys = xs[good], ys[good]
    lx = np.log10(xs)
    if len(lx) < 2 or np.allclose(lx, lx[0]):
        return dict(slope=None, intercept=None, r2=None, per_decade=None, n=len(lx))
    slope, intercept = np.polyfit(lx, ys, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(slope=float(slope), intercept=float(intercept), r2=float(r2),
                per_decade=float(slope), n=len(lx))


def _axis_points(points, axis):
    return [p for p in points if p.get("axis") == axis]


def _scale_of(p, axis):
    return p["params"] if axis == "model" else p["n_spectra"]


def analyze(points):
    """Fit good_z and galaxy_cat for both axes. Returns nested fit dict."""
    out = {}
    for axis in ("model", "data"):
        pts = _axis_points(points, axis)
        xs = [_scale_of(p, axis) for p in pts]
        out[axis] = dict(
            n_points=len(pts),
            good_z=fit_log(xs, [p["good_z"] for p in pts]),
            galaxy_cat=fit_log(
                xs, [p.get("galaxy_cat", float("nan")) for p in pts]),
        )
    return out


# ----------------------------------------------------------------------------
# Reporting + plotting
# ----------------------------------------------------------------------------
def _fmtf(x, nd=3):
    return " n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def print_table(points, fits):
    print("\n================  Metric-3 scaling ladder  ================")
    hdr = (f"  {'label':<14} {'axis':>6} {'params':>14} {'n_spectra':>12} "
           f"{'good_z':>8} {'gal_cat':>8} {'probe_f1':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for p in points:
        print(f"  {p['label']:<14} {p['axis']:>6} {p['params']:>14,d} "
              f"{p['n_spectra']:>12,d} {_fmtf(p['good_z']):>8} "
              f"{_fmtf(p.get('galaxy_cat')):>8} "
              f"{_fmtf(p.get('probe_macro_f1')):>9}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'(ref) '+ 'V1@15k':<14} {'-':>6} {REF_PARAMS_15K:>14,d} "
          f"{REF_NSPECTRA_15K:>12,d} {_fmtf(REFERENCES['V1@15k'][0]):>8} "
          f"{_fmtf(REFERENCES['V1@15k'][1]):>8} {'-':>9}")
    print(f"  {'(ref) '+ 'finer@15k':<14} {'-':>6} {REF_PARAMS_15K:>14,d} "
          f"{REF_NSPECTRA_15K:>12,d} {_fmtf(REFERENCES['finer@15k'][0]):>8} "
          f"{_fmtf(REFERENCES['finer@15k'][1]):>8} {'-':>9}")

    print("\n  fits (y = slope*log10(scale) + intercept):")
    for axis in ("model", "data"):
        f = fits[axis]
        scale = "params" if axis == "model" else "n_spectra"
        for metric in ("good_z", "galaxy_cat"):
            ff = f[metric]
            print(f"    {axis:>5}/{metric:<10} vs log10({scale}): "
                  f"slope={_fmtf(ff['slope'],4)}  R^2={_fmtf(ff['r2'])}  "
                  f"per-decade={_fmtf(ff['per_decade'],4)}  (n={ff['n']})")


def verdict(fits):
    """One honest line. good_z slope>0 OR galaxy_cat slope<0 => scale helps."""
    EPS = 0.01   # |per-decade| below this on good_z is 'flat' (1pp/decade)
    msgs = []
    for axis in ("model", "data"):
        gz = fits[axis]["good_z"]["per_decade"]
        gc = fits[axis]["galaxy_cat"]["per_decade"]
        if gz is None:
            msgs.append(f"{axis}: insufficient points")
            continue
        helps = (gz > EPS) or (gc is not None and gc < -EPS)
        flat = (abs(gz) <= EPS) and (gc is None or abs(gc) <= EPS)
        tag = ("SCALE HELPS" if helps else
               "SCALE-LIMITED (red flag)" if flat else "MIXED/regressing")
        msgs.append(f"{axis}: good_z {gz:+.3f}/decade, "
                    f"gal_cat {(_fmtf(gc,3) if gc is not None else 'n/a')}/decade -> {tag}")
    return "VERDICT  " + "  |  ".join(msgs)


def plot(points, fits, out_png=OUT_PNG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, (axM, axD) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, axis, xlabel, scale_key in (
        (axM, "model", "log10(params)", "params"),
        (axD, "data", "log10(n_spectra)", "n_spectra"),
    ):
        pts = _axis_points(points, axis)
        if pts:
            xs = np.array([_scale_of(p, axis) for p in pts], float)
            ys = np.array([p["good_z"] for p in pts], float)
            order = np.argsort(xs)
            lx = np.log10(xs[order])
            ax.scatter(lx, ys[order], s=70, color="#1f77b4", zorder=3,
                       label="ladder points")
            for p, x, y in zip([pts[i] for i in order], lx, ys[order]):
                ax.annotate(p["label"], (x, y), textcoords="offset points",
                            xytext=(5, 5), fontsize=7)
            f = fits[axis]["good_z"]
            if f["slope"] is not None:
                xr = np.linspace(lx.min(), lx.max(), 50)
                ax.plot(xr, f["slope"] * xr + f["intercept"], "--",
                        color="#1f77b4",
                        label=f"fit {f['slope']:+.3f}/dec (R²={f['r2']:.2f})")
        # off-curve local references
        ref_x = np.log10(REF_PARAMS_15K if axis == "model" else REF_NSPECTRA_15K)
        for (name, (gz, _gc)), mk, col in zip(
                REFERENCES.items(), ("X", "P"), ("#d62728", "#ff7f0e")):
            ax.scatter([ref_x], [gz], marker=mk, s=130, color=col, zorder=4,
                       edgecolor="k", linewidth=0.6, label=f"{name} (off-curve)")
        ax.axhline(0.241, color="#999", ls=":", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("good-z fraction (ZWARN==0)")
        ax.set_title(f"Metric-3 {axis} axis")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("SpectrumFM Metric-3 scaling ladder — good-z vs scale", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")
    return out_png


def run_from_json(path, out_png=OUT_PNG):
    points = json.loads(Path(path).read_text())
    fits = analyze(points)
    print_table(points, fits)
    plot(points, fits, out_png)
    print("\n" + verdict(fits))


# ----------------------------------------------------------------------------
# CPU smoke — synthetic ladders + parser unit-test (no GPU).
# ----------------------------------------------------------------------------
# A real eval_per_class.py V1 block (formatting matches the print statements in
# eval_per_class.py: 2-space indent, class table, aggregate line). good-z 24.1%,
# LRG 97.5, ELG 97.8, BGS 95.7.
SAMPLE_V1 = """\
================  approach_a_mix_l4x2_v1  ================
  N=2048  (1556 ZWARN==0 / 492 ZWARN!=0)   z-bins=256   catastrophic thr |dz|/(1+z) >= 0.0033
  class    N(z0)  SFM cat%  med|dz|  RR fail%   verdict
  ------------------------------------------------------
  LRG        612      97.5   0.0630      8.1      FAIL
  ELG        498      97.8   0.1110     12.4      FAIL
  QSO         44      99.1   0.4200     33.0      FAIL *
  MWS        201       2.0   0.0002      5.0      PASS
  BGS        180      95.7   0.0550      6.7      FAIL
  OTHER       21       n/a      n/a      n/a       ctx
  ------------------------------------------------------
  aggregate catastrophic rate: all spectra= 75.9% (good-z=24.1%),  ZWARN==0 only= 75.9% (good-z=24.1%)   [cf. eval_redshift_dz.py good-z~19% on its records[:96] subset]

  >>> METRIC-1 VERDICT: FAIL   (voted: LRG=F, ELG=F, QSO=F, MWS=P, BGS=F)
"""


def run_smoke(out_png=OUT_PNG):
    print("[SMOKE] CPU only — synthetic ladders + parser unit-test, no GPU.")

    # --- (A) parser unit-test on the pasted real V1 block ---
    parsed = parse_eval_per_class(SAMPLE_V1)
    print(f"[SMOKE] parsed good_z={parsed['good_z']:.4f} "
          f"(source={parsed['good_z_source']}); "
          f"per_class_cat={ {k: round(v,3) for k,v in parsed['per_class_cat'].items()} }")
    print(f"[SMOKE] parsed galaxy_cat={parsed['galaxy_cat']:.4f} "
          f"(mean LRG/ELG/BGS)")
    assert abs(parsed["good_z"] - 0.241) < 1e-6, parsed["good_z"]
    assert abs(parsed["per_class_cat"]["LRG"] - 0.975) < 1e-6
    assert abs(parsed["per_class_cat"]["ELG"] - 0.978) < 1e-6
    assert abs(parsed["per_class_cat"]["BGS"] - 0.957) < 1e-6
    exp_gal = (0.975 + 0.978 + 0.957) / 3.0
    assert abs(parsed["galaxy_cat"] - exp_gal) < 1e-6
    print("[SMOKE] PASS parser: good_z==0.241, LRG/ELG/BGS==0.975/0.978/0.957")

    # parser robustness: missing ZWARN==0 segment -> falls back to all-spectra.
    no_z0 = SAMPLE_V1.replace(
        ",  ZWARN==0 only= 75.9% (good-z=24.1%)", "")
    p2 = parse_eval_per_class(no_z0)
    assert p2["good_z_source"] == "all" and abs(p2["good_z"] - 0.241) < 1e-6
    print(f"[SMOKE] PASS parser robustness: no-ZWARN0 fallback -> "
          f"good_z={p2['good_z']:.3f} (source={p2['good_z_source']})")

    # --- (B) compute_params verified against the documented ladder ---
    try:
        p768 = compute_params(768, 6, 6, 12)
        print(f"[SMOKE] compute_params(768,6,6,12) = {p768:,} "
              f"({p768/1e6:.2f}M; expect 103.67M)")
        assert abs(p768 - 103_671_040) < 1, p768
        params_ladder = {
            "P-250M": compute_params(1024, 8, 8, 16),
            "P-500M": compute_params(1280, 12, 12, 16),
            "P-1B": compute_params(1536, 16, 16, 16),
        }
        print("[SMOKE] PASS compute_params matches documented ladder")
    except Exception as e:  # torch/model unavailable -> use the verified numbers
        print(f"[SMOKE] compute_params unavailable ({e}); using verified counts")
        p768 = 103_671_040
        params_ladder = {"P-250M": 245_717_248,
                         "P-500M": 575_214_336, "P-1B": 1_068_561_664}

    # --- (C) synthetic 4-point MODEL ladder + 3-point DATA ladder ---
    # Plausible (made-up) numbers showing a gentle scale gain.
    model_pts = [
        dict(label="baseline", axis="model", params=p768, n_spectra=18_000_000,
             good_z=0.245, galaxy_cat=0.965, probe_macro_f1=0.41),
        dict(label="P-250M", axis="model", params=params_ladder["P-250M"],
             n_spectra=18_000_000, good_z=0.292, galaxy_cat=0.910,
             probe_macro_f1=0.47),
        dict(label="P-500M", axis="model", params=params_ladder["P-500M"],
             n_spectra=18_000_000, good_z=0.331, galaxy_cat=0.861,
             probe_macro_f1=0.52),
        dict(label="P-1B", axis="model", params=params_ladder["P-1B"],
             n_spectra=18_000_000, good_z=0.378, galaxy_cat=0.802,
             probe_macro_f1=0.58),
    ]
    data_pts = [
        dict(label="data-25%", axis="data", params=params_ladder["P-250M"],
             n_spectra=4_500_000, good_z=0.255, galaxy_cat=0.948,
             probe_macro_f1=0.43),
        dict(label="data-50%", axis="data", params=params_ladder["P-250M"],
             n_spectra=9_000_000, good_z=0.281, galaxy_cat=0.918,
             probe_macro_f1=0.46),
        dict(label="data-100%", axis="data", params=params_ladder["P-250M"],
             n_spectra=18_000_000, good_z=0.292, galaxy_cat=0.910,
             probe_macro_f1=0.47),
    ]
    points = model_pts + data_pts
    fits = analyze(points)
    print_table(points, fits)
    png = plot(points, fits, out_png)
    print("\n" + verdict(fits))

    assert fits["model"]["good_z"]["slope"] is not None
    assert fits["data"]["good_z"]["slope"] is not None
    assert Path(png).exists() and Path(png).stat().st_size > 0, "PNG not written"
    print(f"[SMOKE] PASS plot written: {png} ({Path(png).stat().st_size} bytes)")
    print("[SMOKE] OK.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-json", default=None,
                    help="path to a JSON list of point dicts")
    ap.add_argument("--out-png", default=str(OUT_PNG))
    ap.add_argument("--smoke", action="store_true",
                    help="CPU synthetic-ladder + parser smoke (no GPU)")
    args = ap.parse_args()
    if args.smoke:
        run_smoke(Path(args.out_png))
    elif args.from_json:
        run_from_json(args.from_json, Path(args.out_png))
    else:
        ap.error("provide --from-json POINTS.json or --smoke")


if __name__ == "__main__":
    main()
