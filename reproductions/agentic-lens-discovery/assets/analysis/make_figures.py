#!/usr/bin/env python3
"""Paper figures. Usage: make_figures.py /path/to/jwst-strong-lens-search /path/to/papers/figures

Style: Okabe-Ito subset (validated), thin marks, recessive axes, direct labels.
"""
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO, FIGDIR = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(FIGDIR, exist_ok=True)

BLUE, ORANGE, GREEN, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
GRAY = "#7f7f7f"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.dpi": 200, "savefig.bbox": "tight",
    "font.family": "serif",
})


def load(rel):
    with open(os.path.join(REPO, rel)) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- fig: funnel
def fig_funnel():
    stages = [
        ("NIRCam observations (cal-3, $>$1 ks)", 6848),
        ("Elliptical targets ($r<21$)", 5391),
        ("Flagged by inspection", 2024),
        ("Adversarially verified", 350),
        ("Graded A--C ($\\geq$1 pass)", 22),
        ("Graded A/B ($\\geq$2 passes)", 10),
    ]
    labels = [s[0] for s in stages][::-1]
    vals = np.array([s[1] for s in stages][::-1], float)
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    y = np.arange(len(vals))
    ax.barh(y, vals, height=0.62, color=BLUE, edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlim(1, 2e4)
    ax.set_yticks(y, labels)
    for yi, v in zip(y, vals):
        ax.text(v * 1.25, yi, f"{int(v):,}", va="center", ha="left", fontsize=8.5,
                color="#333")
    ax.set_xlabel("count (log scale)")
    ax.xaxis.grid(True, color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGDIR, "fig_funnel.pdf"))
    plt.close(fig)


# ------------------------------------------------- fig: confidence + personas
def fig_confidence():
    res = load("results/results.csv")
    conf = defaultdict(list)
    for r in res:
        try:
            conf[r["lens_at_center"]].append(float(r["confidence"]))
        except ValueError:
            pass
    ver = load("results/verifications.csv")
    per = defaultdict(Counter)
    for r in ver:
        per[r["persona"]][r["verdict"]] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.3),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    bins = np.arange(0, 101, 5)
    for k, c, lab in (("likely", BLUE, "likely"), ("yes", VERM, "yes")):
        v = [x for x in conf[k] if x > 0]
        ax1.hist(v, bins=bins, histtype="step", lw=1.6, color=c,
                 label=f"“{lab}”")
    ax1.set_yscale("log")
    ax1.set_xlabel("inspector confidence")
    ax1.set_ylabel("cutouts")
    ax1.legend(frameon=False, loc="upper right")
    ax1.set_title("Screening verdicts (non-zero confidence)", loc="left")

    personas = ["artifact", "morphology", "geometry"]
    outcomes = [("pass", GREEN), ("uncertain", ORANGE), ("fail", GRAY)]
    yb = np.arange(len(personas))[::-1]
    h = 0.24
    for j, (out, c) in enumerate(outcomes):
        v = np.array([max(per[p].get(out, 0), 0.4) for p in personas], float)
        ax2.barh(yb + (1 - j) * h, v, h * 0.86, color=c)
        for yi, p in zip(yb, personas):
            n = per[p].get(out, 0)
            # direct-label the outcome name on the top (Artifact) group
            suffix = f" {out}" if p == personas[0] else ""
            ax2.text(max(n, 0.4) * 1.25, yi + (1 - j) * h, f"{n}{suffix}",
                     va="center", fontsize=7.5, color="#333")
    ax2.set_xscale("log")
    ax2.set_xlim(0.4, 9000)
    ax2.set_yticks(yb, [p.capitalize() for p in personas])
    ax2.set_xlabel("judgements (log scale)")
    ax2.set_title("Adversarial verification (350 candidates)", loc="left")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIGDIR, "fig_confidence.pdf"))
    plt.close(fig)


# ------------------------------------------------------------- fig: recovery
def fig_recovery():
    ctl = load("results/control_recovery.csv")
    kl = load("results/known_lens_recovery.csv")

    def er(r):
        try:
            return float(r["einstein_radius"])
        except ValueError:
            return None

    bins = [("$<0.75''$", lambda e: e is not None and e < 0.75),
            ("$0.75$--$1.2''$", lambda e: e is not None and 0.75 <= e < 1.2),
            ("$\\geq 1.2''$", lambda e: e is not None and e >= 1.2),
            ("no $\\theta_{\\rm E}$", lambda e: e is None)]
    fr, ns = [], []
    for _lab, sel in bins:
        sub = [r for r in ctl if sel(er(r))]
        n = len(sub)
        f = sum(1 for r in sub if r["flagged"] == "True")
        ns.append((f, n))
        fr.append(f / n if n else 0)

    srcs = [("COWLS", "COWLS"), ("SIMBAD:gLS", "gLS"), ("SIMBAD:LeG", "LeG"),
            ("SIMBAD:LeI", "LeI"), ("SIMBAD:LeQ", "LeQ"), ("SIMBAD:gLe", "gLe")]
    kfr, kns = [], []
    for src, _lab in srcs:
        sub = [r for r in kl if r["lens_src"] == src]
        n = len(sub)
        f = sum(1 for r in sub if r["flagged"] == "True")
        kns.append((f, n))
        kfr.append(f / n if n else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.2),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    x = np.arange(len(bins))
    ax1.bar(x, fr, 0.55, color=BLUE)
    for xi, (f, n) in zip(x, ns):
        ax1.text(xi, fr[list(x).index(xi)] + 0.03, f"{f}/{n}", ha="center", fontsize=8.5)
    ax1.set_xticks(x, [b[0] for b in bins], fontsize=8)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("fraction flagged")
    ax1.set_title("Injected controls (15/31 = 48%)", loc="left", fontsize=8.5)
    ax1.axhline(15 / 31, color=GRAY, lw=0.8, ls="--")

    x = np.arange(len(srcs))
    ax2.bar(x, kfr, 0.55, color=BLUE)
    for xi, (f, n) in zip(x, kns):
        ax2.text(xi, kfr[list(x).index(xi)] + 0.03, f"{f}/{n}", ha="center", fontsize=8.5)
    ax2.set_xticks(x, [s[1] for s in srcs])
    ax2.set_ylim(0, 1.12)
    ax2.set_title("Catalogued lenses on cutouts (128/239 = 54%)", loc="left",
                  fontsize=8.5)
    ax2.axhline(128 / 239, color=GRAY, lw=0.8, ls="--")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIGDIR, "fig_recovery.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------- fig: cost
def fig_cost():
    rows = list(csv.DictReader(open(os.path.join(HERE, "out", "agent_stats.csv"))))
    # killed flags from MANIFEST
    killed_files = set()
    with open(os.path.join(HERE, "..", "MANIFEST.csv")) as f:
        for r in csv.DictReader(f):
            if "session limit" in r["first_text"]:
                killed_files.add(r["path"])
    roles = ["inspect", "verify", "literature", "catalogue", "notes"]
    lab = {"inspect": "Inspection", "verify": "Verification",
           "literature": "Literature", "catalogue": "Catalogue", "notes": "Notes"}
    agents = Counter()
    killed = Counter()
    tok = Counter()
    for r in rows:
        role = r["role"]
        if role not in roles:
            continue
        agents[role] += 1
        if r["file"] in killed_files:
            killed[role] += 1
        tok[role] += (int(r["in_tok"]) + int(r["out_tok"]) + int(r["cache_create"]))
    print("cost fig killed per role:", dict(killed))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.0))
    y = np.arange(len(roles))[::-1]
    live = np.array([agents[r] - killed[r] for r in roles], float)
    kd = np.array([killed[r] for r in roles], float)
    ax1.barh(y, live, 0.6, color=BLUE, label="completed")
    ax1.barh(y, kd, 0.6, left=live, color=ORANGE, label="killed by usage limit")
    ax1.set_yticks(y, [lab[r] for r in roles])
    for yi, (l, k) in zip(y, zip(live, kd)):
        ax1.text(l + k + 8, yi, f"{int(l + k)}", va="center", fontsize=8.5, color="#333")
    ax1.set_xlabel("subagents")
    ax1.set_xlim(0, 640)
    ax1.legend(frameon=False, loc="lower right", fontsize=7.5)

    mt = np.array([tok[r] / 1e6 for r in roles], float)
    ax2.barh(y, mt, 0.6, color=BLUE)
    for yi, v in zip(y, mt):
        ax2.text(v + 0.6, yi, f"{v:.1f}M", va="center", fontsize=8.5, color="#333")
    ax2.set_yticks(y, ["" for _ in roles])
    ax2.set_xlabel("billable tokens (input + output + cache-write)")
    ax2.set_xlim(0, 33)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIGDIR, "fig_cost.pdf"))
    plt.close(fig)


# ------------------------------------------------------------- copies
def copy_images():
    shutil.copy(os.path.join(REPO, "results/contact_sheet.jpg"),
                os.path.join(FIGDIR, "contact_sheet.jpg"))
    for cid in ["J3440482-522486", "J15199556+2122210", "J16644236-1024898",
                "J23069956+2559453", "J20954380-1094330", "J34707505-219476"]:
        src = os.path.join(REPO, "annotated", f"{cid}.jpg")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(FIGDIR, f"cand_{cid}.jpg"))
        else:
            print("MISSING", src)


if __name__ == "__main__":
    fig_funnel()
    fig_confidence()
    fig_recovery()
    fig_cost()
    copy_images()
    print("figures written to", FIGDIR)
