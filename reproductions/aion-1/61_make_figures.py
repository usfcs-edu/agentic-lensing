"""
61 -- Summary comparison figures (ours vs paper) for the quantitative tasks.

Reads the data/results/task*.json files and renders grouped bar charts comparing
our AION reproduction (per variant) to the paper's printed AION-B numbers. Saves
to figs/. Per-task qualitative figures (redshift posterior, low-data curves,
spectral super-res) are produced by their own scripts (50/25/51).

Run: python 61_make_figures.py
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _config as C

R = C.RESULTS
PT = C.paper_targets()
VAR = C.VARIANTS
COL = {"base": "C0", "large": "C1", "xlarge": "C2"}


def _load(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None


def main():
    C.FIGS.mkdir(parents=True, exist_ok=True)

    # Task 1: galaxy props R2 across configs (base variant) vs paper
    t1 = _load("task1_provabgs.json")
    if t1:
        targs = ["z", "logmass", "age", "logZ", "sSFR"]
        configs = [c for c in ["phot", "phot_image", "phot_spec", "phot_image_spec"] if c in t1]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(targs)); w = 0.8 / (len(configs) + 1)
        for i, cfg in enumerate(configs):
            if "base" not in t1[cfg]:
                continue
            vals = [t1[cfg]["base"]["attn_R2"][t] for t in targs]
            ax.bar(x + i * w, vals, w, label=f"ours {cfg} (B)")
        paper = [PT["galaxy_props_R2"][t] for t in targs]
        ax.bar(x + len(configs) * w, paper, w, label="paper phot+im+spec (B)", color="k", alpha=0.6)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(targs)
        ax.set_ylabel("R²"); ax.set_title("Task 1 — galaxy properties (AION-base) vs paper")
        ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(C.FIGS / "task1_galaxy_props.png", dpi=120); plt.close(fig)

    # Task 1 scaling: R2 vs variant for phot_image_spec (or best available config)
    if t1:
        cfg = "phot_image_spec" if "phot_image_spec" in t1 else (
            "phot_spec" if "phot_spec" in t1 else "phot")
        targs = ["z", "logmass", "age", "logZ", "sSFR"]
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(targs)); w = 0.8 / (len(VAR) + 1)
        for i, v in enumerate(VAR):
            if v in t1.get(cfg, {}):
                vals = [t1[cfg][v]["attn_R2"][t] for t in targs]
                ax.bar(x + i * w, vals, w, label=f"ours {v}", color=COL[v])
        ax.set_xticks(x + 0.3); ax.set_xticklabels(targs); ax.set_ylabel("R²")
        ax.set_title(f"Task 1 — scaling B/L/XL ({cfg})")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(C.FIGS / "task1_scaling.png", dpi=120); plt.close(fig)

    # Task 2: stellar props R2
    t2 = _load("task2_ddpayne.json")
    if t2:
        targs = ["Teff", "logg", "FeH", "vmic"]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(targs)); w = 0.8 / (len(VAR) + 1)
        for i, v in enumerate(VAR):
            if v in t2:
                ax.bar(x + i * w, [t2[v]["attn_R2"][t] for t in targs], w, label=f"ours {v}", color=COL[v])
        pp = PT["stellar_props_R2"]
        ax.bar(x + len(VAR) * w, [pp["teff"], pp["logg"], pp["feh"], pp["vmicro"]], w,
               label="paper DESI+Plx (B)", color="k", alpha=0.6)
        ax.set_xticks(x + 0.3); ax.set_xticklabels(targs); ax.set_ylabel("R²")
        ax.set_title("Task 2 — DESI stellar properties vs paper")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(C.FIGS / "task2_stellar_props.png", dpi=120); plt.close(fig)

    # Task 4: morphology accuracy vs paper
    t4 = _load("task4_gz10.json")
    if t4:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        vs = [v for v in VAR if v in t4]
        ax.bar(vs, [t4[v]["accuracy"] for v in vs], color=[COL[v] for v in vs])
        ax.axhline(PT["morphology_acc"], color="k", ls="--", label=f"paper {PT['morphology_acc']:.3f}")
        ax.set_ylabel("accuracy"); ax.set_title("Task 4 — Galaxy10 morphology")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(C.FIGS / "task4_morphology.png", dpi=120); plt.close(fig)

    print("MAKE_FIGURES_OK ->", C.FIGS)


if __name__ == "__main__":
    main()
