#!/usr/bin/env python
"""05_recovery_plot.py -- paper Fig. 11 style: recovered - truth for all systems.

For each of the 22 parameters, plot (posterior mean - truth) with 68% HPD error
bars across all fitted systems. A faithful reproduction overlays scatter at ~0.

Usage:
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python 05_recovery_plot.py
"""
import os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/fits/system_*_fit.npz")
    ap.add_argument("--out", default="figs/recovery_vs_truth.png")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    pat = args.glob if os.path.isabs(args.glob) else os.path.join(here, args.glob)
    files = sorted(glob.glob(pat))
    if not files:
        print(f"No fits matched {pat}"); return

    labels = None
    means, stds, truths = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        labels = [str(x) for x in d["phys_labels"]]  # arrays are in physical order
        means.append(np.asarray(d["phys_means"]))
        stds.append(np.asarray(d["phys_stds"]))
        truths.append(np.asarray(d["truth_vec"]))
    means = np.vstack(means); stds = np.vstack(stds); truths = np.vstack(truths)
    nsys, npar = means.shape
    diff = means - truths

    ncol = 4
    nrow = int(np.ceil(npar / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.2 * nrow))
    axes = axes.flatten()
    for k in range(npar):
        ax = axes[k]
        x = np.arange(nsys)
        ax.errorbar(x, diff[:, k], yerr=stds[:, k], fmt="o", ms=3,
                    elinewidth=0.8, capsize=2, color="C0")
        ax.axhline(0, color="r", lw=1)
        ax.set_title(labels[k], fontsize=9)
        ax.set_xticks([])
    for k in range(npar, len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"GIGA-Lens recovery: posterior mean - truth ({nsys} systems)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}  ({nsys} systems)")


if __name__ == "__main__":
    main()
