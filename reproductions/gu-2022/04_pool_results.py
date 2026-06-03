#!/usr/bin/env python
"""04_pool_results.py -- pool per-system fits into the paper's Table 2 statistics.

Reads data/fits/system_*_fit.npz and reports, for each of the 22 parameters
(and the 8 lensing params the paper highlights):
  - mean scaled error mu_z = <(mean - truth)/std>  (paper Table 2: should be ~0)
  - <Rhat>, max Rhat  (paper: max Rhat over all systems <= 1.017)
  - <ESS>,  min ESS   (paper: min ESS = 26822, target > 26000)
  - fraction of systems with truth inside the 68% / 95% posterior interval.

CPU-only post-processing (keep the A16s free).

Usage:
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python 04_pool_results.py
  ... --glob 'data/fits/system_*_fit.npz'
"""
import os, glob, argparse
import numpy as np

MASS_LABELS = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y",
               "gamma1", "gamma2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/fits/system_*_fit.npz")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    pat = args.glob if os.path.isabs(args.glob) else os.path.join(here, args.glob)
    files = sorted(glob.glob(pat))
    if not files:
        print(f"No fits matched {pat}")
        return

    labels = None
    ess_all, rhat_all, z_all = [], [], []
    cover68, cover95 = [], []
    sys_ids = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        # arrays (ess/rhat/means/stds/truth) are in the bijector's *physical* leaf
        # order -> label them with phys_labels (NOT param_labels).
        labels = [str(x) for x in d["phys_labels"]]
        ess_all.append(np.asarray(d["ess"]))
        rhat_all.append(np.asarray(d["rhat"]))
        z_all.append(np.asarray(d["z_err"]))
        sys_ids.append(int(d["idx"]))
        # coverage: truth within mean +/- {1,2} sigma
        means = np.asarray(d["phys_means"]); stds = np.asarray(d["phys_stds"])
        truth = np.asarray(d["truth_vec"])
        zz = np.abs((means - truth) / np.where(stds > 0, stds, np.nan))
        cover68.append(zz <= 1.0)
        cover95.append(zz <= 2.0)

    ess = np.vstack(ess_all)     # (Nsys, 22)
    rhat = np.vstack(rhat_all)
    z = np.vstack(z_all)
    c68 = np.vstack(cover68); c95 = np.vstack(cover95)
    nsys = ess.shape[0]

    print(f"Pooled {nsys} systems (ids {sorted(sys_ids)}) from {pat}\n")
    print(f"{'param':12s} {'mu_z':>7s} {'<Rhat>':>7s} {'maxRhat':>8s} "
          f"{'<ESS>':>8s} {'minESS':>8s} {'cov68':>6s} {'cov95':>6s}")
    print("-" * 70)

    def row(name, k):
        print(f"{name:12s} {np.nanmean(z[:, k]):7.3f} {rhat[:, k].mean():7.4f} "
              f"{rhat[:, k].max():8.4f} {ess[:, k].mean():8.0f} "
              f"{ess[:, k].min():8.0f} {c68[:, k].mean():6.2f} {c95[:, k].mean():6.2f}")

    mass_idx = [labels.index(m) for m in MASS_LABELS]
    print("# --- 8 lensing parameters (paper Fig 6 / Table 2) ---")
    for m in MASS_LABELS:
        row(m, labels.index(m))
    print("# --- all 22 parameters ---")
    for k, name in enumerate(labels):
        if name in MASS_LABELS:
            continue
        row(name, k)

    print("\n=== OVERALL (paper targets: maxRhat<=1.017, minESS>=26822, mu_z~0) ===")
    print(f"  max Rhat  (all params, all systems): {rhat.max():.4f}")
    print(f"  min ESS   (all params, all systems): {ess.min():.0f}")
    print(f"  mean ESS  (all params, all systems): {ess.mean():.0f}")
    print(f"  |mu_z| max over params             : {np.nanmax(np.abs(np.nanmean(z,axis=0))):.3f}")
    print(f"  68% coverage (all params)          : {c68.mean():.3f}")
    print(f"  95% coverage (all params)          : {c95.mean():.3f}")
    # mass-only
    print(f"  [mass only] max Rhat={rhat[:,mass_idx].max():.4f}  "
          f"min ESS={ess[:,mass_idx].min():.0f}  "
          f"mean ESS={ess[:,mass_idx].mean():.0f}")


if __name__ == "__main__":
    main()
