"""05 - Summary comparison figure & table: automated redshifts vs Lin et al. 2025.

Reads data/measured_redshifts.csv (written by 04) and makes:
  - figs/05_z_comparison.png : auto vs published z for lens & source, with the
    paper's target ranges marked.
  - prints a compact pass/fail table (lens dz < 0.005 target).
"""
from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"


def main():
    f = DATA / "measured_redshifts.csv"
    if not f.exists():
        raise SystemExit("run 04_measure_redshifts.py first")
    rows = list(csv.DictReader(f.open()))
    for r in rows:
        for k in ("z_lens_pub", "z_lens_auto", "z_src_pub", "z_src_auto",
                  "dz_lens", "dz_src", "lens_snr", "src_snr"):
            r[k] = float(r[k])

    print("AUTOMATED REDSHIFTS vs Lin et al. 2025 (paper, Sect 4.1)")
    print(f"{'system':26} {'zL_auto':>8} {'zL_pub':>7} {'dzL':>8} {'SNR':>5}  "
          f"{'zS_auto':>8} {'zS_pub':>7} {'dzS':>8} {'eng':>8} {'SNR':>5}")
    n_lens_ok = 0
    for r in rows:
        lok = abs(r["dz_lens"]) < 0.005
        n_lens_ok += lok
        print(f"{r['name']:26} {r['z_lens_auto']:8.4f} {r['z_lens_pub']:7.3f} "
              f"{r['dz_lens']:+8.4f} {r['lens_snr']:5.1f}  "
              f"{r['z_src_auto']:8.4f} {r['z_src_pub']:7.3f} {r['dz_src']:+8.4f} "
              f"{r['src_engine']:>8} {r['src_snr']:5.1f}  {'LENS-OK' if lok else ''}")
    print(f"\nLens redshifts within dz<0.005 of paper: {n_lens_ok}/{len(rows)}")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    zl_p = [r["z_lens_pub"] for r in rows]; zl_a = [r["z_lens_auto"] for r in rows]
    zs_p = [r["z_src_pub"] for r in rows]; zs_a = [r["z_src_auto"] for r in rows]
    names = [r["target"] for r in rows]

    ax[0].plot([0, 1.2], [0, 1.2], "k--", lw=0.7, alpha=0.6)
    ax[0].scatter(zl_p, zl_a, c="C3", s=70, zorder=3)
    for x, y, n in zip(zl_p, zl_a, names):
        ax[0].annotate(n, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax[0].set_xlabel("published z_lens (Lin+2025)"); ax[0].set_ylabel("automated z_lens")
    ax[0].set_title("Lens redshifts (auto vs paper)")

    ax[1].plot([0, 3.5], [0, 3.5], "k--", lw=0.7, alpha=0.6)
    ax[1].scatter(zs_p, zs_a, c="C0", s=70, zorder=3)
    for x, y, n in zip(zs_p, zs_a, names):
        ax[1].annotate(n, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax[1].set_xlabel("published z_source (Lin+2025)"); ax[1].set_ylabel("automated z_source")
    ax[1].set_title("Source redshifts (auto vs paper)")
    fig.tight_layout()
    out = FIGS / "05_z_comparison.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
