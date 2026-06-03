#!/usr/bin/env python
"""Foundry II - merge the parsed Table-2, the DR1 z cross-match, and the
FastSpecFit sigma_v recovery into one master comparison table + a summary plot.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)


def load(fn):
    return {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", fn)))}


def f(x):
    return float(x) if x not in ("", "None", None) else None


def main():
    xm = load("foundry_ii_dr1_crossmatch.csv")
    sv = load("foundry_ii_sigmav.csv")
    tab = list(csv.DictReader(open(os.path.join(HERE, "data/foundry_ii_table2.csv"))))

    rows = []
    for t in tab:
        nm = t["name"]
        x = xm[nm]; s = sv.get(nm, {})
        rows.append({
            "name": nm, "section": t["section"],
            "ra_deg": t["ra_deg"], "dec_deg": t["dec_deg"],
            "in_dr1_1.5as": x["matched"],
            "closest_sep_as": x["closest_sep_arcsec"],
            "z_lens_pub": t["z_lens_pub"], "z_lens_dr1": x["z_lens_dr1"],
            "z_lens_match": x["z_lens_match"], "z_lens_zwarn": x["z_lens_dr1_zwarn"],
            "z_source_pub": t["z_source_pub"], "z_source_dr1": x["z_source_dr1"],
            "z_source_match": x["z_source_match"],
            "z_s2_pub": t["z_s2"], "z_s2_dr1": x["z_s2_dr1"], "z_s2_match": x["z_s2_match"],
            "sigma_v_pub": t["sigma_v_pub"], "sigma_v_err_pub": t["sigma_v_err_pub"],
            "vdisp_dr1_fsf": s.get("vdisp_dr1_fsf"), "delta_sigma": s.get("delta_sigma"),
        })

    out = os.path.join(HERE, "data/foundry_ii_master_comparison.csv")
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
    print("Wrote", out)

    # ---- summary plot: pub vs recovered z (lens+source) and sigma_v ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))

    zlp = [f(r["z_lens_pub"]) for r in rows if r["z_lens_match"] == "True"]
    zld = [f(r["z_lens_dr1"]) for r in rows if r["z_lens_match"] == "True"]
    zsp = [f(r["z_source_pub"]) for r in rows if r["z_source_match"] == "True"]
    zsd = [f(r["z_source_dr1"]) for r in rows if r["z_source_match"] == "True"]
    ax[0].plot([0, 3.3], [0, 3.3], "k--", lw=1, alpha=.6)
    ax[0].scatter(zlp, zld, s=40, c="C0", label=f"lens z  (n={len(zlp)})", zorder=3)
    ax[0].scatter(zsp, zsd, s=40, c="C3", marker="^", label=f"source z  (n={len(zsp)})", zorder=3)
    ax[0].set_xlabel("published z (Foundry II / DESI EDR)")
    ax[0].set_ylabel("recovered z (DESI DR1 / Iron)")
    ax[0].set_title("Redshift recovery (|dz|<0.005 match)")
    ax[0].legend(); ax[0].set_xlim(0, 3.3); ax[0].set_ylim(0, 3.3)

    sp = [f(r["sigma_v_pub"]) for r in rows if r["vdisp_dr1_fsf"] not in ("", "None", None)]
    sd = [f(r["vdisp_dr1_fsf"]) for r in rows if r["vdisp_dr1_fsf"] not in ("", "None", None)]
    sp = np.array(sp); sd = np.array(sd)
    lo, hi = 100, 650
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1, alpha=.6)
    ax[1].scatter(sp, sd, s=40, c="C2", zorder=3)
    r_ = np.corrcoef(sp, sd)[0, 1]
    ax[1].set_xlabel(r"published $\sigma_v$ [km/s]  (FastSpecFit, EDR)")
    ax[1].set_ylabel(r"recovered $\sigma_v$ [km/s]  (FastSpecFit, DR1)")
    ax[1].set_title(rf"Velocity dispersion (n={len(sp)}, r={r_:.2f})")
    ax[1].set_xlim(lo, hi); ax[1].set_ylim(lo, hi)

    fig.suptitle("DESI Strong Lens Foundry II - DR1 reproduction of Table 2", fontsize=13)
    fig.tight_layout()
    figp = os.path.join(HERE, "figs/foundry_ii_recovery.png")
    os.makedirs(os.path.dirname(figp), exist_ok=True)
    fig.savefig(figp, dpi=130)
    print("Wrote", figp)


if __name__ == "__main__":
    main()
