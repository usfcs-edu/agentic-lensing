"""08_pool_e1.py — P1b E1 pooling: gate tables, figures, data/e1_report.json.

CPU-only. Reads data/e1_fits/*.npz and reports every pre-registered E1 gate
(README §P1) verbatim, PASS or FAIL, plus the E1d three-way verdict per the
criteria pre-registered in the P1b tasking:

  E1a: median |z(gamma)| > 2 OR 68% coverage < 40%     (artifact REPRODUCED)
  E1b: per scale |zbar(gamma)| < 0.5; cross-scale |gamma_fine - gamma_native|
       < 1 sigma_comb on >= 6/8; width ratio sigma_fine/sigma_native in
       [0.7, 1.5] (gated on the MEDIAN ratio; per-mock table reported)
  E1c: rank-uniformity chi^2 p > 0.01 for each of {theta_E, gamma, e1, e2,
       gamma1, gamma2} (8 bins, ranks thinned to 127 draws); empirical 68%
       coverage in [55, 80]% — the coverage gate is applied POOLED over the
       six mass params (the README wording carries the per-param qualifier
       only on the rank clause; per-param coverage is reported alongside).
  E1d: relaxed ADOPTED iff max_param |zbar| < 0.5 AND pooled 68% coverage in
       [55, 80]% AND kept pixels >= 3x strict; else strict stands (or diag
       if both whiteners fail coverage, failure documented).

Figures -> figs/e1_*.png. Run:
  /raid/benson/.venvs/cgl/bin/python 08_pool_e1.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPRO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO))

from cgl.e1 import (  # noqa: E402
    MASS_LABELS, coverage_flag, rank_uniformity_chi2, sbc_rank, thin_indices,
)
from cgl.paths import DATA, FIGS  # noqa: E402

FITS = DATA / "e1_fits"
N_E1B = 8
SBC_SEEDS = range(64)
E1D_SEEDS = range(100, 116)
RHAT_WARN = 1.05
ESS_WARN = 100.0


def load_fit(path):
    z = np.load(path, allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    labels = [str(x) for x in z["phys_labels"]]
    d = dict(path=str(path), cfg=cfg, labels=labels,
             truth=np.asarray(z["truth_vec"], dtype=np.float64),
             mean=np.asarray(z["post_mean"], dtype=np.float64),
             std=np.asarray(z["post_std"], dtype=np.float64),
             z=np.asarray(z["zscore"], dtype=np.float64),
             cov68=np.asarray(z["cov68"], dtype=bool),
             cov95=np.asarray(z["cov95"], dtype=bool),
             ess=np.asarray(z["ess"], dtype=np.float64),
             rhat=np.asarray(z["rhat"], dtype=np.float64),
             draws=np.asarray(z["phys_draws"], dtype=np.float32),
             map_best_lp=np.asarray(z["map_best_lp"], dtype=np.float64),
             map_pop_gamma=np.asarray(z["map_pop_gamma"], dtype=np.float32),
             map_pop_lp=np.asarray(z["map_pop_lp"], dtype=np.float32))
    d["idx"] = {lab: labels.index(lab) for lab in MASS_LABELS}
    mi = [d["idx"][m] for m in MASS_LABELS]
    d["health"] = dict(
        max_rhat_mass=float(np.max(d["rhat"][mi])),
        min_ess_mass=float(np.min(d["ess"][mi])),
        healthy=bool(np.max(d["rhat"][mi]) < RHAT_WARN
                     and np.min(d["ess"][mi]) > ESS_WARN))
    return d


def get(name):
    p = FITS / f"{name}.npz"
    return load_fit(p) if p.exists() else None


def gamma_stats(fit):
    j = fit["idx"]["gamma"]
    return fit["mean"][j], fit["std"][j], fit["z"][j], bool(fit["cov68"][j])


# --------------------------------------------------------------------------- #
def pool_e1a():
    fits = [get(f"mock{s:03d}_fine_diag_recal") for s in range(N_E1B)]
    fits = {s: f for s, f in enumerate(fits) if f is not None}
    if not fits:
        return dict(status="NO FITS"), {}
    zg, cov, rows, basins = [], [], [], []
    for s, f in fits.items():
        m, sd, z, c68 = gamma_stats(f)
        zg.append(z)
        cov.append(c68)
        rec = f["cfg"].get("recal_info") or {}
        rows.append(dict(seed=s, gamma_truth=f["truth"][f["idx"]["gamma"]],
                         mean=m, std=sd, z=z, cov68=c68,
                         recal_factor=rec.get("rescale"),
                         health=f["health"]))
        # basin splitting: spread of the 4 MAP starts (best-lp and the
        # per-start best-particle gamma)
        bl = f["map_best_lp"]
        bg = [float(f["map_pop_gamma"][k][np.argmax(f["map_pop_lp"][k])])
              for k in range(f["map_pop_gamma"].shape[0])]
        basins.append(dict(seed=s, n_starts=len(bl),
                           best_lp_spread=float(np.max(bl) - np.min(bl)),
                           start_best_gammas=bg,
                           gamma_spread=float(np.max(bg) - np.min(bg))))
    zg = np.array(zg)
    med_abs_z = float(np.median(np.abs(zg)))
    cov68 = float(np.mean(cov))
    gate = bool(med_abs_z > 2.0 or cov68 < 0.40)
    return dict(
        n_fits=len(fits), median_abs_z_gamma=med_abs_z,
        coverage68_gamma=cov68, z_gamma=zg.tolist(),
        gate="median |z(gamma)| > 2 OR 68% coverage < 40%",
        gate_pass=gate,
        verdict=("ARTIFACT REPRODUCED" if gate else
                 "artifact NOT reproduced (diag likelihood looks calibrated)"),
        per_mock=rows, basins=basins), fits


def pool_e1b():
    out = dict(scales={}, cross_scale=[], ablation=[])
    per_scale_fits = {}
    for scale in ("fine", "binned", "native"):
        fits = {s: get(f"mock{s:03d}_{scale}_corr_fitted")
                for s in range(N_E1B)}
        fits = {s: f for s, f in fits.items() if f is not None}
        per_scale_fits[scale] = fits
        if not fits:
            out["scales"][scale] = dict(status="NO FITS")
            continue
        zg = np.array([gamma_stats(f)[2] for f in fits.values()])
        cov = np.array([gamma_stats(f)[3] for f in fits.values()])
        zbar = float(np.mean(zg))
        out["scales"][scale] = dict(
            n_fits=len(fits), zbar_gamma=zbar,
            zbar_gate_pass=bool(abs(zbar) < 0.5),
            z_gamma=zg.tolist(), coverage68_gamma=float(np.mean(cov)),
            median_sigma_gamma=float(np.median(
                [gamma_stats(f)[1] for f in fits.values()])),
            health=[f["health"] for f in fits.values()])
    ff, nf = per_scale_fits.get("fine", {}), per_scale_fits.get("native", {})
    n_agree, ratios = 0, []
    for s in range(N_E1B):
        if s in ff and s in nf:
            mf, sf, _, _ = gamma_stats(ff[s])
            mn, sn, _, _ = gamma_stats(nf[s])
            comb = float(np.hypot(sf, sn))
            agree = bool(abs(mf - mn) < comb)
            n_agree += agree
            ratios.append(sf / sn)
            out["cross_scale"].append(dict(
                seed=s, gamma_fine=mf, sig_fine=sf, gamma_native=mn,
                sig_native=sn, dgamma=mf - mn, sigma_comb=comb,
                agree_1sig=agree, width_ratio=sf / sn))
    n_pairs = len(ratios)
    med_ratio = float(np.median(ratios)) if ratios else float("nan")
    out["cross_scale_gate"] = dict(
        n_agree=n_agree, n_pairs=n_pairs,
        gate=">=6/8 with |dgamma| < 1 sigma_comb",
        gate_pass=bool(n_agree >= 6 and n_pairs >= 8))
    out["width_gate"] = dict(
        median_ratio=med_ratio, ratios=ratios,
        gate="median sigma_fine/sigma_native in [0.7, 1.5] "
             "(per-mock ratios reported)",
        gate_pass=bool(n_pairs > 0 and 0.7 <= med_ratio <= 1.5))
    for s in (0, 1):
        fa = get(f"mock{s:03d}_fine_corr_analytic")
        if fa is not None and s in ff:
            ma, sa, za, _ = gamma_stats(fa)
            mf, sf, zf, _ = gamma_stats(ff[s])
            out["ablation"].append(dict(
                seed=s, gamma_fitted=mf, gamma_analytic=ma,
                dgamma_over_sigma=float((mf - ma) / sf),
                sig_fitted=sf, sig_analytic=sa, z_fitted=zf, z_analytic=za))
    return out, per_scale_fits


def pool_e1c():
    fits = {s: get(f"mock{s:03d}_fine_corr_fitted") for s in SBC_SEEDS}
    fits = {s: f for s, f in fits.items() if f is not None}
    if not fits:
        return dict(status="NO FITS"), {}
    ranks = {m: [] for m in MASS_LABELS}
    cov68 = {m: [] for m in MASS_LABELS}
    cov95 = {m: [] for m in MASS_LABELS}
    unhealthy = []
    for s, f in fits.items():
        for m in MASS_LABELS:
            j = f["idx"][m]
            ranks[m].append(sbc_rank(f["draws"][:, j], f["truth"][j]))
            cov68[m].append(bool(f["cov68"][j]))
            cov95[m].append(bool(f["cov95"][j]))
        if not f["health"]["healthy"]:
            unhealthy.append(dict(seed=s, **f["health"]))
    per_param = {}
    all_p_ok = True
    for m in MASS_LABELS:
        chi2, p, obs = rank_uniformity_chi2(np.array(ranks[m]))
        per_param[m] = dict(chi2=chi2, p=p, bins=obs,
                            p_gate_pass=bool(p > 0.01),
                            coverage68=float(np.mean(cov68[m])),
                            coverage95=float(np.mean(cov95[m])))
        all_p_ok &= p > 0.01
    pooled68 = float(np.mean([np.mean(cov68[m]) for m in MASS_LABELS]))
    cov_ok = 0.55 <= pooled68 <= 0.80
    return dict(
        n_fits=len(fits), per_param=per_param, ranks={m: ranks[m]
                                                      for m in MASS_LABELS},
        pooled_coverage68=pooled68,
        gates=dict(
            rank_p="p > 0.01 for each of 6 mass params",
            rank_p_pass=bool(all_p_ok),
            coverage="pooled 68% coverage in [55, 80]% (per-param reported; "
                     "pooling interpretation documented in module docstring)",
            coverage_pass=bool(cov_ok),
            all_pass=bool(all_p_ok and cov_ok)),
        n_shapelet_fits=sum(1 for f in fits.values()
                            if f["cfg"]["log"]["shapelets"]),
        unhealthy=unhealthy), fits


def pool_e1d():
    arms = {}
    for arm in ("diag", "strict", "relaxed"):
        fits = {s: get(f"e1d{s:03d}_{arm}") for s in E1D_SEEDS}
        fits = {s: f for s, f in fits.items() if f is not None}
        if not fits:
            arms[arm] = dict(status="NO FITS")
            continue
        zbar, cov, widths = {}, [], {}
        for m in MASS_LABELS:
            zs = [f["z"][f["idx"][m]] for f in fits.values()]
            zbar[m] = float(np.mean(zs))
            widths[m] = float(np.median([f["std"][f["idx"][m]]
                                         for f in fits.values()]))
            cov.extend(bool(f["cov68"][f["idx"][m]]) for f in fits.values())
        n_kept = [int(np.sum(f["cfg"]["n_keep_w"]))
                  if f["cfg"]["log"]["likelihood"] == "corr"
                  else int(np.sum(f["cfg"]["n_keep"]))
                  for f in fits.values()]
        arms[arm] = dict(
            n_fits=len(fits), zbar=zbar,
            max_abs_zbar=float(max(abs(v) for v in zbar.values())),
            coverage68_pooled=float(np.mean(cov)),
            median_sigma=widths,
            kept_pixels=int(np.median(n_kept)),
            whitener=fits[min(fits)]["cfg"].get("kernel_info"),
            health=[f["health"] for f in fits.values()])
    verdict = dict(criteria=(
        "relaxed ADOPTED iff max|zbar| < 0.5 AND pooled cov68 in [55,80]% "
        "AND kept >= 3x strict; else strict; else diag if both fail "
        "coverage"))
    ok = {a: (arms[a].get("max_abs_zbar", np.inf) < 0.5
              and 0.55 <= arms[a].get("coverage68_pooled", -1) <= 0.80)
          for a in ("strict", "relaxed") if "zbar" in arms.get(a, {})}
    if "relaxed" in ok and ok["relaxed"] and \
            arms["relaxed"]["kept_pixels"] >= 3 * arms.get(
                "strict", {}).get("kept_pixels", np.inf):
        verdict["adopted"] = "relaxed"
    elif "strict" in ok and ok["strict"]:
        verdict["adopted"] = "strict"
    elif "relaxed" in ok and ok["relaxed"]:
        verdict["adopted"] = "relaxed (strict failed calibration)"
    else:
        verdict["adopted"] = ("diag (both whiteners failed coverage — "
                              "documented failure)" if ok else "UNDECIDED")
    verdict["calibration_ok"] = ok
    return dict(arms=arms, verdict=verdict), None


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_zscore_violins(e1a, e1b, e1d):
    groups, data = [], []
    if "z_gamma" in e1a:
        groups.append("fine\ndiag (E1a)")
        data.append(e1a["z_gamma"])
    for scale in ("fine", "binned", "native"):
        sc = e1b["scales"].get(scale, {})
        if "z_gamma" in sc:
            groups.append(f"{scale}\ncorr (E1b)")
            data.append(sc["z_gamma"])
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.violinplot(data, showmedians=True, widths=0.8)
    ax.axhline(0, color="k", lw=0.8)
    for y in (-2, 2):
        ax.axhline(y, color="crimson", lw=0.8, ls="--")
    ax.set_xticks(range(1, len(groups) + 1), groups)
    ax.set_ylabel(r"$z(\gamma)$ = (post mean $-$ truth)/post std")
    ax.set_title("E1 mock recovery: z-scores of the EPL slope")
    fig.tight_layout()
    fig.savefig(FIGS / "e1_zscore_violins.png", dpi=150)
    plt.close(fig)


def fig_sbc(e1c):
    if "per_param" not in e1c:
        return
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), sharey=True)
    for ax, m in zip(axes.ravel(), MASS_LABELS):
        r = np.array(e1c["ranks"][m])
        ax.hist(r, bins=np.arange(9) * 16, color="#4477aa",
                edgecolor="white")
        n = r.size
        exp = n / 8
        ax.axhline(exp, color="k", lw=0.8, ls="--")
        sd = np.sqrt(exp * (1 - 1 / 8))
        ax.axhspan(exp - 2 * sd, exp + 2 * sd, color="gray", alpha=0.25)
        p = e1c["per_param"][m]["p"]
        ax.set_title(f"{m}  (p={p:.3f})", fontsize=10)
    fig.suptitle(f"E1c SBC ranks (n={e1c['n_fits']} mocks, 127-draw thinning,"
                 " 8 bins; band = ±2σ)")
    fig.tight_layout()
    fig.savefig(FIGS / "e1_sbc_ranks.png", dpi=150)
    plt.close(fig)


def fig_coverage(e1a, e1b, e1c):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars, vals = [], []
    if "coverage68_gamma" in e1a:
        bars.append("γ fine diag\n(E1a)")
        vals.append(e1a["coverage68_gamma"])
    for scale in ("fine", "binned", "native"):
        sc = e1b["scales"].get(scale, {})
        if "coverage68_gamma" in sc:
            bars.append(f"γ {scale} corr\n(E1b)")
            vals.append(sc["coverage68_gamma"])
    if "per_param" in e1c:
        for m in MASS_LABELS:
            bars.append(f"{m}\n(E1c)")
            vals.append(e1c["per_param"][m]["coverage68"])
    if not vals:
        plt.close(fig)
        return
    colors = ["#cc6677" if "diag" in b else "#4477aa" for b in bars]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0.68, color="k", ls="--", lw=0.9, label="nominal 68%")
    ax.axhspan(0.55, 0.80, color="gray", alpha=0.2, label="E1c band")
    ax.axhline(0.40, color="crimson", ls=":", lw=0.9, label="E1a threshold")
    ax.set_xticks(range(len(bars)), bars, fontsize=8)
    ax.set_ylabel("empirical 68% CI coverage")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "e1_coverage.png", dpi=150)
    plt.close(fig)


def fig_cross_scale(e1b, e1a_fits):
    rows = e1b.get("cross_scale", [])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(rows))
    truth = []
    for i, r in enumerate(rows):
        s = r["seed"]
        f = e1a_fits.get(s)
        gt = f["truth"][f["idx"]["gamma"]] if f else np.nan
        truth.append(gt)
        for dx, key, kerr, col, lab in (
                (-0.18, "gamma_fine", "sig_fine", "#4477aa", "fine corr"),
                (0.18, "gamma_native", "sig_native", "#66ccee",
                 "native corr")):
            ax.errorbar(i + dx, r[key], yerr=r[kerr], fmt="o", ms=4,
                        color=col, label=lab if i == 0 else None)
        if f is not None:
            j = f["idx"]["gamma"]
            ax.errorbar(i, f["mean"][j], yerr=f["std"][j], fmt="s", ms=4,
                        color="#cc6677",
                        label="fine diag (E1a)" if i == 0 else None)
    ax.plot(x, truth, "k*", ms=10, label="truth")
    ax.set_xticks(x, [f"mock {r['seed']}" for r in rows])
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("E1b cross-scale EPL slope recovery")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "e1_cross_scale.png", dpi=150)
    plt.close(fig)


def fig_e1d(e1d):
    arms = e1d["arms"]
    have = [a for a in ("diag", "strict", "relaxed") if "zbar" in
            arms.get(a, {})]
    if not have:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    x = np.arange(len(MASS_LABELS))
    for a in have:
        axes[0].plot(x, [arms[a]["zbar"][m] for m in MASS_LABELS], "o-",
                     label=a)
        axes[1].bar(have.index(a), arms[a]["coverage68_pooled"], width=0.6)
        axes[2].bar(have.index(a), arms[a]["kept_pixels"], width=0.6)
    axes[0].axhline(0, color="k", lw=0.8)
    for y in (-0.5, 0.5):
        axes[0].axhline(y, color="crimson", ls="--", lw=0.8)
    axes[0].set_xticks(x, MASS_LABELS, rotation=45, fontsize=8)
    axes[0].set_ylabel(r"$\bar z$ over 16 realizations")
    axes[0].legend(fontsize=8)
    axes[1].axhspan(0.55, 0.80, color="gray", alpha=0.25)
    axes[1].axhline(0.68, color="k", ls="--", lw=0.8)
    axes[1].set_xticks(range(len(have)), have)
    axes[1].set_ylabel("pooled 68% coverage")
    axes[2].set_xticks(range(len(have)), have)
    axes[2].set_ylabel("kept pixels (median)")
    axes[2].set_yscale("log")
    fig.suptitle(f"E1d v2d-realism arms — verdict: "
                 f"{e1d['verdict'].get('adopted', '?')}")
    fig.tight_layout()
    fig.savefig(FIGS / "e1_e1d_compare.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="data/e1_report.json")
    args = ap.parse_args()
    FIGS.mkdir(exist_ok=True)

    e1a, e1a_fits = pool_e1a()
    e1b, _ = pool_e1b()
    e1c, _ = pool_e1c()
    e1d, _ = pool_e1d()

    fig_zscore_violins(e1a, e1b, e1d)
    fig_sbc(e1c)
    fig_coverage(e1a, e1b, e1c)
    fig_cross_scale(e1b, e1a_fits)
    fig_e1d(e1d)

    # strip bulky arrays from the JSON report
    e1c_json = {k: v for k, v in e1c.items() if k != "ranks"}

    report = dict(
        generated_by="08_pool_e1.py",
        deviations=[
            "fit prior == truth-sampling distribution (SBC validity; "
            "gu-2022's broader modelling prior would fail rank uniformity "
            "by construction)",
            "mock 'native product' = 9 native exposures fit jointly (iid); "
            "single-exposure native carries 1/9 the photons and fails the "
            "width-ratio gate by design",
            "near-singular mock fine kernels whitened against the "
            "delta-regularized model K_reg=(K+0.1*delta)/1.1 (conservative; "
            "spectral-floor target is kinked and truncated taps stall at "
            "e_op~0.2; measured M=20 e_op=0.0174 after regularization)",
            "E1a recalibration uses true-model-subtracted residuals over "
            "r>3\" (mocks have no source-free sky; endpoint identical: "
            "sky chi2_pp=1)",
            "E1c coverage gate applied pooled over the 6 mass params "
            "(README wording carries the per-param qualifier only on the "
            "rank clause); per-param coverage reported",
            "07 batcher uses a flock FIFO queue (one process per GPU) "
            "instead of fixed waves (L4 ~2x A16)",
            "QC REMEDIATION 2026-07-06 ~13:55 (mid-batch, 30/135 done): "
            "delta-reg engage rule extended to fall back on e_op-gate "
            "failure, not only on spectrum ratio < 0.02 (fitted fine "
            "kernels' noisy ratio estimate landed just above threshold on "
            "seeds 5/7/9/11 -> un-regularized whiteners stalled at e_op "
            "0.020-0.19, violating the pre-registered 0.02 gate; fits run "
            "with those whiteners are invalid measurements, not gate "
            "results). Out-of-spec kernels + dependent fits quarantined "
            "(data/e1_quarantine/) and re-run under the fixed policy; "
            "rebuilt e_op: seed7 0.0149, seed9 0.0183, seed11 0.0176. "
            "Seed5 remains marginal at 0.0203 (reg already engaged), "
            "retained flagged.",
        ],
        e1a=e1a, e1b=e1b, e1c=e1c_json, e1d=e1d,
    )
    out = REPRO / args.out_json
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out}")

    # ---- console gate summary ------------------------------------------------
    print("\n=== E1 GATE SUMMARY ===")
    if "gate_pass" in e1a:
        print(f"E1a artifact: median|z(gamma)|={e1a['median_abs_z_gamma']:.2f}"
              f" cov68={e1a['coverage68_gamma']:.2f} -> "
              f"{'REPRODUCED' if e1a['gate_pass'] else 'NOT reproduced'}")
    for scale, sc in e1b.get("scales", {}).items():
        if "zbar_gamma" in sc:
            print(f"E1b {scale:6s}: zbar(gamma)={sc['zbar_gamma']:+.3f} "
                  f"(|.|<0.5 {'PASS' if sc['zbar_gate_pass'] else 'FAIL'}) "
                  f"cov68={sc['coverage68_gamma']:.2f}")
    if "gate_pass" in e1b.get("cross_scale_gate", {}):
        cg = e1b["cross_scale_gate"]
        wg = e1b["width_gate"]
        print(f"E1b cross-scale: {cg['n_agree']}/{cg['n_pairs']} within "
              f"1 sigma_comb ({'PASS' if cg['gate_pass'] else 'FAIL'}); "
              f"width ratio median={wg['median_ratio']:.2f} "
              f"({'PASS' if wg['gate_pass'] else 'FAIL'})")
    if "gates" in e1c:
        for m in MASS_LABELS:
            pp = e1c["per_param"][m]
            print(f"E1c {m:8s}: rank p={pp['p']:.4f} "
                  f"({'PASS' if pp['p_gate_pass'] else 'FAIL'}) "
                  f"cov68={pp['coverage68']:.2f}")
        print(f"E1c pooled cov68={e1c['pooled_coverage68']:.3f} "
              f"({'PASS' if e1c['gates']['coverage_pass'] else 'FAIL'}) | "
              f"all gates "
              f"{'PASS' if e1c['gates']['all_pass'] else 'FAIL'}")
    if "verdict" in e1d:
        for a, arm in e1d["arms"].items():
            if "zbar" in arm:
                print(f"E1d {a:8s}: max|zbar|={arm['max_abs_zbar']:.2f} "
                      f"cov68={arm['coverage68_pooled']:.2f} "
                      f"kept={arm['kept_pixels']}")
        print(f"E1d verdict: {e1d['verdict'].get('adopted')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
