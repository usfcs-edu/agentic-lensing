"""11_pool_e2.py — pool the E2 production per-product results into e2_report.json.

Reads data/results/e2_{v3,v3b,v2d}.json (+ e2_v2d_strict_low.json) written by
10_run_e2.py --mode prod and emits data/e2_report.json (gates/numbers/provenance)
with the H1/H2/H3 verdicts.

HEADLINE REFRAME (pre-registered 2026-07-08, before any production result):
  The v3 FINE product is 3.2x-upsampled; its correlated whitener is near-singular
  and the fine-LOW MAP is a demonstrated spectral-zero pathology (rails to
  gamma~1.0 with real-space chi2_pp~72 while the whitened logL improves — see the
  fine-low pre-flight/discriminator diagnostics). Therefore the MONEY comparison
  is gamma_binned(corr, LOW) vs the diagonal-native anchor 1.433 (binned = 2x-
  upsampled, less near-singular). Fine is reported as a SECONDARY characterization
  panel (fine-STEEP corr vs the diagonal-fine 2.585 artifact); fine-LOW is EXCLUDED
  as a characterized whitener limitation, not a failure. Consistent with the
  pre-registered H1 fork ("pathology survives -> it's a finding") and foundry-i's
  independent native/binned-is-the-defensible-headline conclusion.

Pre-registered gate specs (README §P1 E2) + P1c amendments:
  H1  per BOTH-BASIN product (v3b, v2d): steep posterior-mass < 10% OR corrected
        dlogL(steep-low) <= 0 at basin MAPs -> steep is a likelihood artifact.
        ALTERNATIVE: bimodality survives (both basins comparably supported +
        converged) -> genuine/PSF-systematic multimodality. EITHER IS A RESULT.
        Single-basin products (v3 steep-only): H1 skipped, note recorded.
  H2  PRIMARY |gamma_binned(corr,low) - 1.433| < 2 sigma_comb, sigma_comb =
        sqrt(sigma_corr^2 + sigma_native_diag^2). ANCHOR = gamma_native(DIAGONAL)
        = 1.433 [1.400,1.469] (foundry-i headline). Secondary: v2d(relaxed,low) vs
        1.433; fine-STEEP(corr) vs the diagonal-fine 2.585 artifact.
  H3  sigma_gamma(binned,corr,low) and sigma_gamma(v2d relaxed low) each >=
        sigma_gamma(native, DIAGONAL)/1.5 (~0.023). Reference is the DATA-DRIVEN
        diagonal-native width (~0.0345), NOT the prior-pulled correlated-native
        sigma. v2d STRICT-low sigma reported alongside relaxed-low as the
        strict-vs-relaxed pixel-loss information-cost datum.

Usage:  python 11_pool_e2.py [--results-dir data/results] [--out data/e2_report.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cgl.paths import DATA, FOUNDRY_I_DATA, REPRO

# Diagonal-likelihood references (foundry-i).
GAMMA_NATIVE_DIAG = 1.433
GAMMA_NATIVE_DIAG_CI = (1.400, 1.469)
SIGMA_NATIVE_DIAG = 0.5 * (GAMMA_NATIVE_DIAG_CI[1] - GAMMA_NATIVE_DIAG_CI[0])  # ~0.0345
GAMMA_FINE_DIAG_MAP = 2.585    # map_v11_v3cold artifact (diagonal fine MAP; no full HMC)
HMC_V13_V3B = FOUNDRY_I_DATA / "hmc_v13_v3b.npz"   # diagonal binned bimodal chains

# product tags -> (json basename, expected kept-px, human label)
PRODUCTS = {
    "v3":         ("e2_v3",              37519, "fine 260^2 (3.2x, steep-only)"),
    "v3b":        ("e2_v3b",              9273, "binned (2x) — PRIMARY"),
    "v2d":        ("e2_v2d",              1466, "native relaxed whitener"),
    "v2d_strict": ("e2_v2d_strict_low",   487, "native STRICT whitener (low only)"),
}


def _load(results_dir, base):
    p = Path(results_dir) / f"{base}.json"
    return json.load(open(p)) if p.exists() else None


def _basin_summ(b):
    return dict(
        gamma_median=b["gamma_median"], gamma_ci=[b["gamma_q16"], b["gamma_q84"]],
        gamma_std=b["gamma_std"], rhat_max=b["rhat_max"], ess_min=b["ess_min"],
        ess_gamma=b["ess_gamma"], rhat_gamma=b.get("rhat_gamma"),
        logp_best=b["logp_best"], gamma_best=b["gamma_best"],
        thetaE_median=b.get("thetaE_median"), thetaE_std=b.get("thetaE_std"),
        n_draws=b.get("n_draws"))


def diag_binned_gamma():
    """Diagonal binned gamma posterior (bimodal) from hmc_v13_v3b: per-basin
    median/sigma + occupancy, for the correlated-vs-diagonal money figure."""
    if not HMC_V13_V3B.exists():
        return None
    z = np.load(HMC_V13_V3B, allow_pickle=True)
    g = np.asarray(z["mass_gamma"]).reshape(-1)
    split = 1.9
    steep = g[g > split]; low = g[g <= split]
    return dict(n=int(g.size), median_all=float(np.median(g)),
                low_median=float(np.median(low)) if low.size else None,
                low_sigma=float(np.std(low)) if low.size else None,
                steep_median=float(np.median(steep)) if steep.size else None,
                steep_sigma=float(np.std(steep)) if steep.size else None,
                steep_mass=float(steep.size / g.size))


def _h2_row(basin_summ, anchor=GAMMA_NATIVE_DIAG, note=""):
    gm, gs = basin_summ["gamma_median"], basin_summ["gamma_std"]
    sigma_comb = float(np.hypot(gs, SIGMA_NATIVE_DIAG))
    dgamma = abs(gm - anchor)
    return dict(gamma_corr=gm, gamma_corr_std=gs, anchor=anchor,
                dgamma=dgamma, sigma_comb=sigma_comb, z=dgamma / sigma_comb,
                gate_pass=bool(dgamma < 2 * sigma_comb), note=note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DATA / "results"))
    ap.add_argument("--out", default=str(DATA / "e2_report.json"))
    args = ap.parse_args()

    res = {tag: _load(args.results_dir, base)
           for tag, (base, _, _) in PRODUCTS.items()}

    report = dict(
        generated_by="11_pool_e2.py",
        headline_reframe=(
            "PRIMARY money number = gamma_binned(corr, LOW) vs diagonal-native "
            "anchor 1.433. The fine (v3, 3.2x) LOW basin is EXCLUDED as a "
            "characterized near-singular-whitener pathology (fine-low MAP rails to "
            "gamma~1.0 with real-space chi2_pp~72 while whitened logL improves; see "
            "the fine-low pre-flight + discriminator diagnostics). Fine reported as "
            "a secondary panel: fine-STEEP(corr) vs the diagonal-fine 2.585 "
            "artifact. Binned (2x) is less near-singular and is the defensible "
            "cross-scale headline."),
        pre_registered_amendments=[
            "H2/H3 anchor = gamma_native(DIAGONAL) = 1.433 [1.400,1.469], NOT the "
            "correlated-native relaxed fit (native is where the diagonal likelihood "
            "is least wrong; the correlated value-add is largest on the fine/binned).",
            "H2 PRIMARY re-pointed from fine-low to BINNED-low (fine-low excluded, "
            "whitener pathology). Fine-steep vs 2.585 is a secondary characterization.",
            "H3 RE-SPEC: reference sigma = sigma_gamma(native, DIAGONAL) ~ %.4f, NOT "
            "the prior-pulled correlated-native sigma. Correlated-native (relaxed & "
            "strict) sigma reported as the pixel-loss information-cost diagnostic."
            % SIGMA_NATIVE_DIAG],
        anchors=dict(gamma_native_diag=GAMMA_NATIVE_DIAG,
                     gamma_native_diag_ci=list(GAMMA_NATIVE_DIAG_CI),
                     sigma_native_diag=SIGMA_NATIVE_DIAG,
                     gamma_fine_diag_map=GAMMA_FINE_DIAG_MAP),
        diagonal_refs=dict(binned=diag_binned_gamma()),
        products={}, H1={}, H2={}, H3={}, pixel_information={})

    # ---- per-product summaries (basin-subset-robust) --------------------------
    for tag, (base, exp_px, label) in PRODUCTS.items():
        r = res[tag]
        if r is None:
            report["products"][tag] = dict(status="MISSING", label=label)
            continue
        basins = {b: _basin_summ(r["basins"][b]) for b in r["basins"]}
        conv = all(bs["rhat_max"] < 1.01 and bs["ess_min"] >= 1e3
                   for bs in basins.values()) if basins else False
        report["products"][tag] = dict(
            label=r.get("label", label), whiten=r.get("whiten"),
            basins_present=list(basins.keys()), basins=basins,
            converged=bool(conv),
            note=("single-basin (fine-low blocked by near-singular-whitener "
                  "pathology; see fine-low discriminator diagnostic)"
                  if set(basins) != {"low", "steep"} else ""))

    # ---- H1: only for BOTH-BASIN products (v3b, v2d) --------------------------
    for tag in ("v3", "v3b", "v2d", "v2d_strict"):
        r = res[tag]
        if r is None:
            report["H1"][tag] = dict(status="MISSING")
            continue
        both = ("low" in r["basins"]) and ("steep" in r["basins"])
        if not both:
            report["H1"][tag] = dict(
                status="single-basin",
                note="H1 not computed (needs both basins). "
                     + (report["products"][tag].get("note") or ""))
            continue
        h1 = r.get("H1", {})
        dlogL = h1.get("dlogpost_best_steep_minus_low")
        mass_steep = h1.get("laplace_mass_steep", float("nan"))
        artifact = (dlogL is not None and dlogL <= 0.0) or (mass_steep < 0.10)
        report["H1"][tag] = dict(
            dlogpost_best_steep_minus_low=dlogL,
            dlogpost_map_steep_minus_low=h1.get("dlogpost_map_steep_minus_low"),
            dlogZ_laplace=h1.get("dlogZ_laplace_steep_minus_low"),
            laplace_mass_steep=mass_steep,
            steep_map_crossed=h1.get("steep_map_crossed"),
            low_map_crossed=h1.get("low_map_crossed"),
            steep_disfavored=bool(artifact),
            interpretation=("steep basin is a likelihood artifact (disfavored)"
                            if artifact else
                            "bimodality survives the corrected likelihood -> "
                            "genuine/PSF-systematic multimodality (pre-registered "
                            "ALTERNATIVE; candidate flagship P2c target)"))

    # ---- H2: cross-scale unification vs the diagonal-native anchor ------------
    def low(tag):
        r = res[tag]
        return r["basins"]["low"] if (r and "low" in r["basins"]) else None
    def steep(tag):
        r = res[tag]
        return r["basins"]["steep"] if (r and "steep" in r["basins"]) else None

    b = low("v3b")
    report["H2"]["primary_binned_low"] = (
        _h2_row(b, note="THE money number: gamma_binned(corr,low) vs 1.433")
        if b else dict(status="MISSING"))
    bs = steep("v3b")
    report["H2"]["binned_steep"] = (_h2_row(bs, note="binned steep basin vs 1.433")
                                    if bs else dict(status="MISSING"))
    b2 = low("v2d")
    report["H2"]["v2d_relaxed_low"] = (
        _h2_row(b2, note="native relaxed low vs 1.433")
        if b2 else dict(status="MISSING"))
    # optional basin-mass-weighted binned gamma (uses H1 Laplace mass)
    if b and bs and isinstance(report["H1"].get("v3b"), dict) \
            and report["H1"]["v3b"].get("laplace_mass_steep") is not None:
        ms = float(report["H1"]["v3b"]["laplace_mass_steep"])
        if np.isfinite(ms):
            gw = ms * bs["gamma_median"] + (1 - ms) * b["gamma_median"]
            report["H2"]["binned_massweighted"] = dict(
                gamma_massweighted=float(gw), mass_steep=ms,
                note="Laplace-mass-weighted across binned basins (context only)")
    # SECONDARY characterization: fine STEEP vs the diagonal-fine 2.585 artifact
    fs = steep("v3")
    if fs:
        report["H2"]["secondary_fine_steep"] = dict(
            gamma_corr=fs["gamma_median"], gamma_corr_std=fs["gamma_std"],
            gamma_ci=[fs["gamma_q16"], fs["gamma_q84"]],
            gamma_diag_fine_artifact=GAMMA_FINE_DIAG_MAP,
            dgamma_vs_artifact=abs(fs["gamma_median"] - GAMMA_FINE_DIAG_MAP),
            dgamma_vs_anchor=abs(fs["gamma_median"] - GAMMA_NATIVE_DIAG),
            note="does the correlated likelihood move the fine STEEP basin off the "
                 "diagonal-fine 2.585 artifact? (fine-LOW excluded — whitener pathology)")
    else:
        report["H2"]["secondary_fine_steep"] = dict(status="MISSING")

    # ---- H3: honesty (corr widths vs data-driven diagonal-native width) -------
    thr = SIGMA_NATIVE_DIAG / 1.5
    h3 = dict(ref="sigma_gamma(native, DIAGONAL) [re-spec]",
              sigma_native_diag=SIGMA_NATIVE_DIAG, threshold=thr)
    if b:
        h3["binned_low"] = dict(sigma=b["gamma_std"],
                                gate_pass=bool(b["gamma_std"] >= thr))
    if b2:
        h3["v2d_relaxed_low"] = dict(sigma=b2["gamma_std"],
                                     gate_pass=bool(b2["gamma_std"] >= thr))
    bstr = low("v2d_strict")
    if bstr:
        h3["v2d_strict_low_diagnostic"] = dict(
            sigma=bstr["gamma_std"],
            note="strict whitener (487 px) — fewer px -> wider; strict-vs-relaxed "
                 "pixel-loss information-cost (supports info-limited native finding)")
    report["H3"] = h3

    # ---- kept-pixel -> posterior-width table ---------------------------------
    def px_row(tag, basin):
        r = res[tag]
        if not r or basin not in r["basins"]:
            return dict(n_keep_w=PRODUCTS[tag][1], gamma_std=None, status="MISSING")
        return dict(n_keep_w=(r.get("whiten") or {}).get("n_keep_w", PRODUCTS[tag][1]),
                    gamma_std=r["basins"][basin]["gamma_std"], basin=basin)
    report["pixel_information"] = dict(
        fine_steep=px_row("v3", "steep"),
        binned_low=px_row("v3b", "low"),
        v2d_relaxed_low=px_row("v2d", "low"),
        v2d_strict_low=px_row("v2d_strict", "low"),
        note="real-data sigma-inflation story: fewer kept px -> wider gamma posterior")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1, default=float)
    print(f"[11] wrote {args.out}")
    # console summary
    for tag in ("v3b", "v2d"):
        h = report["H1"].get(tag, {})
        if "interpretation" in h:
            print(f"  H1[{tag}]: dlogL(steep-low)={h['dlogpost_best_steep_minus_low']}, "
                  f"mass_steep={h['laplace_mass_steep']:.3f} -> {h['interpretation']}")
        else:
            print(f"  H1[{tag}]: {h.get('status','?')}")
    pm = report["H2"].get("primary_binned_low", {})
    if "z" in pm:
        print(f"  H2 PRIMARY gamma_binned(corr,low)={pm['gamma_corr']:.4f}"
              f"±{pm['gamma_corr_std']:.4f} vs 1.433: z={pm['z']:.2f} "
              f"pass={pm['gate_pass']}")
    for k in ("v2d_relaxed_low",):
        h = report["H2"].get(k, {})
        if "z" in h:
            print(f"  H2 {k} gamma={h['gamma_corr']:.4f} vs 1.433: z={h['z']:.2f} "
                  f"pass={h['gate_pass']}")
    sfs = report["H2"].get("secondary_fine_steep", {})
    if "gamma_corr" in sfs:
        print(f"  H2 secondary fine-steep(corr)={sfs['gamma_corr']:.4f} vs "
              f"diagonal-fine 2.585 (Δ={sfs['dgamma_vs_artifact']:.3f})")
    if "binned_low" in report["H3"]:
        hb = report["H3"]["binned_low"]
        print(f"  H3 sigma_binned_low={hb['sigma']:.4f} >= {thr:.4f}? {hb['gate_pass']}")


if __name__ == "__main__":
    main()
