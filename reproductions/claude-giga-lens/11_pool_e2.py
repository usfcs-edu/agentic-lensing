"""11_pool_e2.py — pool the E2 production per-product results into e2_report.json.

Reads data/results/e2_{v3,v3b,v2d}.json (+ .npz) written by 10_run_e2.py --mode
prod and emits data/e2_report.json (schema mirrors e1_report: gates/numbers/
provenance) with the H1/H2/H3 verdicts.

Pre-registered gate specs (README §P1 E2), with the P1c amendments recorded here:
  H1  per product (steep vs low basin under the correlated likelihood):
        steep posterior-mass < 10% OR corrected dlogL(steep-low) <= 0 at basin
        MAPs -> steep basin is a likelihood artifact. Pre-registered ALTERNATIVE:
        bimodality survives (both basins comparably supported + converged) ->
        genuine/PSF-systematic multimodality, v3b -> flagship P2c target. EITHER
        IS A RESULT.
  H2  |gamma_fine(corr) - anchor| and |gamma_binned(corr) - anchor| < 2 sigma_comb,
        ANCHOR = gamma_native(DIAGONAL) = 1.433 [1.400,1.469] (foundry-i headline;
        native is where the diagonal likelihood is least wrong, rho(1)~0.5 vs 0.8
        fine). The money comparison: does the correlated likelihood pull the fine
        diagonal artifact gamma_fine(diag)=2.585 back into agreement with 1.433?
  H3  (RE-SPEC, pre-registered amendment — rationale below): sigma_gamma(fine,corr)
        >= sigma_gamma(native, DIAGONAL)/1.5. The reference is the DATA-DRIVEN
        diagonal-native width, NOT the correlated-native relaxed sigma (which is
        prior-pulled: the 1466-kept-px relaxed likelihood is information-weak, so
        its width ~ the gamma prior width and would make H3 vacuous). The
        correlated-native sigma is reported separately as the pixel-loss
        information-cost diagnostic.

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


def _load(results_dir, tag):
    p = Path(results_dir) / f"e2_{tag}.json"
    return json.load(open(p)) if p.exists() else None


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DATA / "results"))
    ap.add_argument("--out", default=str(DATA / "e2_report.json"))
    args = ap.parse_args()

    res = {t: _load(args.results_dir, t) for t in ("v3", "v3b", "v2d")}
    report = dict(
        generated_by="11_pool_e2.py",
        pre_registered_amendments=[
            "H2/H3 anchor = gamma_native(DIAGONAL) = 1.433 [1.400,1.469], NOT the "
            "correlated-native relaxed fit (native is where the diagonal likelihood "
            "is least wrong; the correlated value-add is largest on the fine product).",
            "H3 RE-SPEC: reference sigma = sigma_gamma(native, DIAGONAL) ~ %.4f, NOT "
            "the prior-pulled correlated-native sigma (1466-kept-px relaxed likelihood "
            "is information-weak -> width ~ gamma prior width -> H3 vacuous otherwise). "
            "Correlated-native sigma reported separately as the pixel-loss diagnostic."
            % SIGMA_NATIVE_DIAG],
        anchors=dict(gamma_native_diag=GAMMA_NATIVE_DIAG,
                     gamma_native_diag_ci=list(GAMMA_NATIVE_DIAG_CI),
                     sigma_native_diag=SIGMA_NATIVE_DIAG,
                     gamma_fine_diag_map=GAMMA_FINE_DIAG_MAP),
        diagonal_refs=dict(binned=diag_binned_gamma()),
        products={}, H1={}, H2={}, H3={})

    # ---- per-product corr summaries + H1 --------------------------------------
    for tag in ("v3", "v3b", "v2d"):
        r = res[tag]
        if r is None:
            report["products"][tag] = dict(status="MISSING")
            continue
        lo, hi = r["basins"]["low"], r["basins"]["steep"]
        report["products"][tag] = dict(
            label=r["label"], whiten=r["whiten"],
            low=dict(gamma_median=lo["gamma_median"], gamma_ci=[lo["gamma_q16"], lo["gamma_q84"]],
                     gamma_std=lo["gamma_std"], rhat_max=lo["rhat_max"],
                     ess_min=lo["ess_min"], ess_gamma=lo["ess_gamma"],
                     logp_best=lo["logp_best"], gamma_best=lo["gamma_best"]),
            steep=dict(gamma_median=hi["gamma_median"], gamma_ci=[hi["gamma_q16"], hi["gamma_q84"]],
                       gamma_std=hi["gamma_std"], rhat_max=hi["rhat_max"],
                       ess_min=hi["ess_min"], ess_gamma=hi["ess_gamma"],
                       logp_best=hi["logp_best"], gamma_best=hi["gamma_best"]),
            converged=bool(max(lo["rhat_max"], hi["rhat_max"]) < 1.01
                           and min(lo["ess_min"], hi["ess_min"]) >= 1e3))
        h1 = r["H1"]
        dlogL = h1["dlogpost_best_steep_minus_low"]
        mass_steep = h1.get("laplace_mass_steep", float("nan"))
        artifact = (dlogL <= 0.0) or (mass_steep < 0.10)
        report["H1"][tag] = dict(
            dlogpost_best_steep_minus_low=dlogL,
            dlogZ_laplace=h1.get("dlogZ_laplace_steep_minus_low"),
            laplace_mass_steep=mass_steep,
            steep_disfavored=bool(artifact),
            interpretation=("steep basin is a likelihood artifact (disfavored)"
                            if artifact else
                            "bimodality survives the corrected likelihood -> "
                            "genuine/PSF-systematic multimodality; promote v3b to "
                            "flagship P2c target (pre-registered ALTERNATIVE)"))

    # ---- H2: cross-scale unification vs the diagonal-native anchor ------------
    def gpost(tag, basin="low"):
        r = res[tag]
        if r is None:
            return None
        b = r["basins"][basin]
        return b["gamma_median"], b["gamma_std"]

    for tag, name in (("v3", "fine"), ("v3b", "binned")):
        gp = gpost(tag, "low")
        if gp is None:
            report["H2"][name] = dict(status="MISSING")
            continue
        gm, gs = gp
        sigma_comb = float(np.hypot(gs, SIGMA_NATIVE_DIAG))
        dgamma = abs(gm - GAMMA_NATIVE_DIAG)
        report["H2"][name] = dict(
            gamma_corr=gm, gamma_corr_std=gs, anchor=GAMMA_NATIVE_DIAG,
            dgamma=dgamma, sigma_comb=sigma_comb, z=dgamma / sigma_comb,
            gate_pass=bool(dgamma < 2 * sigma_comb),
            gamma_diag_same_product=(GAMMA_FINE_DIAG_MAP if tag == "v3" else None),
            note=("does the correlated likelihood pull the fine diagonal artifact "
                  "2.585 back to the native anchor 1.433?" if tag == "v3" else ""))

    # ---- H3: honesty (fine correlated width vs data-driven native width) ------
    gpf = gpost("v3", "low")
    if gpf is not None:
        sig_fine_corr = gpf[1]
        report["H3"] = dict(
            sigma_fine_corr=sig_fine_corr,
            sigma_native_diag=SIGMA_NATIVE_DIAG,
            ref="sigma_gamma(native, DIAGONAL) [re-spec]",
            threshold=SIGMA_NATIVE_DIAG / 1.5,
            gate_pass=bool(sig_fine_corr >= SIGMA_NATIVE_DIAG / 1.5),
            sigma_native_corr_diagnostic=(res["v2d"]["basins"]["low"]["gamma_std"]
                                          if res["v2d"] else None))

    # ---- pixel-loss information table (the native prior-pull finding) ---------
    report["pixel_information"] = {
        tag: dict(n_keep_w=res[tag]["whiten"]["n_keep_w"] if res[tag] else None,
                  gamma_std_low=res[tag]["basins"]["low"]["gamma_std"] if res[tag] else None)
        for tag in ("v3", "v3b", "v2d")}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1, default=float)
    print(f"[11] wrote {args.out}")
    for tag in ("v3", "v3b", "v2d"):
        if report["H1"].get(tag):
            print(f"  H1[{tag}]: {report['H1'][tag]['interpretation']}")
    for name in ("fine", "binned"):
        h = report["H2"].get(name)
        if h and "z" in h:
            print(f"  H2[{name}]: gamma_corr={h['gamma_corr']:.3f} vs 1.433, "
                  f"z={h['z']:.2f}, pass={h['gate_pass']}")
    if report["H3"]:
        print(f"  H3: sigma_fine_corr={report['H3']['sigma_fine_corr']:.4f} "
              f">= {report['H3']['threshold']:.4f}? {report['H3']['gate_pass']}")


if __name__ == "__main__":
    main()
