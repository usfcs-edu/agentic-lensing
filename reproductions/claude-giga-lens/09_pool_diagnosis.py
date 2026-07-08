"""09_pool_diagnosis.py — P1b diagnosis pass pooling (D1/D2/D3).

CPU-only. Reads data/e1_fits/*.npz (originals + *_deep re-runs) and produces
the diagnosis readout requested after the first E1 readout was committed:

  D1  depth vs calibration: re-pool E1b/E1c with every fit for which a
      *_deep re-run exists (24 chains x 750 burn + 2250 keep, same fit-seed,
      same likelihood/kernel/whitener — only sampler depth changed)
      substituted in. Gates re-evaluated with 08's exact gate code (imported,
      not copied). Fits still unhealthy after deep are listed.
  D2  kernel-source separation: paired fitted-vs-analytic-kernel comparison
      on the 8 E1b mocks (deep budget both arms). Attribution verdict:
      analytic calibrates & fitted does not -> kernel fitting/regularization;
      both fail -> whitening approximation itself.
  D3  E1d verdict per the pre-registered criteria on the depth-controlled
      arm readouts (deep substituted where the original was unhealthy).

The committed first-readout sections of data/e1_report.json are preserved
byte-identical; this script only ADDS a "diagnosis" section.

Run:  /raid/benson/.venvs/cgl/bin/python 09_pool_diagnosis.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO))

from cgl.e1 import MASS_LABELS  # noqa: E402
from cgl.paths import DATA  # noqa: E402

FITS = DATA / "e1_fits"

# import 08_pool_e1 as a module (leading digit -> importlib)
_spec = importlib.util.spec_from_file_location("pool_e1", REPRO / "08_pool_e1.py")
p8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p8)

_orig_get = p8.get


def _get_prefer_deep(name):
    deep = FITS / f"{name}_deep.npz"
    if deep.exists():
        return p8.load_fit(deep)
    return _orig_get(name)


def _strip(d):
    """Drop bulky arrays from a pooled dict (mirrors 08's json slimming)."""
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k != "ranks"}
    return d


def _deep_names():
    return sorted(p.stem[:-5] for p in FITS.glob("*_deep.npz"))


def _health_of(name, prefer_deep):
    f = _get_prefer_deep(name) if prefer_deep else _orig_get(name)
    return None if f is None else f["health"]


def pool_depth_controlled():
    """Re-run 08's E1b/E1c/E1d pooling with *_deep fits substituted."""
    p8.get = _get_prefer_deep
    try:
        e1b, _ = p8.pool_e1b()
        e1c, _ = p8.pool_e1c()
        e1d, _ = p8.pool_e1d()
    finally:
        p8.get = _orig_get
    return e1b, e1c, e1d


def d2_attribution():
    rows = []
    for s in range(p8.N_E1B):
        fit = _get_prefer_deep(f"mock{s:03d}_fine_corr_fitted")
        ana = _get_prefer_deep(f"mock{s:03d}_fine_corr_analytic")
        if fit is None or ana is None:
            continue
        mf, sf, zf, cf = p8.gamma_stats(fit)
        ma, sa, za, ca = p8.gamma_stats(ana)
        rows.append(dict(
            seed=s, z_fitted=zf, z_analytic=za,
            sig_fitted=sf, sig_analytic=sa,
            cov68_fitted=cf, cov68_analytic=ca,
            dgamma_over_sigma=float((mf - ma) / sf),
            healthy_fitted=fit["health"]["healthy"],
            healthy_analytic=ana["health"]["healthy"],
            rhat_fitted=fit["health"]["max_rhat_mass"],
            rhat_analytic=ana["health"]["max_rhat_mass"],
            e_op_analytic=(ana["cfg"].get("whitener_meta") or {}).get("e_op"),
            e_op_fitted=(fit["cfg"].get("whitener_meta") or {}).get("e_op")))
    if not rows:
        return dict(status="NO FITS")

    def _arm(key_z, key_c, key_h):
        h = [r for r in rows if r[key_h]]
        z = np.array([r[key_z] for r in h])
        return dict(
            n_total=len(rows), n_healthy=len(h),
            zbar_gamma_healthy=float(z.mean()) if len(h) else None,
            z_gamma_healthy=z.tolist(),
            cov68_gamma_healthy=(float(np.mean([r[key_c] for r in h]))
                                 if h else None))

    fitted = _arm("z_fitted", "cov68_fitted", "healthy_fitted")
    analytic = _arm("z_analytic", "cov68_analytic", "healthy_analytic")

    def _ok(arm):
        return (arm["n_healthy"] >= 4
                and arm["zbar_gamma_healthy"] is not None
                and abs(arm["zbar_gamma_healthy"]) < 0.5)

    fok, aok = _ok(fitted), _ok(analytic)
    if aok and not fok:
        verdict = ("KERNEL FITTING/REGULARIZATION: analytic-kernel fits "
                   "calibrate, fitted-kernel fits do not")
    elif not aok and not fok:
        verdict = ("WHITENING APPROXIMATION: both kernel arms fail "
                   "calibration -> e_op sensitivity check required")
    elif aok and fok:
        verdict = ("NEITHER: both arms calibrate at depth -> original "
                   "failure attributed to sampler depth, not kernels")
    else:
        verdict = ("INVERTED (fitted ok, analytic not) — unexpected; "
                   "inspect analytic whitener regularization")
    return dict(per_seed=rows, fitted_arm=fitted, analytic_arm=analytic,
                criteria=("arm calibrates iff >=4 healthy fits and "
                          "|zbar(gamma)| < 0.5 over healthy fits"),
                verdict=verdict)


def residual_unhealthy():
    """Deep fits that are still unhealthy (depth did not fix them)."""
    out = []
    for name in _deep_names():
        f = p8.load_fit(FITS / f"{name}_deep.npz")
        if not f["health"]["healthy"]:
            out.append(dict(name=name, **f["health"]))
    return out


def e1c_healthy_only():
    """The D1 persistence criterion: does the E1c rank pathology persist in
    fits that actually reach the health bar (rhat<1.05, ess>100 on mass)?
    Pooled over healthy fits only (deep preferred)."""
    from cgl.e1 import rank_uniformity_chi2, sbc_rank
    ranks = {m: [] for m in MASS_LABELS}
    cov = {m: [] for m in MASS_LABELS}
    n_tot = n_deep = 0
    for s in range(64):
        f = _get_prefer_deep(f"mock{s:03d}_fine_corr_fitted")
        if f is None or not f["health"]["healthy"]:
            continue
        n_tot += 1
        n_deep += f["path"].endswith("_deep.npz")
        for m in MASS_LABELS:
            j = f["idx"][m]
            ranks[m].append(sbc_rank(f["draws"][:, j], f["truth"][j]))
            cov[m].append(bool(f["cov68"][j]))
    per_param = {}
    for m in MASS_LABELS:
        chi2, p, obs = rank_uniformity_chi2(np.array(ranks[m]))
        per_param[m] = dict(chi2=chi2, p=p, bins=obs,
                            p_gate_pass=bool(p > 0.01),
                            coverage68=float(np.mean(cov[m])))
    pooled = float(np.mean([np.mean(cov[m]) for m in MASS_LABELS]))
    return dict(
        note="pre-specified D1 persistence check: rank tests over HEALTHY "
             "fits only (rhat<1.05 & ess>100 on mass params, deep "
             "preferred). Low-n caveat applies (8-bin chi2 at ~n/8 expected "
             "per bin).",
        n_healthy=n_tot, n_deep_used=n_deep, per_param=per_param,
        pooled_coverage68=pooled,
        pooled_coverage_pass=bool(0.55 <= pooled <= 0.80),
        all_rank_pass=bool(all(v["p_gate_pass"]
                               for v in per_param.values())))


def main():
    deep = _deep_names()
    print(f"deep re-runs available: {len(deep)}")

    # complete-batch baseline at standard depth (post-resume; the committed
    # report was pooled before the 15-fit resume pass finished)
    b_e1b, _ = p8.pool_e1b()
    b_e1c, _ = p8.pool_e1c()
    b_e1d, _ = p8.pool_e1d()

    dc_e1b, dc_e1c, dc_e1d = pool_depth_controlled()
    d2 = d2_attribution()
    still_bad = residual_unhealthy()
    e1c_healthy = e1c_healthy_only()

    diagnosis = dict(
        generated_by="09_pool_diagnosis.py",
        deep_budget="two-stage re-preconditioned PHMC (--sampler-stages 2): "
                    "stage 1 = 24 chains x 500 burn + 500 keep with the "
                    "floored-SVI metric, then the momentum metric is "
                    "re-estimated from the pooled cross-chain stage-1 draws "
                    "and stage 2 = 24 chains x 500 burn + 2250 keep "
                    "warm-started from the stage-1 chain ends. Same "
                    "fit-seed, MAP/SVI identical to the originals. "
                    "Validated on the mock002 canary: depth-only 3x gave "
                    "rhat 3.11; staged gives rhat 1.003, ess 13k-22k. "
                    "(An aborted depth-only 3x pass is archived at "
                    "data/e1_quarantine/mock002_fine_corr_fitted_deep3x_"
                    "depthonly.npz and data/logs/e1/batch_diag_master_"
                    "depthonly_aborted.log.)",
        deep_fits=deep,
        baseline_complete=dict(
            note="standard-depth pooling over the COMPLETE batch (incl. "
                 "the 15-fit resume pass that landed after the committed "
                 "first readout)",
            e1b=_strip(b_e1b), e1c=_strip(b_e1c), e1d=_strip(b_e1d)),
        depth_controlled=dict(
            note="*_deep substituted wherever present (targets: unhealthy "
                 "E1b fits, 16 worst-rank E1c fits, unhealthy E1d fits, "
                 "analytic arm seeds 0-7)",
            e1b=_strip(dc_e1b), e1c=_strip(dc_e1c), e1d=_strip(dc_e1d)),
        d2_kernel_attribution=d2,
        e1c_healthy_only=e1c_healthy,
        still_unhealthy_after_deep=still_bad,
    )

    # preserve committed sections byte-identical; add diagnosis
    report_path = DATA / "e1_report.json"
    report = json.loads(report_path.read_text())
    report["diagnosis"] = diagnosis
    report_path.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote diagnosis section -> {report_path}")

    # console summary
    print("\n=== DIAGNOSIS SUMMARY ===")
    for tag, e1b, e1c, e1d in (("baseline", b_e1b, b_e1c, b_e1d),
                               ("depth-ctl", dc_e1b, dc_e1c, dc_e1d)):
        for sc in ("fine", "binned", "native"):
            d = e1b["scales"].get(sc, {})
            if "zbar_gamma" in d:
                print(f"[{tag}] E1b {sc:6s}: zbar={d['zbar_gamma']:+.3f} "
                      f"({'PASS' if d['zbar_gate_pass'] else 'FAIL'}) "
                      f"cov68={d['coverage68_gamma']:.2f}")
        cs, wg = e1b["cross_scale_gate"], e1b["width_gate"]
        print(f"[{tag}] E1b cross-scale {cs['n_agree']}/{cs['n_pairs']} "
              f"({'PASS' if cs['gate_pass'] else 'FAIL'}); width median="
              f"{wg['median_ratio']:.2f} "
              f"({'PASS' if wg['gate_pass'] else 'FAIL'})")
        for m in MASS_LABELS:
            pp = e1c["per_param"][m]
            print(f"[{tag}] E1c {m:8s}: p={pp['p']:.4f} "
                  f"({'PASS' if pp['p_gate_pass'] else 'FAIL'}) "
                  f"cov68={pp['coverage68']:.2f}")
        print(f"[{tag}] E1c pooled cov68={e1c['pooled_coverage68']:.3f} "
              f"({'PASS' if e1c['gates']['coverage_pass'] else 'FAIL'})")
        print(f"[{tag}] E1d verdict: {e1d['verdict'].get('adopted')}")
    print(f"\nD2: {d2.get('verdict', d2)}")
    print(f"\nE1c HEALTHY-ONLY (n={e1c_healthy['n_healthy']}, "
          f"{e1c_healthy['n_deep_used']} deep):")
    for m in MASS_LABELS:
        pp = e1c_healthy["per_param"][m]
        print(f"  {m:8s}: p={pp['p']:.4f} "
              f"({'PASS' if pp['p_gate_pass'] else 'FAIL'}) "
              f"cov68={pp['coverage68']:.2f}")
    print(f"  pooled cov68={e1c_healthy['pooled_coverage68']:.3f} "
          f"({'PASS' if e1c_healthy['pooled_coverage_pass'] else 'FAIL'})")
    print(f"still unhealthy after deep: {len(still_bad)}")


if __name__ == "__main__":
    main()
