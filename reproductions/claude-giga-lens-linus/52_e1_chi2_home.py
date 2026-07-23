#!/usr/bin/env python
"""52_e1_chi2_home.py -- E1 epilogue: real-space chi2/px of the v2d MCLMC
posterior-median model on its HOME product (v2d native, 80^2 @ 0.13"), for
the modeler-summary goodness-of-fit table (papers/sec_summary_modelers.tex).

Convention = the t04 head-to-head machinery pattern (scratchpad
t04_headtohead.py -> data/t04_realspace_headtohead.json): per-dim-median
parameter point, ridge-solved shapelet amplitudes under the DIAGONAL metric,
chi2/px = mean(((Y - model)/masked_err)^2) over the keep mask, arc band
r in [1.2, 4.2] arcsec -- here evaluated on the parity-certified SCENE stack
(the exact `10_anchor_arbitration.build_pm('v2d', diagonal=True)` likelihood
object the E1 fit sampled; F1-F8 machine-precision parity with the old
validated stack is the certified bridge).

Validation ladder (all asserted before the quotable number is written):
  V1  stored chain z_names == mv.model.z_param_names, exact order (KEYED).
  V2  gamma at the pooled per-dim scene-z median == run-json pooled q50
      (<1e-3; the 40_modeler_summary_figs.mclmc_median_model tolerance, and
      the exact same forward path).
  V3  machinery cross-check: the same evaluator run at the WARM-CLOUD per-dim
      scene-z median (the mapped foundry-i anchor cloud, gamma ~1.4334) must
      land within 10% of the OLD-stack t04 anchor home value 1.2510
      (data/t04_realspace_headtohead.json:validation.anchor_home_v2d_chi2_pp).
      Loose on purpose: the two points are medians of different clouds
      (mapped-draw median vs mapped z74-median); the measured value is
      recorded either way.

Writes data/e1_mclmc_home_chi2.json.  GPU required (house rule: the lensing
likelihood never runs on CPU).  Run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=9 GIGALENS_X64=1 \
  /raid/benson/.venvs/cgl2/bin/python 52_e1_chi2_home.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus")
DATA = ROOT / "data"
OUT = DATA / "e1_mclmc_home_chi2.json"
ARC_BAND = (1.2, 4.2)  # arcsec (x1_g0 / t04 convention)
T04 = DATA / "t04_realspace_headtohead.json"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    assert os.environ.get("GIGALENS_X64") == "1", "GIGALENS_X64=1 required"
    t0 = time.time()
    sys.path.insert(0, str(ROOT))
    from cgl2 import guards, paths

    paths.bootstrap_vendor()
    guards.require_vendor_ref()
    guards.require_jax_pin()
    import importlib.util

    import jax.numpy as jnp

    spec = importlib.util.spec_from_file_location(
        "anchor_arb", ROOT / "10_anchor_arbitration.py")
    arb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arb)

    refs = arb.load_refs()
    pm, mv, _ = arb.build_pm("v2d", refs, diagonal=True)
    term = pm.terms[0]

    Y = np.asarray(term.dataset.image, dtype=np.float64)
    masked_err = np.asarray(refs["v2d:masked_err"], dtype=np.float64)
    keep = np.asarray(refs["v2d:keep_mask"]).astype(bool)
    n = Y.shape[0]
    yy, xx = np.indices(Y.shape)
    cen = (n - 1) / 2.0
    r_arc = np.hypot(xx - cen, yy - cen) * 0.13
    band = keep & (r_arc >= ARC_BAND[0]) & (r_arc <= ARC_BAND[1])

    def eval_zmed(zmed: np.ndarray) -> dict:
        u = mv.model.bijector.forward(jnp.asarray(zmed[None, :]))
        gamma = float(np.asarray(u["planes/0/mass/0/gamma"])[0])
        it = term.internals(mv.model.to_params(u))
        resid = (Y - np.asarray(it["model_image"], dtype=np.float64)) / masked_err
        return dict(
            gamma=gamma,
            chi2_pp=float(np.mean(resid[keep] ** 2)),
            chi2_pp_arcband=float(np.mean(resid[band] ** 2)),
            n_keep=int(keep.sum()), n_band=int(band.sum()),
            logL_data_whitened=float(it["logL_data"]))

    # ---- V1: KEYED discipline against the stored chains -------------------- #
    npz = DATA / "mclmc_diag_v2d.npz"
    z = np.load(npz, allow_pickle=True)
    assert [str(s) for s in z["z_names"]] == list(mv.model.z_param_names), \
        "scene-z name/order mismatch vs stored chains"
    pooled = np.concatenate(
        [np.asarray(z["pos_g0"], dtype=np.float64),
         np.asarray(z["pos_g1"], dtype=np.float64)], axis=0).reshape(-1, 46)
    zmed = np.median(pooled, axis=0)

    # ---- the quotable point ------------------------------------------------ #
    res = eval_zmed(zmed)
    jq = json.load(open(DATA / "mclmc_diag_v2d.json"))["gamma"]["pooled_quantiles"]
    dgamma = abs(res["gamma"] - jq["0.5"])
    assert dgamma < 1e-3, (res["gamma"], jq["0.5"])          # V2
    print(f"[e1chi2] MCLMC zmed: gamma {res['gamma']:.6f} (json q50 "
          f"{jq['0.5']:.6f}, d {dgamma:.2e}); chi2/px full "
          f"{res['chi2_pp']:.4f}, arc band {res['chi2_pp_arcband']:.4f} "
          f"({res['n_keep']}/{res['n_band']} px)", flush=True)

    # ---- V3: machinery cross-check at the warm-cloud (anchor) median ------- #
    wz = np.load(DATA / "mclmc_warm_v2d_scenez.npz", allow_pickle=True)
    zmed_w = np.median(np.asarray(wz["Z_scene"], dtype=np.float64), axis=0)
    chk = eval_zmed(zmed_w)
    t04_ref = json.load(open(T04))["validation"]["anchor_home_v2d_chi2_pp"]
    rel = abs(chk["chi2_pp"] - t04_ref) / t04_ref
    print(f"[e1chi2] V3 anchor-cloud zmed: gamma {chk['gamma']:.6f}, chi2/px "
          f"{chk['chi2_pp']:.4f} vs t04 old-stack {t04_ref:.4f} "
          f"(rel {rel:.3%})", flush=True)
    assert rel < 0.10, (chk["chi2_pp"], t04_ref)

    out = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="52_e1_chi2_home.py",
        convention=("t04 head-to-head pattern on the scene stack: per-dim "
                    "scene-z median point; ridge-solved shapelet amps under "
                    "the DIAGONAL (masked-err) metric; chi2/px = "
                    "mean(((Y-model)/masked_err)^2) over keep px; arc band "
                    "r in [1.2,4.2] arcsec; v2d native 80^2 @ 0.13\" (HOME "
                    "product of the E1 fit)"),
        mclmc_v2d_home=res,
        run_json_gamma_q50=jq["0.5"],
        validation=dict(
            V1_keyed="z_names == z_param_names exact",
            V2_gamma_vs_json_q50=dict(d=dgamma, tol=1e-3, ok=True),
            V3_anchor_cloud_check=dict(
                gamma=chk["gamma"], chi2_pp=chk["chi2_pp"],
                chi2_pp_arcband=chk["chi2_pp_arcband"],
                t04_oldstack_anchor_home=t04_ref, rel_diff=rel, tol=0.10,
                note=("different median constructions (mapped-draw median vs "
                      "mapped z74-median) -- consistency check, not identity"),
                ok=True)),
        provenance=dict(
            chains_npz=f"data/mclmc_diag_v2d.npz (md5 {md5(npz)})",
            warm_scenez=f"data/mclmc_warm_v2d_scenez.npz "
                        f"(md5 {md5(DATA / 'mclmc_warm_v2d_scenez.npz')})",
            parity_refs_md5=md5(DATA / "parity_refs.npz"),
            t04_artifact=str(T04.relative_to(ROOT))),
        wall_s=time.time() - t0)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"[e1chi2] wrote {OUT} ({out['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
