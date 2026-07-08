"""03b_build_relaxed_v2d.py — the D3-adopted RELAXED v2d whitener for E2c.

The P1b diagnosis (CAMPAIGN.md, 2026-07-08) ADOPTED the relaxed v2d whitener
for the native-scale real-data fit (E2c): E1d showed the strict M=14 whitener
erodes 91.7% of the 80^2 v2d keep mask (down to 487 px), while the relaxed
(e_op <= 0.05) whitener keeps ~5x more pixels and still calibrates (E1d AMENDED
verdict: max|zbar|=0.492, cov68 0.594, kept 4.9x strict on the mock arm).

This script builds that relaxed whitener on the REAL cutout_v2d.npz product,
using the EXACT construction the E1d relaxed arm used (verified bit-identical
kernel: the E1d mock rho_kernel == data/noise_kernel_v2d.npz rho_kernel):

    e1.build_product_whitener(v2d_rho, m_grid=(3,4,5,6,7,8,10,12), e_target=0.05)
        -> M=10, e_op~0.031, s_floor=0.05 (adaptive)

then erodes the real v2d keep mask by the (2M+1)^2 support and runs the same MC
whiteness audit as 03_build_whiteners.py. Output: data/whitener_v2d_relaxed.npz
(+ appends a 'v2d_relaxed' block to data/whitener_report.json).

Run (CPU-only; numpy/scipy, no likelihood):
    /raid/benson/.venvs/cgl/bin/python 03b_build_relaxed_v2d.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from cgl import e1
from cgl.paths import CUTOUT_V2D, DATA, load_product
from cgl.whiten import erode_keep, make_conv_whitener

# reuse 03's MC-whiteness helper for an identical audit
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_bw03", str((DATA.parent / "03_build_whiteners.py")))
_bw03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bw03)

E_TARGET = 0.05
M_GRID = (3, 4, 5, 6, 7, 8, 10, 12)
N_MC = 500
VAR_GATE = 0.02
OFFDIAG_GATE = 0.01


def main():
    kz = np.load(DATA / "noise_kernel_v2d.npz", allow_pickle=True)
    rho = np.asarray(kz["rho_kernel"], dtype=np.float64)
    kmeta = json.loads(str(kz["meta"]))

    prod = load_product(CUTOUT_V2D)
    err = np.asarray(prod["err_map"], dtype=np.float64)
    keep = np.asarray(prod["keep_mask"], dtype=bool)

    print(f"[03b] building relaxed v2d whitener (e_op<={E_TARGET}) ...",
          flush=True)
    t0 = time.time()
    wh = e1.build_product_whitener(rho, m_grid=M_GRID, e_target=E_TARGET)
    M = int(wh["M"])
    keep_w = erode_keep(keep, M)
    n_keep = int(keep.sum())
    n_e = int(keep_w.sum())
    loss = 1.0 - n_e / n_keep
    print(f"  M={M} e_op={wh['e_op']:.5f} s_floor={wh['s_floor']:.4f} "
          f"reg_lambda={wh['reg_lambda']:.3g} e_target_met={wh['e_target_met']}",
          flush=True)
    print(f"  m_search={wh['m_search']}", flush=True)
    print(f"  pixels {n_keep} -> {n_e} (loss {100*loss:.1f}%); "
          f"strict M=14 kept 487 -> relaxed keeps {n_e} "
          f"({n_e/487:.2f}x strict)", flush=True)

    # ---- MC whiteness audit (same as 03) ------------------------------------
    tm = time.time()
    mc = _bw03.mc_whiteness(wh["h"], rho, err, keep, keep_w, N_MC)
    mc["wall_s"] = time.time() - tm
    mc_var_ok = abs(mc["mean_var"] - 1.0) <= VAR_GATE
    mc_off_ok = mc["worst_abs_lag_corr"] < OFFDIAG_GATE
    mc["var_gate_pass"] = bool(mc_var_ok)
    mc["offdiag_gate_pass"] = bool(mc_off_ok)
    print(f"  MC whiteness ({mc['n_draws']} draws): mean Var = "
          f"{mc['mean_var']:.4f} ({'PASS' if mc_var_ok else 'FAIL'}); "
          f"worst |mean lag corr| = {mc['worst_abs_lag_corr']:.5f} "
          f"({'PASS' if mc_off_ok else 'FAIL'}) [{mc['wall_s']:.0f}s]",
          flush=True)

    wmeta = dict(
        product=str(CUTOUT_V2D), tag="v2d_relaxed",
        arm="relaxed", adopted_by="P1b diagnosis D3 (CAMPAIGN.md 2026-07-08)",
        construction=("cgl.e1.build_product_whitener(noise_kernel_v2d.rho, "
                      f"m_grid={M_GRID}, e_target={E_TARGET}); identical to the "
                      "E1d relaxed arm (E1d mock rho_kernel == real v2d kernel, "
                      "verified max|diff|=0.0)"),
        model_subtracted=True, kernel_npz="noise_kernel_v2d.npz",
        kernel_meta=kmeta,
        M=M, e_op=float(wh["e_op"]), e_target=E_TARGET,
        e_target_met=bool(wh["e_target_met"]),
        s_floor=float(wh["s_floor"]), reg_lambda=float(wh["reg_lambda"]),
        m_search=wh["m_search"], grid=int(wh.get("grid", 512)),
        logdet_per_pix=float(wh["logdet_per_pix"]),
        n_keep=n_keep, n_eroded=n_e, pixel_loss_frac=loss,
        n_eroded_vs_strict=float(n_e / 487.0),
        s_min=float(wh["s_min"]), s_max=float(wh["s_max"]),
        floor_frac=float(wh["floor_frac"]),
        commit=e1._git_head(),
    )
    out = DATA / "whitener_v2d_relaxed.npz"
    np.savez(out, h=wh["h"], keep_w=keep_w, M=M, e_op=float(wh["e_op"]),
             logdet_per_pix=float(wh["logdet_per_pix"]),
             rho_kernel=rho, meta=json.dumps(wmeta))
    print(f"  wrote {out} in {time.time()-t0:.0f}s", flush=True)

    # append to whitener_report.json
    rep_path = DATA / "whitener_report.json"
    rep = json.load(open(rep_path)) if rep_path.exists() else {}
    rep.setdefault("products", {})
    rec = dict(wmeta)
    rec["mc_whiteness"] = mc
    rec["all_gates_pass"] = bool(wh["e_target_met"] and mc_var_ok and mc_off_ok)
    rep["products"]["v2d_relaxed"] = rec
    json.dump(rep, open(rep_path, "w"), indent=1)
    print(f"  appended v2d_relaxed to {rep_path}", flush=True)


if __name__ == "__main__":
    main()
