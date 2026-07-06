"""04_validate_exact_smallgrid.py — THE P1a HARD GATE (conv path vs dense ref).

For v2d (80^2) and v3b (130^2), at 20 prior draws (x ~ prior, z = bij^{-1}(x)):

HARD GATE (pre-registered, README P1): |Delta logL| < 0.1 nat between
  (a) the conv-whitened ridge-marginalized likelihood evaluated through
      cgl.likelihood.build_marg_model(whiten_fn=...) on GPU, and
  (b) the dense-covariance exact reference on CPU (cgl.exact_ref):
      C^{-1} := G_e^T G_e with G_e = R_erode C_h diag(1/masked_err) assembled
      as an explicit sparse matrix, b = X^T C^{-1} R and
      A = X^T C^{-1} X + Lambda via scipy cho_factor — the same statistical
      functional computed by fully independent CPU linear algebra (conv
      semantics, erosion, masking, marginalization, f64 end-to-end).

CONSTANT ACCOUNTING (documented in the report; the dropped log|C| and
n log 2pi reconciled):
  * The conv path's implicit model is u = G_e y ~ N(G_e m(theta), I_{n_e});
    its dropped theta-independent constant is exactly -(n_e/2) log 2pi
    (log det I = 0). The quality of "cov(u) = I" is the Sigma_u audit.
  * The full-data GLS with the PHYSICAL C = D^{1/2} K D^{1/2} on the n_k
    KEPT pixels carries -(n_k/2) log 2pi - 1/2 logdet C. logdet C is
    computed EXACTLY (Cholesky) and compared with the whitener-based
    stationary (Szego) approximation sum_kept 2 log sigma + n_k *
    logdet_per_pix — the gap is reported (it is the correction any future
    cross-whitener/evidence comparison must carry, CAMPAIGN.md P1a watch-out
    (3)).
  * INFORMATIONAL (explicitly NOT the 0.1-nat gate; documented why): the
    exact-GLS marginalized logL differs from the whitened-path logL by the
    whitening misspecification integrated against prior-draw residuals of
    magnitude ||u||^2 ~ 1e5-1e7 nats; with e_op <= 0.02 the achievable
    agreement is O(e_op * ||u||^2) nats, orders of magnitude above 0.1 —
    no implementation could pass a 0.1-nat gate between those two DIFFERENT
    estimators. The 0.1-nat gate is therefore implemented as (a) vs (b)
    above — same estimator, independent implementations — and the
    misspecification scale is REPORTED per draw.

Also: re-verifies the 03 Sigma_u audit numbers, and runs the v3 (260^2)
FFT-circulant-preconditioned CG spot quadratic forms on the real MAP
residual (no dense factor at 260^2).

Outputs: data/exact_ref_report.json

Run (GPU 8):
  GIGALENS_X64=1 CUDA_VISIBLE_DEVICES=8 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /raid/benson/.venvs/cgl/bin/python 04_validate_exact_smallgrid.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO))
os.environ["GIGALENS_X64"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CGL_GPU", "8"))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

from cgl import exact_ref, guards  # noqa: E402
from cgl.paths import (  # noqa: E402
    CUTOUT_V2D, CUTOUT_V3, CUTOUT_V3B, DATA, MODEL_MAP_V3COLD, load_product,
)

GATE_NATS = 0.1
N_DRAWS = 20
PRODUCTS = {
    "v2d": dict(path=CUTOUT_V2D, delta_pix=0.13, supersample=2),
    "v3b": dict(path=CUTOUT_V3B, delta_pix=0.08, supersample=2),
}


def gpu_phase(tag, cfg, seed=12345):
    """Build the conv-whitened marg model and evaluate 20 prior draws.

    Returns per-draw internals (M_det, ret, logL_data, log_prior, logpost_jit)
    plus the model constants needed CPU-side.
    """
    import jax
    import jax.numpy as jnp

    from cgl.likelihood import build_marg_model
    from cgl.whiten import make_conv_whitener

    guards.require_x64()
    guards.require_gpu()
    guards.require_single_device()

    prod = load_product(cfg["path"])
    wz = np.load(DATA / f"whitener_{tag}.npz")
    h = wz["h"].astype(np.float64)
    keep_w = wz["keep_w"].astype(bool)
    masked_err = np.where(prod["keep_mask"],
                          prod["err_map"].astype(np.float64), 1e10)
    whiten_fn = make_conv_whitener(h, 1.0 / masked_err, keep_w)

    model = build_marg_model(
        prod, n_max=6, supersample=cfg["supersample"],
        delta_pix=cfg["delta_pix"],
        exp_time=float(prod["meta"].get("exp_time", 1197.7)),
        whiten_fn=whiten_fn,
        near_xy=tuple(prod["meta"]["nearby_arcsec"]))

    # 20 prior draws -> unconstrained z via the bijector inverse
    xs = model.prior.sample(N_DRAWS, seed=jax.random.PRNGKey(seed))
    z_list = model.bij.inverse(xs)
    z_draws = np.stack([np.asarray(v, dtype=np.float64).reshape(-1)
                        for v in z_list], axis=1)      # (N_DRAWS, 46)

    logpost_jit = jax.jit(model.target_log_prob_fn)
    draws = []
    for i in range(N_DRAWS):
        z = z_draws[i]
        internals = model.marg_internals(z)
        lp_jit = float(logpost_jit(jnp.asarray(z)))
        draws.append(dict(
            z=z, M_det=internals["M_det"], ret=internals["ret"],
            logL_data_gpu=internals["logL_data"],
            log_prior=internals["log_prior"],
            logpost_eager=internals["logpost"], logpost_jit=lp_jit,
            logdetA_gpu=internals["logdetA"],
            u_sq_gpu=float(np.sum(internals["Rw"] ** 2)),
        ))
        print(f"  draw {i:2d}: logL_data(GPU) = {internals['logL_data']:.6f}"
              f"  |jit-eager logpost| = "
              f"{abs(lp_jit - internals['logpost']):.2e}", flush=True)

    return dict(prod=prod, h=h, keep_w=keep_w, masked_err=masked_err,
                Lambda_diag=np.asarray(model.Lambda_diag, dtype=np.float64),
                draws=draws)


def cpu_phase(tag, g, report_03):
    """Dense reference, Sigma_u re-verification, exact GLS + constants."""
    prod = g["prod"]
    img = prod["img"].astype(np.float64)
    err = prod["err_map"].astype(np.float64)
    keep = prod["keep_mask"]
    shape = img.shape
    kz = np.load(DATA / f"noise_kernel_{tag}.npz")
    kmeta = json.loads(str(kz["meta"]))
    guards.assert_model_subtracted_sky(kmeta)
    rho = kz["rho_kernel"].astype(np.float64)

    t0 = time.time()
    G_e, eidx = exact_ref.whitening_operator(
        g["h"], 1.0 / g["masked_err"], g["keep_w"])
    n_e = G_e.shape[0]

    # ---- hard gate: dense whitened marg vs GPU path -------------------------
    per_draw = []
    worst = 0.0
    for i, d in enumerate(g["draws"]):
        ref = exact_ref.dense_whitened_marg(
            G_e, img, d["M_det"], d["ret"], g["Lambda_diag"])
        dl = abs(ref["logL"] - d["logL_data_gpu"])
        worst = max(worst, dl)
        per_draw.append(dict(
            i=i, logL_gpu=d["logL_data_gpu"], logL_dense=ref["logL"],
            abs_dlogL=dl, u_sq_gpu=d["u_sq_gpu"],
            u_sq_dense=float(ref["u"] @ ref["u"]),
            logdetA_gpu=d["logdetA_gpu"], logdetA_dense=ref["logdetA"],
            jit_minus_eager=abs(d["logpost_jit"] - d["logpost_eager"]),
        ))
    gate_pass = bool(worst < GATE_NATS)
    print(f"  hard gate: worst |dlogL| = {worst:.3e} nats over "
          f"{len(per_draw)} draws -> {'PASS' if gate_pass else 'FAIL'} "
          f"(< {GATE_NATS})")

    # ---- Sigma_u audit re-verification ---------------------------------------
    audit = exact_ref.sigma_u_audit(G_e, eidx, rho, err, shape)
    a03 = report_03["products"][tag]["dense_sigma_u"]
    audit_match = (abs(audit["var_min"] - a03["var_min"]) < 1e-9
                   and abs(audit["var_max"] - a03["var_max"]) < 1e-9
                   and abs(audit["offdiag_mean_abs"]
                           - a03["offdiag_mean_abs"]) < 1e-9)
    var_ok = (abs(audit["var_min"] - 1.0) <= 0.02
              and abs(audit["var_max"] - 1.0) <= 0.02)
    off_ok = audit["offdiag_mean_abs"] < 0.01
    print(f"  Sigma_u re-verify: match03={audit_match} "
          f"Var[{audit['var_min']:.4f},{audit['var_max']:.4f}] "
          f"({'PASS' if var_ok else 'FAIL'}), mean|off|="
          f"{audit['offdiag_mean_abs']:.5f} ({'PASS' if off_ok else 'FAIL'})")

    # ---- exact GLS with the physical C + constant accounting ----------------
    keep_idx = np.flatnonzero(keep.reshape(-1))
    n_k = keep_idx.size
    tC = time.time()
    C_kept = exact_ref.dense_C_kept(rho, err, keep_idx, shape)
    gls = exact_ref.ExactGLS(C_kept)
    del C_kept
    print(f"  exact GLS Cholesky: n_k={n_k} logdetC={gls.logdetC:.4f} "
          f"[{time.time()-tC:.0f}s]")

    wz = np.load(DATA / f"whitener_{tag}.npz")
    logdet_per_pix = float(wz["logdet_per_pix"])
    logdetC_szego = float(2.0 * np.sum(np.log(err.reshape(-1)[keep_idx]))
                          + n_k * logdet_per_pix)

    gls_draws = []
    for i, d in enumerate(g["draws"]):
        R_kept = (img - d["M_det"]).reshape(-1)[keep_idx]
        X_kept = d["ret"].reshape(-1, d["ret"].shape[-1])[keep_idx]
        m = gls.marg_loglik(R_kept, X_kept, g["Lambda_diag"])
        conv_aligned = (per_draw[i]["logL_dense"]
                        - 0.5 * n_e * np.log(2.0 * np.pi))
        gls_draws.append(dict(
            i=i, logL_gls_data=m["logL_data"], logL_gls_full=m["logL"],
            quad_gls=m["quad"],
            conv_data=per_draw[i]["logL_dense"],
            conv_aligned_full=float(conv_aligned),
            delta_data=float(per_draw[i]["logL_dense"] - m["logL_data"]),
        ))
    deltas = np.array([d["delta_data"] for d in gls_draws])
    misspec = dict(
        description=("INFORMATIONAL misspecification scale: whitened-path "
                     "data logL minus exact-GLS data logL (different "
                     "estimators; see module docstring for why this is NOT "
                     "the 0.1-nat gate)"),
        mean=float(deltas.mean()), sd=float(deltas.std()),
        min=float(deltas.min()), max=float(deltas.max()),
        mean_abs_frac_of_usq=float(np.mean(
            np.abs(deltas) / np.array([d["u_sq_gpu"]
                                       for d in per_draw]))),
    )
    print(f"  misspec scale (info): dlogL_data mean={misspec['mean']:.1f} "
          f"sd={misspec['sd']:.1f} nats "
          f"(mean |d|/||u||^2 = {misspec['mean_abs_frac_of_usq']:.2e})")

    constants = dict(
        n_e=int(n_e), n_kept=int(n_k),
        conv_dropped_const=float(-0.5 * n_e * np.log(2.0 * np.pi)),
        gls_log2pi_term=float(-0.5 * n_k * np.log(2.0 * np.pi)),
        logdetC_exact=gls.logdetC,
        logdetC_szego_whitener=logdetC_szego,
        szego_gap=float(gls.logdetC - logdetC_szego),
        note=("conv path drops -(n_e/2)log2pi (identity whitened cov, "
              "logdet=0); GLS carries -(n_k/2)log2pi - 1/2 logdetC_exact. "
              "szego_gap = exact logdetC - [sum_kept 2 log sigma + n_k * "
              "whitener logdet_per_pix]; any cross-whitener or evidence "
              "comparison must use the EXACT constants."),
    )

    return dict(
        n_e=int(n_e), gate=dict(threshold=GATE_NATS,
                                worst_abs_dlogL=float(worst),
                                gate_pass=gate_pass),
        per_draw=per_draw, sigma_u=dict(
            match_03=bool(audit_match), var_min=audit["var_min"],
            var_max=audit["var_max"],
            offdiag_mean_abs=audit["offdiag_mean_abs"],
            var_gate_pass=bool(var_ok), offdiag_gate_pass=bool(off_ok)),
        constants=constants, gls_draws=gls_draws, misspec=misspec,
        wall_s=time.time() - t0,
    ), gate_pass and var_ok and off_ok and audit_match


def v3_cg_spotcheck():
    """FFT-circulant-preconditioned CG spot quadratic forms on 260^2 (info).

    Uses the real MAP residual (img - model_map_v3cold): quad = R^T C^{-1} R
    on kept pixels (CG, no dense factor) vs the whitened ||u||^2 on the
    eroded pixels — the fine-grid misspecification + machinery check.
    """
    from scipy.ndimage import correlate

    prod = load_product(CUTOUT_V3)
    img = prod["img"].astype(np.float64)
    err = prod["err_map"].astype(np.float64)
    keep = prod["keep_mask"]
    model = np.load(MODEL_MAP_V3COLD).astype(np.float64)
    kz = np.load(DATA / "noise_kernel_v3.npz")
    rho = kz["rho_kernel"].astype(np.float64)
    wz = np.load(DATA / "whitener_v3.npz")
    h = wz["h"].astype(np.float64)
    keep_w = wz["keep_w"].astype(bool)

    R = img - model
    kidx = np.flatnonzero(keep.reshape(-1))
    t0 = time.time()
    quad, n_iter, rel_resid, info = exact_ref.circulant_cg_quad(
        rho, err, keep, R.reshape(-1)[kidx], tol=1e-8, maxiter=600)
    wall = time.time() - t0

    masked_err = np.where(keep, err, 1e10)
    u = correlate(R / masked_err, h, mode="constant", cval=0.0)
    usq = float(np.sum(u[keep_w] ** 2))
    out = dict(
        residual="img - model_map_v3cold (real MAP residual)",
        n_kept=int(kidx.size), n_eroded=int(keep_w.sum()),
        quad_exact_cg=float(quad), cg_iters=int(n_iter),
        cg_rel_resid=float(rel_resid), cg_info=int(info),
        quad_per_kept=float(quad / kidx.size),
        u_sq_whitened=usq, u_sq_per_eroded=float(usq / keep_w.sum()),
        wall_s=wall,
        note=("informational: exact GLS quad vs whitened quad on the fine "
              "product; also validates the CG machinery (iters, residual)"),
    )
    print(f"  v3 CG: quad/n_kept = {out['quad_per_kept']:.4f} "
          f"({n_iter} iters, rel_resid {rel_resid:.1e}); "
          f"||u||^2/n_eroded = {out['u_sq_per_eroded']:.4f} [{wall:.0f}s]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="data/exact_ref_report.json")
    ap.add_argument("--skip-v3-spot", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    report_03 = json.loads((DATA / "whitener_report.json").read_text())
    report = dict(generated_by="04_validate_exact_smallgrid.py",
                  gate_nats=GATE_NATS, n_draws=N_DRAWS, products={})
    all_pass = True

    for tag, cfg in PRODUCTS.items():
        print(f"\n=== {tag} (GPU phase) ===", flush=True)
        g = gpu_phase(tag, cfg)
        print(f"=== {tag} (CPU dense reference) ===", flush=True)
        rep, ok = cpu_phase(tag, g, report_03)
        report["products"][tag] = rep
        all_pass &= ok

    if not args.skip_v3_spot:
        print("\n=== v3 CG spot check (informational) ===", flush=True)
        report["v3_cg_spotcheck"] = v3_cg_spotcheck()

    report["all_hard_gates_pass"] = bool(all_pass)
    report["wall_s"] = time.time() - t0
    out = REPRO / args.out_json
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    print(f"04 HARD GATES {'PASS' if all_pass else 'FAIL'} "
          f"({report['wall_s']:.0f}s)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
