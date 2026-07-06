"""03_build_whiteners.py — P1a convolutional whiteners for v2d / v3 / v3b.

For each product's fitted noise kernel (02_fit_noise_kernels.py artifacts;
guards.assert_model_subtracted_sky enforced):

  1. build_whitener (cgl.whiten): periodic-embedding spectrum, floored at
     s_floor*mean(S), truncated inverse-sqrt taps, GN + Lawson-IRLS refine.
     M is searched upward until the PRE-REGISTERED gate e_op <= 0.02 holds.
  2. erode_keep: whitened-domain mask (erosion by the (2M+1)^2 support and
     the SAME-conv border margin); pixel loss reported.
  3. Whiteness audits against the PHYSICAL covariance C = D^{1/2} K D^{1/2}
     (unmasked err; fitted kernel K):
       - v2d / v3b: EXACT dense Sigma_u = G_e C G_e^T audit (cgl.exact_ref).
         Gates: per-pixel Var(u) in 1 +/- 0.02; mean |off-diag corr| at
         lags <= 3 below 0.01.
       - v3 (260^2, no dense factor): Monte-Carlo whiteness — 2000 exact
         FFT draws from K scaled by D^{1/2}, pushed through the JAX conv
         whitening path on GPU. Gates: mean Var(u) in 1 +/- 0.02; worst
         |mean off-diag corr| per lag <= 3 below 0.01 (signed means over
         pairs+draws, so MC noise averages out; per-pixel Var spread is
         reported against its sampling band). MC is also run for v2d/v3b
         as a cross-validation of the dense numbers (informational).

Outputs: data/whitener_{v2d,v3,v3b}.npz, data/whitener_report.json

Run (GPU 8):
  GIGALENS_X64=1 CUDA_VISIBLE_DEVICES=8 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /raid/benson/.venvs/cgl/bin/python 03_build_whiteners.py
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
os.environ.setdefault("GIGALENS_X64", "1")

from cgl import guards  # noqa: E402
from cgl.paths import CUTOUT_V2D, CUTOUT_V3, CUTOUT_V3B, DATA, load_product  # noqa: E402
from cgl.whiten import build_whitener, erode_keep, make_conv_whitener  # noqa: E402
from cgl import exact_ref  # noqa: E402

PRODUCTS = {"v2d": CUTOUT_V2D, "v3": CUTOUT_V3, "v3b": CUTOUT_V3B}
M_GRID = (8, 10, 12, 14, 16, 18, 20, 22, 24)
E_OP_GATE = 0.02
VAR_GATE = 0.02
OFFDIAG_GATE = 0.01
S_FLOOR = 0.05
GRID = 512
N_MC = {"v2d": 500, "v3": 2000, "v3b": 500}
LAGS3 = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)
         if (dy, dx) > (0, 0)]


def mc_whiteness(h, rho, err_map, keep_mask, keep_w, n_draws, seed=0,
                 batch=100):
    """MC whiteness through the JAX conv whitening path (GPU).

    Draws exact stationary noise from K (FFT), scales by the UNMASKED err
    map, whitens with the same make_conv_whitener the likelihood uses
    (masked sqrt_d_inv + eroded keep_w), and accumulates per-pixel Var and
    signed lag correlations over eroded pairs.
    """
    import jax
    import jax.numpy as jnp

    H, W = err_map.shape
    masked_err = np.where(keep_mask, err_map, 1e10)
    wf = make_conv_whitener(h, 1.0 / masked_err, keep_w)
    wf_batch = jax.jit(jax.vmap(lambda im: wf(im).reshape(H, W)))

    rng = np.random.default_rng(seed)
    n_e = int(keep_w.sum())
    sum_u2 = np.zeros((H, W))
    lag_sums = {d: 0.0 for d in LAGS3}
    lag_ns = {d: 0 for d in LAGS3}
    pair_masks = {}
    for dy, dx in LAGS3:
        m = np.zeros((H, W), dtype=bool)
        ys = slice(max(0, -dy), min(H, H - dy))
        xs = slice(max(0, -dx), min(W, W - dx))
        m[ys, xs] = keep_w[ys, xs] & np.roll(np.roll(keep_w, -dy, 0), -dx, 1)[ys, xs]
        pair_masks[(dy, dx)] = m

    done = 0
    while done < n_draws:
        b = min(batch, n_draws - done)
        x = exact_ref.sample_stationary_batch(rho, (H, W), b, rng, grid=GRID)
        u = np.asarray(wf_batch(jnp.asarray(x * err_map[None, :, :])))
        sum_u2 += (u ** 2).sum(axis=0)
        for (dy, dx), m in pair_masks.items():
            us = np.roll(np.roll(u, -dy, 1), -dx, 2)
            prod = (u * us)[:, m]
            lag_sums[(dy, dx)] += float(prod.sum())
            lag_ns[(dy, dx)] += prod.size
        done += b

    var_map = sum_u2 / n_draws
    var_kept = var_map[keep_w]
    mean_var = float(var_kept.mean())
    lag_corr = {f"{dy},{dx}": float((lag_sums[(dy, dx)] / lag_ns[(dy, dx)])
                                    / mean_var)
                for (dy, dx) in LAGS3 if lag_ns[(dy, dx)] > 0}
    worst_lag = float(max(abs(v) for v in lag_corr.values()))
    # sampling band for the per-pixel Var spread: Var_hat ~ 1 +/- sqrt(2/N)
    band = float(np.sqrt(2.0 / n_draws))
    return dict(n_draws=int(n_draws), n_e=n_e, mean_var=mean_var,
                var_p01=float(np.percentile(var_kept, 1)),
                var_p99=float(np.percentile(var_kept, 99)),
                var_min=float(var_kept.min()), var_max=float(var_kept.max()),
                var_sampling_sd_expected=band,
                var_spread_sd=float(var_kept.std()),
                worst_abs_lag_corr=worst_lag, lag_corr=lag_corr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="data/whitener_report.json")
    ap.add_argument("--products", nargs="*", default=list(PRODUCTS))
    args = ap.parse_args()
    t0 = time.time()

    report = dict(generated_by="03_build_whiteners.py", s_floor=S_FLOOR,
                  grid=GRID, gates=dict(e_op=E_OP_GATE, var=VAR_GATE,
                                        offdiag=OFFDIAG_GATE),
                  products={})
    all_pass = True

    for tag in args.products:
        print(f"\n=== {tag} ===")
        kz = np.load(DATA / f"noise_kernel_{tag}.npz")
        kmeta = json.loads(str(kz["meta"]))
        guards.assert_model_subtracted_sky(kmeta)
        rho = kz["rho_kernel"].astype(np.float64)
        prod = load_product(PRODUCTS[tag])
        err = prod["err_map"].astype(np.float64)
        keep = prod["keep_mask"]

        # ---- 1. whitener taps: search M upward to the e_op gate ------------
        # ADAPTIVE FLOOR: the fitted rho_model2 kernels are PSD by
        # construction (S_raw_min ~ the delta weight > 0), so the spectral
        # floor exists only as a guard for indefinite (nonparametric E3)
        # kernels. If s_floor=0.05 would ENGAGE on a PSD kernel it BIASES
        # the whitener away from the true spectrum (measured on v3b:
        # Var(u)=0.981, mean|offdiag|=0.0105 — both gate FAILs); choose the
        # largest floor in {0.05, 0.02, 0.01, 0.005} that stays strictly
        # below S_raw_min so flooring is inactive. If the kernel is
        # indefinite (S_raw_min <= 0) keep the hard 0.05 floor.
        S_raw = exact_ref.stationary_spectrum(rho, GRID)
        ratio = float(S_raw.min() / S_raw.mean())
        s_floor_eff = S_FLOOR
        if 0.0 < ratio <= S_FLOOR:
            for cand in (0.05, 0.02, 0.01, 0.005):
                if cand < ratio:
                    s_floor_eff = cand
                    break
            else:
                s_floor_eff = 0.5 * ratio
        print(f"  S_raw min/mean = {ratio:.4f} -> s_floor = {s_floor_eff}")

        chosen = None
        for M in M_GRID:
            w = build_whitener(rho, M, s_floor=s_floor_eff, grid=GRID)
            print(f"  M={M:2d}: e_op={w['e_op']:.5f} "
                  f"(init {w['e_op_init']:.5f})")
            if w["e_op"] <= E_OP_GATE:
                chosen = w
                break
        if chosen is None:
            chosen = w
        M = chosen["M"]
        e_pass = bool(chosen["e_op"] <= E_OP_GATE)

        # ---- 2. eroded whitened-domain mask ---------------------------------
        keep_w = erode_keep(keep, M)
        n_keep = int(keep.sum())
        n_e = int(keep_w.sum())
        loss = 1.0 - n_e / n_keep
        print(f"  chosen M={M} e_op={chosen['e_op']:.5f} "
              f"({'PASS' if e_pass else 'FAIL'} <= {E_OP_GATE}); "
              f"pixels {n_keep} -> {n_e} (loss {100*loss:.1f}%)")

        # ---- 3a. exact dense Sigma_u audit (small grids) --------------------
        audit = None
        if tag in ("v2d", "v3b"):
            ta = time.time()
            masked_err = np.where(keep, err, 1e10)
            G_e, eidx = exact_ref.whitening_operator(
                chosen["h"], 1.0 / masked_err, keep_w)
            audit = exact_ref.sigma_u_audit(G_e, eidx, rho, err, err.shape)
            audit["wall_s"] = time.time() - ta
            var_ok = (abs(audit["var_min"] - 1.0) <= VAR_GATE
                      and abs(audit["var_max"] - 1.0) <= VAR_GATE)
            off_ok = audit["offdiag_mean_abs"] < OFFDIAG_GATE
            audit["var_gate_pass"] = bool(var_ok)
            audit["offdiag_gate_pass"] = bool(off_ok)
            print(f"  dense Sigma_u: Var(u) in [{audit['var_min']:.4f}, "
                  f"{audit['var_max']:.4f}] ({'PASS' if var_ok else 'FAIL'}); "
                  f"mean|offdiag| lags<=3 = {audit['offdiag_mean_abs']:.5f} "
                  f"({'PASS' if off_ok else 'FAIL'}) "
                  f"[{audit['wall_s']:.0f}s]")
        else:
            var_ok = off_ok = True   # gated by MC below for v3

        # ---- 3b. Monte-Carlo whiteness (all products; the gate for v3) ------
        tm = time.time()
        mc = mc_whiteness(chosen["h"], rho, err, keep, keep_w, N_MC[tag])
        mc["wall_s"] = time.time() - tm
        mc_var_ok = abs(mc["mean_var"] - 1.0) <= VAR_GATE
        mc_off_ok = mc["worst_abs_lag_corr"] < OFFDIAG_GATE
        mc["var_gate_pass"] = bool(mc_var_ok)
        mc["offdiag_gate_pass"] = bool(mc_off_ok)
        print(f"  MC whiteness ({mc['n_draws']} draws): mean Var = "
              f"{mc['mean_var']:.4f} ({'PASS' if mc_var_ok else 'FAIL'}); "
              f"worst |mean lag corr| = {mc['worst_abs_lag_corr']:.5f} "
              f"({'PASS' if mc_off_ok else 'FAIL'}) [{mc['wall_s']:.0f}s]")

        if tag == "v3":
            var_ok, off_ok = mc_var_ok, mc_off_ok

        # ---- save ------------------------------------------------------------
        wmeta = dict(
            product=str(PRODUCTS[tag]), tag=tag,
            model_subtracted=True,
            kernel_npz=f"noise_kernel_{tag}.npz", kernel_meta=kmeta,
            M=M, e_op=chosen["e_op"], e_op_gate=E_OP_GATE,
            e_op_gate_pass=e_pass,
            s_floor=s_floor_eff, s_floor_policy_default=S_FLOOR,
            s_raw_min_over_mean=ratio, grid=GRID,
            logdet_per_pix=chosen["logdet_per_pix"],
            n_keep=n_keep, n_eroded=n_e, pixel_loss_frac=loss,
            s_min=chosen["s_min"], s_max=chosen["s_max"],
            floor_frac=chosen["floor_frac"],
        )
        np.savez(DATA / f"whitener_{tag}.npz",
                 h=chosen["h"], keep_w=keep_w, M=M,
                 e_op=chosen["e_op"],
                 logdet_per_pix=chosen["logdet_per_pix"],
                 rho_kernel=rho, meta=json.dumps(wmeta))
        print(f"  wrote whitener_{tag}.npz")

        rep = dict(wmeta)
        rep["dense_sigma_u"] = audit
        rep["mc_whiteness"] = mc
        gate = e_pass and var_ok and off_ok
        rep["all_gates_pass"] = bool(gate)
        report["products"][tag] = rep
        all_pass &= gate

    report["all_gates_pass"] = bool(all_pass)
    report["wall_s"] = time.time() - t0
    out = REPRO / args.out_json
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    print(f"ALL 03 GATES {'PASS' if all_pass else 'FAIL'} "
          f"({report['wall_s']:.0f}s)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
