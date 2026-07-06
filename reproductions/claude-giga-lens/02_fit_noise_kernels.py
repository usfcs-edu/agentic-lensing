"""02_fit_noise_kernels.py — P1a stationary noise kernels for v2d / v3 / v3b.

For each Stage-A product the kernel is fit on MODEL-SUBTRACTED residuals
(guards.assert_model_subtracted_sky enforced on every artifact):

  1. residual = img - model_map (model maps: foundry-i model_map_v3cold.npy /
     model_map_v3b_cold.npy; for v2d none exists on disk, so the gated MAP
     map_v11_v2d_cold.npz (74-dim, n_max=6, chi2=1.234) is RENDERED here via
     the foundry-i _data_lib.build_all path — the 46_noise_audit.render_map
     convention (float32 paper mode, GPU) — and cached to
     data/model_map_v2d_cold.npy).
  2. normalized residual v = residual / err_map (the likelihood's stationary-
     correlation object: Cov(v) = K).
  3. sky region + Background2D detrending exactly as foundry-i 46_noise_audit
     (sky = keep & r>4.5" & |img|<5*median(err); ~1" boxes) — the kernel is
     the drizzle-scale correlation; large-scale background structure is a
     separate finding (46 convention).
  4. masked, mask-deconvolved ACF (cgl.noise.masked_acf_2d, port verified
     bit-identical against 46_noise_audit.masked_autocorr on these inputs).
  5. WLS fit over the fit window (L = 3 native / 8 fine / 4 binned), where
     rho_drz is the NUMERICALLY ENUMERATED drizzle-operator ACF (phase-
     averaged; geometry: square kernel, pixfrac 1.0, NDRIZIM=3; native
     0.1283" -> 0.13" (v2d, r=0.987) / 0.04" (v3, r=3.21); v3b = exact 2x2
     block-sum transform of the fine drizzle covariance).

KERNEL-FAMILY DEVIATION (documented; single-family numbers kept on record):
the plan's one-Gaussian family rho = (1-w) delta + w norm(rho_drz (*) G(s_e))
CANNOT meet the 0.05 gate on the measured ACFs — the model-subtracted,
detrended residuals carry (a) a medium-scale correlated pedestal (~0.15-0.26
out to lag 8 on v3; survives both model subtraction and ~1" detrending) and
(b) diagonal core anisotropy (rotated drizzle frames; ~0.74 vs ~0.70 at the
unit diagonals of v3) plus column-direction stripe correlation on v2d.
Single-family best fits: max|resid| = 0.090 (v2d) / 0.311 (v3) / 0.166 (v3b).
The minimal PSD-by-construction extension that passes is cgl.noise.rho_model2:
    rho = (1-w_d-w_b) delta + w_d norm(rho_drz (*) G2(sig_ey,sig_ex,c_e))
                            + w_b G2(sig_by,sig_bx,c_b)
(bivariate Gaussians; nonnegative mixture of PSD kernels, w_d + w_b <= 1).

GATES (pre-registered): max|rho_fit - rho_meas| <= 0.05 over the fit window,
per product; plus the v3->v3b block-sum cross-check <= 0.05.
Also reported: the drizzle anchor t(1) closed form (r-1)/(r-1/3) vs the
enumerated operator vs the measured axis correlation.

Outputs: data/noise_kernel_{v2d,v3,v3b}.npz, data/noise_kernel_report.json

Run (GPU 8; f32 render mode — matches the 46_noise_audit render convention;
the kernel fitting itself is sanctioned-CPU numpy):
  CUDA_VISIBLE_DEVICES=8 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  XLA_FLAGS=--xla_gpu_autotune_level=0 \
  /raid/benson/.venvs/cgl/bin/python 02_fit_noise_kernels.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO))

from cgl import guards  # noqa: E402
from cgl.noise import (  # noqa: E402
    binned_kernel_from_fine,
    detrend_sky,
    drizzle_acf,
    fit_kernel,
    fit_kernel2,
    masked_acf_2d,
    masked_autocorr_full,
    rho_model2,
)
from cgl.paths import (  # noqa: E402
    CUTOUT_V2D, CUTOUT_V3, CUTOUT_V3B, DATA, FOUNDRY_I, FOUNDRY_I_DATA,
    MODEL_MAP_V3B_COLD, MODEL_MAP_V3COLD, bootstrap_vendor, load_product,
)

NATIVE_PIX = 0.1283          # WFC3/IR native scale (fine-skycell header fact)
PIXFRAC = 1.0                # D001PIXF (verified header fact)
N_FRAMES = 3                 # NDRIZIM
ARC_ANNULUS = (1.2, 4.5)     # arcsec (40/40b/46 convention)

PRODUCTS = {
    "v2d": dict(path=CUTOUT_V2D, window=3, bkg_box=10,
                model="model_map_v2d_cold.npy",
                model_z=FOUNDRY_I_DATA / "map_v11_v2d_cold.npz"),
    "v3": dict(path=CUTOUT_V3, window=8, bkg_box=26,
               model=MODEL_MAP_V3COLD, model_z=None),
    "v3b": dict(path=CUTOUT_V3B, window=4, bkg_box=13,
                model=MODEL_MAP_V3B_COLD, model_z=None),
}
FIT_GATE = 0.05
XCHECK_GATE = 0.05


# --------------------------------------------------------------------------- #
def render_v2d_model(cache_path: Path) -> np.ndarray:
    """Render the v2d gated MAP through the foundry-i forward path (GPU,
    float32 paper mode — the 46_noise_audit.render_map convention)."""
    if cache_path.exists():
        print(f"  using cached {cache_path.name}")
        return np.load(cache_path).astype(np.float64)

    bootstrap_vendor()                       # OUR vendored lib wins
    import jax
    guards.require_gpu()
    guards.require_single_device()
    import jax.numpy as jnp
    from gigalens.jax.simulator import LensSimulator

    sys.path.insert(1, str(FOUNDRY_I))
    import _data_lib as D                    # foundry-i module, read-only

    z = np.load(PRODUCTS["v2d"]["model_z"])
    z_best = z["best_z"].astype(np.float32).reshape(-1)
    assert z_best.size == 74, f"expected the 74-dim (n_max=6) MAP, got {z_best.size}"
    d, prior, phys, prob, sim_config = D.build_all(n_max=6,
                                                   data_file="cutout_v2d.npz")
    sim = LensSimulator(phys, sim_config, bs=2)      # bs=2 dodges the squeeze
    zb = jnp.asarray(np.stack([z_best, z_best]))
    x = prob.bij.forward(list(zb.T))
    model = np.asarray(sim.simulate(x))[0].astype(np.float64)
    np.save(cache_path, model.astype(np.float32))
    print(f"  rendered + cached {cache_path.name}")
    return model


def measure_product(tag: str, cfg: dict, model: np.ndarray) -> dict:
    """Steps 2-4: normalized model-subtracted residual -> detrend -> ACF."""
    prod = load_product(cfg["path"])
    img = prod["img"].astype(np.float64)
    err = prod["err_map"].astype(np.float64)
    keep = prod["keep_mask"]
    meta = prod["meta"]
    delta_pix = float(meta["delta_pix"])
    n_pix = img.shape[0]

    resid = img - model
    v = resid / err

    yy, xx = np.indices(img.shape)
    cen = (n_pix - 1) / 2.0
    r_arc = np.hypot(xx - cen, yy - cen) * delta_pix
    sky = keep & (r_arc > ARC_ANNULUS[1]) & (np.abs(img) < 5.0 * np.median(err))

    v_det, detrend_method = detrend_sky(v, sky, cfg["bkg_box"])

    L = cfg["window"]
    rho_meas, counts = masked_acf_2d(v_det, sky, max_lag=L + 6)

    # ---- port verification: bit-identical to 46_noise_audit on THIS input --
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fi_noise_audit", FOUNDRY_I / "46_noise_audit.py")
    fi = importlib.util.module_from_spec(spec)
    sys.path.insert(1, str(FOUNDRY_I))
    spec.loader.exec_module(fi)
    rho_fi, center_fi = fi.masked_autocorr(v_det, sky)
    rho_full, center, _ = masked_autocorr_full(v_det, sky)
    port_bitmatch = bool(
        center == center_fi
        and np.array_equal(np.isnan(rho_fi), np.isnan(rho_full))
        and np.array_equal(np.nan_to_num(rho_fi), np.nan_to_num(rho_full)))

    chi2_sky_resid = float(np.mean(v[sky] ** 2))
    return dict(
        prod=prod, meta=meta, delta_pix=delta_pix, sky=sky,
        v_det=v_det, err=err,
        rho_meas=rho_meas, counts=counts, L=L,
        detrend_method=detrend_method, port_bitmatch=port_bitmatch,
        n_sky_px=int(sky.sum()), chi2_sky_resid=chi2_sky_resid,
        rho_axis=[float(rho_meas[L + 6, L + 6 + d]) for d in range(0, L + 4)],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="data/noise_kernel_report.json")
    args = ap.parse_args()
    t0 = time.time()

    # ---- model maps ---------------------------------------------------------
    models = {
        "v2d": render_v2d_model(DATA / PRODUCTS["v2d"]["model"]),
        "v3": np.load(PRODUCTS["v3"]["model"]).astype(np.float64),
        "v3b": np.load(PRODUCTS["v3b"]["model"]).astype(np.float64),
    }
    model_sources = {
        "v2d": ("rendered here from foundry-i map_v11_v2d_cold.npz best_z "
                "(74-dim n_max=6 gated MAP, chi2_pp=1.234) via _data_lib."
                "build_all('cutout_v2d.npz') + LensSimulator(bs=2), float32 "
                "paper mode on GPU — the 46_noise_audit.render_map "
                "convention; cached at data/model_map_v2d_cold.npy"),
        "v3": str(MODEL_MAP_V3COLD),
        "v3b": str(MODEL_MAP_V3B_COLD),
    }

    # ---- drizzle anchors (numerically enumerated operator) ------------------
    drz = {}
    r_fine = NATIVE_PIX / 0.04
    r_native = NATIVE_PIX / 0.13
    # generous half-widths: the Gaussian in the model family needs tail room
    drz["v3"] = drizzle_acf(r_fine, PIXFRAC, N_FRAMES, None,
                            max_lag=32, n_phase=32)
    drz["v2d"] = drizzle_acf(r_native, PIXFRAC, N_FRAMES, None,
                             max_lag=32, n_phase=32)
    # v3b: exact 2x2 block-sum transform of the fine drizzle covariance
    cov_fine_drz = drz["v3"]["rho"] * drz["v3"]["var_out_over_var_in"]
    cov_b = binned_kernel_from_fine(cov_fine_drz)          # half-width 15
    rho_drz_b = cov_b / cov_b[15, 15]
    drz["v3b"] = dict(rho=rho_drz_b, t1=float(rho_drz_b[15, 16]),
                      t1_analytic=None, scale_ratio=r_fine / 2.0,
                      pixfrac=PIXFRAC, n_frames=N_FRAMES,
                      note="binned_kernel_from_fine(fine drizzle cov)")

    report = dict(generated_by="02_fit_noise_kernels.py",
                  native_pix=NATIVE_PIX, pixfrac=PIXFRAC, n_frames=N_FRAMES,
                  drizzle_anchor=dict(
                      r_fine=r_fine, r_native=r_native,
                      t1_analytic_fine=drz["v3"]["t1_analytic"],
                      t1_enumerated_fine=drz["v3"]["t1"],
                      t1_enumerated_native=drz["v2d"]["t1"],
                      t1_analytic_native_NOTE=(
                          "closed form (r-1)/(r-1/3) is only valid for r>=2; "
                          "at r=0.987 the enumerated operator is the anchor"),
                      t1_enumerated_binned=drz["v3b"]["t1"]),
                  products={})

    fits = {}
    all_pass = True
    for tag, cfg in PRODUCTS.items():
        print(f"\n=== {tag} ===")
        m = measure_product(tag, cfg, models[tag])
        L = m["L"]
        c = L + 6
        rho_win = m["rho_meas"][c - L:c + L + 1, c - L:c + L + 1]
        cnt_win = m["counts"][c - L:c + L + 1, c - L:c + L + 1]
        # single-family fit kept on record (deviation justification)
        fit1 = fit_kernel(rho_win, cnt_win, drz[tag]["rho"], max_lag=L)
        # primary: two-component bivariate family
        fit = fit_kernel2(rho_win, cnt_win, drz[tag]["rho"], max_lag=L)
        fits[tag] = (m, fit)

        # extended kernel for the whitener / dense-K consumers: the analytic
        # family evaluated out to where every component reaches double-
        # precision zero (L_k=64: Gaussian tails < 1e-13 for sig<=8.5). A
        # hard mid-tail truncation puts ringing into the embedded spectrum
        # and STALLS the whitener taps (measured: v3 e_op plateaued ~0.09 at
        # L_k=30 vs 0.016 untruncated). The drizzle anchor is compactly
        # supported, so zero-padding it is exact.
        L_k = 64
        anchor = np.zeros((2 * L_k + 1, 2 * L_k + 1))
        Ld_anchor = (drz[tag]["rho"].shape[0] - 1) // 2
        anchor[L_k - Ld_anchor:L_k + Ld_anchor + 1,
               L_k - Ld_anchor:L_k + Ld_anchor + 1] = drz[tag]["rho"]
        rho_kernel = rho_model2(*fit["params"], anchor, L_k)

        kmeta = dict(
            product=str(cfg["path"]), tag=tag,
            model_subtracted=True,
            model_source=model_sources[tag],
            detrend=dict(method=m["detrend_method"], box_px=cfg["bkg_box"]),
            sky_region=("keep & r>4.5\" & |img|<5*median(err) "
                        "(46_noise_audit convention)"),
            n_sky_px=m["n_sky_px"],
            chi2_sky_resid=m["chi2_sky_resid"],
            acf_port_bitmatch_46=m["port_bitmatch"],
            normalization="ACF of (img-model)/err_map (Cov(v)=K convention)",
            family=("rho_model2: (1-w_d-w_b) delta "
                    "+ w_d norm(rho_drz (*) G2(sig_ey,sig_ex,c_e)) "
                    "+ w_b G2(sig_by,sig_bx,c_b); see script docstring "
                    "for the documented deviation from the 1-Gaussian plan"),
            fit_window_L=L, kernel_half_width=L_k,
            w_d=fit["w_d"], sig_ey=fit["sig_ey"], sig_ex=fit["sig_ex"],
            c_e=fit["c_e"], w_b=fit["w_b"], sig_by=fit["sig_by"],
            sig_bx=fit["sig_bx"], c_b=fit["c_b"],
            max_abs_resid=fit["max_abs_resid"], rms_resid=fit["rms_resid"],
            gate_le_0p05=fit["gate_le_0p05"],
            fit_single_family=dict(w=fit1["w"], sigma_e=fit1["sigma_e"],
                                   max_abs_resid=fit1["max_abs_resid"],
                                   rms_resid=fit1["rms_resid"],
                                   gate_le_0p05=fit1["gate_le_0p05"]),
            drizzle=dict(scale_ratio=float(drz[tag].get("scale_ratio", 0.0)),
                         pixfrac=PIXFRAC, n_frames=N_FRAMES,
                         t1=drz[tag]["t1"],
                         t1_analytic=drz[tag].get("t1_analytic")),
        )
        guards.assert_model_subtracted_sky(kmeta)

        out = DATA / f"noise_kernel_{tag}.npz"
        np.savez(out,
                 rho_kernel=rho_kernel,
                 params=np.asarray(fit["params"], dtype=np.float64),
                 rho_meas=rho_win, lag_counts=cnt_win,
                 rho_fit=fit["rho_fit"], resid_map=fit["resid_map"],
                 rho_meas_ext=m["rho_meas"], lag_counts_ext=m["counts"],
                 rho_drz=drz[tag]["rho"],
                 meta=json.dumps(kmeta))
        print(f"  w_d={fit['w_d']:.4f} sig_e=({fit['sig_ey']:.3f},"
              f"{fit['sig_ex']:.3f},c={fit['c_e']:+.3f}) "
              f"w_b={fit['w_b']:.4f} sig_b=({fit['sig_by']:.2f},"
              f"{fit['sig_bx']:.2f},c={fit['c_b']:+.3f})")
        print(f"  max|resid|={fit['max_abs_resid']:.4f} "
              f"rms={fit['rms_resid']:.4f} "
              f"-> {'PASS' if fit['gate_le_0p05'] else 'FAIL'} (<= {FIT_GATE})"
              f"   [single-family: {fit1['max_abs_resid']:.4f} "
              f"{'PASS' if fit1['gate_le_0p05'] else 'FAIL'}]")
        print(f"  measured axis rho: {np.round(m['rho_axis'][:6], 4)}")
        print(f"  ACF port bit-match vs 46_noise_audit: {m['port_bitmatch']}")
        print(f"  wrote {out.name}")

        all_pass &= fit["gate_le_0p05"] and m["port_bitmatch"]
        report["products"][tag] = {k: v for k, v in kmeta.items()}
        report["products"][tag]["rho_axis_meas"] = m["rho_axis"]
        report["products"][tag]["rho_axis_fit"] = [
            float(fit["rho_fit"][L, L + d]) for d in range(0, L + 1)]

    # ---- v3 -> v3b block-sum cross-check ------------------------------------
    # The transform is exact for the SAME underlying field, so the fair
    # comparison holds the processing fixed: block-sum the fine DETRENDED
    # residual itself and re-measure the binned ACF (commuting processing).
    # Comparing against the v3b PRODUCT ACF instead convolves the check with
    # detrender non-commutation (Background2D on the two grids removes
    # slightly different pedestal fractions) — that number is reported as
    # informational with this explanation.
    from cgl.noise import block_all, block_sum

    (m3, f3), (m3b, _) = fits["v3"], fits["v3b"]
    rho_fine_model = rho_model2(*f3["params"], drz["v3"]["rho"], 11)
    cov_b_model = binned_kernel_from_fine(rho_fine_model)   # half-width 5
    rho_b_model = cov_b_model / cov_b_model[5, 5]
    Lb = m3b["L"]

    # commuting-processing binned ACF from the fine detrended residual
    resid_det_f = m3["v_det"] * m3["err"]
    resid_b = block_sum(resid_det_f, 2)
    err_q_b = np.sqrt(block_sum(m3["err"] ** 2, 2))
    v_b = resid_b / err_q_b
    sky_b = block_all(m3["sky"], 2)
    rho_b_comm, cnt_b_comm = masked_acf_2d(v_b, sky_b, max_lag=Lb + 2)
    cbm = Lb + 2
    meas_comm = rho_b_comm[cbm - Lb:cbm + Lb + 1, cbm - Lb:cbm + Lb + 1]
    pred_b = rho_b_model[5 - Lb:5 + Lb + 1, 5 - Lb:5 + Lb + 1]
    okc = np.isfinite(meas_comm)
    xcheck = float(np.nanmax(np.abs(
        pred_b - np.where(okc, meas_comm, pred_b))))
    xcheck_pass = bool(xcheck <= XCHECK_GATE)

    # informational: against the v3b PRODUCT ACF (its own detrending)
    cb = Lb + 6
    meas_prod = m3b["rho_meas"][cb - Lb:cb + Lb + 1, cb - Lb:cb + Lb + 1]
    okp = np.isfinite(meas_prod)
    xinfo = float(np.nanmax(np.abs(
        pred_b - np.where(okp, meas_prod, pred_b))))

    all_pass &= xcheck_pass
    report["blocksum_crosscheck"] = dict(
        description=("binned_kernel_from_fine(fitted v3 kernel) vs the "
                     "binned ACF measured from the block-summed fine "
                     "detrended residual (commuting processing) over the "
                     "L=4 window"),
        max_abs_diff=xcheck, gate=XCHECK_GATE, gate_pass=xcheck_pass,
        pred_axis=[float(rho_b_model[5, 5 + d]) for d in range(0, Lb + 1)],
        meas_axis_commuting=[float(meas_comm[Lb, Lb + d])
                             for d in range(0, Lb + 1)],
        meas_axis_v3b_product=[float(meas_prod[Lb, Lb + d])
                               for d in range(0, Lb + 1)],
        info_vs_v3b_product_acf=dict(
            max_abs_diff=xinfo,
            note=("systematically higher prediction (~0.03-0.06 at all "
                  "lags): Background2D detrending does not commute with "
                  "2x2 block-summing — the per-product detrenders remove "
                  "different pedestal fractions. The per-product KERNELS "
                  "are fit to each product's own processed ACF (the object "
                  "the likelihood needs), so this non-commutation does not "
                  "enter the likelihood; the commuting-processing check "
                  "above isolates and validates the block-sum transform "
                  "itself.")),
        var_blocksum_inflation_measured=float(
            binned_kernel_from_fine(
                np.nan_to_num(m3["rho_meas"]))[6, 6] / 4.0),
        var_blocksum_inflation_40c_naive_chi2=3.766,
    )
    print(f"\nblock-sum cross-check (commuting processing): "
          f"max|pred-meas| = {xcheck:.4f} "
          f"-> {'PASS' if xcheck_pass else 'FAIL'} (<= {XCHECK_GATE})")
    print(f"  informational vs v3b product ACF (non-commuting detrend): "
          f"{xinfo:.4f}")

    report["all_gates_pass"] = bool(all_pass)
    report["wall_s"] = time.time() - t0
    out_json = REPRO / args.out_json
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_json}")
    print(f"ALL 02 GATES {'PASS' if all_pass else 'FAIL'} "
          f"({report['wall_s']:.0f}s)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
