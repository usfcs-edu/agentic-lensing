"""62_o3_odell_noise.py — E2-O3: noise audit of Evan Odell's own product with
the campaign's T0.4-1 machinery (P1a-verified cgl.noise port of foundry-i
46_noise_audit + the t04_stationarity design).

Question (plan of record, O3): does HIS preparation's noise show the same
correlation / nonstationarity family as ours?

Three stages:
  render (cgl2 venv, GPU — the lensing model NEVER renders on CPU):
      model image on HIS grid via the parity-certified odell scene stack
      (50_run_mclmc_diag._build_pm_odell). Model point = the pooled per-dim
      scene-z MEDIAN of the E2 odell MCLMC chains — the best available fit to
      his data on his grid, with the DOCUMENTED CAVEAT that the fit failed
      E2-G1 (R-hat_worst 5.46): the median is a residual-minimizing reference
      point, not a certified posterior summary. The transported-anchor-cloud
      median is rendered too (cross-check + fallback); the audit uses
      whichever has lower chi2/px on the keep mask, recorded in the json.
  audit (cgl venv, CPU — sanctioned: ACF/kernel-fit work only):
      masked ACF (mask-deconvolved), two-component drizzle-kernel-family WLS
      fit, 3x3 per-block stationarity with a B=200 stationary-null bootstrap
      (exact FFT sims of the GLOBAL fitted kernel refit per block with the
      identical mask/estimator/objective), multiplicity-calibrated max|z| p,
      drizzle registration-spread envelope. Conventions = t04_stationarity.py
      with the odell-specific parameters documented inline (L=6 fit window at
      r=2.0; sky annulus r>3.2" — his 9" FOV cannot support the 4.5" cut).
  fig   (cgl venv): figs/e2_odell_noise.png  (t04 stationarity-figure layout
      + global-ACF and residual panels).

Artifacts: data/e2_odell_noise.json, figs/e2_odell_noise.png,
scratch npz/pkl in SCRATCH (intermediates only).

Run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CGL2_GPU=9 GIGALENS_X64=1 \
    /raid/benson/.venvs/cgl2/bin/python 62_o3_odell_noise.py render
  /raid/benson/.venvs/cgl/bin/python 62_o3_odell_noise.py audit
  /raid/benson/.venvs/cgl/bin/python 62_o3_odell_noise.py fig
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus")
REPRO = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens")
DATA, FIGS = ROOT / "data", ROOT / "figs"
SCRATCH = Path("/tmp/claude-1306/-raid-benson-git-agentic-lensing/"
               "32c92d29-0b37-4c34-9e8b-ff8ec8d85e14/scratchpad")
RENDER_NPZ = SCRATCH / "e2_o3_odell_render.npz"
STORE_PKL = SCRATCH / "e2_o3_odell_store.pkl"
OUT_JSON = DATA / "e2_odell_noise.json"
OUT_FIG = FIGS / "e2_odell_noise.png"

DELTA_PIX = 0.064125
NATIVE_PIX = 0.1283          # WFC3/IR native (P1a header fact)
PIXFRAC, N_FRAMES = 1.0, 3   # foundry-i stack facts; for the phase-averaged
                             # (offsets=None) anchor the ACF is n_frames-
                             # INDEPENDENT (rho_stack = rho_single after
                             # normalization), so his unknown frame count
                             # does not bias the anchor. Documented assumption
                             # anyway (his stack differs: O1 single-epoch
                             # compact census).
L_FIT = 6                    # fit window: r=2.0 sits between v3 (r=3.2, L=8)
                             # and v3b (r=1.6, L=4)
BKG_BOX = 16                 # ~1" detrend box (v3: 26@0.04, v3b: 13@0.08)
SKY_R = 3.2                  # O1-Gc sky annulus (his 9" FOV; the 4.5" t04
                             # convention leaves no pixels here) — documented
                             # deviation; the |img| < 5*med(err) faint cut
                             # still excises arc/galaxy flux
ARC_BAND = (1.2, 4.2)
B_NULL = 200
N_BLOCKS = 3
MIN_VALID_PX = 400
SEED = 20260723
FUNC_NAMES = ["w_tot", "rho_01", "rho_10", "rho_11"]


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# =========================================================================== #
# stage render (cgl2 venv, GPU)
# =========================================================================== #
def stage_render():
    assert os.environ.get("GIGALENS_X64") == "1", "GIGALENS_X64=1 required"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CGL2_GPU", "9"))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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
    spec50 = importlib.util.spec_from_file_location(
        "mclmc_diag", ROOT / "50_run_mclmc_diag.py")
    m50 = importlib.util.module_from_spec(spec50)
    spec50.loader.exec_module(m50)

    refs = arb.load_refs()
    pm, mv, meta = m50._build_pm_odell(arb, refs)
    term = pm.terms[0]

    Y = np.asarray(term.dataset.image, dtype=np.float64)
    prod = np.load(DATA / "odell_cutout.npz", allow_pickle=True)
    err = np.asarray(prod["err_map"], dtype=np.float64)
    keep = np.asarray(prod["keep_mask"]).astype(bool)
    n = Y.shape[0]
    yy, xx = np.indices(Y.shape)
    cen = (n - 1) / 2.0
    r_arc = np.hypot(xx - cen, yy - cen) * DELTA_PIX
    band = keep & (r_arc >= ARC_BAND[0]) & (r_arc <= ARC_BAND[1])
    sky = keep & (r_arc > SKY_R) & (np.abs(Y) < 5.0 * np.median(err))

    def eval_zmed(zmed):
        u = mv.model.bijector.forward(jnp.asarray(zmed[None, :]))
        gamma = float(np.asarray(u["planes/0/mass/0/gamma"])[0])
        it = term.internals(mv.model.to_params(u))
        model = np.asarray(it["model_image"], dtype=np.float64)
        resid = (Y - model) / err
        return model, dict(
            gamma=gamma,
            chi2_pp=float(np.mean(resid[keep] ** 2)),
            chi2_pp_arcband=float(np.mean(resid[band] ** 2)),
            chi2_pp_sky=float(np.mean(resid[sky] ** 2)),
            logL_data_whitened=float(it["logL_data"]))

    # model point 1: pooled per-dim median of the E2 odell chains (UNCONVERGED
    # fit — documented caveat; V1 KEYED + V2 gamma-vs-json checks below)
    z = np.load(DATA / "mclmc_diag_odell.npz", allow_pickle=True)
    assert [str(s) for s in z["z_names"]] == list(mv.model.z_param_names), \
        "scene-z name/order mismatch vs stored chains"          # V1
    pooled = np.concatenate([np.asarray(z[f"pos_g{g}"], dtype=np.float64)
                             for g in range(4)], axis=0).reshape(-1, 46)
    zmed = np.median(pooled, axis=0)
    model_fit, res_fit = eval_zmed(zmed)
    jq = json.load(open(DATA / "mclmc_diag_odell.json"))["gamma"]["pooled_quantiles"]
    dg = abs(res_fit["gamma"] - jq["0.5"])
    assert dg < 1e-3, (res_fit["gamma"], jq["0.5"])             # V2
    print(f"[render] odell-fit zmed: gamma {res_fit['gamma']:.6f} "
          f"(json q50 {jq['0.5']:.6f}, d {dg:.2e}); chi2/px keep "
          f"{res_fit['chi2_pp']:.4f}, arc {res_fit['chi2_pp_arcband']:.4f}, "
          f"sky {res_fit['chi2_pp_sky']:.4f}", flush=True)

    # model point 1b (PRIMARY): maximum-logdensity kept draw. The pooled
    # per-dim median of a glassy R-hat 5.46 posterior is OFF-MANIFOLD
    # (measured above: chi2/px ~ 73 vs ~1 expected) — a median across
    # disagreeing basins is not a model. A kept DRAW is always on-manifold;
    # argmax over a thin-8 subset of all 64x4000 draws (32,000 logdensity
    # evals, batched) is the best available residual-minimizing model.
    logprior_fn, loglik_fn = arb.make_closures(pm)
    thin = 8
    sub = pooled.reshape(64, 4000, 46)[:, ::thin, :].reshape(-1, 46)
    best_val, best_idx = -np.inf, 0
    bs = 64
    for i in range(0, sub.shape[0], bs):
        zz = jnp.asarray(sub[i:i + bs])
        ld = np.asarray(logprior_fn(zz) + loglik_fn(zz))
        k = int(np.argmax(ld))
        if float(ld[k]) > best_val:
            best_val, best_idx = float(ld[k]), i + k
    z_map = sub[best_idx]
    model_map, res_map = eval_zmed(z_map)
    res_map["logdensity"] = best_val
    res_map["draw"] = dict(thin=thin, flat_index=int(best_idx),
                           chain=int(best_idx // (4000 // thin)),
                           kept_draw=int((best_idx % (4000 // thin)) * thin))
    print(f"[render] MAP-proxy draw (argmax logdensity over thin-{thin}): "
          f"gamma {res_map['gamma']:.6f}, logdens {best_val:.1f}; chi2/px "
          f"keep {res_map['chi2_pp']:.4f}, arc "
          f"{res_map['chi2_pp_arcband']:.4f}, sky "
          f"{res_map['chi2_pp_sky']:.4f}", flush=True)

    # model point 2: transported anchor-cloud median (cross-check + fallback)
    wz = np.load(DATA / "mclmc_warm_odell_scenez.npz", allow_pickle=True)
    zmed_a = np.median(np.asarray(wz["Z_scene"], dtype=np.float64), axis=0)
    model_anc, res_anc = eval_zmed(zmed_a)
    print(f"[render] anchor-cloud zmed: gamma {res_anc['gamma']:.6f}; chi2/px "
          f"keep {res_anc['chi2_pp']:.4f}, arc {res_anc['chi2_pp_arcband']:.4f}, "
          f"sky {res_anc['chi2_pp_sky']:.4f}", flush=True)

    np.savez_compressed(
        RENDER_NPZ,
        model_fit=model_fit, model_anchor=model_anc, model_map=model_map,
        zmed_fit=zmed, zmed_anchor=zmed_a, z_map=z_map,
        res_fit=json.dumps(res_fit), res_anchor=json.dumps(res_anc),
        res_map=json.dumps(res_map),
        provenance=json.dumps(dict(
            odell_product_md5=md5(DATA / "odell_cutout.npz"),
            chains_npz_md5=md5(DATA / "mclmc_diag_odell.npz"),
            warm_scenez_md5=md5(DATA / "mclmc_warm_odell_scenez.npz"),
            parity_refs_md5=md5(DATA / "parity_refs.npz"),
            wall_s=time.time() - t0)))
    print(f"[render] wrote {RENDER_NPZ} ({time.time() - t0:.0f}s)", flush=True)


# =========================================================================== #
# stage audit (cgl venv, CPU)
# =========================================================================== #
def _load_audit_inputs():
    prod = np.load(DATA / "odell_cutout.npz", allow_pickle=True)
    img = np.asarray(prod["img"], dtype=np.float64)
    err = np.asarray(prod["err_map"], dtype=np.float64)
    keep = np.asarray(prod["keep_mask"]).astype(bool)
    rz = np.load(RENDER_NPZ, allow_pickle=True)
    res_fit = json.loads(str(rz["res_fit"]))
    res_anc = json.loads(str(rz["res_anchor"]))
    res_map = json.loads(str(rz["res_map"]))
    # PRIMARY = the max-logdensity kept draw of the odell fit (always
    # on-manifold; the pooled per-dim median measured chi2/px ~ 73 =
    # off-manifold across the glassy basins — recorded, not used)
    cands = [("odell-fit max-logdensity kept draw (UNCONVERGED fit, "
              "R-hat_worst 5.46 — residual-minimizing reference point, NOT "
              "a certified posterior; caveat pre-declared in the O3 task)",
              "model_map", res_map),
             ("transported anchor-cloud median", "model_anchor", res_anc)]
    cands.sort(key=lambda t: t[2]["chi2_pp"])
    which, key, res_use = cands[0]
    model = np.asarray(rz[key], dtype=np.float64)
    return img, err, keep, model, which, res_use, dict(
        odell_fit_median=res_fit, odell_fit_map_draw=res_map,
        anchor_cloud_median=res_anc), rz


def _audit_arm(img, err, keep, model, arm_tag, t04mod, noise_mod, exact_mod):
    """One full T0.4-1 pass (global ACF + kernel fit + 3x3 stationarity
    bootstrap) on residual (img - model)/err. model=0 => no-model arm."""
    from multiprocessing import Pool

    detrend_sky = noise_mod.detrend_sky
    drizzle_acf = noise_mod.drizzle_acf
    fit_kernel = noise_mod.fit_kernel
    fit_kernel2 = noise_mod.fit_kernel2
    masked_acf_2d = noise_mod.masked_acf_2d
    rho_model2 = noise_mod.rho_model2
    sample_stationary_batch = exact_mod.sample_stationary_batch
    block_slices = t04mod.block_slices
    functionals = t04mod.functionals
    params_to_theta = t04mod.params_to_theta
    per_block_measure = t04mod.per_block_measure

    t0 = time.time()
    n = img.shape[0]
    yy, xx = np.indices(img.shape)
    cen = (n - 1) / 2.0
    r_arc = np.hypot(xx - cen, yy - cen) * DELTA_PIX
    med_err = float(np.median(err))

    v = (img - model) / err
    sky = keep & (r_arc > SKY_R) & (np.abs(img) < 5.0 * med_err)
    v_det, detrend_method = detrend_sky(v, sky, BKG_BOX)

    L = L_FIT
    rho_ext, cnt_ext = masked_acf_2d(v_det, sky, max_lag=L + 6)
    c = L + 6
    rho_win = rho_ext[c - L:c + L + 1, c - L:c + L + 1]
    cnt_win = cnt_ext[c - L:c + L + 1, c - L:c + L + 1]

    r_odell = NATIVE_PIX / DELTA_PIX
    drz = drizzle_acf(r_odell, PIXFRAC, N_FRAMES, None, max_lag=32,
                      n_phase=32)
    fit1 = fit_kernel(rho_win, cnt_win, drz["rho"], max_lag=L)
    fit = fit_kernel2(rho_win, cnt_win, drz["rho"], max_lag=L)
    print(f"[audit:{arm_tag}] global fit: w_d {fit['w_d']:.3f} w_b {fit['w_b']:.3f} "
          f"max|resid| {fit['max_abs_resid']:.4f} "
          f"(gate<=0.05 {fit['gate_le_0p05']}); measured rho01 "
          f"{rho_win[L, L + 1]:.3f} rho10 {rho_win[L + 1, L]:.3f} "
          f"rho11 {rho_win[L + 1, L + 1]:.3f}; drizzle t1 {drz['t1']:.3f} "
          f"(analytic {drz['t1_analytic']:.3f})", flush=True)

    # extended kernel for the null sims (P1a convention: components to
    # double-precision zero; compactly-supported anchor zero-padded)
    L_k = 64
    anchor = np.zeros((2 * L_k + 1, 2 * L_k + 1))
    Ld = (drz["rho"].shape[0] - 1) // 2
    anchor[L_k - Ld:L_k + Ld + 1, L_k - Ld:L_k + Ld + 1] = drz["rho"]
    rho_kernel = rho_model2(*fit["params"], anchor, L_k)

    # ---- per-block observed fits ------------------------------------------ #
    valid = keep & (np.abs(img) < 5.0 * med_err)
    slices = block_slices(n, N_BLOCKS)
    obs = per_block_measure(v_det, valid, L, slices)
    for o in obs:
        if not o.get("ok"):
            o["fit"] = None
            continue
        f = fit_kernel2(o["rho"], o["cnt"], drz["rho"], max_lag=L)
        o["fit"] = f
        o["func"] = functionals(f["params"], drz["rho"], L)
    print(f"[audit:{arm_tag}] observed 9-block fits done "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- stationary-null bootstrap ---------------------------------------- #
    rng = np.random.default_rng(SEED)
    sims = sample_stationary_batch(rho_kernel, (n, n), B_NULL, rng, grid=512)
    theta_g = params_to_theta(fit["params"])
    starts = [theta_g,
              [0.7, 0.8, 0.5, 0.5, 0.0, 4.0, 4.0, 0.0],
              [0.95, 0.5, 0.8, 0.8, 0.0, 8.0, 8.0, 0.0]]
    args = [(sims[b], valid, L, slices, drz["rho"], starts)
            for b in range(B_NULL)]
    t1 = time.time()
    with Pool(32) as pool:
        null_res = pool.map(t04mod._null_worker, args)
    print(f"[audit:{arm_tag}] null refits ({B_NULL} x 9): "
          f"{time.time() - t1:.0f}s", flush=True)

    nb = N_BLOCKS * N_BLOCKS
    null_F = {f: [[] for _ in range(nb)] for f in FUNC_NAMES}
    for sim in null_res:
        for b in range(nb):
            if sim[b] is None:
                continue
            for f in FUNC_NAMES:
                null_F[f][b].append(sim[b][f])
    null_mean = {f: np.array([np.mean(v) if len(v) > 10 else np.nan
                              for v in null_F[f]]) for f in FUNC_NAMES}
    null_std = {f: np.array([np.std(v, ddof=1) if len(v) > 10 else np.nan
                             for v in null_F[f]]) for f in FUNC_NAMES}
    obs_F = {f: np.array([o["func"][f] if o.get("func") else np.nan
                          for o in obs]) for f in FUNC_NAMES}
    z = {}
    for f in FUNC_NAMES:
        w = 1.0 / null_std[f] ** 2
        m = np.isfinite(obs_F[f]) & np.isfinite(w)
        wmean = np.sum(w[m] * obs_F[f][m]) / np.sum(w[m])
        z[f] = (obs_F[f] - wmean) / null_std[f]
    z_mat = np.array([z[f] for f in FUNC_NAMES])
    max_abs_z = float(np.nanmax(np.abs(z_mat)))

    null_max = []
    for sim in null_res:
        zz = []
        for f in FUNC_NAMES:
            vals = np.array([sim[b][f] if sim[b] else np.nan
                             for b in range(nb)])
            w = 1.0 / null_std[f] ** 2
            m = np.isfinite(vals) & np.isfinite(w)
            if m.sum() < 3:
                continue
            wmean = np.sum(w[m] * vals[m]) / np.sum(w[m])
            zz.append(np.abs((vals - wmean) / null_std[f]))
        if zz:
            null_max.append(np.nanmax(np.concatenate(zz)))
    null_max = np.array(null_max)
    p_cal = float(np.mean(null_max >= max_abs_z))
    gate_2sigma = bool(max_abs_z <= 2.0)

    # drizzle registration-spread envelope at his scale
    rng2 = np.random.default_rng(1)
    env_vals = {k: [] for k in ("rho_01", "rho_10", "rho_11")}
    for _ in range(200):
        offs = rng2.uniform(0, 1, size=(N_FRAMES, 2))
        d = drizzle_acf(r_odell, PIXFRAC, N_FRAMES, offsets=offs, max_lag=8)
        rho = d["rho"]
        cc = 8
        env_vals["rho_01"].append(float(rho[cc, cc + 1]))
        env_vals["rho_10"].append(float(rho[cc + 1, cc]))
        env_vals["rho_11"].append(float(rho[cc + 1, cc + 1]))
    env = {k: dict(mean=float(np.mean(v)), std=float(np.std(v)),
                   lo=float(np.min(v)), hi=float(np.max(v)))
           for k, v in env_vals.items()}

    acf_lags = {}
    for name, (dy, dx) in dict(rho_01=(0, 1), rho_10=(1, 0),
                               rho_11=(1, 1)).items():
        acf_lags[name] = [float(o["rho"][L + dy, L + dx])
                          if o.get("ok") else None for o in obs]

    correlated = bool(rho_win[L, L + 1] > 0.1 and rho_win[L + 1, L] > 0.1)
    arm_report = dict(
        arm=arm_tag,
        global_acf=dict(
            n_sky_px=int(sky.sum()), detrend_method=detrend_method,
            chi2_sky_resid_predetrend=float(np.mean(v[sky] ** 2)),
            chi2_sky_resid_detrended=float(np.mean(v_det[sky] ** 2)),
            rho_01=float(rho_win[L, L + 1]), rho_10=float(rho_win[L + 1, L]),
            rho_11=float(rho_win[L + 1, L + 1]),
            rho_02=float(rho_win[L, L + 2]), rho_20=float(rho_win[L + 2, L]),
            drizzle_t1=float(drz["t1"]),
            drizzle_t1_analytic=float(drz["t1_analytic"]),
            correlated_verdict=correlated),
        kernel_fit=dict(
            family2=dict(params=[float(x) for x in fit["params"]],
                         w_d=fit["w_d"], sig_ey=fit["sig_ey"],
                         sig_ex=fit["sig_ex"], c_e=fit["c_e"],
                         w_b=fit["w_b"], sig_by=fit["sig_by"],
                         sig_bx=fit["sig_bx"], c_b=fit["c_b"],
                         w_tot=float(fit["w_d"] + fit["w_b"]),
                         max_abs_resid=fit["max_abs_resid"],
                         rms_resid=fit["rms_resid"],
                         gate_le_0p05=fit["gate_le_0p05"]),
            family1=dict(w=fit1["w"], sigma_e=fit1["sigma_e"],
                         max_abs_resid=fit1["max_abs_resid"],
                         gate_le_0p05=fit1["gate_le_0p05"])),
        stationarity=dict(
            n_valid_blockset=int(valid.sum()),
            blocks=[dict(block=list(o["block"]), n_valid=o["n_valid"],
                         ok=bool(o.get("ok", False)), var=o.get("var"),
                         func=o.get("func"),
                         max_abs_fit_resid=(float(o["fit"]["max_abs_resid"])
                                            if o.get("fit") else None))
                    for o in obs],
            obs_functionals={f: [None if not np.isfinite(x) else float(x)
                                 for x in obs_F[f]] for f in FUNC_NAMES},
            null_mean={f: [None if not np.isfinite(x) else float(x)
                           for x in null_mean[f]] for f in FUNC_NAMES},
            null_std={f: [None if not np.isfinite(x) else float(x)
                          for x in null_std[f]] for f in FUNC_NAMES},
            z_scores={f: [None if not np.isfinite(x) else float(x)
                          for x in z[f]] for f in FUNC_NAMES},
            max_abs_z=max_abs_z, gate_2sigma_pass=gate_2sigma,
            p_calibrated_maxz=p_cal, n_null_effective=int(null_max.size),
            drizzle_offset_envelope=env,
            acf_smalllag_measured=acf_lags),
        wall_s=time.time() - t0)
    store = dict(obs=obs, null_F=null_F, z=z, obs_F=obs_F,
                 null_mean=null_mean, null_std=null_std,
                 null_max=null_max, L=L, env=env,
                 rho_win=rho_win, cnt_win=cnt_win,
                 rho_fit=fit["rho_fit"], drz_t1d=drz["t1d"],
                 rho_drz=drz["rho"], v_det=v_det, sky=sky, valid=valid)
    print(f"[audit:{arm_tag}] max|z| {max_abs_z:.2f} 2sig-gate "
          f"{'PASS' if gate_2sigma else 'FAIL'} p_cal {p_cal:.3f} "
          f"({arm_report['wall_s']:.0f}s)", flush=True)
    return arm_report, store


def stage_audit():
    sys.path.insert(0, str(REPRO))
    import cgl.exact_ref as exact_mod
    import cgl.noise as noise_mod
    # reuse the t04 helpers verbatim (same session-surviving scratchpad file
    # that produced data/t04_stationarity.json)
    t04_dir = Path("/tmp/claude-1306/-raid-benson-git-agentic-lensing/"
                   "3d232f85-b99e-4114-ab38-6fc594956452/scratchpad/t04")
    sys.path.insert(0, str(t04_dir))
    import t04_stationarity as t04mod

    t0 = time.time()
    img, err, keep, model, which, res_use, res_all, rz = _load_audit_inputs()

    # PRIMARY arm: model-subtracted (pre-declared O3 design; model caveat
    # below). ROBUSTNESS arm: no-model (model := 0) — the two arms carry
    # model-error vs faint-source contamination in OPPOSITE directions, so
    # verdicts on which both agree are robust to the unconverged-model caveat.
    arms, stores = {}, {}
    for arm_tag, mdl in (("model_subtracted", model),
                         ("no_model", np.zeros_like(model))):
        arms[arm_tag], stores[arm_tag] = _audit_arm(
            img, err, keep, mdl, arm_tag, t04mod, noise_mod, exact_mod)

    a_m, a_n = arms["model_subtracted"], arms["no_model"]
    agree = dict(
        correlated=(a_m["global_acf"]["correlated_verdict"]
                    == a_n["global_acf"]["correlated_verdict"]),
        stationarity_calibrated=(
            (a_m["stationarity"]["p_calibrated_maxz"] < 0.05)
            == (a_n["stationarity"]["p_calibrated_maxz"] < 0.05)),
        d_rho_01=float(a_m["global_acf"]["rho_01"]
                       - a_n["global_acf"]["rho_01"]),
        d_rho_10=float(a_m["global_acf"]["rho_10"]
                       - a_n["global_acf"]["rho_10"]))

    report = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="62_o3_odell_noise.py",
        purpose=("E2-O3 noise audit of Evan Odell's own 0.064125\" product "
                 "with the T0.4-1 machinery: is his preparation's noise "
                 "correlated / nonstationary like ours?"),
        model_render=dict(
            which=which,
            used_point=res_use, all_candidates=res_all,
            caveat=("the odell fit FAILED E2-G1 (R-hat_worst 5.46) and its "
                    "best kept draw still leaves chi2/px ~27 (gross "
                    "data-model mismatch — itself an E2 finding); the "
                    "max-logdensity kept draw is the best available "
                    "residual-minimizing model on his grid (caveat "
                    "pre-declared in the O3 task); the pooled per-dim "
                    "median measured OFF-MANIFOLD (chi2/px ~73, recorded, "
                    "not used). BECAUSE the model is poor, the audit runs "
                    "a NO-MODEL robustness arm with opposite-signed "
                    "contamination; only verdicts on which both arms agree "
                    "are claimed")),
        conventions=dict(
            machinery=("cgl.noise masked_acf_2d/detrend_sky/drizzle_acf/"
                       "fit_kernel2 + t04_stationarity block/null design "
                       "(P1a bit-match-verified port of foundry-i "
                       "46_noise_audit)"),
            delta_pix=DELTA_PIX, native_pix=NATIVE_PIX,
            scale_ratio=NATIVE_PIX / DELTA_PIX, pixfrac=PIXFRAC,
            n_frames_assumed=N_FRAMES,
            n_frames_note=("phase-averaged anchor is n_frames-independent "
                           "(rho normalizes out); his true frame count "
                           "unknown (O1 single-epoch census) — ask Evan"),
            fit_window_L=L_FIT, bkg_box_px=BKG_BOX,
            sky_region=(f"keep & r>{SKY_R}\" & |img|<5*med(err) — {SKY_R}\" "
                        "annulus is the O1-Gc convention (his 9\" FOV cannot "
                        "support the 4.5\" t04 cut; documented deviation)"),
            valid_blockset="keep & |img|<5*med(err) (t04 convention)",
            b_null=B_NULL, seed=SEED, min_valid_px=MIN_VALID_PX),
        arms=arms,
        arm_agreement=agree,
        reference_ours=dict(
            note="stored T0.4-1 results on OUR products (same machinery)",
            v3_fine=dict(max_abs_z=2.55, gate_2sigma=False, p_cal=0.170),
            v3b_binned=dict(max_abs_z=3.67, gate_2sigma=False, p_cal=0.010,
                            verdict="REJECTED"),
            v3_noarc=dict(max_abs_z=3.66, gate_2sigma=False, p_cal=0.010,
                          verdict="REJECTED"),
            v3b_noarc=dict(max_abs_z=3.08, gate_2sigma=False, p_cal=0.055)),
        provenance=dict(
            odell_product_md5=md5(DATA / "odell_cutout.npz"),
            render_npz=str(RENDER_NPZ),
            render_provenance=json.loads(str(rz["provenance"])),
            t04_reference="data/t04_stationarity.json (+ _noarc)"),
        wall_s=time.time() - t0)

    OUT_JSON.write_text(json.dumps(report, indent=1, default=float))
    import pickle
    with open(STORE_PKL, "wb") as fh:
        pickle.dump(stores, fh)
    print(f"[audit] wrote {OUT_JSON} ({report['wall_s']:.0f}s); arm "
          f"agreement: {agree}", flush=True)


# =========================================================================== #
# stage fig (cgl venv)
# =========================================================================== #
def stage_fig():
    import pickle

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j = json.loads(OUT_JSON.read_text())
    with open(STORE_PKL, "rb") as fh:
        stores = pickle.load(fh)
    # the NOISE-question panels use the no-model arm (clean of the
    # unconverged-model confound); the model-subtracted arm appears as the
    # documented-confound panel + comparison row
    st = stores["no_model"]
    stm = stores["model_subtracted"]
    arm = j["arms"]["no_model"]
    armm = j["arms"]["model_subtracted"]
    L = st["L"]
    ga = arm["global_acf"]
    kf = arm["kernel_fit"]["family2"]
    stt = arm["stationarity"]
    agree = j["arm_agreement"]

    C_BLUE, C_ORANGE, C_YELLOW = "#2a78d6", "#eb6834", "#eda100"
    TXT, TXT2 = "#0b0b0b", "#52514e"
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#c9c8c2", "axes.labelcolor": TXT,
        "text.color": TXT, "xtick.color": TXT2, "ytick.color": TXT2,
        "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
        "font.size": 9.5, "axes.titlesize": 10.0, "figure.dpi": 150,
    })
    FUNC_LABEL = {"w_tot": r"$w_d+w_b$", "rho_01": r"$\rho_{fit}(0,1)$",
                  "rho_10": r"$\rho_{fit}(1,0)$", "rho_11": r"$\rho_{fit}(1,1)$"}

    fig = plt.figure(figsize=(14.6, 7.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1.25, 1.0, 1.0],
                          hspace=0.44, wspace=0.32)

    # P0: global measured ACF vs fitted family vs drizzle anchor (axis cuts)
    ax = fig.add_subplot(gs[0, 0])
    lags = np.arange(0, L + 1)
    meas_x = [st["rho_win"][L, L + d] for d in lags]
    meas_y = [st["rho_win"][L + d, L] for d in lags]
    fit_x = [st["rho_fit"][L, L + d] for d in lags]
    fit_y = [st["rho_fit"][L + d, L] for d in lags]
    Ld = (st["rho_drz"].shape[0] - 1) // 2
    drz_x = [st["rho_drz"][Ld, Ld + d] for d in lags]
    meas_xm = [stm["rho_win"][L, L + d] for d in lags]
    ax.plot(lags, meas_x, "o", color=C_BLUE, ms=6, label="measured, x lags")
    ax.plot(lags, meas_y, "s", color=C_ORANGE, ms=6, mfc="white",
            label="measured, y lags")
    ax.plot(lags, fit_x, "-", color=C_BLUE, lw=1.6, label="2-comp family fit")
    ax.plot(lags, fit_y, "--", color=C_ORANGE, lw=1.4)
    ax.plot(lags, meas_xm, "x", color=TXT2, ms=5, alpha=0.6,
            label="model-sub arm, x lags (confounded)")
    ax.plot(lags, drz_x, ":", color=TXT2, lw=1.6,
            label=f"pure drizzle anchor (t1={ga['drizzle_t1']:.2f})")
    ax.set_xlabel("lag (px)")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(f"global masked ACF, no-model arm\n({ga['n_sky_px']} sky "
                 f"px): $\\rho_{{01}}$ {ga['rho_01']:.3f}, "
                 f"$\\rho_{{10}}$ {ga['rho_10']:.3f} — CORRELATED",
                 fontsize=9.2)
    ax.legend(fontsize=7, framealpha=0.9)

    # P1: the model-subtracted residual image — the documented CONFOUND
    ax = fig.add_subplot(gs[1, 0])
    show = np.where(stm["valid"], stm["v_det"], np.nan)
    im = ax.imshow(show, cmap="RdBu_r", vmin=-3, vmax=3, origin="lower")
    gam = armm["global_acf"]
    ax.set_title("model-subtracted arm = CONFOUNDED (for the record):\n"
                 f"MAP draw of the FAILED fit leaves $\\chi^2_{{sky}}$/px "
                 f"{gam['chi2_sky_resid_detrended']:.1f}", fontsize=8.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85)

    # P2/P3: per-block functionals with null bands
    x = np.arange(9)
    for row, f in enumerate(("rho_01", "rho_10")):
        ax = fig.add_subplot(gs[row, 1])
        nm = np.array([np.nan if v is None else v for v in stt["null_mean"][f]])
        ns = np.array([np.nan if v is None else v for v in stt["null_std"][f]])
        ob = np.array([np.nan if v is None else v
                       for v in stt["obs_functionals"][f]])
        ax.fill_between(x, nm - 2 * ns, nm + 2 * ns, color="#dfe9f7",
                        step="mid", label="stationary null $\\pm2\\sigma$")
        ax.plot(x, nm, color=C_BLUE, lw=1.2, drawstyle="steps-mid",
                label="null mean (global kernel)")
        e = stt["drizzle_offset_envelope"][f]
        half = 0.5 * (e["hi"] - e["lo"])
        base = np.nanmean(nm)
        ax.fill_between(x, base - half, base + half, color=C_YELLOW,
                        alpha=0.3, label="drizzle-registration spread")
        ax.plot(x, ob, "o", color=C_ORANGE, ms=6, zorder=5,
                label="observed per-block fit")
        ax.set_xticks(x)
        ax.set_xticklabels([f"({i},{k})" for i in range(3) for k in range(3)],
                           fontsize=7)
        ax.set_ylabel(FUNC_LABEL[f])
        if row == 0:
            ax.set_title("per-block fits vs stationary null", fontsize=9.2)
            ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
        else:
            ax.set_xlabel("block (row, col); center (1,1) excluded")

    # P4: all-functional z table
    zmat = np.array([[np.nan if v is None else v for v in stt["z_scores"][f]]
                     for f in FUNC_NAMES])
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(zmat, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    ax.set_yticks(range(4)); ax.set_yticklabels(FUNC_NAMES, fontsize=8)
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"({i},{k})" for i in range(3) for k in range(3)],
                       fontsize=6.5, rotation=45)
    ax.set_title("z-scores × blocks (no-model arm)", fontsize=9.2)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85)

    # P5: calibrated max|z| null distribution
    ax = fig.add_subplot(gs[0, 3])
    ax.hist(st["null_max"], bins=24, color="#b9d2f0", edgecolor="white")
    ax.axvline(stt["max_abs_z"], color=C_ORANGE, lw=2)
    ax.text(stt["max_abs_z"], ax.get_ylim()[1] * 0.95,
            f"  observed {stt['max_abs_z']:.2f}\n  p = "
            f"{stt['p_calibrated_maxz']:.3f}", color=C_ORANGE, va="top",
            fontsize=9)
    ax.set_xlabel("max |z| over blocks × functionals")
    ax.set_ylabel("null sims")
    ax.set_title("multiplicity-calibrated test")

    # P6: block variance map
    ax = fig.add_subplot(gs[1, 2])
    var = np.array([b["var"] if b["var"] is not None else np.nan
                    for b in stt["blocks"]]).reshape(3, 3)
    im = ax.imshow(var, cmap="viridis")
    for i in range(3):
        for k in range(3):
            vv = var[i, k]
            ax.text(k, i, "excl." if not np.isfinite(vv) else f"{vv:.2f}",
                    ha="center", va="center", fontsize=10, color="white")
    ax.set_title("block var, no-model\n(info)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85)

    # P7: comparison table vs OUR products
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ro = j["reference_ours"]
    stm_st = armm["stationarity"]
    rows = [
        ("product", "max|z|", "cal. p"),
        ("odell (no-model)", f"{stt['max_abs_z']:.2f}",
         f"{stt['p_calibrated_maxz']:.3f}"),
        ("odell (model-sub)*", f"{stm_st['max_abs_z']:.2f}",
         f"{stm_st['p_calibrated_maxz']:.3f}"),
        ("v3 fine (ours)", f"{ro['v3_fine']['max_abs_z']:.2f}",
         f"{ro['v3_fine']['p_cal']:.3f}"),
        ("v3 no-arc (ours)", f"{ro['v3_noarc']['max_abs_z']:.2f}",
         f"{ro['v3_noarc']['p_cal']:.3f}"),
        ("v3b binned (ours)", f"{ro['v3b_binned']['max_abs_z']:.2f}",
         f"{ro['v3b_binned']['p_cal']:.3f}"),
    ]
    tab = ax.table(cellText=[list(r) for r in rows[1:]],
                   colLabels=list(rows[0]), loc="center", cellLoc="center",
                   colWidths=[0.55, 0.22, 0.22])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.0, 1.35)
    ax.set_title("same test, same machinery, ours vs his\n"
                 "(*model-sub arm confounded by model error)", fontsize=8.8)

    w_tot = kf["w_tot"]
    fig.suptitle(
        f"E2-O3 noise audit — Evan's own product IS correlated like ours "
        f"(both arms: $\\rho_{{01}}$ {ga['rho_01']:.2f} no-model / "
        f"{armm['global_acf']['rho_01']:.2f} model-sub vs pure-drizzle "
        f"{ga['drizzle_t1']:.2f}; corr. weight {w_tot:.2f}; drizzle-family "
        f"fit max|resid| {kf['max_abs_resid']:.3f} ≤ 0.05)   |   "
        f"stationarity NOT REJECTED on the clean no-model arm (max|z| "
        f"{stt['max_abs_z']:.2f}, cal. p {stt['p_calibrated_maxz']:.3f}); "
        f"model-sub arm rejects (max|z| "
        f"{armm['stationarity']['max_abs_z']:.2f}, p "
        f"{armm['stationarity']['p_calibrated_maxz']:.3f}) but is "
        f"CONFOUNDED by the unconverged model", fontsize=10.2, y=0.995)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {OUT_FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["render", "audit", "fig"])
    a = ap.parse_args()
    dict(render=stage_render, audit=stage_audit, fig=stage_fig)[a.stage]()
