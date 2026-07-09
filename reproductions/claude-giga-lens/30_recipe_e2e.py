"""30_recipe_e2e.py — the CGL recipe end-to-end on the real HST system.

Demonstrates the full Pillar-1 recipe on the v2d-class HST product of
DESI-165.4754-06.0423 (native 0.13", 80^2, ss2) under the CORRELATED
conv-whitened likelihood (relaxed v2d whitener, D3) — the real-data home of
the P1 machinery. Reuses the P1c production infra in cgl.e2:
  build_target -> make_start (paper MAP) -> map_polish -> laplace_evidence
  -> run_staged (two-stage re-preconditioned PHMC, the P1b recipe).
The paper-MAP seeds are near-optimal, so no Adam warmup is needed (unlike the
cold-start Euclid fits in 31).

Produces the "recipe end-to-end" figure DATA + a 4/5-panel PNG:
  data | MAP model + tangential critical curve | whitened (normalised) residual
  | source-plane reconstruction | gamma posterior.

--sampler defaults to two-stage PHMC; it is the hook for the P2c winner (only
'phmc' is wired here — P2c is not yet delivered). --quick scopes to MAP +
Laplace + a short single-stage HMC when L4 time is tight (the recon's fallback).

Run (pin ONE GPU; L4 8 recommended; never GPU 9 / A16 0-3):
  GIGALENS_X64=1 CUDA_VISIBLE_DEVICES=8 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion" \
  /raid/benson/.venvs/cgl/bin/python 30_recipe_e2e.py --basin low
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2d", choices=["v2d", "v3b", "v3"])
    ap.add_argument("--basin", default="low", choices=["low", "steep"],
                    help="low = the trustworthy gamma~1.43 native basin (P1c anchor)")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--sampler", default="phmc",
                    help="two-stage PHMC (default); hook for the P2c winner")
    ap.add_argument("--chains", type=int, default=24)
    ap.add_argument("--keep", type=int, default=1500)
    ap.add_argument("--stage1-keep", type=int, default=500)
    ap.add_argument("--step-size", type=float, default=0.1)
    ap.add_argument("--num-leapfrog", type=int, default=8,
                    help="leapfrog steps (8 keeps the correlated-model XLA "
                         "compile ~3 min on L4; 16 ~doubles it)")
    ap.add_argument("--map-rounds", type=int, default=4)
    ap.add_argument("--quick", action="store_true",
                    help="MAP + Laplace + short single-stage HMC only")
    ap.add_argument("--ingest", default=None,
                    help="path to a CONVERGED posterior npz to render the "
                         "showcase figure from WITHOUT re-sampling (the ready-to-"
                         "run step for the P1c Job-1 v3b-low posterior). Accepts "
                         "the 10_run_e2.py schema ({basin}_draws + labels [+ "
                         "summary]) or a {draws|low_draws} + labels [+ z_map] npz; "
                         "rebuilds the --tag model, picks the max-logpost kept "
                         "draw (or z_map if present) for the model/critical-curve/"
                         "source panels, uses all draws for the gamma posterior. "
                         "GPU is used only to rebuild+render (no sampler).")
    ap.add_argument("--outdir", default=str(REPRO / "data" / "recipe_e2e"))
    ap.add_argument("--figdir", default=str(REPRO / "figs"))
    return ap.parse_args()


def main():
    args = parse_args()
    if args.sampler != "phmc":
        raise SystemExit(f"--sampler={args.sampler} not wired; only 'phmc' "
                         "(the P2c winner hook is a placeholder). ")
    os.environ["GIGALENS_X64"] = "1"
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion")

    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", str(REPRO / ".jax_cache"))
    import jax.numpy as jnp
    import numpy as np

    from cgl import e2, guards
    guards.require_x64(); guards.require_gpu(); guards.require_single_device()
    print(f"device: {jax.devices()[0]}  tag={args.tag} basin={args.basin}",
          flush=True)

    t_start = time.time()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)

    # ---- correlated marg target (relaxed whitener) + paper-MAP start --------
    target = e2.build_target(args.tag)
    model = target.model
    gi = e2.gamma_index(model); ti = e2.theta_e_index(model)

    if args.ingest:
        return ingest_and_emit(target, model, gi, ti, args, outdir, figdir,
                               t_start, np=np, jnp=jnp)

    z0, rep = e2.make_start(args.tag, args.basin, target)
    print(f"start {rep['artifact']} gamma={rep['gamma']:.4f} "
          f"thetaE={rep['theta_E']:.4f}  whitener={target.whiten_meta}", flush=True)

    mp = e2.map_polish(target, z0, rounds=args.map_rounds, iters=200)
    z_map = mp["z_map"]
    xm = model.to_physical_mass(np.asarray(z_map)[None, :])
    print(f"MAP: logp {mp['logp0']:.1f} -> {mp['logp_map']:.1f} "
          f"gamma_map={float(xm['gamma'][0]):.4f} ({mp['wall_s']:.0f}s)", flush=True)

    lap = e2.laplace_evidence(target, z_map)
    print(f"Laplace: logZ={lap['log_evidence']:.2f} min_eig={lap['min_eig']:.2e} "
          f"n_neg={lap['n_neg']}", flush=True)

    if args.quick:
        st = e2.run_staged(target, z_map, lap["cov"], chains=args.chains, seed=2,
                           stage1_burn=300, stage1_keep=300, burn=200,
                           keep=600, step_size=args.step_size, num_leapfrog=args.num_leapfrog)
        mode = "quick (MAP + Laplace + short two-stage HMC)"
    else:
        st = e2.run_staged(target, z_map, lap["cov"], chains=args.chains, seed=2,
                           stage1_burn=500, stage1_keep=args.stage1_keep,
                           burn=500, keep=args.keep, step_size=args.step_size,
                           num_leapfrog=args.num_leapfrog)
        mode = "full two-stage re-preconditioned PHMC"
    draws = st["draws"]
    print(f"PHMC: R-hat_max={st['rhat'].max():.4f} ess_min={st['ess'].min():.0f} "
          f"ess_gamma={st['ess'][gi]:.0f} accept={st['accept_mean']:.2f} "
          f"({st['wall_s']:.0f}s)", flush=True)

    # ---- posterior mass summary --------------------------------------------
    flat = draws.reshape(-1, draws.shape[-1])
    phys = model.to_physical_mass(flat)
    g = np.asarray(phys["gamma"]); tE = np.asarray(phys["theta_E"])
    gamma_med, gamma_lo, gamma_hi = np.percentile(g, [50, 16, 84])
    print(f"gamma posterior = {gamma_med:.4f} [{gamma_lo:.4f},{gamma_hi:.4f}] "
          f"(P1c native-diag anchor 1.433)", flush=True)

    # ---- figure data: data / model / residual / critical curve / source ----
    Y = np.asarray(model.Y); err = np.asarray(model.masked_err_map)
    keep = np.asarray(model.keep_mask)
    zj = jnp.asarray(z_map, dtype=jnp.float64)
    mimg = np.asarray(model.model_image(zj))
    a_star = np.asarray(model.shapelet_amps(zj), dtype=np.float64)
    resid_norm = np.where(keep, (Y - mimg) / err, np.nan)
    lam_t, extent = crit_field(model, z_map, np=np, jnp=jnp)
    src, src_extent = render_source(model, z_map, np=np, jnp=jnp)
    components = collect_components(model, z_map, np=np, jnp=jnp)

    fig_npz = outdir / f"recipe_{args.tag}_{args.basin}.npz"
    np.savez_compressed(
        fig_npz, data=Y.astype(np.float32), model=mimg.astype(np.float32),
        model_image_map=mimg.astype(np.float32),
        err_map=err.astype(np.float32), keep_mask=keep, psf=np.asarray(
            model.sim_config.kernel, dtype=np.float32), a_star_map=a_star,
        resid_norm=resid_norm.astype(np.float32), keep=keep,
        lam_t=lam_t.astype(np.float32), crit_extent=np.array(extent),
        source=src.astype(np.float32), source_extent=np.array(src_extent),
        gamma_samples=g.astype(np.float32), thetaE_samples=tE.astype(np.float32),
        z_map=z_map, labels=np.array(list(model.index_labels)),
        draws=draws.astype(np.float32))

    summary = dict(
        tag=args.tag, basin=args.basin, mode=mode, sampler=args.sampler,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        whitener=target.whiten_meta, ndim=int(model.ndim), config=vars(args),
        # COOLEST-export fields (consumed by 32_coolest_export.py)
        components=components,
        meta=dict(delta_pix=float(target.cfg["delta_pix"]),
                  num_pix=int(Y.shape[0]), supersample=int(target.cfg["supersample"]),
                  band="HST-F140W-v2d", mag_zero_point=None, subset="hst",
                  system="DESI-165.4754-06.0423"),
        redshifts=dict(z_lens=0.5, z_source=1.0,
                       note="placeholder redshifts; the P1 result is the angular "
                            "mass slope gamma + theta_E, not distances"),
        posterior_mass=dict(
            theta_E=dict(median=float(np.median(tE)), p16=float(np.percentile(tE, 16)),
                         p84=float(np.percentile(tE, 84)), mean=float(np.mean(tE))),
            gamma=dict(median=float(gamma_med), p16=float(gamma_lo),
                       p84=float(gamma_hi), mean=float(np.mean(g)))),
        start=rep, map=dict(logp0=mp["logp0"], logp_map=mp["logp_map"],
                            wall_s=mp["wall_s"]),
        laplace=dict(log_evidence=lap["log_evidence"], min_eig=lap["min_eig"],
                     n_neg=lap["n_neg"]),
        rhat_max=float(st["rhat"].max()), ess_min=float(st["ess"].min()),
        rhat_gamma=float(st["rhat"][gi]), ess_gamma=float(st["ess"][gi]),
        accept_mean=float(st["accept_mean"]), wall_s_sample=st["wall_s"],
        gamma_posterior=dict(median=float(gamma_med), p16=float(gamma_lo),
                             p84=float(gamma_hi), std=float(np.std(g)),
                             native_diag_anchor=1.433),
        thetaE_posterior=dict(median=float(np.median(tE)), std=float(np.std(tE))),
        chi2_pp_diag=float(np.nanmean(resid_norm[keep] ** 2)),
        fig_npz=str(fig_npz))
    summary["wall_s"] = time.time() - t_start
    (outdir / f"recipe_{args.tag}_{args.basin}.json").write_text(
        json.dumps(summary, indent=2, default=float))

    fig_png = figdir / f"recipe_e2e_{args.tag}_{args.basin}.png"
    try:
        make_figure(Y, mimg, resid_norm, lam_t, extent, src, src_extent, g,
                    gamma_med, args, keep, fig_png, np=np)
        print(f"wrote {fig_png}", flush=True)
    except Exception as exc:
        print(f"WARN figure render failed: {exc}", flush=True)

    print(f"\n=== recipe end-to-end ({args.tag}/{args.basin}) ===", flush=True)
    print(f"gamma={gamma_med:.4f} [{gamma_lo:.4f},{gamma_hi:.4f}] "
          f"R-hat={st['rhat'].max():.4f} ess_min={st['ess'].min():.0f} "
          f"chi2_pp={summary['chi2_pp_diag']:.3f}  total {summary['wall_s']:.0f}s",
          flush=True)
    return 0


def load_ingest_draws(path, model, basin, *, np):
    """Load a converged posterior npz -> (draws (T,C,ndim), labels, z_map_or_None).

    Accepts the 10_run_e2.py production schema ({basin}_draws + labels [+ a
    'summary' json string]) OR a plain {draws|<basin>_draws} + labels [+ z_map]
    npz (e.g. this script's own or 31_fit_euclid's). Draws are validated to be in
    the --tag model's 46-dim marg z-space (draws.shape[-1] == model.ndim)."""
    z = np.load(path, allow_pickle=True)
    files = set(z.files)
    if "draws" in files:
        key = "draws"
    elif f"{basin}_draws" in files:
        key = f"{basin}_draws"
    else:
        cand = sorted(k for k in files if k.endswith("_draws"))
        if not cand:
            raise SystemExit(f"{path}: no 'draws' or '*_draws' array (have {files})")
        key = cand[0]
    draws = np.asarray(z[key], dtype=np.float64)
    if draws.ndim == 2:                          # (N, ndim) flat -> (1, N, ndim)
        draws = draws[None]
    if draws.shape[-1] != model.ndim:
        raise SystemExit(f"{path}: draws ndim {draws.shape[-1]} != model.ndim "
                         f"{model.ndim} (wrong --tag?)")
    labels = (np.asarray(z["labels"]) if "labels" in files
              else np.asarray(list(model.index_labels)))
    z_map = np.asarray(z["z_map"], dtype=np.float64) if "z_map" in files else None
    print(f"ingest {path}: draws '{key}' {draws.shape} "
          f"z_map={'present' if z_map is not None else 'derive (max-logpost)'}",
          flush=True)
    return draws, labels, z_map


def ingest_and_emit(target, model, gi, ti, args, outdir, figdir, t_start, *,
                    np, jnp):
    """Render the recipe showcase figure from an ALREADY-CONVERGED posterior npz
    (no sampler). Rebuilds the panels (data | MAP model + tangential critical
    curve | whitened residual | source | gamma posterior) at the max-logpost kept
    draw (or the npz's z_map) and writes the same figure/summary as the sampling
    path. This is the ready-to-run step for the P1c Job-1 v3b-low posterior."""
    draws, labels, z_map = load_ingest_draws(args.ingest, model, args.basin, np=np)
    flat = draws.reshape(-1, draws.shape[-1])

    if z_map is None:
        # max-logpost kept draw (the production npz stores no z_best): batch the
        # log-posterior over a subsample and take the argmax for the point panels.
        idx = np.linspace(0, flat.shape[0] - 1,
                          min(4096, flat.shape[0])).astype(int)
        lp = np.asarray(target.batched_lp(jnp.asarray(flat[idx])))
        z_map = np.asarray(flat[idx[int(np.nanargmax(lp))]], dtype=np.float64)
        logp_rep = float(np.nanmax(lp))
    else:
        logp_rep = float(model.target_log_prob_fn(jnp.asarray(z_map)))
    print(f"representative z: logpost={logp_rep:.2f}", flush=True)

    phys = model.to_physical_mass(flat)
    g = np.asarray(phys["gamma"]); tE = np.asarray(phys["theta_E"])
    gamma_med, gamma_lo, gamma_hi = np.percentile(g, [50, 16, 84])
    ess, rhat = e2.diagnostics(draws) if draws.shape[1] > 1 else (
        np.full(model.ndim, np.nan), np.full(model.ndim, np.nan))
    print(f"gamma posterior = {gamma_med:.4f} [{gamma_lo:.4f},{gamma_hi:.4f}] "
          f"Rhat_max={np.nanmax(rhat):.4f} ess_min={np.nanmin(ess):.0f}", flush=True)

    Y = np.asarray(model.Y); err = np.asarray(model.masked_err_map)
    keep = np.asarray(model.keep_mask)
    zj = jnp.asarray(z_map, dtype=jnp.float64)
    mimg = np.asarray(model.model_image(zj))
    a_star = np.asarray(model.shapelet_amps(zj), dtype=np.float64)
    resid_norm = np.where(keep, (Y - mimg) / err, np.nan)
    lam_t, extent = crit_field(model, z_map, np=np, jnp=jnp)
    src, src_extent = render_source(model, z_map, np=np, jnp=jnp)
    components = collect_components(model, z_map, np=np, jnp=jnp)

    stem = f"recipe_{args.tag}_{args.basin}_ingest"
    fig_npz = outdir / f"{stem}.npz"
    np.savez_compressed(
        fig_npz, data=Y.astype(np.float32), model=mimg.astype(np.float32),
        model_image_map=mimg.astype(np.float32), err_map=err.astype(np.float32),
        keep_mask=keep, psf=np.asarray(model.sim_config.kernel, dtype=np.float32),
        a_star_map=a_star, resid_norm=resid_norm.astype(np.float32), keep=keep,
        lam_t=lam_t.astype(np.float32), crit_extent=np.array(extent),
        source=src.astype(np.float32), source_extent=np.array(src_extent),
        gamma_samples=g.astype(np.float32), thetaE_samples=tE.astype(np.float32),
        z_map=z_map, labels=np.asarray(labels), draws=draws.astype(np.float32))

    summary = dict(
        tag=args.tag, basin=args.basin, mode="ingest (figure from a converged "
        "posterior npz; no re-sampling)", sampler=args.sampler,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ingest_source=str(args.ingest), whitener=target.whiten_meta,
        ndim=int(model.ndim), config=vars(args), components=components,
        meta=dict(delta_pix=float(target.cfg["delta_pix"]),
                  num_pix=int(Y.shape[0]), supersample=int(target.cfg["supersample"]),
                  band=f"HST-{args.tag}", mag_zero_point=None, subset="hst",
                  system="DESI-165.4754-06.0423"),
        redshifts=dict(z_lens=0.5, z_source=1.0,
                       note="placeholder redshifts; the P1 result is gamma + theta_E"),
        logp_representative=logp_rep,
        posterior_mass=dict(
            theta_E=dict(median=float(np.median(tE)), p16=float(np.percentile(tE, 16)),
                         p84=float(np.percentile(tE, 84)), mean=float(np.mean(tE))),
            gamma=dict(median=float(gamma_med), p16=float(gamma_lo),
                       p84=float(gamma_hi), mean=float(np.mean(g)))),
        rhat_max=float(np.nanmax(rhat)), ess_min=float(np.nanmin(ess)),
        rhat_gamma=float(rhat[gi]), ess_gamma=float(ess[gi]),
        gamma_posterior=dict(median=float(gamma_med), p16=float(gamma_lo),
                             p84=float(gamma_hi), std=float(np.std(g)),
                             native_diag_anchor=1.433),
        thetaE_posterior=dict(median=float(np.median(tE)), std=float(np.std(tE))),
        chi2_pp_diag=float(np.nanmean(resid_norm[keep] ** 2)), fig_npz=str(fig_npz))
    summary["wall_s"] = time.time() - t_start
    (outdir / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=float))

    fig_png = figdir / f"recipe_e2e_{args.tag}_{args.basin}_ingest.png"
    try:
        make_figure(Y, mimg, resid_norm, lam_t, extent, src, src_extent, g,
                    gamma_med, args, keep, fig_png, np=np)
        print(f"wrote {fig_png}", flush=True)
    except Exception as exc:
        print(f"WARN figure render failed: {exc}", flush=True)

    print(f"\n=== recipe end-to-end INGEST ({args.tag}/{args.basin}) ===", flush=True)
    print(f"gamma={gamma_med:.4f} [{gamma_lo:.4f},{gamma_hi:.4f}] "
          f"Rhat_max={np.nanmax(rhat):.4f} ess_min={np.nanmin(ess):.0f} "
          f"chi2_pp={summary['chi2_pp_diag']:.3f}  total {summary['wall_s']:.0f}s",
          flush=True)
    print(f"wrote {outdir / (stem + '.json')} + {fig_npz}", flush=True)
    return 0


def collect_components(model, z_map, *, np, jnp):
    """Physical MAP components (mass/shear/4 lens light/source Sersic/shapelet)
    for the COOLEST export, in the same schema 31_fit_euclid.collect_export uses."""
    zj = jnp.asarray(z_map, dtype=jnp.float64)
    x = model.bij.forward(list(zj[None, :].T))
    def sc(d):
        return {k: float(np.asarray(d[k]).reshape(-1)[0]) for k in d}
    return dict(mass=sc(x[0][0]), shear=sc(x[0][1]),
                lens_light=[sc(x[1][i]) for i in range(4)],
                source_sersic=sc(x[2][0]), source_shapelet=sc(x[2][1]))


def crit_field(model, z_map, *, np, jnp, n=401):
    """Tangential eigenvalue field lam_t(x,y) of the MAP mass; lam_t<0 is inside
    the tangential critical curve. Returns (lam_t (n,n), extent [xlo,xhi,ylo,yhi])."""
    from cgl.paths import bootstrap_vendor
    bootstrap_vendor()
    from gigalens.jax.profiles.mass import epl, shear
    m = model.to_physical_mass(np.asarray(z_map)[None, :])
    m = {k: float(np.asarray(m[k]).reshape(-1)[0]) for k in m}
    E, S = epl.EPL(50), shear.Shear()

    def alpha(x, y):
        fx, fy = E.deriv(x, y, m["theta_E"], m["gamma"], m["e1"], m["e2"],
                         m["center_x"], m["center_y"])
        sx, sy = S.deriv(x, y, m["gamma1"], m["gamma2"])
        return fx + sx, fy + sy
    span = max(2.5, 2.5 * m["theta_E"])
    g = jnp.linspace(-span, span, n)
    X, Y = jnp.meshgrid(g, g)
    h = 1e-4
    axp, ayp = alpha(X + h, Y); axm, aym = alpha(X - h, Y)
    ax_yp, ay_yp = alpha(X, Y + h); ax_ym, ay_ym = alpha(X, Y - h)
    a11 = 1.0 - (axp - axm) / (2 * h); a22 = 1.0 - (ay_yp - ay_ym) / (2 * h)
    a12 = -(ax_yp - ax_ym) / (2 * h); a21 = -(ayp - aym) / (2 * h)
    disc = jnp.sqrt(jnp.clip((a11 - a22) ** 2 + 4.0 * a12 * a21, 0.0, None))
    lam_t = 0.5 * (a11 + a22 - disc)
    return np.asarray(lam_t), [-span, span, -span, span]


def render_source(model, z_map, *, np, jnp, n=200, half=1.0):
    """Source-plane reconstruction: source Sersic + shapelets(a*) on a source
    grid. Returns (image (n,n), extent). Best-effort; returns zeros on failure."""
    try:
        from cgl.paths import bootstrap_vendor
        bootstrap_vendor()
        from gigalens.jax.profiles.light import sersic, shapelets
        zj = jnp.asarray(z_map, dtype=jnp.float64)
        x = model.bij.forward(list(zj[None, :].T))
        srcS = {k: float(np.asarray(x[2][0][k]).reshape(-1)[0]) for k in x[2][0]}
        srcSh = {k: float(np.asarray(x[2][1][k]).reshape(-1)[0]) for k in x[2][1]}
        a_star = np.asarray(model.shapelet_amps(zj), dtype=np.float64)
        g = jnp.linspace(-half, half, n)
        X, Y = jnp.meshgrid(g, g)
        img = sersic.SersicEllipse(use_lstsq=False).light(
            X, Y, srcS["R_sersic"], srcS["n_sersic"], srcS["e1"], srcS["e2"],
            srcS["center_x"], srcS["center_y"], Ie=srcS["Ie"])
        shp = shapelets.Shapelets(n_max=model.n_max, use_lstsq=False)
        amps = {name: float(a_star[i]) for i, name in enumerate(shp._amp_names)}
        img = img + shp.light(X, Y, srcSh["center_x"], srcSh["center_y"],
                              srcSh["beta"], **amps)
        return np.asarray(jnp.squeeze(img)), [-half, half, -half, half]
    except Exception as exc:
        print(f"  (source render skipped: {exc})", flush=True)
        return np.zeros((n, n)), [-half, half, -half, half]


def make_figure(Y, mimg, resid, lam_t, extent, src, src_extent, gsamp, gmed,
                args, keep, out, *, np):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [extent[0], extent[1], extent[2], extent[3]]
    span = Y.shape[0]
    im_ext = [-span / 2 * 0.13, span / 2 * 0.13] * 2  # approx arcsec (v2d 0.13")
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.4))
    d = np.where(keep, Y, np.nan)
    vmax = np.nanpercentile(d, 99.5)
    ax[0].imshow(d, origin="lower", cmap="magma", vmin=0, vmax=vmax)
    ax[0].set_title("data (HST v2d)")
    ax[1].imshow(np.where(keep, mimg, np.nan), origin="lower", cmap="magma",
                 vmin=0, vmax=vmax)
    # tangential critical curve as the lam_t=0 contour, mapped to pixel coords
    gy = np.linspace(0, span - 1, lam_t.shape[0])
    ax[1].contour(np.linspace(0, span - 1, lam_t.shape[1]), gy, lam_t,
                  levels=[0.0], colors="cyan", linewidths=1.2)
    ax[1].set_title(r"MAP model + tang. critical curve")
    r = np.nanmax(np.abs(resid))
    m3 = ax[2].imshow(resid, origin="lower", cmap="coolwarm", vmin=-4, vmax=4)
    ax[2].set_title("normalised residual (data-model)/σ")
    plt.colorbar(m3, ax=ax[2], fraction=0.046)
    ax[3].imshow(src, origin="lower", cmap="magma", extent=src_extent)
    ax[3].set_title("source-plane reconstruction")
    ax[4].hist(gsamp, bins=40, color="steelblue", density=True)
    ax[4].axvline(gmed, color="k", lw=1.5, label=f"median {gmed:.3f}")
    ax[4].axvline(1.433, color="crimson", ls="--", lw=1.2,
                  label="native-diag 1.433")
    ax[4].set_xlabel(r"$\gamma$ (mass slope)"); ax[4].legend(fontsize=8)
    ax[4].set_title(r"$\gamma$ posterior")
    for a in ax[:4]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"CGL recipe end-to-end — DESI-165.4754-06.0423 "
                 f"({args.tag}/{args.basin}, correlated likelihood)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
