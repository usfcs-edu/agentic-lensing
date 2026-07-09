"""31_fit_euclid.py — fit a Euclid Q1 VIS lens with the CGL recipe (DIAGONAL).

Euclid VIS is native 0.1"/px -> diagonal RMS -> this is a DIAGONAL-likelihood
(whiten_fn=None) demonstration of the Pillar-2 sampler recipe on independent
real data, NOT the P1 correlated likelihood (CAMPAIGN.md P3 recon entry).

Pipeline (reuses the P1c production infra in cgl.e2):
  load_euclid_vis -> build_marg_model(diagonal, delta_pix=0.1, re-centred prior)
  -> e2.map_polish -> e2.laplace_evidence -> e2.run_staged (two-stage PHMC).
Writes a posterior npz + a JSON with theta_E(ours, area-equivalent from the EPL
fit) vs the published einstein_radius_effective, R-hat / ESS, and the MAP
chi2-per-pixel floor (the foundry-i R0c PSF-oversampling probe).

Run (pin ONE GPU; A16 4-7 or L4 8; never GPU 9 / A16 0-3):
  GIGALENS_X64=1 CUDA_VISIBLE_DEVICES=5 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion" \
  /raid/benson/.venvs/cgl/bin/python 31_fit_euclid.py --target <id> [--pilot]
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
    ap.add_argument("--target", required=True, help="Euclid Q1 id_str")
    ap.add_argument("--subset", default="lens")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES (else inherit)")
    ap.add_argument("--crop-pix", type=int, default=100)
    ap.add_argument("--supersample", type=int, default=1,
                    help="1 is natural for native 0.1\" VIS (PSF at pixel scale) "
                         "and keeps the sim grid small; ss2 quadruples it and "
                         "makes the batched-PHMC XLA compile intractable on L4")
    ap.add_argument("--theta-E-med", type=float, default=1.0)
    ap.add_argument("--theta-E-sig", type=float, default=0.4)
    ap.add_argument("--mass-center-sig", type=float, default=0.3)
    ap.add_argument("--gamma-med", type=float, default=2.0)
    ap.add_argument("--gamma-sig", type=float, default=0.05,
                    help="pin gamma~2 (isothermal-equivalent) to compare with "
                         "the published SIE (fixed gamma=2); 0.25 = free EPL")
    ap.add_argument("--mass-e-sig", type=float, default=0.2,
                    help="mass ellipticity Normal(0,sig); HST-parity 0.1 forbids "
                         "the elliptical Euclid lenses (q~0.56 = |e|~0.28)")
    ap.add_argument("--light-scale", default="auto",
                    help="'auto' calibrates the Sersic Ie prior medians to the "
                         "data flux scale; or a float override")
    ap.add_argument("--shapelet-sigma0-base", type=float, default=5.0,
                    help="source-amp ridge scale (multiplied by light_scale)")
    ap.add_argument("--near-off", type=float, default=4.5,
                    help="LL2/LL3 off-field center (arcsec); isolated targets")
    ap.add_argument("--chains", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--adam-steps", type=int, default=3000)
    ap.add_argument("--adam-lr", type=float, default=3e-2)
    ap.add_argument("--map-rounds", type=int, default=4)
    ap.add_argument("--stage1-burn", type=int, default=300)
    ap.add_argument("--stage1-keep", type=int, default=300)
    ap.add_argument("--burn", type=int, default=300)
    ap.add_argument("--keep", type=int, default=600)
    ap.add_argument("--num-leapfrog", type=int, default=8,
                    help="leapfrog steps; 8 halves per-step cost vs 16 and still "
                         "mixes a well-preconditioned unimodal posterior")
    ap.add_argument("--step-size", type=float, default=0.2)
    ap.add_argument("--pilot", action="store_true",
                    help="MAP + Laplace + chi2 floor only (no sampling)")
    ap.add_argument("--outdir", default=str(REPRO / "data" / "euclid"))
    return ap.parse_args()


def main():
    args = parse_args()
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
    from cgl.euclid_io import load_euclid_vis, load_published_mass
    from cgl.likelihood import build_marg_model

    guards.require_x64()
    guards.require_gpu()
    guards.require_single_device()
    print(f"device: {jax.devices()[0]}  target={args.target}", flush=True)

    t_start = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- data + model -------------------------------------------------------
    product = load_euclid_vis(args.target, args.subset, crop_pix=args.crop_pix,
                              supersample=args.supersample)
    pub = load_published_mass(args.target, args.subset)
    theta_E_pub = float(pub["einstein_radius_effective_median_pdf"])
    near = (args.near_off, args.near_off)

    def _build(ls):
        # shapelet_sigma0 (source-amp ridge prior scale) MUST track light_scale:
        # the amps are marginalized under Normal(0, sigma0/sqrt(i+1)); at the HST
        # sigma0=5 with a Euclid flux ~50x smaller the source is effectively
        # UNregularized and absorbs lensing signal -> biased mass. Scale it with
        # the flux so the source-amp prior matches the data amplitude.
        return build_marg_model(
            product, n_max=6, supersample=product["meta"]["supersample"],
            delta_pix=product["meta"]["delta_pix"],
            exp_time=product["meta"]["exp_time"], whiten_fn=None, near_xy=near,
            theta_E_med=args.theta_E_med, theta_E_sig=args.theta_E_sig,
            mass_center_sig=args.mass_center_sig, light_scale=ls,
            gamma_med=args.gamma_med, gamma_sig=args.gamma_sig,
            mass_e_sig=args.mass_e_sig,
            shapelet_sigma0=args.shapelet_sigma0_base * ls)

    # ---- calibrate the light (Ie) prior scale to the data flux scale --------
    # The Sersic Ie LogNormal(sig=0.5) prior is informative: with light_scale=1
    # (HST-F140W medians) a Euclid-VIS model is ~O(100) too bright and the MAP
    # cannot escape the prior. Calibrate light_scale so the prior-median lens
    # light peak ~ the data peak (one linear rescale; MAP + the sig=0.5 prior
    # fine-tune within a factor ~e^0.5). Mass params are flux-independent.
    if args.light_scale == "auto":
        m1 = _build(1.0)
        z1 = make_start(m1, theta_E=theta_E_pub, light_scale=1.0, near=near,
                        jnp=jnp, np=np)
        peak1 = float(np.max(np.abs(np.asarray(m1.m_det(jnp.asarray(z1))))))
        keep = np.asarray(product["keep_mask"])
        data_peak = float(np.percentile(np.asarray(product["img"])[keep], 99.9))
        light_scale = max(data_peak, 1e-6) / max(peak1, 1e-30)
        print(f"light_scale=auto: data_peak={data_peak:.4g} model_peak(1)="
              f"{peak1:.4g} -> light_scale={light_scale:.4g}", flush=True)
    else:
        light_scale = float(args.light_scale)
    args.light_scale_resolved = float(light_scale)

    model = _build(light_scale)
    print(f"model ndim={model.ndim} n_keep={int(product['meta']['n_keep'])} "
          f"theta_E_pub(eff)={theta_E_pub:.4f}", flush=True)

    # e2.* expect a target with .model / .batched_lp / .conversion_factor
    import types
    target = types.SimpleNamespace(
        model=model, batched_lp=jax.jit(jax.vmap(model.target_log_prob_fn)),
        conversion_factor=product["meta"]["delta_pix"] ** 2, tag=args.target)

    # ---- start: prior-median physical pytree, theta_E at published value ----
    z0 = make_start(model, theta_E=theta_E_pub, light_scale=light_scale,
                    near=near, jnp=jnp, np=np)
    lp0 = float(model.target_log_prob_fn(jnp.asarray(z0)))
    print(f"start logp={lp0:.3f}", flush=True)
    assert np.isfinite(lp0), "start logp not finite"

    # ---- Adam warmup (cold start) then L-BFGS polish ------------------------
    # e2.map_polish is L-BFGS-only; its zoom line-search STALLS from a cold,
    # far-from-optimum start (P1c seeded near-optimal paper MAPs). Adam is
    # robust to the cold start; L-BFGS then sharpens the mode for the Laplace
    # metric (map_polish keeps z_adam if L-BFGS cannot improve it).
    z_adam, adam_lps, adam_s = adam_warmup(
        model, z0, args.adam_steps, args.adam_lr, jnp=jnp, np=np)
    print(f"Adam {args.adam_steps}: logp {adam_lps[0]:.2f} -> {adam_lps[-1]:.2f} "
          f"({adam_s:.1f}s)", flush=True)
    mp = e2.map_polish(target, z_adam, rounds=args.map_rounds, iters=200)
    z_map = mp["z_map"]
    print(f"L-BFGS polish: logp {mp['logp0']:.2f} -> {mp['logp_map']:.2f} "
          f"({mp['wall_s']:.1f}s)", flush=True)

    chi2 = chi2_per_pixel(model, z_map, int(product["meta"]["n_keep"]), np=np, jnp=jnp)
    phys_map = mass_phys(model, z_map, np=np, jnp=jnp)
    tE_eff_map, tE_detail = theta_E_eff(model, phys_map, np=np, jnp=jnp)
    export = collect_export(model, z_map, product, np=np, jnp=jnp)
    verdict = ("PSF-OVERSAMPLING SUSPECT (chi2_pp floor >> 1 with a good model; "
               "resample PSF to 0.1\" per foundry-i 40d)" if chi2 > 2.0 else
               "PSF sampling OK (chi2_pp floor ~1)")
    print(f"MAP chi2_pp={chi2:.4f}  [{verdict}]", flush=True)
    print(f"MAP mass: theta_E(gigalens)={phys_map['theta_E']:.4f} "
          f"gamma={phys_map['gamma']:.4f} q={tE_detail['q']:.3f}  "
          f"theta_E_eff(ours)={tE_eff_map:.4f} vs pub {theta_E_pub:.4f} "
          f"({100*(tE_eff_map/theta_E_pub-1):+.1f}%)", flush=True)

    summary = dict(
        target=args.target, subset=args.subset,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=vars(args), meta=product["meta"],
        theta_E_pub_eff=theta_E_pub,
        theta_E_pub_eff_1sig=[pub["einstein_radius_effective_lower_1_sigma"],
                              pub["einstein_radius_effective_upper_1_sigma"]],
        map=dict(logp_start=lp0, logp_adam=adam_lps[-1], logp_map=mp["logp_map"],
                 round_lps=mp["round_lps"], adam_wall_s=adam_s,
                 lbfgs_wall_s=mp["wall_s"]),
        map_chi2_per_pixel=chi2, psf_verdict=verdict,
        map_mass_phys=phys_map, map_theta_E_eff=tE_eff_map,
        theta_E_eff_detail=tE_detail,
        map_theta_E_offset_pct=100 * (tE_eff_map / theta_E_pub - 1),
        light_scale_resolved=float(light_scale),
        components=export["components"],
        redshifts=dict(z_lens=0.5, z_source=1.0,
                       note="PyAutoLens fixed defaults; only angular theta_E "
                            "is meaningful"),
    )
    # MAP-image arrays for COOLEST export (CPU-only downstream, no GPU rebuild)
    map_arrays = dict(
        z_map=export["z_map"], data=export["data"].astype(np.float32),
        err_map=export["err_map"].astype(np.float32),
        keep_mask=export["keep_mask"], psf=export["psf"].astype(np.float32),
        model_image_map=export["model_image_map"].astype(np.float32),
        a_star_map=export["a_star_map"])

    if args.pilot:
        summary["mode"] = "pilot"
        summary["wall_s"] = time.time() - t_start
        _write(outdir, args.target, summary, map_arrays, np=np)
        print(f"\nPILOT done ({summary['wall_s']:.0f}s)", flush=True)
        return 0

    # ---- Laplace metric + two-stage PHMC ------------------------------------
    lap = e2.laplace_evidence(target, z_map)
    print(f"Laplace: min_eig={lap['min_eig']:.3e} n_neg={lap['n_neg']} "
          f"logZ={lap['log_evidence']:.2f}", flush=True)
    st = e2.run_staged(
        target, z_map, lap["cov"], chains=args.chains, seed=args.seed,
        stage1_burn=args.stage1_burn, stage1_keep=args.stage1_keep,
        burn=args.burn, keep=args.keep, step_size=args.step_size,
        num_leapfrog=args.num_leapfrog)
    draws = st["draws"]                                   # (T, C, ndim)
    print(f"PHMC: R-hat max={st['rhat'].max():.4f} ess_min={st['ess'].min():.0f} "
          f"accept={st['accept_mean']:.2f} ({st['wall_s']:.0f}s)", flush=True)

    # ---- posterior mass summary + theta_E_eff posterior ---------------------
    flat = draws.reshape(-1, draws.shape[-1])
    labels = list(model.index_labels)
    gi = labels.index("mass.gamma")
    ti = labels.index("mass.theta_E")
    mass = model.to_physical_mass(flat)
    # per-draw theta_E_eff (subsample for speed)
    idx = np.linspace(0, flat.shape[0] - 1, min(200, flat.shape[0])).astype(int)
    tE_eff_samples = np.array([
        theta_E_eff(model, {k: float(mass[k][j]) for k in mass}, np=np, jnp=jnp)[0]
        for j in idx])
    tE_eff_med = float(np.median(tE_eff_samples))
    tE_eff_lo, tE_eff_hi = np.percentile(tE_eff_samples, [16, 84])

    def stat(a):
        return dict(median=float(np.median(a)),
                    p16=float(np.percentile(a, 16)),
                    p84=float(np.percentile(a, 84)),
                    mean=float(np.mean(a)), std=float(np.std(a)))

    summary.update(dict(
        mode="full",
        sampler="two-stage re-preconditioned PHMC (cgl.e2.run_staged)",
        rhat_max=float(st["rhat"].max()), ess_min=float(st["ess"].min()),
        rhat_gamma=float(st["rhat"][gi]), ess_gamma=float(st["ess"][gi]),
        rhat_theta_E=float(st["rhat"][ti]), ess_theta_E=float(st["ess"][ti]),
        accept_mean=float(st["accept_mean"]), step_final=float(st["step_final"]),
        stage1_rhat_max=st["stage1_rhat_max"], stage1_ess_min=st["stage1_ess_min"],
        wall_s_sample=st["wall_s"],
        posterior_mass=dict(theta_E=stat(mass["theta_E"]), gamma=stat(mass["gamma"]),
                            e1=stat(mass["e1"]), e2=stat(mass["e2"]),
                            center_x=stat(mass["center_x"]),
                            center_y=stat(mass["center_y"]),
                            gamma1=stat(mass["gamma1"]), gamma2=stat(mass["gamma2"])),
        theta_E_eff_posterior=dict(median=tE_eff_med, p16=float(tE_eff_lo),
                                   p84=float(tE_eff_hi), n_subsample=len(idx)),
        theta_E_eff_vs_pub_pct=100 * (tE_eff_med / theta_E_pub - 1),
        laplace=dict(log_evidence=lap["log_evidence"], min_eig=lap["min_eig"],
                     n_neg=lap["n_neg"]),
    ))
    summary["wall_s"] = time.time() - t_start
    _write(outdir, args.target, summary, dict(
        draws=draws.astype(np.float32), labels=np.array(labels),
        theta_E_eff_samples=tE_eff_samples,
        rhat=st["rhat"], ess=st["ess"], **map_arrays), np=np)

    print(f"\n=== {args.target} ===", flush=True)
    print(f"theta_E_eff(ours)={tE_eff_med:.4f} [{tE_eff_lo:.4f},{tE_eff_hi:.4f}] "
          f"vs pub {theta_E_pub:.4f} ({summary['theta_E_eff_vs_pub_pct']:+.1f}%)",
          flush=True)
    print(f"gamma={mass['gamma'].mean():.3f}+/-{mass['gamma'].std():.3f}  "
          f"R-hat_max={st['rhat'].max():.4f}  ess_min={st['ess'].min():.0f}  "
          f"chi2_pp={chi2:.3f}", flush=True)
    print(f"total {summary['wall_s']:.0f}s", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_start(model, *, theta_E, light_scale, near, jnp, np, source_R=0.2,
               e0=0.05):
    """Prior-median physical pytree -> z start; theta_E at the published value,
    SMALL nonzero ellipticity (e0), no shear, source at centre. LL2/LL3 parked
    off-field. e0 must be > 0: at e1=e2=0 the gigalens c=sqrt(e1^2+e2^2) and
    phi=atan2(e2,e1) have singular (NaN) gradients (the 0/0 at the ellipticity
    origin), which stalls the optimiser at a finite-logp / NaN-grad point."""
    nx, ny = float(near[0]), float(near[1])
    def a(v):
        return jnp.asarray([float(v)], dtype=jnp.float64)
    x = [
        [dict(theta_E=a(theta_E), gamma=a(2.0), e1=a(e0), e2=a(e0),
              center_x=a(0.0), center_y=a(0.0)),
         dict(gamma1=a(1e-3), gamma2=a(1e-3))],
        [dict(R_sersic=a(0.4), n_sersic=a(4.0), e1=a(e0), e2=a(e0),
              center_x=a(0.0), center_y=a(0.0), Ie=a(5.0 * light_scale)),
         dict(R_sersic=a(1.0), n_sersic=a(1.5), e1=a(e0), e2=a(e0),
              center_x=a(0.0), center_y=a(0.0), Ie=a(2.0 * light_scale)),
         dict(R_sersic=a(0.3), n_sersic=a(1.0), e1=a(e0), e2=a(e0),
              center_x=a(nx), center_y=a(ny), Ie=a(1.0 * light_scale)),
         dict(R_sersic=a(0.6), n_sersic=a(1.0), e1=a(e0), e2=a(e0),
              center_x=a(nx), center_y=a(ny), Ie=a(0.5 * light_scale))],
        [dict(R_sersic=a(source_R), n_sersic=a(1.0), e1=a(e0), e2=a(e0),
              center_x=a(0.0), center_y=a(0.0), Ie=a(2.0 * light_scale)),
         dict(beta=a(0.1), center_x=a(0.0), center_y=a(0.0))],
    ]
    z_list = model.bij.inverse(x)
    z = np.array([float(np.asarray(v).reshape(-1)[0]) for v in z_list],
                 dtype=np.float64)
    return z


def adam_warmup(model, z0, steps, lr0, *, jnp, np):
    """Cosine-decayed Adam on -logpost from a cold start. Returns
    (z_final, [logp_start, logp_end], wall_s). jax is imported locally: the
    module must NOT import jax before main() pins the GPU."""
    import functools
    import jax
    import optax
    lp = model.target_log_prob_fn
    sched = optax.cosine_decay_schedule(lr0, int(steps), alpha=0.02)
    opt = optax.adam(sched)

    @functools.partial(jax.jit, static_argnums=(1,))
    def run(z, n):
        def step(carry, _):
            zz, s = carry
            v, g = jax.value_and_grad(lambda q: -lp(q))(zz)
            u, s = opt.update(g, s, zz)
            return (optax.apply_updates(zz, u), s), v
        (zf, _), vals = jax.lax.scan(step, (z, opt.init(z)), length=n)
        return zf, vals
    t0 = time.time()
    zf, vals = run(jnp.asarray(z0, dtype=jnp.float64), int(steps))
    zf = np.asarray(jax.block_until_ready(zf), dtype=np.float64)
    vals = np.asarray(vals)
    return zf, [float(-vals[0]), float(-vals[-1])], time.time() - t0


def mass_phys(model, z, *, np, jnp):
    """Physical mass+shear params at a single z-vector."""
    m = model.to_physical_mass(np.asarray(z)[None, :])
    return {k: float(np.asarray(m[k]).reshape(-1)[0]) for k in m}


def collect_export(model, z_map, product, *, np, jnp):
    """All physical MAP components + arrays the COOLEST export needs (so
    32_coolest_export runs on CPU without rebuilding the jax model).

    components: nested dict {mass, shear, lens_light[4], source_sersic,
    source_shapelet} of physical scalars; a_star_map: the 28 ridge-marginalized
    shapelet amplitudes at the MAP z (marginal mode, NOT sampled)."""
    zj = jnp.asarray(z_map, dtype=jnp.float64)
    x = model.bij.forward(list(zj[None, :].T))         # nested physical pytree
    def sc(d):
        return {k: float(np.asarray(d[k]).reshape(-1)[0]) for k in d}
    comp = dict(
        mass=sc(x[0][0]), shear=sc(x[0][1]),
        lens_light=[sc(x[1][i]) for i in range(4)],
        source_sersic=sc(x[2][0]), source_shapelet=sc(x[2][1]))
    a_star = np.asarray(model.shapelet_amps(zj), dtype=np.float64)
    mimg = np.asarray(model.model_image(zj), dtype=np.float64)
    return dict(components=comp, a_star_map=a_star, model_image_map=mimg,
                z_map=np.asarray(z_map, dtype=np.float64),
                data=np.asarray(model.Y, dtype=np.float64),
                err_map=np.asarray(product["err_map"], dtype=np.float64),
                keep_mask=np.asarray(product["keep_mask"], dtype=bool),
                psf=np.asarray(product["psf"], dtype=np.float64))


def chi2_per_pixel(model, z, n_keep, *, np, jnp):
    """Reduced chi2 of the FULL model (Sersic + marginalized shapelets) at z.

    Uses the whitened residual Rw and design Xw with the marginal-mode amps
    a*: chi2 = ||Rw - Xw a*||^2 / n_keep (diagonal whitener -> pixel chi2)."""
    it = model.marg_internals(np.asarray(z, dtype=np.float64))
    Rw = np.asarray(it["Rw"]); Xw = np.asarray(it["Xw"]); a = np.asarray(it["a_star"])
    resid = Rw - Xw @ a
    return float(np.sum(resid ** 2) / max(1, n_keep))


def theta_E_eff(model, mass, *, np, jnp, n=451):
    """Area-equivalent Einstein radius from the fitted EPL+shear: find the
    region inside the tangential critical curve (smaller Jacobian eigenvalue
    < 0) on a fine grid, theta_E_eff = sqrt(area/pi). This is PyAutoLens's
    einstein_radius_effective definition, computed via the gigalens EPL/shear
    deflection so it is convention-consistent with our fit."""
    from cgl.paths import bootstrap_vendor
    bootstrap_vendor()
    from gigalens.jax.profiles.mass import epl, shear
    E = epl.EPL(50)
    S = shear.Shear()

    def alpha(x, y):
        fx, fy = E.deriv(x, y, mass["theta_E"], mass["gamma"], mass["e1"],
                         mass["e2"], mass["center_x"], mass["center_y"])
        sx, sy = S.deriv(x, y, mass["gamma1"], mass["gamma2"])
        return fx + sx, fy + sy

    span = max(3.0, 3.0 * mass["theta_E"])
    dx = 2 * span / (n - 1)
    g = jnp.linspace(-span, span, n)
    X, Y = jnp.meshgrid(g, g)
    h = 1e-4
    axp, ayp = alpha(X + h, Y); axm, aym = alpha(X - h, Y)
    ayp2, axp2 = None, None
    ax_yp, ay_yp = alpha(X, Y + h); ax_ym, ay_ym = alpha(X, Y - h)
    dax_dx = (axp - axm) / (2 * h); day_dx = (ayp - aym) / (2 * h)
    dax_dy = (ax_yp - ax_ym) / (2 * h); day_dy = (ay_yp - ay_ym) / (2 * h)
    # Jacobian A = I - d(alpha)/d(x); eigenvalues of the 2x2 field
    a11 = 1.0 - dax_dx; a22 = 1.0 - day_dy
    a12 = -dax_dy; a21 = -day_dx
    tr = a11 + a22
    det2 = (a11 - a22) ** 2 + 4.0 * a12 * a21
    disc = jnp.sqrt(jnp.clip(det2, 0.0, None))
    lam_t = 0.5 * (tr - disc)                     # smaller eigenvalue (tangential)
    inside = np.asarray(lam_t < 0.0)
    area = float(inside.sum()) * dx * dx
    tE = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
    c = float(np.hypot(mass["e1"], mass["e2"]))
    q = (1 - c) / (1 + c)
    return tE, dict(q=q, area=area, span=span, grid=n)


def _write(outdir, target, summary, arrays, *, np):
    (outdir / f"{target}_fit.json").write_text(json.dumps(summary, indent=2,
                                                          default=float))
    if arrays is not None:
        np.savez_compressed(outdir / f"{target}_posterior.npz", **arrays)
    print(f"wrote {outdir / (target + '_fit.json')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
