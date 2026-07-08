"""06_fit_mock.py — P1b E1 single-mock fit driver (one process, ONE GPU).

Fits one 05 drizzle-mock product (or one E1d realization) with the 22/23-dim
gu-2022 model family under the diagonal or the correlated (conv-whitened)
likelihood, using the reduced-budget MAP -> SVI -> PHMC+ChEES recipe.
All design decisions are documented in cgl/e1.py's module docstring.

Corr fits with --kernel fitted run the real-data two-pass recipe in-process:
quick diag MAP -> model-subtract -> masked ACF -> WLS kernel fit -> whitener
(adaptive s_floor, M-search to e_op<=0.02); cached at data/e1_kernels/.

E1d arms (--e1d): the mock is data/mocks/e1d_{seed}.npz (native scale, REAL
v2d fitted-kernel noise, ported v2d keep-mask geometry). --whitener-arm:
  diag    : diagonal likelihood (exact marginal err map)
  strict  : the P1a v2d whitener taps (M=14, e_op=0.0177), mock-geometry erosion
  relaxed : build_whitener target e_op<=0.05, smallest M (the P1a OPEN-FLAG arm)

Run (example, GPU 8):
  CUDA_VISIBLE_DEVICES=8 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /raid/benson/.venvs/cgl/bin/python 06_fit_mock.py --seed 0 --scale fine \
      --likelihood corr --kernel fitted --out data/e1_fits/mock000_fine_corr_fitted.npz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO))


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="mock seed")
    ap.add_argument("--mocks-dir", default=None)
    ap.add_argument("--scale", choices=["fine", "binned", "native"],
                    default="fine")
    ap.add_argument("--likelihood", choices=["diag", "corr"], default="diag")
    ap.add_argument("--kernel", choices=["fitted", "analytic"],
                    default="fitted", help="corr kernel source")
    ap.add_argument("--err", choices=["exact", "recal40b"], default="exact")
    ap.add_argument("--e1d", action="store_true",
                    help="fit an E1d realization (data/mocks/e1d_SEED.npz)")
    ap.add_argument("--whitener-arm", choices=["diag", "strict", "relaxed"],
                    default=None, help="E1d arm")
    ap.add_argument("--dtype", choices=["mixed", "f32", "f64"],
                    default="mixed")
    ap.add_argument("--out", required=True)
    # budgets (reduced-budget recipe defaults from the approved plan)
    ap.add_argument("--map-particles", type=int, default=100)
    ap.add_argument("--map-steps", type=int, default=300)
    ap.add_argument("--map-starts", type=int, default=1)
    ap.add_argument("--svi-particles", type=int, default=128)
    ap.add_argument("--svi-steps", type=int, default=500)
    ap.add_argument("--chains", type=int, default=24)
    ap.add_argument("--burn", type=int, default=250)
    ap.add_argument("--keep", type=int, default=750)
    ap.add_argument("--sampler-stages", type=int, default=1,
                    choices=(1, 2),
                    help="2 = re-preconditioned second PHMC stage (metric "
                         "from pooled cross-chain stage-1 draws; diagnosis "
                         "pass). --burn/--keep apply to the final stage.")
    ap.add_argument("--stage1-burn", type=int, default=500)
    ap.add_argument("--stage1-keep", type=int, default=500)
    ap.add_argument("--step-size", type=float, default=0.3)
    ap.add_argument("--max-leapfrog", type=int, default=30)
    ap.add_argument("--fit-seed", type=int, default=0)
    ap.add_argument("--kernel-only", action="store_true",
                    help="run only the two-pass kernel fit, then exit")
    return ap.parse_args(argv)


def main():
    args = parse_args()

    # ---- process env BEFORE any jax import ---------------------------------
    os.environ["GIGALENS_X64"] = "0" if args.dtype == "f32" else "1"
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # INCIDENT NOTE (pilot, 2026-07-06): SVI at 300 ELBO particles appeared
    # to livelock the XLA compile (600+ threads) on both device classes.
    # Diagnosis: NOT the P0 priority-fusion livelock — the 300-particle ELBO
    # graph needs ~13-16 GB (A16 OOMs outright; the L4 sits at its ceiling
    # and the rematerialization search crawls). Adding
    # --xla_disable_hlo_passes=priority-fusion made ALL compiles minutes-slow
    # and is NOT used. Fix: SVI ELBO batch capped at 128 particles (default
    # below; documented deviation from the plan's ~300 — affects only ELBO
    # gradient noise, i.e. preconditioner quality).
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_autotune_level" not in flags:
        flags += " --xla_gpu_autotune_level=0"
    os.environ["XLA_FLAGS"] = flags.strip()

    import numpy as np

    from cgl import e1
    from cgl.paths import DATA

    import jax

    if args.dtype != "f32":
        # flip x64 BEFORE the first jnp array (05/cgl.likelihood contract)
        jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir",
                      str(DATA / "jax_cache_e1"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    t_all = time.time()
    mocks_dir = Path(args.mocks_dir) if args.mocks_dir else e1.MOCKS_DIR
    if args.e1d:
        mock_path = mocks_dir / f"e1d_{args.seed:03d}.npz"
        scale = "native"
    else:
        mock_path = mocks_dir / f"mock_{args.seed:03d}.npz"
        scale = args.scale
    mock = e1.load_mock(mock_path)
    shapelets = bool(mock["meta"].get("has_shapelets", False)) and \
        scale == "fine"
    likelihood = args.likelihood
    if args.e1d:
        likelihood = "diag" if args.whitener_arm == "diag" else "corr"

    e1.E1_FITS.mkdir(parents=True, exist_ok=True)
    e1.E1_KERNELS.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # shapelet-marg fits reduce particle counts (design-tensor memory:
    # X_w is (B, 43264, 15); 300 particles would be ~7.8 GB f32)
    map_particles = min(args.map_particles, 64) if shapelets \
        else args.map_particles
    svi_particles = min(args.svi_particles, 64) if shapelets \
        else args.svi_particles

    log = dict(argv=vars(args), mock=str(mock_path), scale=scale,
               likelihood=likelihood, shapelets=shapelets,
               commit=e1._git_head(),
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    print(f"[06] {out_path.name}: scale={scale} like={likelihood} "
          f"kernel={args.kernel} err={args.err} dtype={args.dtype} "
          f"shapelets={shapelets}", flush=True)

    # ---- whitener ------------------------------------------------------------
    whitener = None
    kernel_info = None
    t_kernel = 0.0
    if likelihood == "corr":
        t0 = time.time()
        if args.e1d:
            if args.whitener_arm == "strict":
                wz = np.load(DATA / "whitener_v2d.npz")
                whitener = dict(h=wz["h"].astype(np.float64),
                                M=int(wz["M"]), e_op=float(wz["e_op"]),
                                s_floor=float(json.loads(
                                    str(wz["meta"]))["s_floor"]))
                kernel_info = dict(source="whitener_v2d.npz (P1a strict)",
                                   M=whitener["M"], e_op=whitener["e_op"])
            else:  # relaxed
                wh = e1.build_product_whitener(
                    mock["rho_kernel"], m_grid=(3, 4, 5, 6, 7, 8, 10, 12),
                    e_target=0.05)
                whitener = wh
                kernel_info = dict(source="build_whitener(v2d rho, e_op<=0.05)",
                                   M=int(wh["M"]), e_op=float(wh["e_op"]),
                                   s_floor=float(wh["s_floor"]),
                                   m_search=wh["m_search"],
                                   e_target_met=bool(wh["e_target_met"]))
        elif args.kernel == "analytic":
            wh = e1.analytic_whitener(mock, scale)
            whitener = wh
            kernel_info = dict(source="analytic mock kernel",
                               M=int(wh["M"]), e_op=float(wh["e_op"]),
                               s_floor=float(wh["s_floor"]),
                               m_search=wh["m_search"])
        else:
            cache = e1.E1_KERNELS / f"kernel_{args.seed:03d}_{scale}.npz"
            if cache.exists():
                kz = np.load(cache)
                whitener = dict(h=kz["h"].astype(np.float64), M=int(kz["M"]),
                                e_op=float(kz["e_op"]),
                                s_floor=float(kz["s_floor"]))
                whitener["rho_fit"] = kz["rho_kernel"]
                kernel_info = dict(source=f"cache {cache.name}",
                                   M=whitener["M"], e_op=whitener["e_op"],
                                   reg_lambda=float(kz["reg_lambda"]),
                                   max_abs_resid=float(kz["max_abs_resid"]))
            else:
                # two-pass: quick diag MAP -> model-subtract -> fit kernel
                print("[06] two-pass kernel fit: quick diag MAP...",
                      flush=True)
                m_diag = e1.build_e1_model(
                    mock, scale=scale, likelihood="diag", err_mode="exact",
                    shapelets=shapelets, dtype_mode=args.dtype)
                qm = e1.run_map(m_diag, n_particles=64, steps=150,
                                seed=args.fit_seed + 900)
                prods = m_diag.render_products(qm["best_z"])
                kp = e1.kernel_pass(mock, scale, prods, out_path=cache)
                whitener = dict(h=kp["h"], M=kp["M"], e_op=kp["e_op"],
                                s_floor=kp["s_floor"],
                                rho_fit=kp["rho_fit"])
                kernel_info = dict(
                    source="two-pass fitted", family=kp["family"],
                    M=kp["M"], e_op=kp["e_op"], s_floor=kp["s_floor"],
                    reg_lambda=kp["reg_lambda"],
                    max_abs_resid=kp["max_abs_resid"],
                    kernel_gate_le_0p05=kp["gate_le_0p05"],
                    m_search=kp["m_search"], chi2_sky=kp["chi2_sky"],
                    quick_map_chi2=qm["chi2_best"],
                    e_target_met=kp["e_target_met"])
                del m_diag
        t_kernel = time.time() - t0
        print(f"[06] whitener ready: M={whitener['M']} "
              f"e_op={whitener['e_op']:.4f} [{t_kernel:.0f}s]", flush=True)
        if args.kernel_only:
            print("[06] --kernel-only: done")
            return 0

    # ---- model + pipeline ------------------------------------------------------
    model = e1.build_e1_model(
        mock, scale=scale, likelihood=likelihood, whitener=whitener,
        err_mode=args.err, shapelets=shapelets, dtype_mode=args.dtype)

    maps = []
    for k in range(args.map_starts):
        mres = e1.run_map(model, n_particles=map_particles,
                          steps=args.map_steps,
                          seed=args.fit_seed + 1000 * k)
        maps.append(mres)
        print(f"[06] MAP start {k}: stage1={mres['best_lp_stage1']:.1f} "
              f"lbfgs={['%.1f' % v for v in mres['lbfgs_lp']]} "
              f"best_lp={mres['best_lp']:.2f} "
              f"chi2={mres['chi2_best']:.4f} [{mres['wall_s']:.0f}s]",
              flush=True)
    best_idx = int(np.argmax([m["best_lp"] for m in maps]))
    map_best = maps[best_idx]

    svi = e1.run_svi(model, map_best["best_z"], n_vi=svi_particles,
                     steps=args.svi_steps, seed=args.fit_seed + 1)
    print(f"[06] SVI: -ELBO={svi['neg_elbo']:.2f} [{svi['wall_s']:.0f}s]",
          flush=True)

    if args.sampler_stages == 2:
        hmc = e1.run_chees_staged(model, svi["loc"], svi["cov"],
                                  chains=args.chains, burn=args.burn,
                                  keep=args.keep, step_size=args.step_size,
                                  max_leapfrog=args.max_leapfrog,
                                  seed=args.fit_seed + 2,
                                  stage1_burn=args.stage1_burn,
                                  stage1_keep=args.stage1_keep)
        print(f"[06] HMC stage1: rhat_max={hmc['stage1_rhat_max']:.2f} "
              f"ess_min={hmc['stage1_ess_min']:.0f} "
              f"[{hmc['stage1_wall_s']:.0f}s]", flush=True)
    else:
        hmc = e1.run_chees(model, svi["loc"], svi["cov"], chains=args.chains,
                           burn=args.burn, keep=args.keep,
                           step_size=args.step_size,
                           max_leapfrog=args.max_leapfrog,
                           seed=args.fit_seed + 2)
    print(f"[06] HMC: draws {hmc['draws'].shape} "
          f"[{hmc['wall_s']:.0f}s, {hmc['n_floored']} floored eigs]",
          flush=True)

    draws = hmc["draws"]                      # (T, C, ndim)
    ess, rhat = e1.diagnostics(draws)
    flat = draws.reshape(-1, model.ndim)
    labels, phys = model.to_physical(flat)
    truth_vec = model.truth_vector(labels)
    summ = e1.summarize_fit(labels, phys, truth_vec)

    mass_idx = [labels.index(m) for m in e1.MASS_LABELS]
    print(f"[06] === recovery (6 mass params) ===")
    for j in mass_idx:
        print(f"  {labels[j]:10s} truth={truth_vec[j]:8.4f} "
              f"mean={summ['mean'][j]:8.4f} std={summ['std'][j]:8.4f} "
              f"z={summ['z'][j]:6.2f} cov68={int(summ['cov68'][j])} "
              f"ess={ess[j]:7.0f} rhat={rhat[j]:.4f}", flush=True)

    # grad-count estimate: ChEES trajectory trace (max leapfrog per step)
    if hmc["max_traj"].size:
        mean_leap = float(np.mean(hmc["max_traj"] /
                                  np.maximum(hmc["step_size_hist"], 1e-8)))
    else:
        mean_leap = float("nan")

    wall_total = time.time() - t_all
    cfg_json = dict(
        log=log, kernel_info=kernel_info,
        recal_info=model.recal_info, whitener_meta=model.whitener_meta,
        budgets=dict(map_particles=map_particles, map_steps=args.map_steps,
                     map_starts=args.map_starts, svi_particles=svi_particles,
                     svi_steps=args.svi_steps, chains=args.chains,
                     burn=args.burn, keep=args.keep,
                     step_size=args.step_size,
                     max_leapfrog=args.max_leapfrog,
                     sampler_stages=args.sampler_stages,
                     stage1_burn=(args.stage1_burn
                                  if args.sampler_stages == 2 else None),
                     stage1_keep=(args.stage1_keep
                                  if args.sampler_stages == 2 else None)),
        stage1_diag=(dict(rhat_max=hmc.get("stage1_rhat_max"),
                          ess_min=hmc.get("stage1_ess_min"))
                     if args.sampler_stages == 2 else None),
        n_data=int(model.n_data), n_keep=model.n_keep,
        n_keep_w=model.n_keep_w,
        timing=dict(kernel_s=t_kernel,
                    map_s=[m["wall_s"] for m in maps],
                    svi_s=svi["wall_s"], hmc_s=hmc["wall_s"],
                    total_s=wall_total),
        map_stage1_lp=[m["best_lp_stage1"] for m in maps],
        map_lbfgs_lp=[m["lbfgs_lp"] for m in maps],
        mean_leapfrog_est=mean_leap,
        svi_neg_elbo=svi["neg_elbo"], n_floored=hmc["n_floored"],
        arc_snr=float(mock["meta"].get("arc_snr", np.nan)),
        truth_gamma=float(mock["flat"].get("gamma", np.nan)),
    )

    np.savez(
        out_path,
        phys_draws=phys.astype(np.float32), phys_labels=labels,
        truth_vec=truth_vec,
        mass_idx=np.array(mass_idx),
        post_mean=summ["mean"], post_std=summ["std"], zscore=summ["z"],
        cov68=summ["cov68"], cov95=summ["cov95"],
        q16=summ["q16"], q84=summ["q84"],
        ess=ess, rhat=rhat,
        map_pop_lp=np.stack([m["pop_lp"] for m in maps]),
        map_pop_gamma=np.stack([
            model.to_physical(m["pop_z"])[1][:, labels.index("gamma")]
            for m in maps]).astype(np.float32),
        map_best_lp=np.array([m["best_lp"] for m in maps]),
        map_best_chi2=np.array([m["chi2_best"] for m in maps]),
        config=json.dumps(cfg_json),
    )
    print(f"[06] saved -> {out_path} ({wall_total:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
