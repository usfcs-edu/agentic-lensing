"""05_gen_drizzle_mocks.py — P1a self-consistent drizzle mock trio generator.

Per mock (seeds 0-7; truth from the gu-2022 Eq.(8) sampler, ported with
attribution into cgl.mocks.sample_truth):

  1. noiseless scene (*) PSF rendered by gigalens on the fine grid
     (214^2 @ 0.04", supersample=1, kernel = the vendored gigalens TinyTim
     asset psf.npy DECLARED at 0.04"/px — self-consistent convention,
     guards.assert_psf_sampling-compatible),
  2. NINE dithered native exposures (70^2 @ 0.12") at ALL integer fine
     offsets {0,1,2}^2 via exact 3x3 block-sum (detector pixelation).
     DEVIATION from the plan's 3 exposures {(0,0),(1,2),(2,1)}, documented
     with measurements: the 3-frame stack is period-3 shift-variant (NOT a
     convolution), so no effective fine PSF exists and the render-check
     gate failed at 2-27 sigma (data/mocks_report_3frame.json) — an
     E1-killing simulator-mismatch floor. The full-phase dither makes the
     stacked operator EXACTLY the separable 3x3-tent convolution (PSF_eff
     exact) with the SAME 1-D noise-correlation tent (3-|d|)/3.
  3. iid noise per native pixel (gu-2022 add_noise: sigma_bkg=0.2,
     t_exp=100 s, G=1),
  4. drizzle back (square kernel, pixfrac 1, r=3 exact 3x3 drop; operator =
     the cgl.noise 1-D overlap enumeration, one implementation) -> fine
     product 208^2 + binned 104^2 (2x2 block-sum) + the native exposures,
     each with EXACT analytic covariance (per-pixel variance propagated
     through the operator; stationary correlation kernel from
     cgl.noise.drizzle_acf at the actual offsets — support |lag|<=2 fine,
     <=1 binned, iid native).
  5. effective fine PSF = PSF pushed noiselessly through the same pipeline.

GATE (pre-registered): noiseless render-vs-drizzle agreement < 0.05 sigma on
every kept pixel (40c render-check convention): the fine-grid gigalens render
with kernel=PSF_eff must reproduce the drizzled noiseless product. keep_mask
excludes r < 0.2" around the lens center — mirroring the core mask carried
by EVERY real Stage-A product (40/40b: CORE_MASK_ARC=0.20).

Outputs: data/mocks/mock_{seed:03d}.npz, data/mocks_report.json

Run (GPU 8):
  GIGALENS_X64=1 CUDA_VISIBLE_DEVICES=8 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /raid/benson/.venvs/cgl/bin/python 05_gen_drizzle_mocks.py
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

from cgl import guards  # noqa: E402
from cgl.mocks import (  # noqa: E402
    CROP0, EXP_TIME, FINE_PIX, FINE_RENDER, N_FINE, NATIVE_PIX, OFFSETS_FINE,
    R_INT, SIGMA_BKG, DrizzleMockPipeline, add_noise_native,
    effective_fine_psf, sample_truth,
)
from cgl.noise import block_sum  # noqa: E402
from cgl.paths import DATA, VENDOR_SRC, bootstrap_vendor  # noqa: E402

GATE_SIGMA = 0.05
CORE_MASK_ARC = 0.20              # mirrors 40/40b real-product core mask
N_MOCKS = 8


def parse_range(spec):
    """'8-63' -> [8..63]; '5' -> [5]; '1,3,7-9' -> [1,3,7,8,9]."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def build_sims(psf, psf_eff, with_shapelets=False):
    bootstrap_vendor()
    import jax

    # cgl.likelihood flips x64 at ITS import time; this script renders
    # without it, so honor the env var here (before any jnp array exists —
    # the cgl.mocks/noise imports above are numpy-only).
    if os.environ.get("GIGALENS_X64") == "1":
        jax.config.update("jax_enable_x64", True)

    guards.require_x64()
    guards.require_gpu()
    guards.require_single_device()

    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.simulator import LensSimulator
    from gigalens.model import PhysicalModel
    from gigalens.simulator import SimulatorConfig

    pm_full = PhysicalModel([epl.EPL(50), shear.Shear()],
                            [sersic.SersicEllipse(use_lstsq=False)],
                            [sersic.SersicEllipse(use_lstsq=False)])
    pm_ll = PhysicalModel([epl.EPL(50), shear.Shear()],
                          [sersic.SersicEllipse(use_lstsq=False)], [])
    cfg = SimulatorConfig(delta_pix=FINE_PIX, num_pix=FINE_RENDER,
                          supersample=1, kernel=psf)
    cfg_eff = SimulatorConfig(delta_pix=FINE_PIX, num_pix=FINE_RENDER,
                              supersample=1, kernel=psf_eff)
    sims = dict(
        full=LensSimulator(pm_full, cfg, bs=1),
        ll=LensSimulator(pm_ll, cfg, bs=1),
        full_eff=LensSimulator(pm_full, cfg_eff, bs=1),
    )
    if with_shapelets:
        # E1c marg arm: additive shapelet source component (n_max=4),
        # rendered separately (linear) so the Sersic-only sims are untouched.
        from gigalens.jax.profiles.light import shapelets as shp_profile

        from cgl.e1 import SHAPELET_NMAX

        pm_shp = PhysicalModel(
            [epl.EPL(50), shear.Shear()], [],
            [shp_profile.Shapelets(n_max=SHAPELET_NMAX, use_lstsq=False,
                                   interpolate=False)])
        sims["shp"] = LensSimulator(pm_shp, cfg, bs=1)
        sims["shp_eff"] = LensSimulator(pm_shp, cfg_eff, bs=1)
    return sims


def arc_snr(arc_only, err_map):
    """gu-2022 integrated arc SNR diagnostic (ported convention)."""
    mask = arc_only > err_map
    if mask.sum() == 0:
        return 0.0
    return float(arc_only[mask].sum() / np.sqrt((err_map[mask] ** 2).sum()))


def gen_e1d(seeds, out_json):
    """E1d realism arm (P1b): 16 native-scale realizations with noise drawn
    from the REAL v2d fitted kernel on the mock native scene, using the REAL
    v2d keep-mask geometry (central 70^2 port) so whitener erosion loss is
    realistic. Single frame at offset (2,2) — the exactly-centered frame,
    matching the cropped v2d mask center (34.5, 34.5)."""
    t0 = time.time()
    outdir = DATA / "mocks"
    outdir.mkdir(parents=True, exist_ok=True)

    psf = np.load(Path(VENDOR_SRC) / "gigalens" / "assets" / "psf.npy")
    psf = (psf / psf.sum()).astype(np.float64)
    pipe = DrizzleMockPipeline()
    psf_eff = effective_fine_psf(psf, pipe, half=(psf.shape[0] // 2) + 3)
    sims = build_sims(psf, psf_eff)

    from cgl import exact_ref
    from cgl.paths import CUTOUT_V2D, load_product

    kz = np.load(DATA / "noise_kernel_v2d.npz")
    kmeta = json.loads(str(kz["meta"]))
    guards.assert_model_subtracted_sky(kmeta)
    rho = kz["rho_kernel"].astype(np.float64)

    v2d = load_product(CUTOUT_V2D)
    n_v2d = v2d["keep_mask"].shape[0]
    c0 = (n_v2d - 70) // 2                       # 5: central 70^2 port
    keep_port = v2d["keep_mask"][c0:c0 + 70, c0:c0 + 70].copy()
    oy = ox = 2                                  # exactly-centered frame
    c_glob = (FINE_RENDER - 1) / 2.0
    cy = (c_glob - 1.0 - oy) / 3.0
    cx = (c_glob - 1.0 - ox) / 3.0
    yy, xx = np.indices((70, 70))
    r_arc = np.hypot(yy - cy, xx - cx) * NATIVE_PIX
    keep = keep_port & (r_arc > CORE_MASK_ARC)

    import jax

    report = dict(generated_by="05_gen_drizzle_mocks.py --e1d-seeds",
                  kernel="data/noise_kernel_v2d.npz rho_kernel",
                  mask=(f"real v2d keep_mask central 70^2 port "
                        f"[{c0}:{c0+70})^2 & r>{CORE_MASK_ARC}\""),
                  n_keep=int(keep.sum()), frame=[oy, ox], mocks=[])
    for seed in seeds:
        rng = np.random.default_rng(seed)
        truth, flat = sample_truth(rng)
        truth64 = jax.tree_util.tree_map(np.float64, truth)
        scene = np.asarray(sims["full"].simulate(truth64),
                           dtype=np.float64).reshape(FINE_RENDER, FINE_RENDER)
        truth_ll = [truth64[0], truth64[1], []]
        ll_only = np.asarray(sims["ll"].simulate(truth_ll),
                             dtype=np.float64).reshape(FINE_RENDER,
                                                       FINE_RENDER)
        sub = scene[oy:oy + 210, ox:ox + 210]
        model = block_sum(sub, 3)
        err = np.sqrt(SIGMA_BKG ** 2 + np.clip(model, 0.0, None) / EXP_TIME)
        u = exact_ref.sample_stationary_batch(rho, (70, 70), 1, rng,
                                              grid=512)[0]
        img = model + err * u
        arc = block_sum(np.clip(scene - ll_only, 0.0, None)[oy:oy + 210,
                                                            ox:ox + 210], 3)
        snr = arc_snr(arc, err)
        meta = dict(
            e1d=True, seed=seed, generated_by="05_gen_drizzle_mocks.py",
            delta_pix=NATIVE_PIX, frame_offset=[oy, ox],
            psf_pixel_scale=FINE_PIX, model_subtracted=True,
            noise_model=("stationary draw from the REAL v2d fitted kernel "
                         "(exact_ref.sample_stationary_batch, grid 512) "
                         "scaled by the exact per-pixel native err map — "
                         "marginal variance exact, correlation = v2d"),
            mask_port=report["mask"], n_keep=int(keep.sum()),
            sigma_bkg=SIGMA_BKG, exp_time=EXP_TIME, arc_snr=snr,
            truth_sampler="cgl.mocks.sample_truth (gu-2022 Eq.(8))",
        )
        np.savez(outdir / f"e1d_{seed:03d}.npz",
                 img=img, err_map=err, keep_mask=keep, rho_kernel=rho,
                 psf=psf, psf_eff=psf_eff,
                 truth_json=json.dumps(truth),
                 flat_keys=list(flat.keys()),
                 flat_vals=np.array(list(flat.values()), dtype=np.float64),
                 meta=json.dumps(meta))
        print(f"[e1d {seed:03d}] theta_E={flat['theta_E']:.3f} "
              f"gamma={flat['gamma']:.3f} arc_snr~{snr:7.1f} "
              f"u_var={float(np.var(u)):.3f}", flush=True)
        report["mocks"].append(dict(seed=seed, theta_E=flat["theta_E"],
                                    gamma=flat["gamma"], arc_snr=snr,
                                    u_var=float(np.var(u))))
    report["wall_s"] = time.time() - t0
    (REPRO / out_json).write_text(json.dumps(report, indent=2))
    print(f"wrote {REPRO / out_json} ({report['wall_s']:.0f}s)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_MOCKS)
    ap.add_argument("--seeds", default=None,
                    help="explicit seed list/range (e.g. '8-63'); "
                         "overrides --n")
    ap.add_argument("--shapelet-seeds", default=None,
                    help="seeds that get the ADDITIVE shapelet source "
                         "component (E1c marg arm; n_max=4, ridge-prior amps)")
    ap.add_argument("--e1d-seeds", default=None,
                    help="generate E1d native realizations (real-v2d-kernel "
                         "noise) instead of trios")
    ap.add_argument("--out-json", default="data/mocks_report.json")
    args = ap.parse_args()
    t0 = time.time()

    if args.e1d_seeds:
        return gen_e1d(parse_range(args.e1d_seeds), args.out_json)

    seeds = parse_range(args.seeds) if args.seeds else list(range(args.n))
    shp_seeds = set(parse_range(args.shapelet_seeds)) \
        if args.shapelet_seeds else set()

    outdir = DATA / "mocks"
    outdir.mkdir(parents=True, exist_ok=True)

    psf = np.load(Path(VENDOR_SRC) / "gigalens" / "assets" / "psf.npy")
    psf = (psf / psf.sum()).astype(np.float64)

    pipe = DrizzleMockPipeline()
    rho_fine = pipe.fine_rho(max_lag=4)
    rho_binned = pipe.binned_rho(rho_fine)
    psf_eff = effective_fine_psf(psf, pipe, half=(psf.shape[0] // 2) + 3)

    sims = build_sims(psf, psf_eff, with_shapelets=bool(shp_seeds))

    # product geometry: scene (0,0) arcsec sits at global fine pixel
    # (FINE_RENDER-1)/2 = 106.5 -> product pixel 104.5 of 208 (center 103.5)
    center_px = (FINE_RENDER - 1) / 2.0 - CROP0        # 104.5, product coords
    yy, xx = np.indices((N_FINE, N_FINE))
    r_arc = np.hypot(yy - center_px, xx - center_px) * FINE_PIX
    keep_fine = r_arc > CORE_MASK_ARC
    keep_binned = block_sum(keep_fine.astype(np.float64), 2) == 4.0

    meta_common = dict(
        generated_by="05_gen_drizzle_mocks.py",
        delta_pix_fine=FINE_PIX, delta_pix_native=NATIVE_PIX,
        delta_pix_binned=2 * FINE_PIX, scale_ratio=R_INT, pixfrac=1.0,
        offsets_fine=[list(o) for o in OFFSETS_FINE],
        offsets_deviation=("9 full-phase dithers instead of the plan's 3 "
                           "Latin offsets: the 3-frame stack is period-3 "
                           "shift-variant (no exact effective PSF; render "
                           "check failed at 2-27 sigma, see "
                           "data/mocks_report_3frame.json); full-phase "
                           "dithering makes the stack exactly the 3x3-tent "
                           "convolution with the same 1-D noise tent"),
        fine_render=FINE_RENDER, crop=[CROP0, CROP0 + N_FINE],
        scene_center_product_px=center_px,
        sigma_bkg=SIGMA_BKG, exp_time=EXP_TIME, gain=1.0,
        psf_source=("vendored gigalens-sean assets/psf.npy (TinyTim), "
                    "renormalized, DECLARED at 0.04\"/px, supersample=1 — "
                    "self-consistent mock convention"),
        psf_pixel_scale=FINE_PIX,
        core_mask_arc=CORE_MASK_ARC,
        truth_sampler=("cgl.mocks.sample_truth — ported with attribution "
                       "from gu-2022/01_gen_mocks.py Eq.(8) sim dist"),
        model_subtracted=True,
        noise_model=("EXACT analytic: iid native (sigma_bkg^2 + "
                     "clip(model,0)/t_exp), variance propagated through the "
                     "enumerated drizzle operator; stationary correlation = "
                     "cgl.noise.drizzle_acf at the actual offsets "
                     "(constant-sigma limit; support <=2 fine, <=1 binned)"),
        drizzle_operator=("cgl.noise.drizzle_overlap_matrix_1d separable "
                          "pair per frame, row-normalized, equal weights "
                          "(cgl.mocks.DrizzleMockPipeline)"),
    )

    report = dict(generated_by="05_gen_drizzle_mocks.py",
                  gate_sigma=GATE_SIGMA, n_mocks=len(seeds),
                  seeds=list(seeds),
                  shapelet_seeds=sorted(shp_seeds),
                  rho_fine_axis=[float(rho_fine[4, 4 + d]) for d in range(3)],
                  psf_eff_shape=list(psf_eff.shape), mocks=[])
    worst_all = 0.0
    all_pass = True

    import jax

    for seed in seeds:
        rng = np.random.default_rng(seed)
        truth, flat = sample_truth(rng)
        shp = None
        if seed in shp_seeds:
            from cgl.e1 import (SHAPELET_NMAX, SHAPELET_SIGMA0,
                                sample_shapelet_truth)

            beta, amps = sample_shapelet_truth(rng)
            names = [f"amp{str(i).zfill(len(str(amps.size)))}"
                     for i in range(amps.size)]
            shp = dict(beta=beta, amps=amps, names=names,
                       n_max=SHAPELET_NMAX, sigma0=SHAPELET_SIGMA0)
        # strong-f64 leaves force the f32 coordinate grids to promote (the
        # 01_parity_harness gu2022_forward_check recipe; grids are hardcoded
        # float32 in the vendored simulator)
        truth64 = jax.tree_util.tree_map(np.float64, truth)

        scene = np.asarray(sims["full"].simulate(truth64), dtype=np.float64)
        scene = scene.reshape(FINE_RENDER, FINE_RENDER)
        if shp is not None:
            # shapelet CENTER TIED to the source Sersic center (the E1c fit
            # model's convention; see cgl.e1.build_truth_prior)
            # amp leaves must be (bs,) arrays: the non-interpolating shapelet
            # profile einsums the amp stack as (depth, bs)
            shp_params = [truth64[0], [dict(
                center_x=truth64[2][0]["center_x"],
                center_y=truth64[2][0]["center_y"],
                beta=np.full(1, shp["beta"], dtype=np.float64),
                **{n: np.full(1, a, dtype=np.float64)
                   for n, a in zip(shp["names"], shp["amps"])})]]
            shp_scene = np.asarray(sims["shp"].simulate(shp_params),
                                   dtype=np.float64).reshape(FINE_RENDER,
                                                             FINE_RENDER)
            scene = scene + shp_scene
        truth_ll = [truth64[0], truth64[1], []]
        ll_only = np.asarray(sims["ll"].simulate(truth_ll), dtype=np.float64)
        ll_only = ll_only.reshape(FINE_RENDER, FINE_RENDER)

        natives_clean = pipe.natives_from_fine(scene)
        noisy_nats, err_nats = [], []
        for nat in natives_clean:
            noisy, err = add_noise_native(nat, rng)
            noisy_nats.append(noisy)
            err_nats.append(err)

        fine_model = pipe.drizzle(natives_clean)
        fine_img = pipe.drizzle(noisy_nats)
        fine_var = pipe.fine_var(err_nats)
        fine_err = np.sqrt(fine_var)

        binned_model = block_sum(fine_model, 2)
        binned_img = block_sum(fine_img, 2)
        binned_err = np.sqrt(pipe.binned_var(fine_var, rho_fine))

        # ---- render check: fine render with PSF_eff vs drizzled noiseless --
        direct = np.asarray(sims["full_eff"].simulate(truth64),
                            dtype=np.float64).reshape(FINE_RENDER,
                                                      FINE_RENDER)
        if shp is not None:
            direct = direct + np.asarray(
                sims["shp_eff"].simulate(shp_params),
                dtype=np.float64).reshape(FINE_RENDER, FINE_RENDER)
        direct = direct[CROP0:CROP0 + N_FINE, CROP0:CROP0 + N_FINE]
        dev = (direct - fine_model) / fine_err
        worst = float(np.max(np.abs(dev[keep_fine])))
        ok = worst < GATE_SIGMA
        worst_all = max(worst_all, worst)
        all_pass &= ok

        arc_fine = pipe.drizzle(pipe.natives_from_fine(
            np.clip(scene - ll_only, 0.0, None)))
        snr = arc_snr(arc_fine, fine_err)

        meta = dict(meta_common)
        meta.update(seed=seed,
                    render_check_max_abs_dev_sigma=worst,
                    render_check_pass=bool(ok), arc_snr=snr)
        extra = {}
        if shp is not None:
            meta.update(has_shapelets=True, shapelet_beta=shp["beta"],
                        shapelet_sigma0=shp["sigma0"],
                        shapelet_n_max=shp["n_max"],
                        shapelet_center="tied to source Sersic center",
                        shapelet_amp_prior="N(0, sigma0/sqrt(i+1)) — the "
                                           "exact ridge prior the E1c fit "
                                           "marginalizes under")
            extra["shp_amps"] = shp["amps"]
        path = outdir / f"mock_{seed:03d}.npz"
        np.savez(
            path,
            **extra,
            img=fine_img.astype(np.float64),
            model=fine_model.astype(np.float64),
            err_map=fine_err.astype(np.float64),
            keep_mask=keep_fine,
            rho_kernel=rho_fine,
            binned_img=binned_img.astype(np.float64),
            binned_model=binned_model.astype(np.float64),
            binned_err=binned_err.astype(np.float64),
            binned_keep=keep_binned,
            rho_kernel_binned=rho_binned,
            native_img=np.stack(noisy_nats),
            native_model=np.stack(natives_clean),
            native_err=np.stack(err_nats),
            offsets=np.asarray(OFFSETS_FINE),
            psf=psf, psf_eff=psf_eff,
            truth_json=json.dumps(truth),
            flat_keys=list(flat.keys()),
            flat_vals=np.array(list(flat.values()), dtype=np.float64),
            meta=json.dumps(meta),
        )
        print(f"[{seed:03d}] theta_E={flat['theta_E']:.3f} "
              f"gamma={flat['gamma']:.3f} arc_snr~{snr:7.1f} "
              f"render_check={worst:.4f} sigma "
              f"{'PASS' if ok else 'FAIL'} -> {path.name}", flush=True)
        report["mocks"].append(dict(
            seed=seed, theta_E=flat["theta_E"], gamma=flat["gamma"],
            arc_snr=snr, render_check_max_dev_sigma=worst,
            render_check_pass=bool(ok),
            has_shapelets=bool(shp is not None)))

    report["worst_render_check_sigma"] = worst_all
    report["all_gates_pass"] = bool(all_pass)
    report["wall_s"] = time.time() - t0
    (REPRO / args.out_json).write_text(json.dumps(report, indent=2))
    print(f"\nworst render check over {len(seeds)} mocks: "
          f"{worst_all:.4f} sigma -> "
          f"{'PASS' if all_pass else 'FAIL'} (< {GATE_SIGMA})")
    print(f"wrote {REPRO / args.out_json} ({report['wall_s']:.0f}s)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
