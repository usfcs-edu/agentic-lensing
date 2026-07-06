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


def build_sims(psf, psf_eff):
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
    return dict(
        full=LensSimulator(pm_full, cfg, bs=1),
        ll=LensSimulator(pm_ll, cfg, bs=1),
        full_eff=LensSimulator(pm_full, cfg_eff, bs=1),
    )


def arc_snr(arc_only, err_map):
    """gu-2022 integrated arc SNR diagnostic (ported convention)."""
    mask = arc_only > err_map
    if mask.sum() == 0:
        return 0.0
    return float(arc_only[mask].sum() / np.sqrt((err_map[mask] ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_MOCKS)
    ap.add_argument("--out-json", default="data/mocks_report.json")
    args = ap.parse_args()
    t0 = time.time()

    outdir = DATA / "mocks"
    outdir.mkdir(parents=True, exist_ok=True)

    psf = np.load(Path(VENDOR_SRC) / "gigalens" / "assets" / "psf.npy")
    psf = (psf / psf.sum()).astype(np.float64)

    pipe = DrizzleMockPipeline()
    rho_fine = pipe.fine_rho(max_lag=4)
    rho_binned = pipe.binned_rho(rho_fine)
    psf_eff = effective_fine_psf(psf, pipe, half=(psf.shape[0] // 2) + 3)

    sims = build_sims(psf, psf_eff)

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
                  gate_sigma=GATE_SIGMA, n_mocks=args.n,
                  rho_fine_axis=[float(rho_fine[4, 4 + d]) for d in range(3)],
                  psf_eff_shape=list(psf_eff.shape), mocks=[])
    worst_all = 0.0
    all_pass = True

    import jax

    for seed in range(args.n):
        rng = np.random.default_rng(seed)
        truth, flat = sample_truth(rng)
        # strong-f64 leaves force the f32 coordinate grids to promote (the
        # 01_parity_harness gu2022_forward_check recipe; grids are hardcoded
        # float32 in the vendored simulator)
        truth64 = jax.tree_util.tree_map(np.float64, truth)

        scene = np.asarray(sims["full"].simulate(truth64), dtype=np.float64)
        scene = scene.reshape(FINE_RENDER, FINE_RENDER)
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
        path = outdir / f"mock_{seed:03d}.npz"
        np.savez(
            path,
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
            render_check_pass=bool(ok)))

    report["worst_render_check_sigma"] = worst_all
    report["all_gates_pass"] = bool(all_pass)
    report["wall_s"] = time.time() - t0
    (REPRO / args.out_json).write_text(json.dumps(report, indent=2))
    print(f"\nworst render check over {args.n} mocks: "
          f"{worst_all:.4f} sigma -> "
          f"{'PASS' if all_pass else 'FAIL'} (< {GATE_SIGMA})")
    print(f"wrote {REPRO / args.out_json} ({report['wall_s']:.0f}s)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
