"""01_parity_harness.py — pre-registered parity gates A–E (the campaign gate).

Builds foundry-i's `_hmc_lib_marg.build_model_marg` (imported by path,
UNMODIFIED) and our `cgl.likelihood.build_marg_model` on the SAME vendored
gigalens-sean (multinode-2025 @ 58ec9a7) and the SAME inputs (cutout_F140W
SCI/WHT + empirical_psf.npy, mirrored into a product dict following the
original's data block line-by-line), then checks at z_ref =
map_marg_pd.npz['qz_refined'] and 3 seeded perturbations z_ref + 0.1 N(0,1)
(np.random.default_rng seeds 0,1,2):

  HARD GATES (README §Verification, frozen at P0):
    A  forward/deterministic model image  max|d|/max|img| <= 1e-12  (z_ref)
    B  |d log-posterior|                 <= 1e-8   (all 4 points)
    C  gradient rel-L2                   <= 1e-8   (all 4 points)
    D  delta-kernel conv-whitener == diagonal path |d logp| <= 1e-10
       (our stack only, all 4 points)
    E  our -1/2 logdetA vs numpy slogdet(densified A) <= 1e-10  (z_ref)

  INFORMATIONAL (report-only; investigate before accepting > 1e-5):
    - our/foundry-i logp at z_ref vs the value stored in map_marg_pd.npz
      (stored value was computed on the PIP gigalens era stack + A16 GPU)
    - gu-2022 system_000 mock forward image re-rendered with the vendored lib
      vs the stored (float32-era) model image.

Foundry-i's namespace does not export its M_det simulator, so gate A rebuilds
it in this harness from THEIR returned bij + sim_config with the shared
vendored classes (documented in the report); their shapelet_amps(z) — fully
their code end-to-end — provides the independent amplitude/image check.

Writes data/parity_report.json; exits nonzero on any hard-gate failure.

Run (GPU 8, an L4; never GPU 9):
  GIGALENS_X64=1 CUDA_VISIBLE_DEVICES=8 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  XLA_FLAGS=--xla_gpu_autotune_level=0 \
  /raid/benson/.venvs/cgl/bin/python 01_parity_harness.py
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

# ---- env BEFORE jax import (x64 + single pinned GPU) ------------------------
os.environ["GIGALENS_X64"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CGL_GPU", "8"))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

REPRO = Path(__file__).resolve().parent

from cgl.paths import (  # noqa: E402
    CUTOUT_F140W, DATA, EMPIRICAL_PSF, EXPECTED_VENDOR_REF, FOUNDRY_I,
    GU2022_MOCK000, MAP_MARG_PD, bootstrap_vendor,
)

bootstrap_vendor()                      # vendored gigalens first on sys.path
sys.path.insert(1, str(FOUNDRY_I))      # for `import _hmc_lib_marg` (unpatched)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", str(REPRO / ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from cgl import guards  # noqa: E402

EXP_TIME = 1197.7  # _hmc_lib_marg.EXP_TIME (v2-class F140W)

GATE_TOL = {"A": 1e-12, "B": 1e-8, "C": 1e-8, "D": 1e-10, "E": 1e-10}
INFO_TOL = 1e-5


def build_parity_product():
    """Mirror _hmc_lib_marg's data block (lines 227-247) into a product dict.

    err_map is computed with the SAME jnp ops on-device so the bits match the
    original exactly; the mask is folded in by build_marg_model (err -> 1e10).
    """
    with fits.open(CUTOUT_F140W) as h:
        sci = h["SCI"].data.astype(np.float64)
        wht = h["WHT"].data.astype(np.float64)
    sky = float(np.median(sci))
    img = (sci - sky).astype(np.float64)
    background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
    num_pix = img.shape[0]
    err_map = np.asarray(
        jnp.sqrt(background_rms ** 2 +
                 jnp.clip(jnp.asarray(img, dtype=jnp.float64), 0.0, np.inf)
                 / EXP_TIME))
    yy, xx = np.indices(img.shape)
    r_center = np.sqrt((xx - num_pix // 2) ** 2 + (yy - num_pix // 2) ** 2)
    keep_mask = r_center > 1.5
    psf = np.load(EMPIRICAL_PSF).astype(np.float64)
    # NOTE meta deliberately does NOT declare psf_pixel_scale: empirical_psf.npy
    # predates the foundry-i delta_pix-sampling fix and the parity reference
    # (_hmc_lib_marg + map_marg_pd.npz) is defined on that legacy convention.
    meta = dict(
        delta_pix=0.13, num_pix=int(num_pix), supersample=2, exp_time=EXP_TIME,
        sky=sky, background_rms=background_rms,
        source="parity mirror of foundry-i _hmc_lib_marg data block "
               "(cutout_F140W.fits SCI/WHT + empirical_psf.npy)",
    )
    return dict(img=img, err_map=err_map, keep_mask=keep_mask, psf=psf,
                meta=meta)


def rebuild_foundry_m_det(m_fi):
    """Deterministic-image closure for the foundry-i stack.

    _hmc_lib_marg does not export sim_det, so rebuild it from THEIR bij +
    THEIR sim_config with the shared vendored classes (documented harness-side
    reconstruction; the fully-their-code cross-check is shapelet_amps)."""
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.simulator import LensSimulator
    from gigalens.model import PhysicalModel

    phys_det = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)] * 4,
        [sersic.SersicEllipse(use_lstsq=False)],
    )
    sim_det = LensSimulator(phys_det, m_fi.sim_config, bs=1)
    inv_cf = 1.0 / float(sim_det.conversion_factor)

    @jax.jit
    def m_det(z):
        x = m_fi.bij.forward(list(z[None, :].T))
        return jnp.squeeze(sim_det.simulate(x)) * inv_cf

    return m_det


def gu2022_forward_check():
    """Re-render gu-2022 system_000's model image with the vendored lib
    following 01_gen_mocks.py's gigalens path; report max|d|/max|model|."""
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.simulator import LensSimulator
    from gigalens.model import PhysicalModel
    from gigalens.simulator import SimulatorConfig

    d = np.load(GU2022_MOCK000, allow_pickle=True)
    truth = json.loads(str(d["truth_json"]))
    # json floats are weak-typed: in an x64 session the f32 coordinate grids
    # would keep the light image f32 while the PSF kernel is f64 (conv dtype
    # error). Cast to strong float64 (the mock itself was an f32-era render).
    truth = jax.tree_util.tree_map(np.float64, truth)
    stored = np.asarray(d["model"], dtype=np.float64)  # float32-era render
    cfg = SimulatorConfig(delta_pix=float(d["delta_pix"]),
                          num_pix=int(d["num_pix"]),
                          supersample=int(d["supersample"]),
                          kernel=np.asarray(d["psf"], dtype=np.float32))
    pm = PhysicalModel([epl.EPL(50), shear.Shear()],
                       [sersic.SersicEllipse(use_lstsq=False)],
                       [sersic.SersicEllipse(use_lstsq=False)])
    sim = LensSimulator(pm, cfg, bs=1)
    rendered = np.asarray(sim.simulate(truth), dtype=np.float64)
    max_rel = float(np.max(np.abs(rendered - stored)) / np.max(np.abs(stored)))
    return dict(
        mock=str(GU2022_MOCK000), max_abs_diff=float(np.max(np.abs(rendered - stored))),
        max_model=float(np.max(np.abs(stored))), max_rel_diff=max_rel,
        note="stored model was rendered float32 on the pip multi-node lib; "
             "expect ~f32-level agreement (informational tol 1e-5)")


def main():
    t_start = time.time()
    print(f"devices: {jax.devices()}  x64={jax.config.jax_enable_x64}", flush=True)
    guards.require_x64()
    guards.require_gpu()
    guards.require_single_device()

    # ---- inputs / reference points -----------------------------------------
    ref = np.load(MAP_MARG_PD)
    z_ref = np.asarray(ref["qz_refined"], dtype=np.float64)
    stored_logp = float(ref["logp"])
    points = [("z_ref", z_ref)]
    for s in (0, 1, 2):
        rng = np.random.default_rng(s)
        points.append((f"z_pert{s}", z_ref + 0.1 * rng.standard_normal(z_ref.size)))

    product = build_parity_product()

    # ---- build both stacks on the SAME vendored lib -------------------------
    print("building foundry-i _hmc_lib_marg.build_model_marg (vendored lib) ...",
          flush=True)
    import _hmc_lib_marg  # noqa: E402  (foundry-i module, unpatched)
    assert str(Path(_hmc_lib_marg.__file__).resolve().parent) == str(FOUNDRY_I)
    import gigalens
    print(f"  gigalens resolved at: {gigalens.__file__}", flush=True)
    t0 = time.time()
    m_fi = _hmc_lib_marg.build_model_marg(x64=True)
    print(f"  built in {time.time()-t0:.1f}s (ndim={m_fi.ndim})", flush=True)

    print("building cgl.likelihood.build_marg_model (diagonal path) ...", flush=True)
    from cgl.likelihood import build_marg_model
    from cgl.whiten import make_conv_whitener
    t0 = time.time()
    m_cgl = build_marg_model(product, n_max=6, supersample=2, delta_pix=0.13,
                             exp_time=EXP_TIME)
    print(f"  built in {time.time()-t0:.1f}s (ndim={m_cgl.ndim})", flush=True)

    print("building cgl model with delta-kernel conv whitener (gate D) ...",
          flush=True)
    sqrt_d_inv = np.asarray(1.0 / m_cgl.masked_err_map)
    m_conv = build_marg_model(
        product, n_max=6, supersample=2, delta_pix=0.13, exp_time=EXP_TIME,
        whiten_fn=make_conv_whitener(np.ones((1, 1)), sqrt_d_inv,
                                     np.asarray(product["keep_mask"])))

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {
            "python": sys.version.split()[0],
            "hostname": platform.node(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "device": str(jax.devices()[0]),
            "device_kind": jax.devices()[0].device_kind,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
            "x64": bool(jax.config.jax_enable_x64),
            "vendored_gigalens_ref": EXPECTED_VENDOR_REF,
            "gigalens_file": gigalens.__file__,
            "foundry_module": _hmc_lib_marg.__file__,
        },
        "reference": {
            "map_marg_pd": str(MAP_MARG_PD),
            "z_ref_key": "qz_refined",
            "stored_logp": stored_logp,
            "perturbation": "z_ref + 0.1*default_rng(seed).standard_normal(46), seeds 0,1,2",
        },
        "gates": {}, "informational": {}, "data_parity": {},
    }

    # ---- data-array parity (diagnostic, not a gate) --------------------------
    report["data_parity"] = {
        "max_abs_diff_observed_image": float(jnp.max(jnp.abs(
            m_fi.prob_model.observed_image - m_cgl.Y))),
        "max_abs_diff_masked_err_map": float(jnp.max(jnp.abs(
            m_fi.prob_model.err_map - m_cgl.masked_err_map))),
    }
    print(f"data parity: dY={report['data_parity']['max_abs_diff_observed_image']:.3e} "
          f"dErr={report['data_parity']['max_abs_diff_masked_err_map']:.3e}", flush=True)

    # ---- Gate A: forward/deterministic model image at z_ref ------------------
    print("gate A: forward images at z_ref ...", flush=True)
    fi_m_det = rebuild_foundry_m_det(m_fi)
    zj = jnp.asarray(z_ref, dtype=jnp.float64)
    M_fi = np.asarray(fi_m_det(zj))
    M_cgl = np.asarray(m_cgl.m_det(zj))
    scale = float(np.max(np.abs(M_fi)))
    a_mdet = float(np.max(np.abs(M_fi - M_cgl)) / scale)

    amps_fi = np.asarray(m_fi.shapelet_amps(zj))          # fully their code
    internals = m_cgl.marg_internals(z_ref)
    amps_cgl = internals["a_star"]
    d_amps = float(np.max(np.abs(amps_fi - amps_cgl)))
    rel_amps = d_amps / float(np.max(np.abs(amps_fi)))

    full_cgl = np.asarray(m_cgl.model_image(zj))
    full_fi = M_fi + np.sum(internals["ret"] * amps_fi[None, None, :], axis=-1)
    a_full = float(np.max(np.abs(full_cgl - full_fi)) / np.max(np.abs(full_fi)))

    gateA = max(a_mdet, a_full)
    report["gates"]["A"] = {
        "description": "forward model image max|d|/max|img| at z_ref",
        "threshold": GATE_TOL["A"], "achieved": gateA,
        "pass": bool(gateA <= GATE_TOL["A"]),
        "detail": {
            "m_det_rel": a_mdet, "full_model_rel": a_full,
            "max_abs_m_det": scale,
            "shapelet_amps_max_abs_diff": d_amps,
            "shapelet_amps_max_rel_diff": rel_amps,
            "note": "foundry m_det rebuilt harness-side from their bij+sim_config "
                    "(module exports no image fn); amps use their shapelet_amps.",
        },
    }
    print(f"  m_det rel={a_mdet:.3e}  full rel={a_full:.3e}  "
          f"amps max|d|={d_amps:.3e}", flush=True)

    # ---- Gates B & C: logpost + grad at all 4 points --------------------------
    print("gates B/C: logpost + grad at 4 points ...", flush=True)
    g_fi = jax.jit(jax.grad(m_fi.target_log_prob_fn))
    g_cgl = jax.jit(jax.grad(m_cgl.target_log_prob_fn))
    per_point = {}
    b_worst, c_worst = 0.0, 0.0
    for name, z in points:
        zj = jnp.asarray(z, dtype=jnp.float64)
        lp_fi = float(m_fi.target_log_prob_fn(zj))
        lp_cgl = float(m_cgl.target_log_prob_fn(zj))
        d_lp = abs(lp_fi - lp_cgl)
        gf = np.asarray(g_fi(zj), dtype=np.float64)
        gc = np.asarray(g_cgl(zj), dtype=np.float64)
        denom = max(np.linalg.norm(gf), np.linalg.norm(gc))
        rel_g = float(np.linalg.norm(gf - gc) / denom)
        lp_conv = float(m_conv.target_log_prob_fn(zj))
        d_conv = abs(lp_cgl - lp_conv)
        per_point[name] = {
            "logp_foundry": lp_fi, "logp_cgl": lp_cgl, "abs_dlogp": d_lp,
            "grad_norm_foundry": float(np.linalg.norm(gf)),
            "grad_norm_cgl": float(np.linalg.norm(gc)),
            "grad_rel_l2": rel_g,
            "logp_cgl_convwhiten": lp_conv, "abs_dlogp_conv_vs_diag": d_conv,
        }
        b_worst = max(b_worst, d_lp)
        c_worst = max(c_worst, rel_g)
        print(f"  {name:8s} lp_fi={lp_fi:+.9f} lp_cgl={lp_cgl:+.9f} "
              f"|d|={d_lp:.3e}  grad_relL2={rel_g:.3e}  |d_conv|={d_conv:.3e}",
              flush=True)
    report["points"] = per_point
    report["gates"]["B"] = {
        "description": "|d log-posterior| foundry vs cgl, worst of 4 points",
        "threshold": GATE_TOL["B"], "achieved": b_worst,
        "pass": bool(b_worst <= GATE_TOL["B"]),
    }
    report["gates"]["C"] = {
        "description": "gradient rel-L2 foundry vs cgl, worst of 4 points",
        "threshold": GATE_TOL["C"], "achieved": c_worst,
        "pass": bool(c_worst <= GATE_TOL["C"]),
    }

    # ---- Gate D: delta-kernel conv whitener == diagonal path ------------------
    d_worst = max(per_point[n]["abs_dlogp_conv_vs_diag"] for n, _ in points)
    report["gates"]["D"] = {
        "description": "delta-kernel conv-whitener vs diagonal |d logp|, worst of 4",
        "threshold": GATE_TOL["D"], "achieved": d_worst,
        "pass": bool(d_worst <= GATE_TOL["D"]),
    }

    # ---- Gate E: -1/2 logdetA vs numpy slogdet at z_ref -----------------------
    A = internals["A"]
    sign, logdet_np = np.linalg.slogdet(np.asarray(A, dtype=np.float64))
    assert sign > 0, f"A not PD?! slogdet sign={sign}"
    e_val = abs(-0.5 * internals["logdetA"] - (-0.5 * logdet_np))
    report["gates"]["E"] = {
        "description": "|(-1/2 logdetA)_chol - (-1/2 slogdet(A))_numpy| at z_ref",
        "threshold": GATE_TOL["E"], "achieved": float(e_val),
        "pass": bool(e_val <= GATE_TOL["E"]),
        "detail": {"logdetA_chol": internals["logdetA"], "logdetA_numpy": float(logdet_np),
                   "cond_A": float(np.linalg.cond(A))},
    }
    print(f"gate E: logdetA chol={internals['logdetA']:.12f} "
          f"numpy={logdet_np:.12f} |d(-1/2 logdet)|={e_val:.3e}", flush=True)

    # ---- informational cross-checks -------------------------------------------
    info = {}
    lp_ref_cgl = per_point["z_ref"]["logp_cgl"]
    lp_ref_fi = per_point["z_ref"]["logp_foundry"]
    info["stored_map_logp"] = {
        "stored": stored_logp,
        "cgl_at_z_ref": lp_ref_cgl, "abs_diff_cgl": abs(lp_ref_cgl - stored_logp),
        "foundry_at_z_ref": lp_ref_fi, "abs_diff_foundry": abs(lp_ref_fi - stored_logp),
        "note": "stored logp came from the pip-gigalens-era stack on an A16; "
                "both of today's values run the vendored lib on an L4, so "
                "abs_diff_foundry isolates lib/arch drift from port drift.",
        "tol_informational": INFO_TOL,
    }
    if abs(lp_ref_cgl - stored_logp) > INFO_TOL:
        print(f"  WARN informational: |cgl - stored| = "
              f"{abs(lp_ref_cgl - stored_logp):.3e} > {INFO_TOL:g} "
              f"(foundry-vs-stored = {abs(lp_ref_fi - stored_logp):.3e})", flush=True)
    try:
        info["gu2022_forward"] = gu2022_forward_check()
        ok = info["gu2022_forward"]["max_rel_diff"] <= INFO_TOL
        print(f"gu-2022 forward re-render: max_rel_diff="
              f"{info['gu2022_forward']['max_rel_diff']:.3e} "
              f"({'ok' if ok else 'WARN > 1e-5'})", flush=True)
    except Exception as exc:  # informational must never kill the harness
        info["gu2022_forward"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  WARN gu-2022 informational check errored: {exc}", flush=True)
    report["informational"] = info

    # ---- verdict + report ------------------------------------------------------
    all_pass = all(g["pass"] for g in report["gates"].values())
    report["all_hard_gates_pass"] = bool(all_pass)
    report["wall_s"] = time.time() - t_start

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "parity_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}", flush=True)

    print("\n=== PARITY SUMMARY (hard gates, pre-registered) ===", flush=True)
    print(f"{'gate':>4s}  {'achieved':>12s}  {'threshold':>10s}  verdict", flush=True)
    for k in ("A", "B", "C", "D", "E"):
        g = report["gates"][k]
        print(f"{k:>4s}  {g['achieved']:12.3e}  {g['threshold']:10.0e}  "
              f"{'PASS' if g['pass'] else 'FAIL'}", flush=True)
    print(f"\ninformational: |logp_cgl - stored_map_logp| = "
          f"{info['stored_map_logp']['abs_diff_cgl']:.3e}  "
          f"(foundry-vs-stored {info['stored_map_logp']['abs_diff_foundry']:.3e})",
          flush=True)
    if "max_rel_diff" in info.get("gu2022_forward", {}):
        print(f"informational: gu-2022 forward max_rel_diff = "
              f"{info['gu2022_forward']['max_rel_diff']:.3e}", flush=True)
    print(f"\nALL HARD GATES {'PASS' if all_pass else 'FAIL'}  "
          f"({report['wall_s']:.0f}s)", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
