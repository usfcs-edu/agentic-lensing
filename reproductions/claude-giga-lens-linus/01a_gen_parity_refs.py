"""01a_gen_parity_refs.py — OLD-STACK reference artifacts for gates F1-F7.

RUNS UNDER THE OLD VENV (/raid/benson/.venvs/cgl/bin/python): the validated
claude-giga-lens stack (vendored gigalens-sean @58ec9a7 + cgl @P0-frozen).
Reference-artifact parity design per ledger D3: both stacks are package
`gigalens`, so the old side DUMPS and the scene side COMPARES — no dual
import. Follows ../claude-giga-lens/01_parity_harness.py's bootstrap pattern.

For each product (foundry-i cutout_v2d, cutout_v3b) and each of 4 points
(z_ref = map_marg_pd['qz_refined'] + 3 seeded 0.1*N(0,1) perturbations,
np.random.default_rng seeds 0,1,2), evaluates on the OLD stack:

  * constrained params (46 labels, canonical cgl2.param_map order)
  * forward deterministic image M_det (no-conversion-factor units)
  * design-matrix columns ret (h, w, 28)  [gigalens lstsq_simulate `ret`]
  * diagonal ridge-marginalized internals: b, A, a_star, logdetA, logL_data,
    whitened chi2 (pre-marg and at a*), log_prior, logpost
  * gradient of the diagonal-marg logL_data wrt the 46 CONSTRAINED params
    (bijector-free — the cross-stack-comparable object)
  * forward-fixed-amps references (amps frozen at a_ref = a_star(z_ref)):
    model image, masked Gaussian loglik + chi2 (the stock-ImageLikelihoodTerm
    functional) + its constrained-space gradient
  * convention arrays: float32 coordinate grids (img_X/img_Y), the UNFLIPPED
    lenstronomy subgrid PSF kernel, masks, error maps, Lambda_diag, near_xy

plus mirror-certification numbers: the harness-side mirror of
cgl.likelihood.build_marg_model's per-eval pipeline (needed because that
module exposes no x-space gradient surface) is validated against the frozen
model at every point (|d logL_data|, |d logpost|) before its gradients are
trusted.

Writes  data/parity_refs.npz  with a full provenance dict.

Run (L4, GPU 9):
  GIGALENS_X64=1 CUDA_VISIBLE_DEVICES=9 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /raid/benson/.venvs/cgl/bin/python 01a_gen_parity_refs.py
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

# ---- env BEFORE jax import (x64 + single pinned GPU) ------------------------
os.environ["GIGALENS_X64"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CGL2_GPU", "9"))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # for cgl2.param_map (numpy-only label map)

from cgl.paths import (  # noqa: E402  (old campaign package, cgl venv)
    CUTOUT_V2D, CUTOUT_V3B, EXPECTED_VENDOR_REF, MAP_MARG_PD,
    NEARBY_GALAXY_LOC, bootstrap_vendor, load_product,
)

bootstrap_vendor()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from cgl import guards  # noqa: E402
from cgl.likelihood import _flat_named46, build_marg_model  # noqa: E402
from cgl2.param_map import OLD_LABELS_46  # noqa: E402  (labels only)

PRODUCTS = {
    "v2d": dict(cutout=CUTOUT_V2D, delta_pix=0.13, supersample=2),
    "v3b": dict(cutout=CUTOUT_V3B, delta_pix=0.08, supersample=2),
}
N_MAX = 6
EXP_TIME = 1197.7

_MASS = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y"]
_SHEAR = ["gamma1", "gamma2"]
_SERSIC = ["R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y", "Ie"]
_SHP = ["beta", "center_x", "center_y"]
_IDX = {lab: i for i, lab in enumerate(OLD_LABELS_46)}


def x_tree_from_vec(v):
    """(46,) canonical-order jnp vector -> old-model params pytree, leaves (1,)."""
    def comp(prefix, names):
        return {n: v[_IDX[f"{prefix}.{n}"]][None] for n in names}
    mass = comp("mass", _MASS)
    sh = comp("shear", _SHEAR)
    LL = [comp(f"LL{i}", _SERSIC) for i in range(4)]
    srcS = comp("srcS", _SERSIC)
    shp = comp("srcShp", _SHP)
    return [[mass, sh], LL, [srcS, shp]]


def make_mirror(model, product):
    """Harness-side mirror of cgl.likelihood.build_marg_model's per-eval
    pipeline as functions of the CONSTRAINED 46-vector (canonical order).

    Documented reconstruction (the old module exposes no x-space surface);
    certified against the frozen model per point before its grads are used.
    """
    from gigalens.jax.profiles.light import sersic, shapelets
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.simulator import LensSimulator
    from gigalens.model import PhysicalModel
    from objax.functional import average_pool_2d

    phys_det = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)] * 4,
        [sersic.SersicEllipse(use_lstsq=False)])
    sim_det = LensSimulator(phys_det, model.sim_config, bs=1)
    inv_cf = 1.0 / float(sim_det.conversion_factor)
    phys_shp = PhysicalModel(
        [epl.EPL(50), shear.Shear()], [],
        [shapelets.Shapelets(n_max=N_MAX, use_lstsq=True, interpolate=False)])
    sim_shp = LensSimulator(phys_shp, model.sim_config, bs=1)
    ss = int(model.sim_config.supersample)

    Y = model.Y
    sqrtW = 1.0 / model.masked_err_map
    Wcol = sqrtW[..., None]
    Lam = jnp.asarray(model.Lambda_diag)
    err_raw = jnp.asarray(np.asarray(product["err_map"], dtype=np.float64))
    keep = jnp.asarray(np.asarray(product["keep_mask"], dtype=bool))

    def design_ret(x):
        bx, by = sim_shp._beta(x[0])
        im = jnp.zeros((0, *sim_shp.img_X.shape))
        for lm, p in zip(sim_shp.phys_model.source_light, [x[2][1]]):
            im = jnp.concatenate((im, lm.light(bx, by, **p)), axis=0)
        im = jnp.nan_to_num(im)
        im = jnp.transpose(im, (3, 0, 1, 2))
        ret = jax.lax.conv_general_dilated(
            im, sim_shp.kernel, (1, 1), padding="SAME",
            feature_group_count=sim_shp.depth,
            dimension_numbers=("NCHW", "HWOI", "NCHW"))
        if ss != 1:
            ret = average_pool_2d(ret, size=(ss, ss), padding="SAME")
        return jnp.squeeze(jnp.transpose(ret, (0, 2, 3, 1)), axis=0)

    def parts(v):
        x = x_tree_from_vec(v)
        M_det = jnp.squeeze(sim_det.simulate(x)) * inv_cf
        ret = design_ret(x)
        Xw = jnp.reshape(ret * Wcol, (-1, sim_shp.depth))
        Rw = jnp.reshape((Y - M_det) * sqrtW, (-1,))
        return M_det, ret, Xw, Rw

    def logL_marg(v):
        _, _, Xw, Rw = parts(v)
        b = Xw.T @ Rw
        A = Xw.T @ Xw + jnp.diag(Lam)
        chol = jnp.linalg.cholesky(A)
        a = jax.scipy.linalg.cho_solve((chol, True), b)
        logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        return -0.5 * jnp.sum(Rw ** 2) + 0.5 * jnp.dot(b, a) - 0.5 * logdetA

    def ll_fwd(v, a_ref):
        M_det, ret, _, _ = parts(v)
        mi = M_det + jnp.sum(ret * a_ref[None, None, :], axis=-1)
        d2 = ((mi - Y) / err_raw) ** 2
        chi2 = jnp.sum(jnp.where(keep, d2, 0.0))
        norm = jnp.sum(jnp.where(keep, jnp.log(2 * jnp.pi * err_raw ** 2), 0.0))
        return -0.5 * (chi2 + norm), chi2, mi

    return dict(
        logL_marg=jax.jit(logL_marg),
        grad_marg=jax.jit(jax.grad(logL_marg)),
        ll_fwd=jax.jit(ll_fwd),
        grad_fwd=jax.jit(jax.grad(lambda v, a: ll_fwd(v, a)[0])),
        img_X32=np.asarray(sim_det.img_X)[:, :, 0],
        img_Y32=np.asarray(sim_det.img_Y)[:, :, 0],
        kernel_unflipped=np.asarray(sim_shp.kernel)[::-1, ::-1, 0, 0],
        conversion_factor=float(sim_det.conversion_factor),
    )


def main():
    t0 = time.time()
    print(f"devices: {jax.devices()}  x64={jax.config.jax_enable_x64}", flush=True)
    guards.require_x64()
    guards.require_gpu()
    guards.require_single_device()

    ref = np.load(MAP_MARG_PD)
    z_ref = np.asarray(ref["qz_refined"], dtype=np.float64)
    points = [("z_ref", z_ref)]
    for s in (0, 1, 2):
        rng = np.random.default_rng(s)
        points.append((f"z_pert{s}", z_ref + 0.1 * rng.standard_normal(z_ref.size)))

    nb = np.load(NEARBY_GALAXY_LOC)
    near_xy = (float(nb["arcsec_x"]), float(nb["arcsec_y"]))

    out = {}
    prov = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="01a_gen_parity_refs.py",
        stack="cgl (claude-giga-lens P0-frozen) on vendored gigalens-sean",
        vendored_gigalens_sean_ref=EXPECTED_VENDOR_REF,
        python=sys.version.split()[0], hostname=platform.node(),
        jax=jax.__version__, numpy=np.__version__,
        device=str(jax.devices()[0]),
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
        XLA_FLAGS=os.environ.get("XLA_FLAGS"),
        z_ref_source=str(MAP_MARG_PD) + "['qz_refined']",
        perturbation="z_ref + 0.1*default_rng(seed).standard_normal(46), seeds 0,1,2",
        near_xy=list(near_xy), n_max=N_MAX, exp_time=EXP_TIME,
        labels_order="cgl2.param_map.OLD_LABELS_46 (canonical)",
        products={t: dict(cutout=str(c["cutout"]), delta_pix=c["delta_pix"],
                          supersample=c["supersample"])
                  for t, c in PRODUCTS.items()},
        mirror_note=("x-space gradients come from a harness-side mirror of "
                     "cgl.likelihood.build_marg_model's pipeline; certified "
                     "per point against the frozen model (see "
                     "'<tag>:mirror_dlogL' / '<tag>:mirror_dlogpost')"),
        forward_ref_note=("forward refs freeze the 28 shapelet amps at "
                          "a_ref = a_star(z_ref) per product; loglik = "
                          "-1/2(chi2+norm) masked Gaussian over keep_mask "
                          "with the RAW err_map (stock ImageLikelihoodTerm "
                          "functional)"),
        points=[name for name, _ in points],
    )
    out["labels"] = np.array(OLD_LABELS_46)
    for name, z in points:
        out[f"z:{name}"] = z

    for tag, cfg in PRODUCTS.items():
        print(f"\n=== {tag}: building old-stack model ===", flush=True)
        prod = load_product(cfg["cutout"])
        t1 = time.time()
        model = build_marg_model(
            prod, n_max=N_MAX, supersample=cfg["supersample"],
            delta_pix=cfg["delta_pix"], exp_time=EXP_TIME, near_xy=near_xy)
        print(f"  built in {time.time()-t1:.1f}s (ndim={model.ndim})", flush=True)
        assert sorted(model.index_labels) == sorted(OLD_LABELS_46), \
            "old model labels do not match the canonical 46 set"
        # old-z-index -> canonical-index permutation (z vectors are in the old
        # model's own packing; x/grad vectors are stored canonically)
        mirror = make_mirror(model, prod)

        out[f"{tag}:img"] = np.asarray(prod["img"], dtype=np.float64)
        out[f"{tag}:err_raw"] = np.asarray(prod["err_map"], dtype=np.float64)
        out[f"{tag}:keep_mask"] = np.asarray(prod["keep_mask"], dtype=bool)
        out[f"{tag}:masked_err"] = np.asarray(model.masked_err_map)
        out[f"{tag}:psf_raw"] = np.asarray(prod["psf"], dtype=np.float64)
        out[f"{tag}:img_X32"] = mirror["img_X32"]
        out[f"{tag}:img_Y32"] = mirror["img_Y32"]
        out[f"{tag}:kernel_unflipped"] = mirror["kernel_unflipped"]
        out[f"{tag}:conversion_factor"] = np.float64(mirror["conversion_factor"])
        out[f"{tag}:Lambda_diag"] = np.asarray(model.Lambda_diag)

        a_ref = None
        for pname, z in points:
            zj = jnp.asarray(z)
            internals = model.marg_internals(z)
            labeled = dict(_flat_named46(model.bij.forward(list(zj[None, :].T))))
            v46 = np.array([labeled[lab] for lab in OLD_LABELS_46],
                           dtype=np.float64)
            vj = jnp.asarray(v46)

            # mirror certification (values must match the frozen model)
            lm = float(mirror["logL_marg"](vj))
            d_logL = abs(lm - internals["logL_data"])
            d_logpost = abs((lm + internals["log_prior"]) - internals["logpost"])
            lp_jit = float(model.target_log_prob_fn(zj))
            d_jit = abs(internals["logpost"] - lp_jit)

            if a_ref is None:  # z_ref is always the first point
                a_ref = np.asarray(internals["a_star"], dtype=np.float64)
                out[f"{tag}:a_ref"] = a_ref
            aj = jnp.asarray(a_ref)
            llf, chi2f, mi_f = mirror["ll_fwd"](vj, aj)
            gm = np.asarray(mirror["grad_marg"](vj), dtype=np.float64)
            gf = np.asarray(mirror["grad_fwd"](vj, aj), dtype=np.float64)

            k = f"{tag}:{pname}"
            out[f"{k}:x46"] = v46
            out[f"{k}:M_det"] = internals["M_det"]
            out[f"{k}:ret"] = internals["ret"]
            out[f"{k}:b"] = internals["b"]
            out[f"{k}:A"] = internals["A"]
            out[f"{k}:a_star"] = internals["a_star"]
            out[f"{k}:logdetA"] = np.float64(internals["logdetA"])
            out[f"{k}:logL_data"] = np.float64(internals["logL_data"])
            out[f"{k}:log_prior"] = np.float64(internals["log_prior"])
            out[f"{k}:logpost"] = np.float64(internals["logpost"])
            out[f"{k}:chi2w_premarg"] = np.float64(
                float(np.sum(internals["Rw"] ** 2)))
            out[f"{k}:chi2w_resid"] = np.float64(float(np.sum(
                (internals["Rw"] - internals["Xw"] @ internals["a_star"]) ** 2)))
            out[f"{k}:model_image_marg"] = (
                internals["M_det"] + np.sum(
                    internals["ret"] * internals["a_star"][None, None, :],
                    axis=-1))
            out[f"{k}:model_image_fwd"] = np.asarray(mi_f)
            out[f"{k}:loglik_fwd"] = np.float64(float(llf))
            out[f"{k}:chi2_fwd"] = np.float64(float(chi2f))
            out[f"{k}:grad_marg_x46"] = gm
            out[f"{k}:grad_fwd_x46"] = gf
            out[f"{k}:mirror_dlogL"] = np.float64(d_logL)
            out[f"{k}:mirror_dlogpost"] = np.float64(d_logpost)
            out[f"{k}:jit_minus_eager_logpost"] = np.float64(d_jit)

            print(f"  {pname:8s} logL_data={internals['logL_data']:+.6f} "
                  f"mirror|d|={d_logL:.2e} fwd_ll={float(llf):+.6f} "
                  f"|grad_marg|={np.linalg.norm(gm):.3e}", flush=True)
            if d_logL > 1e-6:
                raise RuntimeError(
                    f"mirror certification FAILED at {tag}/{pname}: "
                    f"|d logL_data| = {d_logL:.3e} > 1e-6 — the mirror does "
                    "not reproduce the frozen model; refusing to dump grads")

    out["provenance"] = np.array(json.dumps(prov))
    dest = HERE / "data" / "parity_refs.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(f"\nwrote {dest} ({dest.stat().st_size/1e6:.1f} MB) "
          f"in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
