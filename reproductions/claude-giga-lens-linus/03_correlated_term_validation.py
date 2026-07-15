"""03_correlated_term_validation.py — correlated term vs exact dense reference.

Two validation layers on a 32x32 toy scene (same structural class as the
parity model: EPL+Shear, Sersic lens light, Sersic+Shapelets source), a real
truncated whitener (built by cgl2.whiten.build_whitener from a stationary
delta+Gaussian kernel, e_op <= 0.02), and 4 evaluation points (truth + 3
prior-ish jitters):

  GATE A (<= 0.1 nat, the PLAN §10 threshold): the GPU/JAX
    CorrelatedImageLikelihoodTerm marg logL vs the DENSE-CHOLESKY exact
    reference computed with independent numpy/scipy float64 linear algebra:
    G_e = R_erode C_h diag(sqrt_d_inv) assembled as an explicit sparse
    matrix; u = G_e r; Xw = G_e X; A = Xw^T Xw + Lambda via scipy
    cho_factor. Same statistical functional, no shared code with the JAX
    path. (Both consume the SAME M_det/ret rendered by the scene simulator,
    so the gate isolates the whiten+marginalize machinery; the renders
    themselves are certified by F1/F2.)

  GATE B (<= 1e-10): delta-kernel identities —
    (i)  forward-mode CorrelatedImageLikelihoodTerm == stock
         ImageLikelihoodTerm (loglik AND chi2), and
    (ii) marg-mode delta-conv whitening == the diagonal multiply path.

Dense-reference helpers are COPIED WITH ATTRIBUTION from
../claude-giga-lens/cgl/exact_ref.py (conv_matrix / whitening_operator /
dense_whitened_marg — the 04_validate_exact_smallgrid machinery).

Writes data/correlated_term_validation.json; exits nonzero on failure.

Run (L4 GPU 9; CPU also allowed for this toy):
  GIGALENS_X64=1 CUDA_VISIBLE_DEVICES=9 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /raid/benson/.venvs/cgl2/bin/python 03_correlated_term_validation.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("GIGALENS_X64", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CGL2_GPU", "9")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = Path(__file__).resolve().parent

from cgl2 import guards, paths  # noqa: E402

paths.bootstrap_vendor()
guards.require_vendor_ref()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402

from cgl2 import scene_build  # noqa: E402
from cgl2.correlated import (CorrelatedImageData, delta_bundle,  # noqa: E402
                             make_whitener_bundle)
from cgl2.marg import marg_loglik  # noqa: E402
from cgl2.whiten import build_whitener, erode_keep  # noqa: E402

GATE_A_NATS = 0.1
GATE_B = 1e-10


# --------------------------------------------------------------------------- #
# dense reference — COPIED WITH ATTRIBUTION from
# /raid/benson/git/agentic-lensing/reproductions/claude-giga-lens/cgl/exact_ref.py
# (conv_matrix / whitening_operator / dense_whitened_marg; unchanged math)
# --------------------------------------------------------------------------- #
def _shift_1d(n, k):
    if abs(k) >= n:
        return sp.csr_matrix((n, n))
    return sp.eye(n, n, k=k, format="csr")


def conv_matrix(h, shape):
    h = np.asarray(h, dtype=np.float64)
    kh, kw = h.shape
    if kh % 2 != 1 or kw % 2 != 1:
        raise ValueError("kernel must be odd-shaped")
    My, Mx = kh // 2, kw // 2
    H, W = shape
    A = sp.csr_matrix((H * W, H * W))
    for iy in range(kh):
        Ey = _shift_1d(H, iy - My)
        for ix in range(kw):
            w = h[iy, ix]
            if w == 0.0:
                continue
            A = A + w * sp.kron(Ey, _shift_1d(W, ix - Mx), format="csr")
    return A.tocsr()


def whitening_operator(h, sqrt_d_inv, keep_w):
    sqrt_d_inv = np.asarray(sqrt_d_inv, dtype=np.float64)
    keep_w = np.asarray(keep_w, dtype=bool)
    Ch = conv_matrix(h, sqrt_d_inv.shape)
    G = Ch @ sp.diags(sqrt_d_inv.reshape(-1))
    idx = np.flatnonzero(keep_w.reshape(-1))
    return G.tocsr()[idx], idx


def dense_whitened_marg(G_e, Y, M_det, ret, Lambda_diag):
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    M = np.asarray(M_det, dtype=np.float64).reshape(-1)
    depth = ret.shape[-1]
    X = np.asarray(ret, dtype=np.float64).reshape(-1, depth)
    u = G_e @ (Y - M)
    Xw = G_e @ X
    b = Xw.T @ u
    A = Xw.T @ Xw + np.diag(np.asarray(Lambda_diag, dtype=np.float64))
    c, low = cho_factor(A, lower=True)
    a_star = cho_solve((c, low), b)
    logdetA = 2.0 * float(np.sum(np.log(np.diag(c))))
    logL = -0.5 * float(u @ u) + 0.5 * float(b @ a_star) - 0.5 * logdetA
    return dict(logL=logL, a_star=a_star, logdetA=logdetA, u=u, A=A)


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print(f"devices: {jax.devices()} x64={jax.config.jax_enable_x64}", flush=True)
    guards.require_x64()
    guards.require_gpu()   # CGL2_ALLOW_CPU=1 sanctions the toy on CPU

    toy = scene_build.build_toy_scene(n_pix=32, n_max=4, seed=7)
    prod, sim_cfg = toy.product, toy.sim_cfg
    H = prod["img"].shape[0]

    # --- stationary kernel + truncated whitener (e_op-certified) -------------
    rho = np.zeros((9, 9))
    rho[4, 4] = 1.0
    gy, gx = np.mgrid[-4:5, -4:5]
    g = np.exp(-(gx ** 2 + gy ** 2) / (2 * 0.9 ** 2))
    rho = 0.55 * rho + 0.45 * g / g.sum() * 4.0   # delta + broad positive part
    rho = rho / rho[4, 4]
    wres = build_whitener(rho, M=5, grid=128)
    print(f"whitener: e_op={wres['e_op']:.4f} (init {wres['e_op_init']:.4f})",
          flush=True)
    keep_w = erode_keep(prod["keep_mask"], 5)
    sdi = 1.0 / prod["err_map"].astype(np.float64)
    bundle = make_whitener_bundle(wres["h"], sdi, keep_w, e_op=wres["e_op"],
                                  logdet_per_pix=wres["logdet_per_pix"],
                                  source="03 toy delta+gaussian rho, M=5")

    ds = CorrelatedImageData(prod["img"], sim_cfg, bundle,
                             error_map=prod["err_map"],
                             mask=prod["keep_mask"], sees="all",
                             corr_mode="marg",
                             Lambda_diag=toy.marg.Lambda_diag)
    term = ds.build_term(toy.marg.model, ds.resolve_sees(toy.marg.model))

    # points: truth + 3 jitters of the constrained values
    rng = np.random.default_rng(11)
    pts = [dict(toy.truth_marg)]
    for _ in range(3):
        p = {}
        for k, v in toy.truth_marg.items():
            jit_scale = 0.03 if not k.endswith(("Ie", "beta", "R_sersic",
                                                "n_sersic")) else 0.0
            p[k] = float(v) * (1.0 + 0.02 * rng.standard_normal()) \
                + jit_scale * rng.standard_normal()
        pts.append(p)

    G_e, _ = whitening_operator(wres["h"], sdi, keep_w)
    report = dict(points=[], whitener=dict(e_op=float(wres["e_op"]), M=5,
                                           n_keep=int(keep_w.sum())),
                  gate_a_nats=GATE_A_NATS, gate_b=GATE_B)
    worst_a = 0.0
    for i, p in enumerate(pts):
        params = toy.marg.model.to_params(
            {k: jnp.asarray(np.float64(v)) for k, v in p.items()})
        it = term.internals(params)
        ref = dense_whitened_marg(G_e, prod["img"], it["M_det"], it["ret"],
                                  toy.marg.Lambda_diag)
        dl = abs(ref["logL"] - it["logL_data"])
        worst_a = max(worst_a, dl)
        report["points"].append(dict(
            i=i, logL_jax=it["logL_data"], logL_dense=ref["logL"],
            abs_dlogL=dl,
            dlogdetA=abs(ref["logdetA"] - it["logdetA"]),
            occam_vs_numpy_slogdet=abs(
                -0.5 * it["logdetA"]
                + 0.5 * np.linalg.slogdet(it["A"])[1])))
        print(f"  pt{i}: logL_jax={it['logL_data']:+.6f} "
              f"dense={ref['logL']:+.6f} |d|={dl:.3e}", flush=True)

    # --- GATE B: delta-kernel identities -------------------------------------
    fwd_params = toy.fwd.model.to_params(
        {k: jnp.asarray(np.float64(v)) for k, v in toy.truth_fwd.items()})
    ds_stock = scene_build.build_image_data(prod, sim_cfg, sees="all",
                                            mode="forward")
    t_stock = ds_stock.build_term(toy.fwd.model,
                                  ds_stock.resolve_sees(toy.fwd.model))
    ll_s, chi2_s = t_stock.log_like(fwd_params)
    ds_d = CorrelatedImageData(prod["img"], sim_cfg,
                               delta_bundle(prod["err_map"],
                                            prod["keep_mask"]),
                               error_map=prod["err_map"],
                               mask=prod["keep_mask"], sees="all",
                               corr_mode="forward")
    t_d = ds_d.build_term(toy.fwd.model, ds_d.resolve_sees(toy.fwd.model))
    ll_d, chi2_d = t_d.log_like(fwd_params)
    b_fwd = max(abs(float(ll_s) - float(ll_d)),
                abs(float(chi2_s) - float(chi2_d)))

    masked_err = np.where(prod["keep_mask"], prod["err_map"], 1e10)
    ds_dd = CorrelatedImageData(prod["img"], sim_cfg,
                                delta_bundle(None, None,
                                             masked_err=masked_err),
                                error_map=prod["err_map"],
                                mask=prod["keep_mask"], sees="all",
                                corr_mode="marg",
                                Lambda_diag=toy.marg.Lambda_diag)
    t_dd = ds_dd.build_term(toy.marg.model,
                            ds_dd.resolve_sees(toy.marg.model))
    params0 = toy.marg.model.to_params(
        {k: jnp.asarray(np.float64(v)) for k, v in toy.truth_marg.items()})
    it0 = t_dd.internals(params0)
    sdim = 1.0 / masked_err
    Rw_m = jnp.reshape((jnp.asarray(prod["img"]) - it0["M_det"]) * sdim, (-1,))
    Xw_m = jnp.reshape(jnp.asarray(it0["ret"]) * sdim[..., None],
                       (-1, it0["ret"].shape[-1]))
    ll_mult, _, _ = marg_loglik(Xw_m, Rw_m,
                                jnp.asarray(toy.marg.Lambda_diag))
    b_marg = abs(float(ll_mult) - it0["logL_data"])

    gate_a = bool(worst_a <= GATE_A_NATS)
    gate_b = bool(max(b_fwd, b_marg) <= GATE_B)
    report["gateA"] = dict(threshold=GATE_A_NATS, worst_abs_dlogL=worst_a,
                           gate_pass=gate_a)
    report["gateB"] = dict(threshold=GATE_B,
                           delta_forward_vs_stock=float(b_fwd),
                           delta_marg_conv_vs_multiply=float(b_marg),
                           gate_pass=gate_b)
    report["all_pass"] = bool(gate_a and gate_b)
    report["wall_s"] = time.time() - t0

    dest = HERE / "data" / "correlated_term_validation.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nGATE A (dense exact ref): worst |dlogL| = {worst_a:.3e} nats "
          f"-> {'PASS' if gate_a else 'FAIL'} (< {GATE_A_NATS})")
    print(f"GATE B (delta identities): fwd={b_fwd:.3e} marg={b_marg:.3e} "
          f"-> {'PASS' if gate_b else 'FAIL'} (< {GATE_B})")
    print(f"wrote {dest} ({report['wall_s']:.0f}s)")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
