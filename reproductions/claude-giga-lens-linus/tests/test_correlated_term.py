"""CorrelatedImageData/Term on the 16x16 CPU toy (CGL2_ALLOW_CPU=1).

Covers: delta-kernel forward identity vs the stock ImageLikelihoodTerm,
delta-conv marg whitening == diagonal multiply, the Occam term vs numpy
slogdet, event_size semantics, and validate-at-construction raises.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from cgl2 import scene_build
from cgl2.correlated import CorrelatedImageData, delta_bundle
from cgl2.marg import marg_loglik


def _params(ns, labeled):
    return ns.model.to_params({k: jnp.asarray(np.float64(v))
                               for k, v in labeled.items()})


def test_delta_forward_equals_stock(toy):
    prod, cfg = toy.product, toy.sim_cfg
    ds_stock = scene_build.build_image_data(prod, cfg, sees="all",
                                            mode="forward")
    t_stock = ds_stock.build_term(toy.fwd.model,
                                  ds_stock.resolve_sees(toy.fwd.model))
    ds_d = CorrelatedImageData(prod["img"], cfg,
                               delta_bundle(prod["err_map"], prod["keep_mask"]),
                               error_map=prod["err_map"],
                               mask=prod["keep_mask"], sees="all",
                               corr_mode="forward")
    t_d = ds_d.build_term(toy.fwd.model, ds_d.resolve_sees(toy.fwd.model))
    p = _params(toy.fwd, toy.truth_fwd)
    ll_s, chi2_s = t_stock.log_like(p)
    ll_d, chi2_d = t_d.log_like(p)
    assert abs(float(ll_s) - float(ll_d)) <= 1e-10
    assert abs(float(chi2_s) - float(chi2_d)) <= 1e-10


def test_marg_delta_conv_equals_multiply_and_occam(toy):
    prod, cfg = toy.product, toy.sim_cfg
    masked_err = np.where(prod["keep_mask"], prod["err_map"], 1e10)
    ds = CorrelatedImageData(prod["img"], cfg,
                             delta_bundle(None, None, masked_err=masked_err),
                             error_map=prod["err_map"],
                             mask=prod["keep_mask"], sees="all",
                             corr_mode="marg",
                             Lambda_diag=toy.marg.Lambda_diag)
    term = ds.build_term(toy.marg.model, ds.resolve_sees(toy.marg.model))
    it = term.internals(_params(toy.marg, toy.truth_marg))
    # (i) conv-delta whitening == diagonal multiply
    sdi = 1.0 / masked_err
    Rw = jnp.reshape((jnp.asarray(prod["img"]) - it["M_det"]) * sdi, (-1,))
    Xw = jnp.reshape(jnp.asarray(it["ret"]) * sdi[..., None],
                     (-1, it["ret"].shape[-1]))
    ll_mult, a_mult, logdet_mult = marg_loglik(
        Xw, Rw, jnp.asarray(toy.marg.Lambda_diag))
    assert abs(float(ll_mult) - it["logL_data"]) <= 1e-10
    # (ii) Occam -1/2 logdetA vs numpy slogdet (toy A is well-conditioned)
    sign, ld = np.linalg.slogdet(it["A"])
    assert sign > 0
    assert abs(-0.5 * it["logdetA"] - (-0.5 * ld)) <= 1e-10
    # (iii) log_like agrees with internals (jit-vs-eager ULP band)
    ll, chi2 = term.log_like(_params(toy.marg, toy.truth_marg))
    assert abs(float(ll) - it["logL_data"]) < 1e-8
    assert abs(float(chi2) - it["chi2w_resid"]) < 1e-8


def test_event_size_is_kept_whitened_dof(toy):
    prod, cfg = toy.product, toy.sim_cfg
    keep = prod["keep_mask"].copy()
    keep[:4] = False
    b = delta_bundle(prod["err_map"], keep)
    ds = CorrelatedImageData(prod["img"], cfg, b,
                             error_map=prod["err_map"],
                             mask=prod["keep_mask"], sees="all",
                             corr_mode="forward")
    assert ds.event_size == int(keep.sum())


def test_validation_raises(toy):
    prod, cfg = toy.product, toy.sim_cfg
    with pytest.raises(ValueError):   # marg without Lambda
        CorrelatedImageData(prod["img"], cfg,
                            delta_bundle(prod["err_map"], prod["keep_mask"]),
                            error_map=prod["err_map"], mask=prod["keep_mask"],
                            sees="all", corr_mode="marg")
    bad = delta_bundle(np.ones((8, 8)), np.ones((8, 8), bool))
    with pytest.raises(ValueError):   # wrong-shape whitener
        CorrelatedImageData(prod["img"], cfg, bad,
                            error_map=prod["err_map"], mask=prod["keep_mask"],
                            sees="all", corr_mode="forward")
    with pytest.raises(ValueError):   # bad mode
        CorrelatedImageData(prod["img"], cfg,
                            delta_bundle(prod["err_map"], prod["keep_mask"]),
                            error_map=prod["err_map"], mask=prod["keep_mask"],
                            sees="all", corr_mode="lstsq")
    with pytest.raises(ValueError):   # Lambda length mismatch
        ds = CorrelatedImageData(prod["img"], cfg,
                                 delta_bundle(prod["err_map"],
                                              prod["keep_mask"]),
                                 error_map=prod["err_map"],
                                 mask=prod["keep_mask"], sees="all",
                                 corr_mode="marg",
                                 Lambda_diag=np.ones(3))
        ds.build_term(toy.marg.model, ds.resolve_sees(toy.marg.model))


def test_checkpoint_flag_exactness(toy):
    """checkpoint=True/False must be numerically identical (remat re-runs
    identical ops)."""
    prod, cfg = toy.product, toy.sim_cfg
    masked_err = np.where(prod["keep_mask"], prod["err_map"], 1e10)
    outs = []
    for ckpt in (True, False):
        ds = CorrelatedImageData(prod["img"], cfg,
                                 delta_bundle(None, None,
                                              masked_err=masked_err),
                                 error_map=prod["err_map"],
                                 mask=prod["keep_mask"], sees="all",
                                 corr_mode="marg",
                                 Lambda_diag=toy.marg.Lambda_diag,
                                 checkpoint=ckpt)
        term = ds.build_term(toy.marg.model, ds.resolve_sees(toy.marg.model))
        ll, chi2 = term.log_like(_params(toy.marg, toy.truth_marg))
        outs.append((float(ll), float(chi2)))
    assert outs[0] == outs[1]
