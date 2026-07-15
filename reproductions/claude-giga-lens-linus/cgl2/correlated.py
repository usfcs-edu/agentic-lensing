"""Correlated-noise likelihood as a scene-API Dataset/LikelihoodTerm (PLAN §4).

``CorrelatedImageData`` subclasses the vendored ``ImageData`` (the documented
dataset seam, scene_prob_model.py); ``CorrelatedImageLikelihoodTerm``
subclasses ``LikelihoodTerm``. The vendor stays UNPATCHED.

Statistical model (the validated claude-giga-lens P1 construction):
noise covariance C = D^{1/2} K D^{1/2} with a stationary kernel K; the frozen
whitener bundle provides truncated conv taps h with spectrum ~ K^{-1/2}
(certificate e_op = max|S_K H^2 - 1| <= 0.02) and an eroded keep-mask (kept
whitened pixels never touch masked data or the zero-padded border). Whitening
u = keep .* conv2d_SAME(h, sqrt_d_inv .* r) is applied IDENTICALLY to the
residual and to every design column (cgl2.whiten; jax.checkpoint behind a
flag, default ON — the <=200 MB/particle memory fix).

Modes:
  * mode="marg" (default; use with a cgl2.scene_build "marg"-view model):
    ONE ``lstsq_simulate(..., return_stacked=True)`` render per log_like
    (their single-forward-eval contract; reuses their fused conv-pool +
    remat_basis pipeline verbatim). The stacked columns split into
    deterministic (amplitude-carrying, use_lstsq=False) columns -> M_det and
    lstsq columns -> design matrix X. Generalized-ridge (Gaussian-prior)
    marginalization of the lstsq amplitudes with prior precision Lambda:
        logL = -1/2||Rw||^2 + 1/2 b.a* - 1/2 logdet(A),  A = Xw^T Xw + Lambda
    including the -1/2 logdet A Gaussian-evidence/Occam term gigalens's own
    lstsq path omits (the owed foundry-i upstream fix). theta-independent
    normalization constants are DROPPED — the old validated convention
    (foundry-i _hmc_lib_marg); absolute normalization needs the constants
    accounting of claude-giga-lens 04_validate_exact_smallgrid.
  * mode="forward" (use with a "forward"-view model): one ``simulate()``
    render; non-marginalized whitened chi^2 with the FULL Gaussian
    normalization  logL = -1/2 (chi2_w + n_keep log 2pi - 2 sum_keep log
    sqrt_d_inv + n_keep * logdet_per_pix)  so a delta-kernel bundle is
    EXACTLY the stock ``ImageLikelihoodTerm`` (gate F5). For a truncated h
    the kernel term uses the bundle's stationary (Szego) logdet_per_pix — a
    constant, irrelevant within a fixed whitener.

``reports_chi2 = True``: the chi2 channel returns the whitened chi^2
(mode="marg": at the marginal-mode amplitudes, ||Rw - Xw a*||^2;
mode="forward": ||Rw||^2). ``event_size`` = kept whitened dof =
keep_mask.sum(), so ProbModel's reduced-chi2 channel is chi2_w per kept
whitened pixel. Masked reductions use where()/exact zeroing, never a
float-mask multiply of the likelihood (their jaxlib miscompile note); the
DIAGONAL-parity configuration (delta taps + sqrt_d_inv = 1/masked_err with
err->1e10 inside the mask, keep = all) instead reproduces the old stack's
down-weighting convention bit-for-bit — that is the F2/F4 parity anchor.
"""
from __future__ import annotations

import hashlib
import types
from typing import Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from cgl2 import guards, paths
from cgl2.marg import marg_loglik
from cgl2.whiten import make_conv_whitener, make_grouped_design_whitener

paths.bootstrap_vendor()

from gigalens.jax.scene import LensModel  # noqa: E402
from gigalens.jax.scene_prob_model import ImageData, LikelihoodTerm  # noqa: E402


# --------------------------------------------------------------------------- #
# whitener bundles
# --------------------------------------------------------------------------- #
def make_whitener_bundle(h_taps, sqrt_d_inv, keep_mask, *, e_op,
                         logdet_per_pix=0.0, source="") -> Dict:
    """Assemble + validate a frozen whitener bundle dict.

    Keys: h_taps (2M+1,2M+1) f64; sqrt_d_inv (H,W) f64 (per-pixel D^{-1/2});
    keep_mask (H,W) bool (kept whitened dof); e_op (certificate);
    logdet_per_pix (stationary log|K| per pixel; 0 for a delta kernel);
    kernel_hash (sha256 over the arrays); source (provenance string).
    """
    b = dict(
        h_taps=np.asarray(h_taps, dtype=np.float64),
        sqrt_d_inv=np.asarray(sqrt_d_inv, dtype=np.float64),
        keep_mask=np.asarray(keep_mask, dtype=bool),
        e_op=float(e_op), logdet_per_pix=float(logdet_per_pix),
        source=str(source),
    )
    hsh = hashlib.sha256()
    for k in ("h_taps", "sqrt_d_inv", "keep_mask"):
        hsh.update(np.ascontiguousarray(b[k]).tobytes())
    b["kernel_hash"] = hsh.hexdigest()
    guards.assert_whitener_bundle(b)
    if b["h_taps"].ndim != 2 or b["h_taps"].shape[0] % 2 != 1:
        raise ValueError("h_taps must be odd-shaped 2-D")
    if b["sqrt_d_inv"].shape != b["keep_mask"].shape:
        raise ValueError("sqrt_d_inv / keep_mask shapes disagree")
    if not np.all(np.isfinite(b["h_taps"])):
        raise ValueError("h_taps has non-finite values")
    if not np.all(np.isfinite(b["sqrt_d_inv"])):
        raise ValueError("sqrt_d_inv has non-finite values")
    if int(b["keep_mask"].sum()) == 0:
        raise ValueError("keep_mask keeps zero whitened pixels")
    return b


def delta_bundle(err_map, keep_mask=None, *, masked_err=None,
                 source="delta-kernel diagonal") -> Dict:
    """Delta-kernel (1x1 taps) bundles — the two diagonal-limit anchors.

    * ``delta_bundle(err_raw, keep_mask)``: sqrt_d_inv = 1/err over the data
      grid, keep = the data mask -> ``mode='forward'`` term identical to the
      stock ImageLikelihoodTerm (gate F5).
    * ``delta_bundle(None, None, masked_err=masked_err)``: sqrt_d_inv =
      1/masked_err (err->1e10 inside the mask), keep = ALL pixels -> the OLD
      stack's diagonal down-weighting convention bit-for-bit (parity anchor
      for F2/F4; old gate-D analogue).
    """
    if masked_err is not None:
        sdi = 1.0 / np.asarray(masked_err, dtype=np.float64)
        keep = np.ones_like(sdi, dtype=bool)
    else:
        err = np.asarray(err_map, dtype=np.float64)
        keep = (np.ones_like(err, dtype=bool) if keep_mask is None
                else np.asarray(keep_mask, dtype=bool))
        sdi = np.where(keep, 1.0 / np.where(keep, err, 1.0), 0.0)
    return make_whitener_bundle(np.ones((1, 1)), sdi, keep, e_op=0.0,
                                logdet_per_pix=0.0, source=source)


def load_whitener_bundle(npz_path, *, sqrt_d_inv, source=None) -> Dict:
    """Load a frozen claude-giga-lens whitener npz (keys h/keep_w/M/e_op/
    logdet_per_pix) into the cgl2 bundle layout. ``sqrt_d_inv`` is the
    per-product D^{-1/2} (1/masked_err or 1/err — caller's convention; the
    bundles are D-free by construction)."""
    z = np.load(npz_path, allow_pickle=True)
    return make_whitener_bundle(
        np.asarray(z["h"], dtype=np.float64), sqrt_d_inv,
        np.asarray(z["keep_w"]).astype(bool), e_op=float(z["e_op"]),
        logdet_per_pix=float(z["logdet_per_pix"]),
        source=source or str(npz_path))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CorrelatedImageData(ImageData):
    """One imaging observation with correlated (stationary-kernel) noise.

    Validate-at-construction, raise-never-default (their M3 pattern): the
    whitener bundle is checked by ``make_whitener_bundle``/
    ``guards.assert_whitener_bundle`` (e_op certificate, shapes, finiteness)
    and must match the image shape. ``event_size`` = kept whitened dof.

    Args (beyond ImageData's): whitener (bundle dict), corr_mode
    ('marg'|'forward'), Lambda_diag ((k,) ridge precision of the lstsq
    amplitudes, REQUIRED for 'marg'), checkpoint (jax.checkpoint on the
    whitening convs, default True), Y_units_cf: divide the rendered image by
    conversion_factor before the residual in 'forward' mode when the model's
    amplitudes are in old no-cf units (default False: scene-forward units).
    """

    def __init__(self, image, sim_config, whitener, *, error_map=None,
                 background_rms=None, exp_time=None, mask=None, sees=None,
                 corr_mode: str = "marg", Lambda_diag=None,
                 checkpoint: bool = True):
        super().__init__(image, sim_config, error_map=error_map,
                         background_rms=background_rms, exp_time=exp_time,
                         mask=mask, sees=sees, mode=None)
        if corr_mode not in ("marg", "forward"):
            raise ValueError(f"corr_mode must be 'marg'/'forward'; got {corr_mode!r}")
        wb = dict(whitener)
        guards.assert_whitener_bundle(wb)
        for key in ("sqrt_d_inv", "keep_mask"):
            if tuple(np.asarray(wb[key]).shape) != tuple(self.image.shape):
                raise ValueError(
                    f"whitener {key} shape {np.asarray(wb[key]).shape} != image "
                    f"shape {tuple(self.image.shape)} — no silent broadcast")
        self.whitener = wb
        self.corr_mode = corr_mode
        self.checkpoint = bool(checkpoint)
        if corr_mode == "marg":
            if Lambda_diag is None:
                raise ValueError("corr_mode='marg' requires Lambda_diag (the "
                                 "amplitude prior precision; no silent default)")
            Lambda_diag = np.asarray(Lambda_diag, dtype=np.float64)
            if Lambda_diag.ndim != 1 or not np.all(Lambda_diag > 0):
                raise ValueError("Lambda_diag must be a positive 1-D vector")
        self.Lambda_diag = Lambda_diag
        # event_size := kept whitened dof (overrides ImageData's masked-pixel
        # count; ProbModel's reduced-chi2 divides by this).
        self.event_size = int(np.asarray(wb["keep_mask"], dtype=bool).sum())

    def build_term(self, model: LensModel, seen: List, *,
                   default_mode: str = "lstsq") -> "CorrelatedImageLikelihoodTerm":
        # default_mode (lstsq/forward) is the ProbModel-level amplitude-mode
        # default; the correlated term's mode is fixed at dataset construction.
        return CorrelatedImageLikelihoodTerm(self, model, seen, self.corr_mode)


# --------------------------------------------------------------------------- #
# LikelihoodTerm
# --------------------------------------------------------------------------- #
class CorrelatedImageLikelihoodTerm(LikelihoodTerm):
    """Whitened (correlated-noise) Gaussian likelihood of one CorrelatedImageData.

    mode='marg': ONE lstsq_simulate(return_stacked=True) render; ridge
    marginalization with the Occam -1/2 logdet A term (module docstring).
    mode='forward': ONE simulate() render; whitened chi2 + full normalization.
    """

    reports_chi2 = True

    def __init__(self, dataset: CorrelatedImageData, model: LensModel,
                 seen: List, mode: str):
        if mode not in ("marg", "forward"):
            raise ValueError(f"mode must be 'marg' or 'forward'; got {mode!r}")
        self.dataset = dataset
        self.mode = mode
        self.event_size = dataset.event_size
        self.simulator = dataset._build_simulator(model, seen)

        wb = dataset.whitener
        h = jnp.asarray(wb["h_taps"], dtype=jnp.float64)
        sdi = jnp.asarray(wb["sqrt_d_inv"], dtype=jnp.float64)
        keep = np.asarray(wb["keep_mask"], dtype=bool)
        self._whiten_R = make_conv_whitener(h, sdi, jnp.asarray(keep),
                                            checkpoint=dataset.checkpoint)
        self._whiten_X = make_grouped_design_whitener(
            h, sdi, jnp.asarray(keep), checkpoint=dataset.checkpoint)

        # Gaussian normalization over the kept whitened dof (constant):
        # log det(2 pi C) ~= n_k log 2pi - 2 sum_keep log(sqrt_d_inv)
        #                    + n_k * logdet_per_pix   (Szego; 0 for delta taps)
        n_k = int(keep.sum())
        sdi_np = np.asarray(wb["sqrt_d_inv"], dtype=np.float64)
        self._norm_const = float(
            n_k * np.log(2.0 * np.pi)
            - 2.0 * np.sum(np.log(sdi_np[keep]))
            + n_k * float(wb.get("logdet_per_pix", 0.0)))

        if mode == "marg":
            # column split from the simulator's own light bookkeeping:
            # use_lstsq columns are marginalized; the rest carry amplitudes.
            det_idx, marg_idx, off = [], [], 0
            for i, j, comp, d in self.simulator._light:
                cols = list(range(off, off + d))
                (marg_idx if comp.profile.use_lstsq else det_idx).extend(cols)
                off += d
            self._det_idx = np.asarray(det_idx, dtype=int)
            self._marg_idx = np.asarray(marg_idx, dtype=int)
            k = len(marg_idx)
            if k == 0:
                raise ValueError("mode='marg' but the model has no use_lstsq "
                                 "light columns to marginalize")
            lam = np.asarray(dataset.Lambda_diag, dtype=np.float64)
            if lam.shape != (k,):
                raise ValueError(f"Lambda_diag shape {lam.shape} != ({k},) "
                                 "(one precision per marginalized column)")
            self._Lambda = jnp.asarray(lam)

    # -- single-sample cores -------------------------------------------------
    def _marg_single(self, stacked_hwc):
        ds = self.dataset
        M_det = jnp.sum(stacked_hwc[..., self._det_idx], axis=-1)
        ret = stacked_hwc[..., self._marg_idx]
        Rw = self._whiten_R(ds.image - M_det)
        Xw = self._whiten_X(ret)
        logL, a_star, logdetA = marg_loglik(Xw, Rw, self._Lambda)
        chi2 = jnp.sum((Rw - Xw @ a_star) ** 2)
        return logL, chi2

    def _forward_single(self, im_hw):
        ds = self.dataset
        Rw = self._whiten_R(ds.image - im_hw)
        chi2 = jnp.sum(Rw ** 2)
        return -0.5 * (chi2 + self._norm_const), chi2

    # -- LikelihoodTerm surface ----------------------------------------------
    def log_like(self, params, *, simulator=None):
        """(log_like, chi2_w) — both from a SINGLE forward evaluation."""
        sim = self.simulator if simulator is None else simulator
        ds = self.dataset
        if self.mode == "marg":
            stacked = sim.lstsq_simulate(params, ds.image, ds.error_map,
                                         ds.mask, return_stacked=True)
            ll, chi2 = jax.vmap(self._marg_single)(stacked)
            bs = stacked.shape[0]
        else:
            im = sim.simulate(params)
            im = jnp.reshape(im, (-1, *ds.image.shape))
            ll, chi2 = jax.vmap(self._forward_single)(im)
            bs = im.shape[0]
        if bs == 1:  # mirror the stock term's unbatched squeeze
            return ll[0], chi2[0]
        return ll, chi2

    # -- debug / parity surface ----------------------------------------------
    def internals(self, params) -> Dict:
        """Eager op-by-op intermediates of one (unbatched) marg log_like eval
        (numpy). ULP-level fusion differences vs the jitted path are possible
        and expected (the old marg_internals caveat)."""
        if self.mode != "marg":
            raise RuntimeError("internals() is a marg-mode surface")
        ds = self.dataset
        stacked = self.simulator.lstsq_simulate(
            params, ds.image, ds.error_map, ds.mask, return_stacked=True)
        s = stacked[0]
        M_det = jnp.sum(s[..., self._det_idx], axis=-1)
        ret = s[..., self._marg_idx]
        Rw = self._whiten_R(ds.image - M_det)
        Xw = self._whiten_X(ret)
        b = Xw.T @ Rw
        A = Xw.T @ Xw + jnp.diag(self._Lambda)
        chol = jnp.linalg.cholesky(A)
        a_star = jax.scipy.linalg.cho_solve((chol, True), b)
        logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        logL = (-0.5 * jnp.sum(Rw ** 2) + 0.5 * jnp.dot(b, a_star)
                - 0.5 * logdetA)
        return dict(
            stacked=np.asarray(s), M_det=np.asarray(M_det),
            ret=np.asarray(ret), Rw=np.asarray(Rw), Xw=np.asarray(Xw),
            b=np.asarray(b), A=np.asarray(A), chol=np.asarray(chol),
            a_star=np.asarray(a_star), logdetA=float(logdetA),
            logL_data=float(logL),
            chi2w_premarg=float(jnp.sum(Rw ** 2)),
            chi2w_resid=float(jnp.sum((Rw - Xw @ a_star) ** 2)),
            model_image=np.asarray(
                M_det + jnp.sum(ret * a_star[None, None, :], axis=-1)),
        )
