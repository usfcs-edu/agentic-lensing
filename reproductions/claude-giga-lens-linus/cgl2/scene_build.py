"""Scene-API factories for the foundry-i 46-dim config class (parity harness).

Builds the vendored gigalens-linus LensModel/ImageData for the model the old
validated stack ran (cgl.likelihood.build_marg_model / foundry-i
_hmc_lib_marg): EPL(50)+Shear deflector, 4 SersicEllipse lens lights,
SersicEllipse + Shapelets(n_max=6) source, priors verbatim.

Scene layout (the single definition; cgl2.param_map._PREFIX_MAP mirrors it):
    Plane 0 (lens):   mass=[EPL, Shear], light=[LL0, LL1, LL2, LL3]
    Plane 1 (source): light=[srcS, srcShp]   (deflection_ratio -> default 1.0)

Two views of the SAME physics:
  * view="marg"    — THE model of record. Sersics render as rank-4 depth-1
    basis columns WITH their sampled Ie (ParitySersicEllipse(as_basis=True)),
    shapelets are use_lstsq=True unit columns; one
    ``lstsq_simulate(return_stacked=True)`` render yields M_det (sum of the 5
    Sersic columns) + the 28-column design matrix, in the old no-conversion-
    factor units (identity Ie map, param_map reconciliation #3).
  * view="forward" — all light amplitude-carrying and rank-3 so the stock
    ``SceneSimulator.simulate`` / ``ImageLikelihoodTerm`` path runs (gates
    F1/F3/F5). Shapelet amps are FIXED constants (`amp_consts`, e.g. the
    old-stack marginal-mode a* at z_ref). ``simulate`` multiplies by
    conversion_factor, so forward-view amplitudes are in old/cf units
    (param_map reconciliation #3).

PARITY-CONVENTION SUBCLASSES (cgl2-side; the vendor stays UNPATCHED).
The scene API's native profile/grid conventions differ from the validated
58ec9a7 stack in three DOCUMENTED ways; parity models therefore use local
subclasses/overrides that reproduce the old conventions exactly, and the
harness ALSO reports the native-convention deltas:
  1. Sersic bn approximant: old ``bn = 1.9992 n - 0.3271`` (Capaccioli-style
     linear fit) vs new ``bn = exp(0.6950 + ln n - 0.1789/n)`` (MacArthur+
    2003) — a ~1e-3-relative MODEL difference. ParitySersicEllipse uses the
     old bn.
  2. Shapelet Hermite normalization: old computes the prefactor
     1/sqrt(2^n sqrt(pi) n!) in FLOAT32 (jnp.arange(dtype=float32)) and the
     H_n recurrence on 2z; new uses an exact-f64 normalized phi recurrence —
     a ~1e-8-relative numerics difference. ParityShapelets replicates the old
     op sequence bit-for-bit.
  3. Coordinate grid dtype: old LensSimulatorInterface.get_coords returns
     FLOAT32 grid values (lenstronomy PixelGrid + .astype(float32)); the new
     LensWCS.pixel_grid is float64 — ~1e-7-relative coordinate rounding.
     ``apply_parity_conventions`` overrides a SceneSimulator instance's
     img_X/img_Y (and its PSF kernel: old = lenstronomy subgrid_kernel,
     NO re-normalization; new re-normalizes to sum 1) BEFORE its first
     trace. PSF convolution itself stays the vendor's FFT path (that is part
     of what F1/F2 certify); spatial-vs-FFT roundoff is the residual.
"""
from __future__ import annotations

import functools
import types
from typing import Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from cgl2 import guards, paths

paths.bootstrap_vendor()

import tensorflow_probability.substrates.jax as tfp  # noqa: E402
from gigalens.jax.profiles.light import sersic, shapelets  # noqa: E402
from gigalens.jax.profiles.mass import epl, shear  # noqa: E402
from gigalens.jax.scene import Component, LensModel, Plane  # noqa: E402
from gigalens.jax.scene_prob_model import ImageData  # noqa: E402
from gigalens.jax.scene_simulator import SceneSimulator  # noqa: E402
from gigalens.simulator import SimulatorConfig  # noqa: E402
from jax import jit  # noqa: E402

tfd = tfp.distributions

DEFAULT_EXP_TIME = 1197.7  # v2-class F140W exposure time (foundry-i EXP_TIME)


# --------------------------------------------------------------------------- #
# parity profiles (old-stack conventions; vendor classes subclassed, unpatched)
# --------------------------------------------------------------------------- #
class ParitySersicEllipse(sersic.SersicEllipse):
    """SersicEllipse with the OLD stack's bn (docstring item 1).

    ``as_basis=True`` returns a rank-4 (1, H, W, batch) Ie-scaled column so it
    can be stacked with lstsq basis columns in ``lstsq_simulate`` (the marg
    view); ``as_basis=False`` is the stock rank-3 forward-render shape.
    Ops replicate gigalens-sean@58ec9a7 light/sersic.py SersicEllipse.light
    line-for-line (distance() is inherited: verified identical in both
    stacks).
    """

    def __init__(self, as_basis=False, **kwargs):
        super().__init__(use_lstsq=False, **kwargs)
        self._as_basis = bool(as_basis)

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, R_sersic, n_sersic, e1, e2, center_x, center_y, Ie=None):
        R = self.distance(x, y, center_x, center_y, e1, e2)
        bn = 1.9992 * n_sersic - 0.3271          # OLD approximant (item 1)
        ret = jnp.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))
        out = Ie * ret
        return out[jnp.newaxis, ...] if self._as_basis else out


class ParityShapelets(shapelets.Shapelets):
    """Shapelets with the OLD stack's basis numerics (docstring item 2).

    Replicates gigalens-sean@58ec9a7 light/shapelets.py light()
    (interpolate=False path) op-for-op: float32 Hermite prefactor, H_n
    recurrence on 2z in a preallocated buffer, separate Gaussian envelope.
    N1/N2 enumeration is identical in both stacks (verified).
    """

    def __init__(self, n_max, use_lstsq=False, **kwargs):
        super().__init__(n_max, use_lstsq=use_lstsq, **kwargs)
        N = jnp.arange(0, self.n_max + 1, dtype=jnp.float32)  # f32 ON PURPOSE
        self.prefactor = 1.0 / jnp.sqrt(
            2 ** N * jnp.sqrt(float(jnp.pi)) * jnp.exp(jax.lax.lgamma(N + 1)))
        self._N1l = [int(v) for v in np.asarray(self.N1)]
        self._N2l = [int(v) for v in np.asarray(self.N2)]

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, center_x, center_y, beta, **amp):
        x = (x - center_x) / beta
        y = (y - center_y) / beta
        z = jnp.stack([x, y], axis=0)
        ret = jnp.ones((self.n_max + 1, *z.shape))
        ret = ret.at[0].set(jnp.ones_like(z))
        ret = ret.at[1].set(2 * z)
        for n in range(2, self.n_max + 1):
            ret = ret.at[n].set(2 * (z * ret[n - 1] - (n - 1) * ret[n - 2]))
        ret = jnp.einsum("i,i...->i...", self.prefactor, ret)
        XX, YY = ret[:, 0, ...], ret[:, 1, ...]
        fac = jnp.exp(-(x ** 2 + y ** 2) / 2)
        basis = fac * XX[self._N1l, ...] * YY[self._N2l, ...]
        if self.use_lstsq:
            return basis
        amps = jnp.stack([amp[k] for k in self._amp_names], axis=0)
        if jnp.ndim(amps) == 1:      # fixed (constant) amplitudes
            return jnp.einsum("i,i...->...", amps, basis)
        return jnp.einsum("ij,i...j->...j", amps, basis)


# --------------------------------------------------------------------------- #
# old-stack grid + kernel conventions (docstring item 3)
# --------------------------------------------------------------------------- #
def sean_coords(delta_pix: float, num_pix: int, supersample: int):
    """The old stack's float32 supersampled coordinate grids (img_X, img_Y).

    Replicates gigalens-sean@58ec9a7 LensSimulatorInterface.get_coords with
    transform_pix2angle = (delta_pix * I)/supersample: lenstronomy PixelGrid,
    final .astype(float32). Returns two (n*ss, n*ss) float32 arrays.
    """
    from lenstronomy.Data.pixel_grid import PixelGrid

    transform = np.eye(2) * delta_pix / supersample
    lo = np.arange(0, supersample * num_pix, dtype=np.float32)
    lo = np.min(lo - np.mean(lo))
    ra0, dec0 = np.squeeze(transform @ ([[lo], [lo]]))
    grid = PixelGrid(nx=supersample * num_pix, ny=supersample * num_pix,
                     ra_at_xy_0=ra0, dec_at_xy_0=dec0,
                     transform_pix2angle=np.array(transform))
    return (grid._x_grid.astype(np.float32), grid._y_grid.astype(np.float32))


def sean_subgrid_kernel(psf: np.ndarray, supersample: int) -> np.ndarray:
    """The old stack's PSF kernel: lenstronomy subgrid_kernel(odd=True),
    NO re-normalization (the new stack divides by sum), UNFLIPPED orientation
    (the old stack pre-flips then cross-correlates == true convolution with
    THIS kernel; the new FFT path convolves directly)."""
    from lenstronomy.Util.kernel_util import subgrid_kernel

    psf = np.asarray(psf, dtype=np.float64)
    if supersample == 1:
        return psf
    return np.asarray(subgrid_kernel(psf, supersample, odd=True),
                      dtype=np.float64)


def apply_parity_conventions(sim: SceneSimulator, *, img_X=None, img_Y=None,
                             kernel=None) -> SceneSimulator:
    """Override a SceneSimulator instance's grid/PSF with old-stack-convention
    arrays (instance attributes only; the vendor class is untouched). MUST be
    called before the simulator's first traced call — its jitted methods
    capture these arrays at trace time."""
    if img_X is not None:
        sim.img_X = jnp.asarray(np.asarray(img_X))[..., jnp.newaxis]
    if img_Y is not None:
        sim.img_Y = jnp.asarray(np.asarray(img_Y))[..., jnp.newaxis]
    if kernel is not None:
        k = jnp.asarray(np.asarray(kernel, dtype=np.float64))
        sim.flat_kernel = k[jnp.newaxis, jnp.newaxis, :, :]
    return sim


# --------------------------------------------------------------------------- #
# priors (verbatim hyperparameters from cgl.likelihood._build_prior /
# foundry-i _hmc_lib_marg._build_prior)
# --------------------------------------------------------------------------- #
def _sersic_priors(R_med, R_sig, Ie_med, Ie_sig, n_lo=0.5, n_hi=8.0,
                   cx_mean=0.0, cy_mean=0.0, c_sig=0.05, ie_scale=1.0):
    return dict(
        R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
        n_sersic=tfd.Uniform(n_lo, n_hi),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        center_x=tfd.Normal(cx_mean, c_sig), center_y=tfd.Normal(cy_mean, c_sig),
        Ie=tfd.LogNormal(jnp.log(Ie_med * ie_scale), Ie_sig),
    )


def build_scene_model(*, view: str = "marg", n_max: int = 6,
                      near_xy: Sequence[float],
                      amp_consts: Optional[Dict[str, float]] = None,
                      ie_scale: float = 1.0,
                      shapelet_sigma0: float = 5.0,
                      parity_profiles: bool = True) -> types.SimpleNamespace:
    """Build the foundry-i 46-dim config class as a scene-API LensModel.

    Args:
        view: "marg" (model of record; rank-4 Sersic basis + lstsq shapelets)
            or "forward" (rank-3 forward light; shapelet amps FIXED constants).
        n_max: shapelet order (28 columns at 6).
        near_xy: (arcsec_x, arcsec_y) LL2/LL3 prior centers (the nearby
            galaxy) — REQUIRED, no silent default (pass the foundry-i
            nearby_galaxy_loc values; the parity harness reads them from the
            reference artifact so both stacks agree bit-for-bit).
        amp_consts: {"amp00": val, ...} fixed shapelet amplitudes, REQUIRED
            for view="forward" (in scene-forward units = old-units / cf).
        ie_scale: multiplies the five Sersic Ie prior MEDIANS (use
            1/conversion_factor for a forward view whose params are in
            scene-forward units; priors are irrelevant to likelihood-only
            parity gates but kept unit-consistent).
        shapelet_sigma0: ridge scale — Lambda_ii = (i+1)/sigma0**2.
        parity_profiles: True (default) = the old-convention subclasses
            (ParitySersicEllipse/ParityShapelets — the parity-certified
            configuration); False = the vendor's NATIVE profiles (new bn,
            exact-f64 shapelet basis) for the informational native-convention
            arm. Only view='forward' supports False (the marg view needs the
            rank-4 basis Sersic, which only the parity subclass provides).

    Returns SimpleNamespace with: model, components (dict), planes,
    Lambda_diag (marg view), det_idx/marg_idx column indices into the stacked
    basis, n_max, depths, view.
    """
    if view not in ("marg", "forward"):
        raise ValueError(f"view must be 'marg' or 'forward'; got {view!r}")
    guards.require_x64()
    guards.assert_scene_config_certified(["EPL", "Shear"],
                                         ["SersicEllipse", "Shapelets"])

    near_x, near_y = float(near_xy[0]), float(near_xy[1])
    as_basis = view == "marg"
    if not parity_profiles and view != "forward":
        raise ValueError("parity_profiles=False is only supported for "
                         "view='forward' (see docstring)")

    def _mk_sersic():
        if parity_profiles:
            return ParitySersicEllipse(as_basis=as_basis)
        return sersic.SersicEllipse(use_lstsq=False)

    mass = Component(epl.EPL(50), dict(
        theta_E=tfd.LogNormal(jnp.log(2.5), 0.25),
        gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 2.7),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.02), center_y=tfd.Normal(0.0, 0.02),
    ))
    ext_shear = Component(shear.Shear(), dict(
        gamma1=tfd.Normal(0.0, 0.05), gamma2=tfd.Normal(0.0, 0.05)))

    ll_hyper = [
        dict(R_med=0.4, R_sig=0.3, Ie_med=5.0, Ie_sig=0.5, c_sig=0.02),
        dict(R_med=2.0, R_sig=0.3, Ie_med=2.0, Ie_sig=0.5, c_sig=0.02),
        dict(R_med=0.3, R_sig=0.3, Ie_med=1.0, Ie_sig=0.5,
             cx_mean=near_x, cy_mean=near_y, c_sig=0.1),
        dict(R_med=0.6, R_sig=0.3, Ie_med=0.5, Ie_sig=0.5,
             cx_mean=near_x, cy_mean=near_y, c_sig=0.1),
    ]
    LLs = [Component(_mk_sersic(),
                     _sersic_priors(ie_scale=ie_scale, **h)) for h in ll_hyper]

    srcS = Component(_mk_sersic(), dict(
        R_sersic=tfd.LogNormal(jnp.log(0.5), 0.3),
        n_sersic=tfd.Uniform(0.5, 6.0),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
        Ie=tfd.LogNormal(jnp.log(2.0 * ie_scale), 0.5),
    ))

    shp_priors: Dict[str, object] = dict(
        beta=tfd.LogNormal(jnp.log(0.1), 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05))
    if view == "marg":
        shp = ParityShapelets(n_max=n_max, use_lstsq=True)
    else:
        shp = (ParityShapelets(n_max=n_max, use_lstsq=False) if parity_profiles
               else shapelets.Shapelets(n_max=n_max, use_lstsq=False))
        if amp_consts is None:
            raise ValueError("view='forward' requires amp_consts (fixed "
                             "shapelet amplitudes; no silent default)")
        missing = [a for a in shp._amp_names if a not in amp_consts]
        if missing:
            raise ValueError(f"amp_consts missing {missing}")
        shp_priors.update({a: float(amp_consts[a]) for a in shp._amp_names})
    srcShp = Component(shp, shp_priors)

    planes = [
        Plane(mass=[mass, ext_shear], light=LLs),
        Plane(light=[srcS, srcShp]),   # deflection_ratio -> documented 1.0
    ]
    model = LensModel(planes)

    depth = (n_max + 1) * (n_max + 2) // 2
    i_idx = jnp.arange(depth, dtype=jnp.float64)
    Lambda_diag = (i_idx + 1.0) / (shapelet_sigma0 * shapelet_sigma0)

    # stacked-basis column bookkeeping (marg view): columns follow the
    # (plane, light) order — LL0..LL3 (0..3), srcS (4), shapelets (5..4+depth)
    det_idx = np.arange(5)
    marg_idx = np.arange(5, 5 + depth)

    return types.SimpleNamespace(
        model=model, view=view, n_max=n_max, depth=depth,
        components=dict(mass=mass, shear=ext_shear, LLs=LLs, srcS=srcS,
                        srcShp=srcShp),
        planes=planes, Lambda_diag=np.asarray(Lambda_diag),
        det_idx=det_idx, marg_idx=marg_idx, near_xy=(near_x, near_y),
    )


# --------------------------------------------------------------------------- #
# data / simulator-config helpers
# --------------------------------------------------------------------------- #
def load_product(npz_path) -> dict:
    """foundry-i Stage-A product dict (img/err_map/keep_mask/psf/meta),
    float64-preserving (mirror of cgl.paths.load_product, old campaign)."""
    import json

    z = np.load(npz_path)
    meta = json.loads(str(z["meta"]))
    return dict(img=np.asarray(z["img"]), err_map=np.asarray(z["err_map"]),
                keep_mask=np.asarray(z["keep_mask"]).astype(bool),
                psf=np.asarray(z["psf"]), meta=meta)


def sim_config_for(product: dict, *, delta_pix: float, supersample: int,
                   psf: Optional[np.ndarray] = None) -> SimulatorConfig:
    """SimulatorConfig for a product; PSF handed at delta_pix (the simulator
    supersamples internally via subgrid_kernel — the foundry-i incident
    convention, enforced by assert_psf_sampling when meta declares a scale)."""
    meta = product.get("meta") or {}
    if "delta_pix" in meta and abs(float(meta["delta_pix"]) - delta_pix) > 1e-12:
        raise ValueError(f"delta_pix={delta_pix} but meta says {meta['delta_pix']}")
    if "supersample" in meta and int(meta["supersample"]) != int(supersample):
        raise ValueError(f"supersample={supersample} but meta says "
                         f"{meta['supersample']}")
    if "psf_delta_pix" in meta:
        guards.assert_psf_sampling(float(meta["psf_delta_pix"]), delta_pix)
    kernel = np.asarray(product["psf"] if psf is None else psf,
                        dtype=np.float64)
    return SimulatorConfig(delta_pix=float(delta_pix),
                           num_pix=int(product["img"].shape[0]),
                           supersample=int(supersample), kernel=kernel)


def build_image_data(product: dict, sim_cfg: SimulatorConfig, *,
                     sees="all", mode: Optional[str] = None) -> ImageData:
    """Stock ImageData: image + raw error map + keep mask (§3.4/§3.8)."""
    return ImageData(np.asarray(product["img"], dtype=np.float64), sim_cfg,
                     error_map=np.asarray(product["err_map"], dtype=np.float64),
                     mask=np.asarray(product["keep_mask"], dtype=bool),
                     sees=sees, mode=mode)


# --------------------------------------------------------------------------- #
# toy scene (unit tests + 03 small-grid exact reference; CPU-safe)
# --------------------------------------------------------------------------- #
def build_toy_scene(*, n_pix: int = 32, n_max: int = 4, supersample: int = 1,
                    delta_pix: float = 0.2, noise_sigma: float = 0.05,
                    seed: int = 0) -> types.SimpleNamespace:
    """Small EPL+Shear / Sersic / Sersic+Shapelets scene for validation toys.

    Same structural class as the parity model (1 lens-light Sersic instead of
    4). Returns namespace with: marg (marg-view namespace: model/components/
    Lambda_diag/det_idx/marg_idx), fwd (forward-view twin with the same truth
    amps as constants), sim_cfg, product (img/err_map/keep_mask/psf/meta;
    img = forward truth render + seeded white noise), truth_labeled (the
    constrained truth values by scene unique key), amp_truth (n_k,).
    """
    guards.require_x64()
    rng = np.random.default_rng(seed)
    depth = (n_max + 1) * (n_max + 2) // 2

    # small centered Gaussian PSF at delta_pix, sum=1
    yy, xx = np.mgrid[-3:4, -3:4]
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.1 ** 2))
    psf = (psf / psf.sum()).astype(np.float64)
    sim_cfg = SimulatorConfig(delta_pix=delta_pix, num_pix=n_pix,
                              supersample=supersample, kernel=psf)

    def _mk(view, amp_consts=None):
        as_basis = view == "marg"
        mass = Component(epl.EPL(50), dict(
            theta_E=tfd.LogNormal(jnp.log(1.2), 0.15),
            gamma=tfd.TruncatedNormal(2.0, 0.2, 1.2, 2.6),
            e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
            center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05)))
        sh = Component(shear.Shear(), dict(gamma1=tfd.Normal(0.0, 0.05),
                                           gamma2=tfd.Normal(0.0, 0.05)))
        LL = Component(ParitySersicEllipse(as_basis=as_basis),
                       _sersic_priors(0.6, 0.3, 2.0, 0.5, c_sig=0.05))
        srcS = Component(ParitySersicEllipse(as_basis=as_basis),
                         _sersic_priors(0.25, 0.3, 1.5, 0.5, n_hi=6.0,
                                        c_sig=0.1))
        shp_priors = dict(beta=tfd.LogNormal(jnp.log(0.15), 0.1),
                          center_x=tfd.Normal(0.0, 0.05),
                          center_y=tfd.Normal(0.0, 0.05))
        if view == "marg":
            shp = ParityShapelets(n_max=n_max, use_lstsq=True)
        else:
            shp = ParityShapelets(n_max=n_max, use_lstsq=False)
            shp_priors.update({a: float(amp_consts[a])
                               for a in shp._amp_names})
        srcShp = Component(shp, shp_priors)
        model = LensModel([Plane(mass=[mass, sh], light=[LL]),
                           Plane(light=[srcS, srcShp])])
        return types.SimpleNamespace(
            model=model, view=view,
            components=dict(mass=mass, shear=sh, LLs=[LL], srcS=srcS,
                            srcShp=srcShp),
            Lambda_diag=np.asarray((np.arange(depth) + 1.0) / 25.0),
            det_idx=np.arange(2), marg_idx=np.arange(2, 2 + depth),
            n_max=n_max, depth=depth)

    # truth (constrained, scene unique keys) + truth amps
    amp_truth = 0.5 * rng.standard_normal(depth) / np.sqrt(np.arange(depth) + 1.0)
    cf = float(delta_pix) ** 2
    truth = {
        "planes/0/mass/0/theta_E": 1.2, "planes/0/mass/0/gamma": 2.05,
        "planes/0/mass/0/e1": 0.06, "planes/0/mass/0/e2": -0.04,
        "planes/0/mass/0/center_x": 0.01, "planes/0/mass/0/center_y": -0.02,
        "planes/0/mass/1/gamma1": 0.02, "planes/0/mass/1/gamma2": -0.01,
        "planes/0/light/0/R_sersic": 0.7, "planes/0/light/0/n_sersic": 3.1,
        "planes/0/light/0/e1": 0.08, "planes/0/light/0/e2": 0.03,
        "planes/0/light/0/center_x": 0.01, "planes/0/light/0/center_y": -0.02,
        "planes/0/light/0/Ie": 2.1,
        "planes/1/light/0/R_sersic": 0.22, "planes/1/light/0/n_sersic": 1.4,
        "planes/1/light/0/e1": -0.1, "planes/1/light/0/e2": 0.06,
        "planes/1/light/0/center_x": 0.05, "planes/1/light/0/center_y": 0.03,
        "planes/1/light/0/Ie": 1.4,
        "planes/1/light/1/beta": 0.16, "planes/1/light/1/center_x": 0.05,
        "planes/1/light/1/center_y": 0.03,
    }
    mv = _mk("marg")
    amp_names = mv.components["srcShp"].profile._amp_names
    fv = _mk("forward", amp_consts={a: float(amp_truth[i] / cf)
                                    for i, a in enumerate(amp_names)})
    truth_fwd = dict(truth)
    for kk in list(truth_fwd):
        if kk.endswith("/Ie"):
            truth_fwd[kk] = truth_fwd[kk] / cf

    from gigalens.jax.scene_simulator import SceneSimulator as _SS

    sim = _SS(fv.model, sim_cfg)
    params_t = fv.model.to_params({k: jnp.asarray(np.float64(v))
                                   for k, v in truth_fwd.items()})
    img_clean = np.asarray(sim.simulate(params_t), dtype=np.float64)
    img = img_clean + noise_sigma * rng.standard_normal(img_clean.shape)
    err = np.full_like(img, noise_sigma)
    keep = np.ones_like(img, dtype=bool)
    product = dict(img=img, err_map=err, keep_mask=keep, psf=psf,
                   meta=dict(delta_pix=delta_pix, supersample=supersample,
                             toy=True, seed=seed))
    return types.SimpleNamespace(
        marg=mv, fwd=fv, sim_cfg=sim_cfg, product=product,
        truth_marg=truth, truth_fwd=truth_fwd, amp_truth=amp_truth,
        conversion_factor=cf, img_clean=img_clean)
