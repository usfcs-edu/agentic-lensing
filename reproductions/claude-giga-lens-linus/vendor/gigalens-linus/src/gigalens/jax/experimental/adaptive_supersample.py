"""EXPERIMENTAL: adaptive per-pixel super/sub-sampling for the scene simulator.

Uniform supersampling pays ``ss**2`` light evaluations on EVERY pixel, but the
pixels that need sub-pixel quadrature are the few where the unconvolved model
varies inside a pixel by more than the noise — arcs, cusps, the lens-light
center. This module renders each native pixel with its own quadrature:

* **factor 2 / 4 / 8** — ``f x f`` sub-pixel points, averaged (same quadrature
  as uniform ``supersample=f``);
* **factor 1**      — one point at the pixel center (the ``supersample=1`` rule);
* **factor 1/2 / 1/4** — one point shared by an aligned ``2x2`` / ``4x4`` block
  (SUBsampling: the block is deep in the noise, one sample is enough).

**What drives the factor map.** The per-pixel criterion is the LIKELIHOOD damage
of a quadrature error: an error ``delta`` in pixel ``i`` biases chi^2 by
``(delta/sigma_i)^2``, and ``|delta|`` is bounded by (a fraction of) the pixel's
flux, so the model-free driver is the observed **SNR = image / error_map**
(:meth:`AdaptiveGrid.from_snr`). SNR — not raw brightness — because the noise
map is what converts a render error into likelihood units, and because
``SNR << 1`` over a whole neighborhood *proves* subsampling is harmless there.
The map is frozen at construction (data-derived, never model-derived): a static
quadrature keeps shapes static under jit and keeps the likelihood smooth in the
parameters, which gradient-based samplers require. The map is dilated so that
during burn-in a slightly mis-placed model arc still lands on refined pixels.

**PSF: bin-first.** Points are scattered/averaged to the native grid FIRST and
convolved with the native-resolution kernel (the ``supersample=1`` conv path) —
NOT the ``subgrid_kernel`` supersampled convolution of the uniform pipeline.
This is exact for the flux redistribution of whatever image was rendered (PSF
wings from bright regions land on subsampled dark pixels exactly), but it drops
sub-pixel PSF structure. NOTE the two conventions genuinely differ where the
unconvolved model is cuspy: on the bundled demo at truth the gap reaches
~4 sigma at the lens-light peak. That gap pre-exists this module — stock
gigalens itself switches convention between ``supersample=1`` (native kernel)
and ``supersample >= 2`` (subgrid kernel) — but at depths where it matters the
choice must match how the PSF was measured (an empirical star-stacked PSF is
already pixel-integrated, which is what bin-first assumes). Measure BOTH modes
of :func:`compare_to_reference` on a new system before trusting posteriors.

Typical use::

    grid = AdaptiveGrid.from_snr(observed_img, error_map)
    sim  = AdaptiveSceneSimulator(model, sim_config, grid)      # sim_config.supersample == 1
    img  = sim.simulate(params)

    # or end-to-end through the ProbModel stack:
    ds   = AdaptiveImageData(observed_img, sim_config, background_rms=.2,
                             exp_time=100, sees='all')
    prob = ProbModel(model, ds, mode='forward')

    plot_factor_map(grid)                                       # sanity-check the tiers
    rep  = compare_to_reference(sim, params, error_map=error_map)
    assert rep['max_abs_delta_over_sigma'] < 0.1                # the falsifier

The default SNR thresholds are STARTING POINTS calibrated on the bundled demo
system (see ``tests/test_adaptive_supersample.py``); for a new instrument/depth,
run :func:`compare_to_reference` at a fiducial model and tighten the levels
until the per-pixel ``|delta|/sigma`` map is below your tolerance everywhere.

**Known limitation — model flux outside the refined regions.** The map is
frozen from the DATA, so it is only accurate for models whose flux lies within
``dilate`` pixels of where the data shows flux. Two situations violate that:
(a) early burn-in proposals displaced by more than ``dilate`` native pixels,
and (b) model features the data does not show at all (e.g. a predicted faint
counter-image in a noise-dominated region) — both are rendered at the coarse
tiers, and their likelihood/gradient carries the corresponding quadrature
error. When that matters (point-source flux ratios, exploratory burn-in),
disable subsampling by ending ``snr_levels`` with ``(-np.inf, 1.0)`` (factor-1
sky costs accuracy nothing — only speed) or increase ``dilate``; the
:func:`compare_to_reference` falsifier only certifies the params you test.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Optional, Sequence, Tuple

import numpy as np
from jax import jit
from jax import numpy as jnp

from gigalens.jax.scene import LensModel
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.jax.scene_prob_model import ImageData
from gigalens.jax.simulator import _convolve_components
from gigalens.simulator import SimulatorConfig

__all__ = [
    "AdaptiveGrid",
    "AdaptiveSceneSimulator",
    "AdaptiveImageData",
    "compare_to_reference",
    "plot_factor_map",
    "ALLOWED_FACTORS",
    "DEFAULT_SNR_LEVELS",
]

# Powers of two only: factors >= 1 reproduce the uniform-supersample quadrature
# exactly (same sub-pixel points), and the sub-1 factors tile the image in
# aligned blocks whose size divides the supersample offsets' alignment.
ALLOWED_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)#, 8.0)
_SUBSAMPLE_BLOCK = {0.5: 2, 0.25: 4}

# (min_snr, factor), descending; first level with snr >= min_snr wins. The last
# level must be the (-inf, floor) catch-all. Calibrated on the bundled demo
# system (2026-07-12, truth params, unconvolved uniform-ss quadrature vs ss=16):
# per-factor error first exceeds 0.05 sigma at smoothed/dilated SNR 8.9 (factor
# 1), 11.9 (factor 2), 14.7 (factor 4); ss=8 stays below 0.023 sigma everywhere;
# 2x2 / 4x4 blocks stay below 0.1 sigma for SNR < 1.0 / < 0.5. The levels below
# sit inside those bands, giving ~2x margin against the 0.1 sigma falsifier in
# tests/test_adaptive_supersample.py. Treat as a starting point elsewhere:
# error at fixed factor scales with flux x profile curvature, so a cuspier or
# deeper system moves the bands — re-derive with compare_to_reference.
DEFAULT_SNR_LEVELS = (
    # (14.0, 8.0),
    (10.0, 4.0),
    (7.0, 2.0),
    (2.0, 1.0),
    (1.0, 0.5),
    (-np.inf, 0.25),
)


# DEFAULT_SNR_LEVELS = (
#     # (14.0, 8.0),
#     (10.0, 4.0),
#     (7.0, 2.0),
#     (1.0, 1.0),
#     (0.4, 0.5),
#     (-np.inf, 0.25),
# )

def _validate_subsample_alignment(fm: np.ndarray, factor: float, b: int):
    """Every pixel with a sub-1 ``factor`` must belong to a FULLY-inside, aligned,
    uniformly-``factor`` ``b x b`` block — one evaluation point legitimately
    represents exactly that block. Anything else raises (no silent promotion of a
    hand-built map: a caller who placed a 0.5 pixel at an odd offset believes it
    is being subsampled; quietly rendering it at full resolution would hide the
    mistake and quietly changing coverage would corrupt a careful spec)."""
    sel = fm == factor
    if not sel.any():
        return
    H, W = fm.shape
    Hb, Wb = (H // b) * b, (W // b) * b
    ok = np.zeros_like(sel)
    if Hb and Wb:
        blocks = sel[:Hb, :Wb].reshape(Hb // b, b, Wb // b, b)
        full = blocks.all(axis=(1, 3))
        ok[:Hb, :Wb] = np.kron(full, np.ones((b, b), dtype=bool))
    bad = sel & ~ok
    if bad.any():
        i, j = np.argwhere(bad)[0]
        raise ValueError(
            f"AdaptiveGrid factor_map: pixel (y={i}, x={j}) has subsample factor "
            f"{factor} but is not part of an aligned, fully-{factor} {b}x{b} block "
            f"(blocks start at indices divisible by {b} and must lie fully inside "
            "the image). Build the map with AdaptiveGrid.from_snr / "
            "AdaptiveGrid.quantize, which align blocks by construction.")


class AdaptiveGrid:
    """A validated per-native-pixel quadrature spec: ``factor_map[y, x]`` in
    ``{1/4, 1/2, 1, 2, 4}``. Construct via :meth:`from_snr` (data-driven) or pass
    an explicit map (validated strictly — see :func:`_validate_subsample_alignment`).
    """

    def __init__(self, factor_map):
        fm = np.asarray(factor_map, dtype=np.float64)
        if fm.ndim != 2:
            raise ValueError(
                f"AdaptiveGrid factor_map must be 2D (H, W); got shape {fm.shape}.")
        if not np.all(np.isfinite(fm)):
            raise ValueError(
                "AdaptiveGrid factor_map contains non-finite values — every pixel "
                "needs an explicit factor from " + repr(ALLOWED_FACTORS) + ".")
        bad = ~np.isin(fm, ALLOWED_FACTORS)
        if bad.any():
            vals = np.unique(fm[bad])[:8]
            raise ValueError(
                f"AdaptiveGrid factor_map contains unsupported factors {vals} — "
                f"allowed: {ALLOWED_FACTORS} (powers of two: >=1 reproduce uniform "
                "supersampling exactly, <1 are aligned-block subsampling).")
        for f, b in _SUBSAMPLE_BLOCK.items():
            _validate_subsample_alignment(fm, f, b)
        fm.setflags(write=False)  # frozen: the simulator bakes it into static arrays
        self.factor_map = fm

    # -- data-driven construction ---------------------------------------------------
    @classmethod
    def from_snr(cls, image, error_map, *, snr_levels=DEFAULT_SNR_LEVELS,
                 smooth: int = 1, dilate: int = 2) -> "AdaptiveGrid":
        """Factor map from the observed per-pixel SNR (``image / error_map``).

        ``smooth`` (half-width of a box mean, 0 = off) suppresses single-pixel
        noise excursions so sky pixels don't get promoted by a 3-sigma fluctuation;
        it only drives the factor-1/subsample boundary — the supersample tiers
        (factor >= 2) threshold ``max(raw, smoothed)`` SNR, so a genuine compact
        peak (cusp, point source) is never averaged below its refinement
        threshold by its dark neighborhood. ``dilate`` (half-width of a grey
        max-filter, 0 = off) expands every bright tier outward so a model arc
        slightly displaced from the data's arc — early burn-in — still lands on
        refined pixels, and demotion to subsampling only happens where a whole
        neighborhood is quiet. Both act on the SNR field BEFORE thresholding.

        ``snr_levels``: descending ``(min_snr, factor)`` pairs, first match wins;
        the last pair must be the ``(-inf, floor)`` catch-all. Pass e.g.
        ``((8.0, 2.0), (-np.inf, 1.0))`` to disable factor-4 refinement and all
        subsampling.
        """
        from scipy.ndimage import maximum_filter, uniform_filter

        image = np.asarray(image, dtype=np.float64)
        error_map = np.asarray(error_map, dtype=np.float64)
        if image.ndim != 2 or image.shape != error_map.shape:
            raise ValueError(
                f"from_snr: image {image.shape} and error_map {error_map.shape} "
                "must be identical 2D shapes.")
        if not np.all(np.isfinite(image)):
            raise ValueError("from_snr: image contains non-finite values.")
        if not (np.all(np.isfinite(error_map)) and np.all(error_map > 0)):
            raise ValueError(
                "from_snr: error_map must be finite and strictly positive — it is "
                "the per-pixel noise sigma that converts a render error into "
                "likelihood units.")

        levels = [(float(s), float(f)) for s, f in snr_levels]
        if len(levels) < 2:
            raise ValueError("from_snr: snr_levels needs at least two levels.")
        thr = [s for s, _ in levels]
        fac = [f for _, f in levels]
        if any(t2 >= t1 for t1, t2 in zip(thr, thr[1:])):
            raise ValueError(
                f"from_snr: snr_levels thresholds must be strictly descending; got {thr}.")
        if any(f2 >= f1 for f1, f2 in zip(fac, fac[1:])):
            raise ValueError(
                f"from_snr: snr_levels factors must be strictly descending; got {fac}.")
        if not np.isneginf(thr[-1]):
            raise ValueError(
                "from_snr: the last snr_level must be (-inf, floor_factor) so every "
                "pixel is assigned explicitly — no silent default tier.")
        unsupported = [f for f in fac if f not in ALLOWED_FACTORS]
        if unsupported:
            raise ValueError(
                f"from_snr: factors {unsupported} not in {ALLOWED_FACTORS}.")

        snr_raw = image / error_map
        snr = (uniform_filter(snr_raw, size=2 * int(smooth) + 1, mode="nearest")
               if smooth else snr_raw)
        # Two driver fields. Smoothing exists to stop NOISE from promoting sky
        # pixels at the low thresholds — but a box mean also averages a genuine
        # single-pixel peak (a cusp, a point source) down by up to the box area,
        # demoting exactly the pixels this module exists to refine. So the
        # SUPERSAMPLE tiers (factor >= 2) threshold max(raw, smoothed): a real
        # compact peak keeps its raw SNR, while single-pixel noise cannot reach
        # those high thresholds (their calibrated values sit at many sigma). The
        # factor-1/subsample boundary keeps the smoothed field, where noise
        # suppression is load-bearing (raw |noise| > 1 sigma is common).
        hi = np.maximum(snr, snr_raw)
        if dilate:
            size = 2 * int(dilate) + 1
            snr = maximum_filter(snr, size=size, mode="nearest")
            hi = maximum_filter(hi, size=size, mode="nearest")
        desired = np.select(
            [(hi if f >= 2 else snr) >= t for t, f in levels], fac)
        return cls(cls.quantize(desired))

    @staticmethod
    def quantize(desired: np.ndarray) -> np.ndarray:
        """Snap a per-pixel DESIRED factor field to a valid map: sub-1 factors are
        kept only where they tile an aligned uniform block, everything else is
        PROMOTED (never demoted) to the next valid factor — quantization may only
        add resolution, so it cannot introduce quadrature error."""
        d = np.asarray(desired, dtype=np.float64)
        if d.ndim != 2:
            raise ValueError(f"quantize: desired factor field must be 2D; got {d.shape}.")
        if not np.all(np.isfinite(d)):
            raise ValueError(
                "quantize: desired factor field contains non-finite values — a NaN "
                "in the SNR driver would otherwise surface as an opaque "
                "'unsupported factors' error downstream.")
        H, W = d.shape
        out = np.where(d >= 1.0, d, 1.0)
        # Coarser blocks override finer ones where they fully agree (a block that
        # is uniformly <= 0.25 is also uniformly <= 0.5, so apply 2x2 first).
        for f, b in sorted(_SUBSAMPLE_BLOCK.items(), reverse=True):
            Hb, Wb = (H // b) * b, (W // b) * b
            if not (Hb and Wb):
                continue
            ok = (d[:Hb, :Wb].reshape(Hb // b, b, Wb // b, b) <= f).all(axis=(1, 3))
            mask = np.kron(ok, np.ones((b, b), dtype=bool))
            out[:Hb, :Wb] = np.where(mask, f, out[:Hb, :Wb])
        return out

    # -- bookkeeping ------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        return self.factor_map.shape

    @property
    def n_points(self) -> int:
        """Total light-evaluation points (the compute cost driver; compare with
        ``H * W * ss**2`` for uniform supersampling)."""
        fm = self.factor_map
        n = int(sum((fm == f).sum() * int(f) ** 2
                    for f in ALLOWED_FACTORS if f >= 1))
        n += int(sum((fm == f).sum() // b ** 2 for f, b in _SUBSAMPLE_BLOCK.items()))
        return n

    def __repr__(self):
        fm = self.factor_map
        tiers = {f: int((fm == f).sum()) for f in ALLOWED_FACTORS if (fm == f).any()}
        return (f"AdaptiveGrid(shape={fm.shape}, n_points={self.n_points}, "
                f"pixels_per_factor={tiers})")


def _build_point_cloud(factor_map: np.ndarray, wcs):
    """Static quadrature arrays for a validated factor map on a NATIVE-resolution
    ``LensWCS``. Returns ``(pts_x, pts_y, c_point, c_pix, c_weight)``:

    * ``pts_x/pts_y`` — (N,) angular coordinates of the light-evaluation points;
    * the contribution triplets — (M,) parallel arrays scattering point samples
      to flat native-pixel indices: ``pixel[c_pix] += c_weight * value[c_point]``.

    Weights: a factor-``f>=1`` pixel averages its ``f*f`` points (weight
    ``1/f**2`` — the same box quadrature as uniform ``supersample=f``); a
    subsample block copies its single point to each covered pixel (weight 1),
    i.e. the surface brightness is treated as constant across the block.
    """
    H, W = factor_map.shape
    px, py, cpt, cpix, cw = [], [], [], [], []
    n_pts = 0
    for f in (int(f) for f in ALLOWED_FACTORS if f >= 1):
        ii, jj = np.nonzero(factor_map == float(f))
        if ii.size == 0:
            continue
        off = (np.arange(f) + 0.5) / f - 0.5
        ox, oy = np.meshgrid(off, off)
        x = (jj[:, None] + ox.ravel()[None, :]).ravel()
        y = (ii[:, None] + oy.ravel()[None, :]).ravel()
        px.append(x)
        py.append(y)
        cpt.append(n_pts + np.arange(x.size))
        cpix.append(np.repeat(ii * W + jj, f * f))
        cw.append(np.full(x.size, 1.0 / (f * f)))
        n_pts += x.size
    for f, b in _SUBSAMPLE_BLOCK.items():
        sel = factor_map == f
        if not sel.any():
            continue
        Hb, Wb = (H // b) * b, (W // b) * b
        full = sel[:Hb, :Wb].reshape(Hb // b, b, Wb // b, b).all(axis=(1, 3))
        bi, bj = np.nonzero(full)
        i0, j0 = bi * b, bj * b
        px.append(j0 + (b - 1) / 2.0)
        py.append(i0 + (b - 1) / 2.0)
        di, dj = np.meshgrid(np.arange(b), np.arange(b), indexing="ij")
        pix = ((i0[:, None] + di.ravel()[None, :]) * W
               + (j0[:, None] + dj.ravel()[None, :])).ravel()
        cpt.append(np.repeat(n_pts + np.arange(i0.size), b * b))
        cpix.append(pix)
        cw.append(np.ones(pix.size))
        n_pts += i0.size
    pts_px = np.concatenate(px)
    pts_py = np.concatenate(py)
    ra, dec = wcs.pix2angle(pts_px, pts_py)
    return (np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64),
            np.concatenate(cpt).astype(np.int32), np.concatenate(cpix).astype(np.int32),
            np.concatenate(cw))


class AdaptiveSceneSimulator(SceneSimulator):
    """A :class:`SceneSimulator` whose quadrature is an :class:`AdaptiveGrid`.

    Light is evaluated on the grid's point cloud via the ``trace(positions=...)``
    seam, scattered/averaged to the native pixel grid, and PSF-convolved there
    with the native-resolution kernel (bin-first; see the module docstring for
    when that approximation must be measured). The lstsq amplitude solve is
    inherited unchanged — only the basis rendering (``_render_basis_nchw``)
    changes.
    """

    def __init__(self, model: LensModel, sim_config: SimulatorConfig,
                 grid: AdaptiveGrid, sees=None):
        if not isinstance(grid, AdaptiveGrid):
            raise TypeError(
                "AdaptiveSceneSimulator: pass an AdaptiveGrid (wrap a raw factor "
                "map in AdaptiveGrid(factor_map) so it is validated), got "
                f"{type(grid).__name__}.")
        if int(sim_config.supersample) != 1:
            raise ValueError(
                f"AdaptiveSceneSimulator: sim_config.supersample must be 1 (got "
                f"{sim_config.supersample}) — the factor map IS the supersampling "
                "spec; a global supersample on top would silently compose two "
                "quadratures (and resize the PSF kernel twice).")
        super().__init__(model, sim_config, sees=sees)
        ny, nx = self.wcs.n_y, self.wcs.n_x
        if grid.shape != (ny, nx):
            raise ValueError(
                f"AdaptiveSceneSimulator: factor map shape {grid.shape} != image "
                f"grid (n_y={ny}, n_x={nx}) — the map is a per-NATIVE-pixel spec "
                "and must match sim_config.num_pix exactly.")
        self.grid = grid
        pts_x, pts_y, c_point, c_pix, c_weight = _build_point_cloud(
            grid.factor_map, self.wcs)
        # Trailing singleton axis, like the stored (H, W, 1) grid: the params'
        # batch axis broadcasts onto it at trace time.
        self._pts_x = jnp.asarray(pts_x)[:, jnp.newaxis]
        self._pts_y = jnp.asarray(pts_y)[:, jnp.newaxis]
        self._c_point = jnp.asarray(c_point)
        self._c_pix = jnp.asarray(c_pix)
        self._c_weight = jnp.asarray(c_weight)
        self.n_points = int(pts_x.size)
        if self.n_points != grid.n_points:
            raise AssertionError(
                f"internal invariant broken: built {self.n_points} evaluation "
                f"points but AdaptiveGrid.n_points says {grid.n_points} — "
                "_build_point_cloud and the per-tier arithmetic have diverged.")

    def replace_config(self, sim_config) -> "AdaptiveSceneSimulator":
        return type(self)(self.model, sim_config, self.grid, sees=self._sees_ref)

    def _scatter_native(self, vals):
        """(…, N, bs) point samples → (…, H, W, bs) native pixels (weighted
        scatter-add over the precomputed static contribution triplets)."""
        contrib = (vals[..., self._c_point, :]
                   * self._c_weight.astype(vals.dtype)[:, jnp.newaxis])
        ny, nx = self.wcs.n_y, self.wcs.n_x
        flat = jnp.zeros(vals.shape[:-2] + (ny * nx, vals.shape[-1]), vals.dtype)
        flat = flat.at[..., self._c_pix, :].add(contrib)
        return flat.reshape(vals.shape[:-2] + (ny, nx, vals.shape[-1]))

    def _eval_point_lights(self, p, no_deflection):
        """Trace the point cloud and evaluate every rendered light component on
        it — the shared front half of both render paths (forward sums the
        contributions, lstsq stacks them). Returns ``(contribs, depths)`` with
        each contrib shaped ``(..., N, bs)``."""
        positions = self.trace(p, no_deflection,
                               positions=(self._pts_x, self._pts_y))
        contribs, depths = [], []
        for i, j, comp, d in self._light:
            bx, by = positions[i]
            contribs.append(self._render_light(comp, bx, by, p["planes"][i]["light"][j]))
            depths.append(d)
        return contribs, depths

    @functools.partial(jit, static_argnums=(0, 2))
    def simulate(self, params, no_deflection: bool = False):
        contribs, _ = self._eval_point_lights(params, no_deflection)
        img = functools.reduce(jnp.add, contribs)                # (N, bs)
        img = self._scatter_native(img)                          # (H, W, bs)
        img = jnp.transpose(img, (2, 0, 1))[:, jnp.newaxis]      # (bs, 1, H, W)
        ret = self._psf_convolve(img, lambda im, k: _convolve_components(im, k))
        return jnp.squeeze(ret) * self.conversion_factor

    def _render_basis_nchw(self, p, no_deflection):
        # The adaptive counterpart of the base pipeline: same stacking and
        # batch-matching contract, point-cloud quadrature, native-res conv, no pool.
        contribs, depths = self._eval_point_lights(p, no_deflection)
        img = jnp.concatenate(self._match_light_batch(contribs), axis=0)  # (ncomp, N, bs)
        img = self._scatter_native(img)                                   # (ncomp, H, W, bs)
        img = jnp.transpose(img, (3, 0, 1, 2))                            # (bs, ncomp, H, W)
        return self._psf_convolve(img, lambda im, k: _convolve_components(im, k, depths))


class AdaptiveImageData(ImageData):
    """An :class:`ImageData` whose likelihood term renders adaptively.

    ``adaptive_grid=None`` derives the grid from this dataset's own image and
    noise map via :meth:`AdaptiveGrid.from_snr` (``snr_levels``/``smooth``/
    ``dilate`` forwarded); pass an explicit :class:`AdaptiveGrid` to override.
    The grid lives on the DATASET because it is data-derived — the simulator
    still receives only a quadrature spec, so the model/data split of the
    Phase-A seam is preserved.
    """

    def __init__(self, image, sim_config, *, adaptive_grid: Optional[AdaptiveGrid] = None,
                 snr_levels: Sequence = DEFAULT_SNR_LEVELS, smooth: int = 1,
                 dilate: int = 2, **kwargs):
        super().__init__(image, sim_config, **kwargs)
        if adaptive_grid is None:
            adaptive_grid = AdaptiveGrid.from_snr(
                np.asarray(self.image), np.asarray(self.error_map),
                snr_levels=snr_levels, smooth=smooth, dilate=dilate)
        elif not isinstance(adaptive_grid, AdaptiveGrid):
            raise TypeError(
                "AdaptiveImageData: adaptive_grid must be an AdaptiveGrid (or None "
                f"to derive one from the data), got {type(adaptive_grid).__name__}.")
        if adaptive_grid.shape != tuple(self.image.shape):
            raise ValueError(
                f"AdaptiveImageData: adaptive_grid shape {adaptive_grid.shape} != "
                f"image shape {tuple(self.image.shape)}.")
        self.adaptive_grid = adaptive_grid

    def _build_simulator(self, model: LensModel, seen) -> AdaptiveSceneSimulator:
        # The one seam ImageData exposes for swapping the rendering quadrature;
        # the stock ImageLikelihoodTerm carries the adaptive simulator unchanged.
        return AdaptiveSceneSimulator(
            model, self.sim_config, self.adaptive_grid, sees=seen)


def compare_to_reference(adaptive_sim: AdaptiveSceneSimulator, params, *,
                         reference_supersample: int = 8, error_map=None,
                         psf_mode: str = "bin_first",
                         no_deflection: bool = False) -> dict:
    """Render ``adaptive_sim`` and a uniform high-supersample reference at the
    same ``params`` and report the per-pixel difference — the calibration /
    falsification tool for a factor map.

    ``psf_mode`` picks the reference's PSF convention, and the two modes answer
    DIFFERENT questions — measure both before trusting a new system:

    * ``"bin_first"`` (default): supersampled quadrature, binned to native
      pixels, convolved with the native-resolution kernel — the SAME convention
      the adaptive simulator uses (and the same one the stock pipeline uses at
      ``supersample=1``). This isolates the QUADRATURE error of the factor map:
      with ``error_map``, any pixel of ``delta_over_sigma`` above your tolerance
      (0.1 is a sensible default — chi^2 bias < 0.01 per pixel) means the map is
      mis-assigned there; tighten ``snr_levels``/``dilate`` and re-derive.
    * ``"subgrid"``: the stock uniform pipeline at ``supersample >= 2`` —
      supersampled ``subgrid_kernel`` PSF convolution before pooling. The extra
      delta this mode shows is the bin-first-vs-subgrid PSF CONVENTION gap, a
      modeling choice that pre-exists this module (stock gigalens switches
      convention between ``supersample=1`` and ``>= 2``). It is NOT reducible by
      refining the factor map. Measured on the bundled demo at truth: up to
      ~4 sigma at the lens-light cusp — at that depth the convention choice
      matters and should be made deliberately (match how the data was reduced /
      how the PSF was measured: a star-stacked empirical PSF is already
      pixel-integrated, which is what bin-first assumes).

    Returns a dict with ``adaptive``, ``reference``, ``delta`` (adaptive −
    reference), ``psf_mode``, ``n_points``, ``n_points_reference``, and — when
    ``error_map`` is given — ``delta_over_sigma`` and
    ``max_abs_delta_over_sigma``.
    """
    ss = int(reference_supersample)
    a = np.asarray(adaptive_sim.simulate(params, no_deflection))
    if psf_mode == "subgrid":
        ref_cfg = dataclasses.replace(adaptive_sim.sim_config, supersample=ss)
        ref_sim = SceneSimulator(adaptive_sim.model, ref_cfg,
                                 sees=adaptive_sim._sees_ref)
        r = np.asarray(ref_sim.simulate(params, no_deflection))
    elif psf_mode == "bin_first":
        ref_cfg = dataclasses.replace(adaptive_sim.sim_config, supersample=ss,
                                      kernel=None)
        ref_sim = SceneSimulator(adaptive_sim.model, ref_cfg,
                                 sees=adaptive_sim._sees_ref)
        r_unc = jnp.asarray(ref_sim.simulate(params, no_deflection))
        if adaptive_sim.flat_kernel is None:
            r = np.asarray(r_unc)
        else:
            lead = r_unc.shape[:-2]
            img4 = r_unc.reshape((-1, 1) + r_unc.shape[-2:])
            conv = adaptive_sim._psf_convolve(
                img4, lambda im, k: _convolve_components(im, k))
            r = np.asarray(conv.reshape(lead + r_unc.shape[-2:]))
    else:
        raise ValueError(
            f"psf_mode must be 'bin_first' or 'subgrid'; got {psf_mode!r}")
    out = {
        "adaptive": a,
        "reference": r,
        "delta": a - r,
        "psf_mode": psf_mode,
        "n_points": adaptive_sim.n_points,
        "n_points_reference": adaptive_sim.wcs.n_y * adaptive_sim.wcs.n_x * ss ** 2,
    }
    if error_map is not None:
        err = np.asarray(error_map)
        out["delta_over_sigma"] = out["delta"] / err
        out["max_abs_delta_over_sigma"] = float(np.max(np.abs(out["delta_over_sigma"])))
    return out


# ColorBrewer RdBu steps (colorblind- and print-safe per ColorBrewer): a
# diverging scale — cool/blue = subsampled, neutral gray = native, warm/red =
# supersampled — with darkness carrying magnitude on each arm. Identity is never
# color-alone: the colorbar labels every tier explicitly.
_FACTOR_COLORS = {
    0.25: "#0571b0",
    0.5: "#92c5de",
    1.0: "#f0efec",
    2.0: "#f4a582",
    4.0: "#d6604d",
    8.0: "#b2182b",
}
_FACTOR_LABELS = {
    0.25: "1/4  (4x4 block)",
    0.5: "1/2  (2x2 block)",
    1.0: "1  (native)",
    2.0: "2",
    4.0: "4",
    8.0: "8",
}


def plot_factor_map(grid, ax=None, title="Per-pixel sampling factor"):
    """Heat-map of a factor map: which pixels are super-/sub-sampled and by how
    much. ``grid`` is an :class:`AdaptiveGrid` or a raw factor map array (raw
    arrays are validated by wrapping). Returns ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if not isinstance(grid, AdaptiveGrid):
        grid = AdaptiveGrid(grid)
    fm = grid.factor_map
    factors = list(ALLOWED_FACTORS)
    # Map factors to consecutive integer levels so the colormap is uniform even
    # though the factors are geometrically spaced.
    lvl = np.zeros(fm.shape)
    for k, f in enumerate(factors):
        lvl[fm == f] = k
    cmap = ListedColormap([_FACTOR_COLORS[f] for f in factors])
    norm = BoundaryNorm(np.arange(len(factors) + 1) - 0.5, cmap.N)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
    else:
        fig = ax.figure
    im = ax.imshow(lvl, cmap=cmap, norm=norm, origin="lower",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(factors)), shrink=0.85)
    cbar.set_label("sampling factor")
    cbar.ax.set_yticklabels([_FACTOR_LABELS[f] for f in factors])
    ax.set_title(f"{title}\n{grid.n_points} eval points "
                 f"({grid.n_points / fm.size:.2f} per native pixel)")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    return fig, ax
