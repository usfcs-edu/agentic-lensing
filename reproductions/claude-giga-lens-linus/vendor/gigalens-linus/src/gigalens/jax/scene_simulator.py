"""Unified simulator over a :class:`~gigalens.jax.scene.LensModel` (Phase 3).

One ``render`` core + a pluggable **trace** (the only thing that differs between
single-deflector and multi-plane, §2.6):

- ``DeflectionRatioTrace`` — one mass plane; ``β_p = θ − deflection_ratio_p · α(θ)``.
- ``MultiPlaneTrace``      — recursive ray trace; distances from the model's own
  cosmology (``cosmo.distance_matrix``), so plane redshifts / cosmology are
  differentiable. The recursion is adapted from ``multiplane_exp`` but consumes the
  canonical params dict and the gigalens distance matrix.

The PSF convolution, supersample pooling, precision pipeline, and the lstsq solver are
**reused verbatim** from ``gigalens.jax.simulator`` (load-bearing; validated). Params
come from ``LensModel.to_params(...)`` — the full structured dict (free + constants).
"""
from __future__ import annotations

import functools
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import jit
from lenstronomy.Util.kernel_util import subgrid_kernel
from objax.constants import ConvPadding
from objax.functional import average_pool_2d

from gigalens.simulator import LensWCS, SimulatorConfig
from gigalens.jax.simulator import (_convolve_components,
                                    _convolve_pool_components,
                                    _solve_normal_eq_with_fallback)
from gigalens.jax.scene import LensModel

# Fused PSF-convolution + average-pooling (spectral fold; see
# gigalens.jax.simulator._rfft_convolve_pool_same). Module-level so the
# equivalence gate (wip/validate_fold_stack.py) can toggle it around fresh
# traces; read at TRACE time.
_FUSE_CONV_POOL = True
# Deployment threshold (measured 2026-07-09, A100-40GB, production conv f32):
# at ss=4 the fold gains 8.7% grad time at ~equal peak memory, but at ss=2 it
# gains ~0-3% while REGRESSING peak memory ~20% (the fold's spectral-gather
# intermediate outweighs the small-stride c2r savings, and peak memory is the
# MAP batch-size binding constraint, compute-profiling C-9). Fuse only when the
# stride is large enough to pay: supersample >= _FUSE_CONV_POOL_MIN_SS.
_FUSE_CONV_POOL_MIN_SS = 3


def _as_float_or_none(x):
    """Best-effort scalar float for a construction-time geometry value (a number or a
    length-1 array). Returns None for anything non-numeric (e.g. a tfd.Distribution if a
    redshift were left free), so the caller skips the static consistency check."""
    try:
        return float(jnp.asarray(x).ravel()[0])
    except (TypeError, ValueError):
        return None


class SceneSimulator:
    def __init__(
        self,
        model: LensModel,
        sim_config: SimulatorConfig,
        sees=None,
    ):
        self.model = model
        self.sim_config = sim_config
        # Per-dataset light view (§2.5). ``sees`` is an optional collection of the
        # model's light Components matched BY OBJECT IDENTITY; when given, only those
        # components are rendered, while the trace still runs through the FULL shared
        # mass (the mass loops below use ``model.planes`` directly, independent of
        # ``self._light``). ``sees=None`` renders all light (the single-image default
        # used by Phase-3/4 callers; the no-silent-coupling rule lives in ProbModel,
        # §3.5, since that is where multiple datasets are visible).
        self._sees_ids = None if sees is None else {id(c) for c in sees}
        self._sees_ref = sees  # original Components, for replace_config reconstruction

        # --- precision (mirrors gigalens.jax.simulator.LensSimulator) ---
        # SimulatorConfig normalizes these to numpy dtypes (or None) in __post_init__,
        # so we consume dtypes directly and cast with a bare ``astype`` below.
        prec = getattr(sim_config, "likelihood_precision", None) or jnp.dtype("float64")
        self.likelihood_precision = prec
        self.high_precision = prec == jnp.dtype("float64")
        self.conv_precision = getattr(sim_config, "conv_precision", None)    # dtype | None
        self.basis_precision = getattr(sim_config, "basis_precision", None)  # dtype | None
        # Opt-in remat of the basis+conv+pool pipeline (see lstsq_simulate).
        self.remat_basis = bool(getattr(sim_config, "remat_basis", False))

        # --- geometry / grid (reuse LensWCS, like the existing simulator) ---
        self.supersample = int(sim_config.supersample)
        self.transform_pix2angle = (
            jnp.eye(2) * sim_config.delta_pix
            if sim_config.transform_pix2angle is None
            else jnp.asarray(sim_config.transform_pix2angle)
        )
        self.conversion_factor = jnp.linalg.det(self.transform_pix2angle)
        self.wcs = LensWCS(
            n=sim_config.num_pix, supersample=self.supersample,
            transform_pix2angle=sim_config.transform_pix2angle,
            pix_scale=sim_config.delta_pix, shift=sim_config.angular_shift,
        )
        img_X, img_Y = self.wcs.pixel_grid()
        # Trailing SINGLETON batch axis (H, W, 1): the sample/chain batch is carried by
        # the params (each leaf is (batch,)) and broadcasts onto this axis at trace time,
        # so one simulator handles any batch size with no per-batch rebuild (the old
        # pre-tiling to a fixed ``bs`` was incidental, not load-bearing -- see
        # wip/bench_bs_flexibility.py for the equivalence gate).
        self.img_X = img_X[..., jnp.newaxis]
        self.img_Y = img_Y[..., jnp.newaxis]

        # --- light bookkeeping: ordered (plane_idx, light_idx, component, depth) ---
        # When a ``sees`` view is given, restrict the rendered light to those
        # components (matched by identity). The (plane_idx, light_idx) addressing into
        # the params dict is preserved, so the trace and to_params layout are unchanged;
        # only which light contributes to the rendered image is filtered.
        self._light = []
        seen_ids_used = set()
        for i, plane in enumerate(model.planes):
            for j, comp in enumerate(plane.light):
                # Non-rendering light (renders=False, e.g. a point source) is a lensed
                # emitter that produces no pixels: skip it entirely so it never enters the
                # render sum, the lstsq basis, or the ``depth`` total. Skipped BEFORE the
                # sees filter, so a stray non-renderer never counts even if a caller built
                # this simulator with sees=None (imaging ``sees`` resolution already
                # excludes non-renderers, so a resolved ``_sees_ids`` won't contain one).
                if not getattr(comp.profile, "renders", True):
                    continue
                if self._sees_ids is not None and id(comp) not in self._sees_ids:
                    continue
                self._light.append((i, j, comp, comp.profile.depth))
                seen_ids_used.add(id(comp))
        if self._sees_ids is not None:
            missing = self._sees_ids - seen_ids_used
            if missing:
                raise ValueError(
                    "sees references light Component(s) that are not in this model's "
                    "planes (matched by object identity); pass the same Component "
                    "instances used to build the LensModel.")
        self.depth = sum(d for *_, d in self._light)

        # --- per-light "fully fixed?" mask (used by lstsq_simulate's batch guard) ---
        # A light component renders at the batch size of its OWN params: when every shape
        # parameter is fixed to a constant the component is sample-independent and renders
        # at batch-1, so it may be safely replicated across a sample batch; a component
        # with any free parameter must render at the full batch size. We precompute, from
        # the model's public ``constants`` dict, which light entries are fully fixed, so
        # lstsq_simulate can replicate ONLY those and refuse to silently broadcast a
        # mis-batched free component. (lstsq amplitudes are solved, not in
        # ``profile.params``, so they never count toward fixed-ness.)
        consts = getattr(model, "constants", {}) or {}
        self._light_fully_fixed = []
        for i, j, comp, _ in self._light:
            node = consts.get("planes", {}).get(i, {}).get("light", {}).get(j, {})
            self._light_fully_fixed.append(
                all(pname in node for pname in comp.profile.params))

        # --- single PSF kernel (subgrid + flip), depth-broadcast for conv ---
        self.flat_kernel = None
        if sim_config.kernel is not None:
            if self.supersample == 1:
                raw = jnp.asarray(sim_config.kernel)
                k2d = raw / raw.sum()
            else:
                k2d = subgrid_kernel(sim_config.kernel, self.supersample, odd=True, num_iter=100)
                k2d = k2d / k2d.sum()
            kernel = k2d[:, :, jnp.newaxis, jnp.newaxis]
            self.flat_kernel = jnp.transpose(jnp.asarray(kernel), (2, 3, 0, 1))

        # Per-plane "is this plane lensed?" flag: a plane's light is lensed iff some
        # EARLIER plane carries mass (same index-order convention as
        # LensModel.source_plane_light / _validate_geometry). Used by the
        # deflection_ratio trace to decide which light planes get deflected.
        self._lensed_plane = [
            any(model.planes[j].has_mass for j in range(i)) for i in range(len(model.planes))
        ]

        # --- trace mode from structure (printed, never silent: §0) ---
        n_mass_planes = sum(1 for p in model.planes if p.has_mass)
        if n_mass_planes >= 2:
            if model.cosmo is None:
                raise ValueError("multi-plane (>=2 mass planes) requires a cosmology.")
            self.trace_mode = "multiplane"
        else:
            self.trace_mode = "deflection_ratio"

        # Single-deflector + cosmology: the deflection_ratio scaling is defined relative
        # to the lens (cosmo.z_lens). If the mass plane sits at a different redshift, the
        # scaling is silently wrong -- catch the mismatch loudly (no silent inconsistency).
        if self.trace_mode == "deflection_ratio" and model.cosmo is not None:
            mass_planes = [i for i, p in enumerate(model.planes) if p.has_mass]
            if mass_planes:
                z_mass = _as_float_or_none(model.planes[mass_planes[0]].redshift)
                z_cosmo_lens = _as_float_or_none(model.cosmo.profile.z_lens)
                if (z_mass is not None and z_cosmo_lens is not None
                        and abs(z_mass - z_cosmo_lens) > 1e-6 * max(1.0, abs(z_cosmo_lens))):
                    raise ValueError(
                        f"single-deflector model: the mass plane redshift ({z_mass}) must "
                        f"match the cosmology's z_lens ({z_cosmo_lens}); the deflection-ratio "
                        "scaling is defined relative to the lens redshift (no silent mismatch).")

    # ------------------------------------------------------------------ precision
    def _psf_convolve(self, img, conv_fn):
        if self.flat_kernel is None:
            return img
        if self.conv_precision is not None:
            cdt = self.conv_precision
            return conv_fn(img.astype(cdt), self.flat_kernel.astype(cdt)).astype(img.dtype)
        return conv_fn(img, self.flat_kernel.astype(img.dtype))

    def _psf_convolve_pool(self, img, depths=None):
        """Fused PSF convolution + supersample average-pooling (spectral fold),
        honoring ``conv_precision`` exactly like ``_psf_convolve``. NOTE: with
        conv_precision=float32 the pooling now also runs in float32 (it was
        float64 after the cast-back in the unfused path) — covered by the
        production-dtype gate in wip/validate_fold_stack.py."""
        if self.conv_precision is not None:
            cdt = self.conv_precision
            return _convolve_pool_components(
                img.astype(cdt), self.flat_kernel.astype(cdt), self.supersample,
                depths).astype(img.dtype)
        return _convolve_pool_components(
            img, self.flat_kernel.astype(img.dtype), self.supersample, depths)

    # ------------------------------------------------------------------ light basis
    def _render_light(self, comp, bx, by, lp):
        """Evaluate one light component's basis, optionally at a fixed basis_precision.

        When ``basis_precision`` is float32 the deflected positions and the light params
        are cast to float32 so the per-component basis (and its VJP) run in float32; the
        result is left float32 and promoted back to float64 downstream by the float64
        noise/observed arrays at the gram/reduction (so the solve and chi^2 stay float64).
        ``None`` leaves the dtype untouched (byte-identical)."""
        if self.basis_precision is not None:
            dt = self.basis_precision
            bx = bx.astype(dt)
            by = by.astype(dt)
            lp = {k: (v.astype(dt) if hasattr(v, "astype") else v)
                  for k, v in lp.items()}
        return comp.profile.light(bx, by, **lp)

    # ------------------------------------------------------------------ deflection
    def _alpha(self, params, x, y):
        fx, fy = jnp.zeros_like(x), jnp.zeros_like(y)
        for i, plane in enumerate(self.model.planes):
            for j, comp in enumerate(plane.mass):
                dfx, dfy = comp.profile.deriv(x, y, **params["planes"][i]["mass"][j])
                fx = fx + dfx
                fy = fy + dfy
        return fx, fy

    # ------------------------------------------------------------------ traces
    def _trace_deflection_ratio(self, params, no_deflection: bool, X, Y):
        # Single deflector. A lensed light plane's source position is β = θ − r·α, where
        # the scaling r is EITHER given explicitly (``deflection_ratio`` geometry, no
        # cosmology) OR, when a cosmology is present, derived from it as
        # ``deflection_ratio(z_source)`` -- the (D_ls/D_s) ratio normalized to the
        # cosmology's z_source_ref, evaluated at the cosmo params (so it stays
        # differentiable in them, like the multiplane distance matrix). Non-lensed light
        # (foreground / lens-plane) gets no deflection.
        if no_deflection:
            return {i: (X, Y) for i, *_ in self._light}
        fx, fy = self._alpha(params, X, Y)
        cosmo = self.model.cosmo
        positions = {}
        for i in range(len(self.model.planes)):
            geom = params["planes"][i].get("geometry", {})
            if "deflection_ratio" in geom:
                dr = geom["deflection_ratio"]
            elif cosmo is not None and self._lensed_plane[i] and "redshift" in geom:
                dr = jnp.squeeze(
                    cosmo.profile.deflection_ratio(geom["redshift"], **params["cosmo"]))
            else:
                positions[i] = (X, Y)  # foreground / lens-plane light
                continue
            positions[i] = (X - dr * fx, Y - dr * fy)
        return positions

    def _trace_multiplane(self, params, no_deflection: bool, X, Y):
        planes = self.model.planes
        N = len(planes)
        if no_deflection:
            return {i: (X, Y) for i in range(N)}
        zs = [params["planes"][i]["geometry"]["redshift"] for i in range(N)]
        D = self.model.cosmo.profile.distance_matrix(zs, **params["cosmo"])
        ax, ay = X, Y                            # physical deflection angle (plane 0)
        x = jnp.zeros_like(X)
        y = jnp.zeros_like(Y)
        traj_x, traj_y = [x], [y]
        for k in range(N):
            dT = D[k + 1, k]
            x = x + ax * dT
            y = y + ay * dT
            traj_x.append(x)
            traj_y.append(y)
            if k == N - 1:
                break
            thx, thy = x / D[k + 1, 0], y / D[k + 1, 0]
            rx, ry = 0.0, 0.0
            for j, comp in enumerate(planes[k].mass):
                dx, dy = comp.profile.deriv(thx, thy, **params["planes"][k]["mass"][j])
                rx = rx + dx
                ry = ry + dy
            scale = D[-1, 0] / D[-1, k + 1]
            ax = ax - rx * scale
            ay = ay - ry * scale
        return {i: (traj_x[i + 1] / D[i + 1, 0], traj_y[i + 1] / D[i + 1, 0])
                for i in range(N)}

    def trace(self, params, no_deflection: bool = False,
              positions=None) -> Dict[int, Tuple[Any, Any]]:
        """Ray-trace image-plane positions to every plane's source position.

        ``positions`` (optional) is an ``(x, y)`` pair of image-plane angular
        coordinates to trace instead of this simulator's stored pixel grid — the
        entry point for non-grid consumers (lens-equation solving for point
        sources, catalog image positions, adaptive supersampling). The arrays are
        used exactly like the stored grid: each must carry a trailing axis that
        broadcasts against the params' trailing batch axis (the stored grid is
        ``(H, W, 1)``; e.g. pass shape ``(N, 1)`` for N points). ``None`` (default,
        the rendering path) traces the stored supersampled grid — identical to the
        pre-``positions`` behavior.
        """
        if positions is None:
            X, Y = self.img_X, self.img_Y
        else:
            X, Y = positions
        if self.trace_mode == "multiplane":
            return self._trace_multiplane(params, no_deflection, X, Y)
        return self._trace_deflection_ratio(params, no_deflection, X, Y)

    # ------------------------------------------------------------------ render
    @functools.partial(jit, static_argnums=(0, 2))
    def simulate(self, params, no_deflection: bool = False):
        """Forward image: sum each (amplitude-carrying) light component. Non-lstsq."""
        positions = self.trace(params, no_deflection)
        # Accumulate the rendered light; the trailing batch axis comes from the params
        # (broadcast onto the singleton grid), so no fixed batch size is needed here.
        img = None
        for i, j, comp, _ in self._light:
            bx, by = positions[i]
            contrib = self._render_light(comp, bx, by, params["planes"][i]["light"][j])
            img = contrib if img is None else img + contrib
        img = jnp.transpose(img, (2, 0, 1))[:, jnp.newaxis, :, :]
        if (_FUSE_CONV_POOL and self.flat_kernel is not None
                and self.supersample >= _FUSE_CONV_POOL_MIN_SS):
            ret = self._psf_convolve_pool(img)
        else:
            ret = self._psf_convolve(img, lambda im, k: _convolve_components(im, k))
            ret = (average_pool_2d(ret, size=self.supersample, padding=ConvPadding.SAME)
                   if self.supersample != 1 else ret)
        return jnp.squeeze(ret) * self.conversion_factor

    def _match_light_batch(self, contribs):
        """Align light components' trailing batch axis before lstsq concatenation.

        Every contrib has shape ``(n_basis, h, w, batch)``; the batch axis is set by each
        component's params, so a fully-fixed component renders at batch-1 while a free one
        renders at the sample batch ``bs``. Concatenating along the basis axis needs a
        shared batch axis, so a fully-fixed (sample-independent) component is REPLICATED
        across the batch -- an exact operation, since its basis does not depend on the
        sample. Every other disagreement is refused rather than silently broadcast:

          * a component with FREE parameters that nonetheless rendered at batch-1 under a
            batch-``bs`` evaluation (would silently produce a sample-independent profile);
          * any batch size that is neither 1 nor ``bs`` (a genuine mis-batch).

        This runs at trace time (shapes are static under jit), so a violation raises at
        compile time rather than corrupting the fit.
        """
        sizes = [c.shape[-1] for c in contribs]
        bs = max(sizes)
        if bs == 1:
            return contribs  # single-sample / all-fixed eval: nothing to align (no-op)
        out = []
        for (i, j, comp, _), c, n, fixed in zip(
                self._light, contribs, sizes, self._light_fully_fixed):
            if n == bs:
                out.append(c)
            elif n == 1 and fixed:
                out.append(jnp.broadcast_to(c, c.shape[:-1] + (bs,)))
            elif n == 1:
                raise ValueError(
                    f"scene light component (plane {i}, light {j}, {comp.profile}) has "
                    f"free parameters but rendered at batch 1 during a batch-{bs} "
                    "evaluation; refusing to broadcast it, which would silently yield a "
                    "sample-independent light profile. This signals a mis-batched free "
                    "parameter upstream (check to_params / the sampled inputs).")
            else:
                raise ValueError(
                    f"scene light component (plane {i}, light {j}, {comp.profile}) "
                    f"rendered at batch {n}, incompatible with the evaluation batch {bs} "
                    "(expected 1 or bs); refusing to broadcast a mis-batched light "
                    "profile.")
        return out

    def _render_basis_nchw(self, p, no_deflection):
        """Render the stacked per-component basis on the detector grid: (bs, ncomp, h, w).

        Stack each component's basis along the component axis. Each component
        inherits its trailing batch axis from its own params, so a fixed
        (constant-param) component renders at batch-1 while a free/sampled one
        renders at the full sample batch. Concatenation needs a shared batch
        axis: ``_match_light_batch`` replicates the genuinely fully-fixed
        (sample-independent) components and refuses any other mismatch, so a
        mis-batched profile fails loudly instead of being silently broadcast
        (unlike the forward ``simulate`` path, which sums and so broadcasts).

        Overridable seam: this is the whole basis+conv+pool pipeline of
        ``lstsq_simulate`` (the solve tail is quadrature-agnostic), so a variant
        quadrature (e.g. the experimental adaptive supersampler) overrides ONLY
        this method and inherits the solve unchanged.
        """
        positions = self.trace(p, no_deflection)
        contribs, depths = [], []
        for i, j, comp, d in self._light:
            bx, by = positions[i]
            contribs.append(self._render_light(comp, bx, by, p["planes"][i]["light"][j]))
            depths.append(d)
        img = jnp.concatenate(self._match_light_batch(contribs), axis=0)
        img = jnp.transpose(img, (3, 0, 1, 2))  # bs, ncomp, h, w
        if (_FUSE_CONV_POOL and self.flat_kernel is not None
                and self.supersample >= _FUSE_CONV_POOL_MIN_SS):
            return self._psf_convolve_pool(img, depths)
        ret = self._psf_convolve(img, lambda im, k: _convolve_components(im, k, depths))
        return (average_pool_2d(ret, size=(self.supersample, self.supersample), padding="SAME")
                if self.supersample != 1 else ret)

    def replace_config(self, sim_config) -> "SceneSimulator":
        """Rebuild this simulator with a different ``sim_config``, preserving its
        concrete type and construction arguments (subclasses with extra state
        override). Used by ``ProbModel.with_map_remat`` so a config tweak never
        silently downgrades a subclass simulator to the base quadrature."""
        return type(self)(self.model, sim_config, sees=self._sees_ref)

    @functools.partial(jit, static_argnums=(0, 5, 6, 7))
    def lstsq_simulate(self, params, observed_image, err_map, mask=None,
                       return_stacked=False, return_coeffs=False, no_deflection=False):
        """lstsq image: render each component's basis, solve amplitudes, sum."""
        if mask is None:
            mask = jnp.ones_like(observed_image)
        else:
            mask = mask.astype(jnp.float32)

        stacked_nchw = functools.partial(
            self._render_basis_nchw, no_deflection=no_deflection)

        # remat_basis (SimulatorConfig): recompute the basis+conv+pool pipeline in
        # the backward pass instead of storing its supersampled/complex
        # intermediates — only the pooled (bs, ncomp, h_det, w_det) output and the
        # params are saved. Mathematically exact (identical ops re-run). Costs
        # recompute on single evals; pays on MEMORY-bound batched MAP/SVI (the
        # OOM frontier is the binding constraint there, compute-profiling C-9).
        ret = (jax.checkpoint(stacked_nchw)(params) if self.remat_basis
               else stacked_nchw(params))            # bs, ncomp, h, w
        if return_stacked:
            return jnp.transpose(ret, (0, 2, 3, 1))  # bs, h, w, ncomp (public layout)

        # NOTE (2026-07-09): a channels-first normal-equation layout (contracting
        # the gram directly from (bs, ncomp, h, w), no full-tensor transpose) was
        # implemented and FALSIFIED — XLA already compiles both layouts to
        # bitwise-identical executables (equivalence diff exactly 0.0, speed
        # delta 0.0% at (200,ss2) n_max 15/30). The C-14 "copies/transposes"
        # trace class comes from the FFT ops' internal separable lowering, not
        # from this transpose. Removed per its pre-registered <2% falsifier;
        # see compute-profiling.md and git history for the variant.
        retc = jnp.transpose(ret, (0, 2, 3, 1))  # bs, h, w, ncomp
        W = (1 / err_map)[..., jnp.newaxis]
        Y = jnp.reshape(observed_image * mask * jnp.squeeze(W), (1, -1, 1))
        X = jnp.reshape((retc * mask[jnp.newaxis, ..., jnp.newaxis] * W), (retc.shape[0], -1, self.depth))
        Xt = jnp.transpose(X, (0, 2, 1))
        coeffs = _solve_normal_eq_with_fallback(Xt @ X, Xt @ Y)[..., 0]
        if return_coeffs:
            return coeffs
        return jnp.squeeze(jnp.sum(retc * coeffs[:, jnp.newaxis, jnp.newaxis, :], axis=-1))
