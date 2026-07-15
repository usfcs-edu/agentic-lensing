"""Unified probabilistic model over a scene + observations (Phase 4 + Phase 5 + Phase A).

``ProbModel(model, datasets)`` is the single code path that replaces today's
``ForwardProbModel`` / ``BackwardProbModel`` / ``*MultiModel`` / ``JointModel`` family.
It consumes a :class:`~gigalens.jax.scene.LensModel` (the parameter source of truth)
and one or more observations, and sums their log-likelihood terms. Phase 5 adds the
multi-dataset ``sees`` loop: each dataset carries its own ``sim_config`` (grid + PSF)
and its own ``sees`` view into the model's light, while the shared lens mass +
cosmology are traced for every dataset.

**Phase A (dataset seam).** ``ProbModel`` no longer knows what an observation *is*:
each :class:`Dataset` subclass owns its data validation, its ``sees`` resolution,
and the construction of its compiled :class:`LikelihoodTerm` (params → log-likelihood).
``ProbModel`` resolves sees, runs the cross-observation validators, builds one term per
observation, and sums the terms. The imaging :class:`ImageData` (masked Gaussian pixel
likelihood via a per-dataset :class:`~gigalens.jax.scene_simulator.SceneSimulator`) is
the first — currently only — instance; future observation types (point-source
positions/fluxes, time delays, kinematics, visibilities) subclass ``Dataset``
without touching ``ProbModel``.

Amplitude modes mirror the existing classes:
  - ``"lstsq"``   : light amplitudes solved per dataset (was ``BackwardProbModel``).
  - ``"forward"`` : light amplitudes sampled (was ``ForwardProbModel``).
``mode`` may be set per ``ImageData``; the ``ProbModel(mode=...)`` argument is the
default for datasets that don't specify one (back-compat with the Phase-5 global mode).

The cross-dataset validators (§3.5 sees-required, §3.6 dead-entities, §3.7
forward-mode flux-sharing) live HERE rather than on ``LensModel``, because they are
properties of the (model, datasets) pair — a ``Component`` is "dead" or "shared across
bands" only relative to the full set of datasets, which the model does not see.
(§3.5 resolution itself is delegated to each ``Dataset``, since what an observation
may legally *see* — light components for imaging — is a property of its type.)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

import numpy as np
from jax import numpy as jnp

from gigalens.jax.scene import LensModel
from gigalens.jax.scene_simulator import SceneSimulator


class Dataset(ABC):
    """One observation of the scene: a validated DATA holder plus the recipe for
    building its likelihood engine against a :class:`LensModel`.

    Contract for subclasses (the imaging :class:`ImageData` is the reference instance):

    * **Validate at construction; raise, never default** (the M3/§3.4 pattern). Every
      numeric input that enters the likelihood must be checked finite/physical in
      ``__init__`` — a NaN sigma or an unphysical exposure time must fail at build
      time, not poison the fit silently downstream.
    * ``sees`` — the model entities this observation constrains (§3.5), matched by
      object identity and resolved by :meth:`resolve_sees`. Imaging sees light
      Components; a future observation type declares (and enforces) what it may
      legally see.
    * ``event_size`` — the number of independent data points entering the likelihood.
    * The observation must stay model-free (no state tied to a particular
      ``LensModel``), so one observation can be reused against several models
      (e.g. a ``fix_to`` restriction). All per-(model, observation) compiled state
      lives on the :class:`LikelihoodTerm` returned by :meth:`build_term`.
    """

    sees: Optional[Sequence] = None
    event_size: int = 0

    @abstractmethod
    def resolve_sees(self, model: LensModel) -> List:
        """Resolve ``sees`` to a list of model Components (identity-matched), raising
        on anything this observation type may not legally see (§3.5)."""

    @abstractmethod
    def build_term(self, model: LensModel, seen: List, *,
                   default_mode: str = "lstsq") -> "LikelihoodTerm":
        """Build the compiled per-(model, observation) likelihood term.

        ``default_mode`` is the ProbModel-level amplitude-mode default; observation
        types with no amplitude-mode concept ignore it."""


class LikelihoodTerm(ABC):
    """Compiled engine for one observation against one model: params → log-likelihood.

    ``log_like(params)`` returns ``(log_like, chi2)`` with batch dims following the
    params' trailing-batch convention. ``chi2`` is the Gaussian Σ(r/σ)² for terms that
    have one (``reports_chi2 = True``) and ``None`` otherwise; ProbModel aggregates the
    reported chi2s into the reduced-χ² diagnostic channel. Both values must come from a
    SINGLE forward evaluation — splitting them into two methods would simulate the
    model twice per likelihood call.
    """

    reports_chi2: bool = False
    event_size: int = 0

    @abstractmethod
    def log_like(self, params):
        """Return ``(log_like, chi2_or_None)`` for one structured params dict."""


class ImageData(Dataset):
    """One imaging observation: image + camera (``sim_config``, incl. PSF) + noise + mask.

    Noise must be given as ``error_map`` OR (``background_rms`` and ``exp_time``);
    neither raises (§3.4 — no silent default). ``error_map`` and ``mask`` must match
    the image shape exactly, and the mask must keep at least one pixel (§3.8–§3.9 —
    broadcastable-but-wrong shapes and all-masked datasets fail silently otherwise).
    ``sees`` declares the per-dataset view
    into the model's light (a list of light Components matched by identity, or the
    explicit string ``"all"``). It is required and non-empty; ProbModel enforces this
    (§3.5) since the sees rules are cross-dataset. ``sees`` is stored verbatim here and
    resolved against the model by ProbModel.

    ``mode`` optionally fixes this dataset's amplitude mode (``"lstsq"``/``"forward"``);
    ``None`` (default) inherits the ProbModel-level mode.
    """

    def __init__(
        self,
        image,
        sim_config,
        *,
        error_map=None,
        background_rms=None,
        exp_time=None,
        mask=None,
        sees: Optional[Sequence] = None,
        mode: Optional[str] = None,
    ):
        self.image = jnp.asarray(image)
        self.sim_config = sim_config
        if error_map is not None:
            self.error_map = jnp.asarray(error_map)
        elif background_rms is not None and exp_time is not None:
            # Validate the physical inputs of the derived-noise model directly (M3): a
            # nonsensical exp_time/background_rms must raise here, NOT be left to the
            # downstream sigma check — the background term can mask a negative exp_time
            # and yield a finite-but-wrong sigma that slips through silently.
            bg = jnp.asarray(background_rms)
            et = jnp.asarray(exp_time)
            if not (bool(jnp.all(jnp.isfinite(bg))) and bool(jnp.all(bg >= 0))):
                raise ValueError(
                    "ImageData background_rms must be finite and non-negative (M3) — it is "
                    "a noise RMS.")
            if not (bool(jnp.all(jnp.isfinite(et))) and bool(jnp.all(et > 0))):
                raise ValueError(
                    "ImageData exp_time must be finite and strictly positive (M3) — a "
                    "non-positive exposure time is unphysical and gives a NaN/zero sigma.")
            for _nm, _arr in (("background_rms", bg), ("exp_time", et)):
                if _arr.ndim != 0 and _arr.shape != self.image.shape:
                    raise ValueError(
                        f"ImageData {_nm} must be a scalar or match the image shape "
                        f"{tuple(self.image.shape)}; got {tuple(_arr.shape)} (§3.8). A "
                        "wrong-but-broadcastable shape would silently derive a wrong "
                        "sigma for every pixel.")
            self.error_map = jnp.sqrt(
                bg ** 2 + jnp.clip(self.image, 0.0, np.inf) / et)
        else:
            raise ValueError(
                "ImageData needs either error_map or (background_rms and exp_time) "
                "— no silent noise default (§3.4).")
        # §3.8: the noise map must agree with the image shape EXACTLY. Verified
        # (2026-07-10, pre-guard kernel): an (N, N, 1) map against an (N, N) image was
        # accepted silently and the likelihood residual broadcast to (N, N, N) — a
        # 512-element pseudo-likelihood for an 8x8 image; an (N,) map crashed with a
        # cryptic IndexError deep in the finite-noise check. Both become this one
        # actionable error.
        if self.error_map.shape != self.image.shape:
            raise ValueError(
                f"ImageData error_map shape {tuple(self.error_map.shape)} != image "
                f"shape {tuple(self.image.shape)} (§3.8) — a broadcastable-but-wrong "
                "noise map silently misweights the likelihood; pass a per-pixel sigma "
                "of exactly the image shape.")
        self.mask = (jnp.ones_like(self.image, dtype=bool) if mask is None
                     else jnp.asarray(mask, dtype=bool))
        # §3.8 (mask leg): a wrong-shaped boolean mask does not necessarily error —
        # an (N,) mask boolean-indexes the FIRST AXIS of an (N, N) image (selecting
        # whole rows), so the finite-noise guard and the likelihood would silently
        # run over the wrong pixel set.
        if self.mask.shape != self.image.shape:
            raise ValueError(
                f"ImageData mask shape {tuple(self.mask.shape)} != image shape "
                f"{tuple(self.image.shape)} (§3.8) — a wrong-shaped mask silently "
                "selects the wrong pixels; pass a boolean mask of exactly the image "
                "shape.")
        self.event_size = int(jnp.count_nonzero(self.mask))
        # §3.9: a dataset whose mask excludes EVERY pixel contributes nothing to the
        # likelihood. Among other live datasets that is perfectly silent (its chi2/ll
        # terms are identically zero), so the model quietly ignores an observation the
        # user believes is constraining it — raise instead.
        if self.event_size == 0:
            raise ValueError(
                "ImageData mask excludes every pixel (event_size == 0) (§3.9) — this "
                "dataset would contribute nothing to the likelihood, silently. Unmask "
                "the pixels it should constrain, or remove the dataset.")
        self._validate_finite_noise()  # M3: no silent NaN/zero-sigma propagation
        self.sees = sees
        if mode not in (None, "lstsq", "forward"):
            raise ValueError(
                f"ImageData mode must be None (inherit), 'lstsq' or 'forward'; got {mode!r}")
        self.mode = mode

    def _validate_finite_noise(self):
        """Every pixel entering the likelihood must have a finite observed value and a
        finite, strictly-positive noise sigma — no silent NaN/zero-sigma propagation.

        Real incident (misuse_register M3): a NaN-containing ``error_map`` was accepted
        silently and ``lstsq_simulate`` returned an all-NaN image with no warning. A
        single NaN/inf sigma poisons the entire Gaussian likelihood; a zero/negative
        sigma makes it infinite/ill-defined. Checked over the UNMASKED region only:
        masked-out pixels are excluded from the likelihood by design, so NaN/inf padding
        there is legitimate.

        Observed-image values are NOT required to be positive: background subtraction and
        other reduction steps legitimately yield negative pixels (only the noise sigma is
        constrained positive). We therefore check the image for finiteness, not sign.
        """
        m = self.mask
        img_m = self.image[m]
        err_m = self.error_map[m]
        if not bool(jnp.all(jnp.isfinite(img_m))):
            raise ValueError(
                "ImageData observed image has non-finite (NaN/inf) values in the unmasked "
                "region — no silent NaN propagation (M3). Fix the image or mask those "
                "pixels out.")
        if not bool(jnp.all(jnp.isfinite(err_m))):
            raise ValueError(
                "ImageData error_map has non-finite (NaN/inf) values in the unmasked "
                "region — no silent NaN propagation (M3): a NaN/inf sigma poisons the "
                "whole likelihood. Fix the noise map or mask those pixels out.")
        if not bool(jnp.all(err_m > 0)):
            raise ValueError(
                "ImageData error_map has non-positive (<=0) values in the unmasked region "
                "(M3): the per-pixel noise sigma must be strictly positive — a zero or "
                "negative sigma gives an infinite/ill-defined Gaussian likelihood.")

    # -- sees resolution (§3.5; imaging sees light Components) ---------------------
    def resolve_sees(self, model: LensModel) -> List:
        """Resolve this dataset's ``sees`` to a list of model light Components (§3.5).

        ``sees`` must be supplied and non-empty (no silent "sees everything"). Imaging
        observes only RENDERABLE light: the explicit string ``"all"`` opts in to every
        renderable light Component (``renders=True``), and a non-rendering emitter (e.g. a
        point source, ``renders=False``) is excluded — it is a lensed emitter but has no
        pixel representation, so imaging cannot render it. References in an explicit list
        are matched by object identity against ``model.light_components``; an unknown
        Component, or a non-rendering one, raises (a wiring bug, not a degraded model)."""
        sees = self.sees
        if sees is None:
            raise ValueError(
                "ImageData.sees is required and must be non-empty (§3.5) — no silent "
                "'sees everything' default, which would couple every band's light. "
                "Pass an explicit list of light Components, or sees='all'.")
        all_light = model.light_components
        renderable = [c for c in all_light if getattr(c.profile, "renders", True)]
        if isinstance(sees, str):
            if sees != "all":
                raise ValueError(
                    f"ImageData.sees as a string must be 'all'; got {sees!r}.")
            if not renderable:
                raise ValueError(
                    "ImageData.sees='all' but the model has no renderable light Components "
                    "(§3.5) — non-rendering emitters (e.g. point sources) are not imaged.")
            return list(renderable)
        seen = list(sees)
        if len(seen) == 0:
            raise ValueError(
                "ImageData.sees is empty (§3.5) — a dataset that renders no light is a "
                "wiring bug; give it the light Components it observes or sees='all'.")
        model_ids = {id(c) for c in all_light}
        renderable_ids = {id(c) for c in renderable}
        for c in seen:
            if id(c) not in model_ids:
                raise ValueError(
                    "ImageData.sees references a Component that is not a light Component "
                    "of this model (matched by object identity); pass the same "
                    "instances used to build the LensModel (§3.5).")
            if id(c) not in renderable_ids:
                raise ValueError(
                    "ImageData.sees references a non-rendering light Component "
                    f"({c.profile}) — imaging cannot render an emitter with renders=False "
                    "(e.g. a point source). It is observed by a non-imaging dataset "
                    "(PointSourceData) instead; remove it from this ImageData's sees.")
        return seen

    def build_term(self, model: LensModel, seen: List, *,
                   default_mode: str = "lstsq") -> "ImageLikelihoodTerm":
        mode = default_mode if self.mode is None else self.mode
        return ImageLikelihoodTerm(self, model, seen, mode)

    def _build_simulator(self, model: LensModel, seen: List) -> SceneSimulator:
        """The simulator this dataset's likelihood term renders with — the ONE
        seam a dataset subclass overrides to swap the rendering quadrature (e.g.
        the experimental adaptive supersampler) without re-plumbing the term.
        Called exactly once, by ``ImageLikelihoodTerm.__init__``."""
        return SceneSimulator(model, self.sim_config, sees=seen)


class ImageLikelihoodTerm(LikelihoodTerm):
    """Masked Gaussian pixel likelihood of one imaging :class:`ImageData`.

    Owns the dataset's :class:`SceneSimulator` (its grid + PSF, restricted to its seen
    light, tracing the full shared mass) and computes Σ(r/σ)² plus the Gaussian
    log-normalization over the unmasked pixels.
    """

    reports_chi2 = True

    def __init__(self, dataset: ImageData, model: LensModel, seen: List, mode: str):
        if mode not in ("lstsq", "forward"):
            raise ValueError(f"mode must be 'lstsq' or 'forward'; got {mode!r}")
        self.dataset = dataset
        self.mode = mode
        self.event_size = dataset.event_size
        self.simulator = dataset._build_simulator(model, seen)

    def _model_image(self, params, simulator):
        ds = self.dataset
        if self.mode == "lstsq":
            return simulator.lstsq_simulate(params, ds.image, ds.error_map, ds.mask)
        return simulator.simulate(params)

    def log_like(self, params, *, simulator=None):
        """``(log_like, chi2)`` for this dataset's image; masked Gaussian.

        ``simulator`` (keyword-only) optionally overrides the term's own simulator —
        the back-compat escape hatch used by inference algorithms that build a
        simulator at a specific batch size (see ``ProbModel.log_like``)."""
        sim = self.simulator if simulator is None else simulator
        ds = self.dataset
        im = self._model_image(params, sim)
        mask = ds.mask.astype(bool)
        diff = (im - ds.image) / ds.error_map
        # Apply the mask with jnp.where, not a multiply. A float mask that
        # broadcast-multiplies a higher-rank array -- diff is (bs, H, W), mask is
        # (H, W) -- triggers a jaxlib 0.9.1 GPU-jit miscompile of the implicit
        # rank-expanding `arr * mask` (silently inflates the reduced chi^2). where()
        # selects instead of multiplying and is not affected.
        chi2 = jnp.sum(jnp.where(mask, diff ** 2, 0.0), axis=(-2, -1))
        norm = jnp.sum(
            jnp.where(mask, jnp.log(2 * jnp.pi * ds.error_map ** 2), 0.0),
            axis=(-2, -1))
        return -0.5 * (chi2 + norm), chi2


class ProbModel:
    def __init__(
        self,
        model: LensModel,
        datasets: Union[Dataset, Sequence[Dataset]],
        mode: str = "lstsq",
    ):
        if mode not in ("lstsq", "forward"):
            raise ValueError(f"mode must be 'lstsq' or 'forward'; got {mode!r}")
        self.model = model
        self.datasets: List[Dataset] = (
            [datasets] if isinstance(datasets, Dataset) else list(datasets))
        if len(self.datasets) < 1:
            raise ValueError("ProbModel needs at least one dataset.")
        self.mode = mode
        self.prior = model.prior
        self.bij = model.bijector
        # Canonical column→name map for the flat sampler ``z`` (delegates to the
        # LensModel; see ``gigalens.jax.scene._z_param_names``). Use THIS for any
        # z-column labeling — never reconstruct names from param/insertion order.
        self.z_param_names = model.z_param_names
        self.event_size = sum(d.event_size for d in self.datasets)

        # Resolve each observation's ``sees`` (type-specific, §3.5) and run the
        # cross-observation validators (§3.6–3.7). This is the home for these checks:
        # only ProbModel sees both the model and all observations.
        self._resolved_sees: List[List] = [
            d.resolve_sees(model) for d in self.datasets]
        self._validate_dead_entities()      # §3.6
        self._validate_forward_flux_sharing()  # §3.7 (imaging-only)
        self._validate_no_pointsource_imaging_mix()  # point-source terms not yet joinable

        # One LikelihoodTerm per observation: the compiled per-(model, observation)
        # engine. For imaging this wraps a SceneSimulator on that dataset's grid + PSF,
        # restricted to its seen light, tracing through the full shared mass.
        self.terms: List[LikelihoodTerm] = [
            d.build_term(model, seen, default_mode=mode)
            for d, seen in zip(self.datasets, self._resolved_sees)]

        # The (log_like, red_chi2) diagnostic channel consumed by the whole inference
        # stack is defined over the Gaussian (chi²-reporting) terms. A model with no
        # such term has no reduced χ² — refuse loudly rather than emit a vacuous 0/0;
        # revisit the diagnostics channel when the first purely non-Gaussian
        # observation set becomes a real use case.
        self._chi2_event_size = sum(
            t.event_size for t in self.terms if t.reports_chi2)
        if self._chi2_event_size == 0:
            raise NotImplementedError(
                "ProbModel requires at least one chi²-reporting (Gaussian) term: the "
                "(log_like, red_chi2) diagnostic channel has no meaning without one.")

    @property
    def simulators(self) -> List[SceneSimulator]:
        """Per-dataset ``SceneSimulator``s of the imaging terms, in dataset order
        (back-compat: Phase-5 code and the inference stack index this list)."""
        return [t.simulator for t in self.terms
                if isinstance(t, ImageLikelihoodTerm)]

    def with_map_remat(self) -> "ProbModel":
        """A twin ProbModel for batched-MAP use (the only wired consumer is
        ``ModellingSequence.MAP``; SVI is NOT wired): ``remat_basis`` forced ON
        for imaging simulators in the MEASURED-benefit regime (supersample below
        the fused-conv threshold).

        Rationale (compute-profiling C-16/C-17): batched MAP is memory-bound —
        remat cuts peak ~46% and doubles the feasible per-device batch at
        ss=2 — but at compute saturation the recompute is a real cost, and at
        ss >= _FUSE_CONV_POOL_MIN_SS recomputing the fused spectral-fold conv
        measured +23%; there remat stays whatever the user configured. Shares
        model/datasets with self (shallow copies); returns ``self`` unchanged if
        no simulator would change. Non-imaging terms are untouched (remat is a
        property of the image-rendering pipeline). Exactness: remat re-runs
        identical ops (gates in gigalens wip/validate_map_throughput.py and
        wip/validate_remat_default.py)."""
        import copy as _copy
        import dataclasses as _dc
        from gigalens.jax.scene_simulator import _FUSE_CONV_POOL_MIN_SS
        new_terms, changed = [], False
        for d, term in zip(self.datasets, self.terms):
            sim = getattr(term, "simulator", None)
            if (sim is not None and sim.supersample < _FUSE_CONV_POOL_MIN_SS
                    and not sim.remat_basis):
                new_term = _copy.copy(term)
                # replace_config (not a bare SceneSimulator(...)) so a subclass
                # simulator (e.g. the experimental adaptive supersampler) keeps its
                # quadrature — a config tweak must never silently change the physics.
                new_term.simulator = sim.replace_config(
                    _dc.replace(d.sim_config, remat_basis=True))
                new_terms.append(new_term)
                changed = True
            else:
                new_terms.append(term)
        if not changed:
            return self
        twin = _copy.copy(self)
        twin.terms = new_terms
        return twin

    # -- back-compat surface for the gigalens_research pipeline (G1 A) ------------
    @property
    def observed_image(self):
        """The single dataset's observed image (read-only).

        The gigalens_research pipeline (``ModellingSequence.MAP`` loss normalization,
        ``model_card``, ``InferenceContext.hash``, ``posterior``) reads this off the old
        prob models. The scene ProbModel is built per dataset, so for a single-dataset
        model this is unambiguous; with multiple datasets it raises (the multi-band
        pipeline is out of the G1 slice scope — use the per-dataset images explicitly)."""
        if len(self.datasets) != 1:
            raise AttributeError(
                "observed_image is only defined for a single-dataset ProbModel; this "
                f"model has {len(self.datasets)} datasets — read per-dataset images.")
        return self.datasets[0].image

    @property
    def error_map(self):
        """The single dataset's per-pixel noise σ (read-only; see ``observed_image``)."""
        if len(self.datasets) != 1:
            raise AttributeError(
                "error_map is only defined for a single-dataset ProbModel; this model "
                f"has {len(self.datasets)} datasets — read per-dataset error maps.")
        return self.datasets[0].error_map

    @property
    def num_pixels(self) -> int:
        """Total image-pixel count summed over all imaging datasets (mask-independent).

        Used by ``ModellingSequence.MAP`` to normalize the loss. Multi-dataset-safe:
        for a single dataset this equals ``observed_image.size``, so single-band
        behavior is unchanged; with multiple datasets it sums each band's pixels rather
        than raising (unlike ``observed_image``). Note this counts ALL pixels, not the
        masked event count (``event_size``) — it matches the historical normalizer.
        Non-imaging observations are excluded (they have no pixels); this is a LITERAL
        imaging-pixel count and is 0 for a model with no imaging dataset (e.g. a
        point-source-only ProbModel). Inference must not divide by it directly — use
        :attr:`loss_normalization`, which is defined for any dataset mix."""
        return int(sum(int(d.image.size) for d in self.datasets
                       if isinstance(d, ImageData)))

    @property
    def loss_normalization(self) -> int:
        """Denominator for the MAP mean-loss — a SCALE, not a fit statistic.

        This exists so the inference layer never has to reach into pixels: it asks the
        ProbModel for a normalization defined over ANY observation mix. Imaging-pixel
        count when imaging is present (preserving the historical MAP normalizer exactly),
        else the total observable count (:attr:`event_size`) — the point-source-only
        analogue, so MAP never divides by zero. Always ≥ 1 for a valid ProbModel (it has
        at least one dataset with a positive event_size)."""
        n_pix = self.num_pixels
        return n_pix if n_pix > 0 else int(self.event_size)

    @property
    def high_precision(self) -> bool:
        """Working likelihood precision (True => float64), for inference to read INSTEAD of
        reaching into ``simulators[0]``.

        Resolved from the imaging simulator when there is one (its own precision decision,
        unchanged); otherwise — a model with no imaging simulator, e.g. point-source-only —
        from a dataset-declared ``likelihood_precision`` if any, else the global x64 flag.
        Generalizes to any future non-imaging observation and cannot ``IndexError`` on an
        empty simulator list."""
        sims = self.simulators
        if sims:
            return bool(sims[0].high_precision)
        for d in self.datasets:
            lp = getattr(d, "likelihood_precision", None)
            if lp is not None:
                return lp == "float64"
        import jax
        return bool(jax.config.jax_enable_x64)

    # -- cross-observation validators (§3.6–3.7) -----------------------------------
    def _validate_dead_entities(self):
        """§3.6: a light Component that NO observation sees contributes nothing → raise."""
        seen_ids = set()
        for seen in self._resolved_sees:
            seen_ids.update(id(c) for c in seen)
        dead = [c for c in self.model.light_components if id(c) not in seen_ids]
        if dead:
            raise ValueError(
                f"{len(dead)} light Component(s) are seen by no dataset (§3.6) — a "
                "wiring bug: either add them to some dataset's sees, or remove them "
                f"from the model. Offending profiles: {[c.profile for c in dead]}.")

    def _validate_no_pointsource_imaging_mix(self):
        """Refuse a ProbModel that mixes a point-source dataset with an imaging one.

        A point-source :class:`Dataset` (marked ``is_pointsource``) currently returns a
        HAND-TUNED, unnormalized penalized loss for its ``log_like`` — its scale is set by
        arbitrary weights, not per-observable inverse-variances — so it is NOT commensurable
        with the calibrated Gaussian pixel likelihood of an :class:`ImageData` term. Summing
        the two (as ``ProbModel.log_like`` does) would produce a posterior whose relative
        weighting of the image vs the point-source constraints is meaningless, silently.

        The dataset seam is deliberately built so the two CAN be joined later; the blocker
        is the weighting, not the architecture. Recast the point-source weights as real
        inverse-variances (a genuine reduced χ²) and this guard can be lifted. Fires ONLY
        when a point-source dataset is present, so all-imaging models are unaffected.
        """
        has_ps = any(getattr(d, "is_pointsource", False) for d in self.datasets)
        has_img = any(isinstance(d, ImageData) for d in self.datasets)
        if has_ps and has_img:
            raise NotImplementedError(
                "This ProbModel mixes a point-source dataset (PointSourceData) with an "
                "imaging dataset (ImageData), which is not supported yet. The point-source "
                "term's log_like is an unnormalized, hand-tuned penalized loss (weights, not "
                "inverse-variances), so it is not commensurable with the Gaussian pixel "
                "likelihood — summing them would weight image vs point-source constraints "
                "arbitrarily and silently. Model them separately for now; joint fitting will "
                "be enabled once the point-source weights are recast as real per-observable "
                "inverse-variances (a genuine reduced χ²).")

    def _validate_forward_flux_sharing(self):
        """§3.7 (imaging-only): a non-lstsq light Component seen by >1 imaging dataset
        would render identical flux in every band (unphysical across filters) → raise.
        lstsq components are fine (their amplitude is solved per dataset)."""
        count: dict = {}
        comp_of: dict = {}
        for d, seen in zip(self.datasets, self._resolved_sees):
            if not isinstance(d, ImageData):
                continue
            for c in seen:
                count[id(c)] = count.get(id(c), 0) + 1
                comp_of[id(c)] = c
        # Only renderable components render flux, so only they can clash across bands.
        # (ImageData.sees resolution already excludes non-renderers, so this is defensive.)
        bad = [comp_of[k] for k, n in count.items()
               if n > 1 and getattr(comp_of[k].profile, "renders", True)
               and not comp_of[k].profile.use_lstsq]
        if bad:
            raise ValueError(
                f"{len(bad)} forward-mode (use_lstsq=False) light Component(s) are "
                "seen by more than one dataset (§3.7); they would render IDENTICAL "
                "flux in every band, which is unphysical across filters. Share only "
                "the geometry via shared(...) and give each band its own profile, or "
                "set use_lstsq=True so the amplitude is solved per dataset. Offending "
                f"profiles: {[c.profile for c in bad]}.")

    # -- core ---------------------------------------------------------------------
    def constrained(self, z):
        """The constrained-space params dict for a flat unconstrained ``z`` — the single,
        model-agnostic representation of constrained space (delegates to
        :meth:`LensModel.constrained`). Grouped/coupled priors appear as their individual
        per-site absolute values, not as internal k-vectors."""
        return self.model.constrained(z)

    def unconstrained(self, params):
        """The flat unconstrained ``z`` for a structured constrained params dict — the
        inverse of :meth:`constrained` (delegates to :meth:`LensModel.unconstrained`). Use
        it to seed a sampler at a physical point; ``unconstrained(constrained(z)) == z``."""
        return self.model.unconstrained(params)

    def log_like(self, z, *, simulator=None):
        """Log-likelihood and reduced chi-squared, summed over the observation terms.

        Batch dims are ``z.shape[:-1]`` (scalar for a single MCMC position,
        ``(n_samples,)`` for batched MAP) — same convention as the existing ProbModel.

        The reduced χ² is defined over the Gaussian (chi²-reporting) terms:
        ``Σ chi2 / Σ event_size`` with both sums over those terms only. With imaging
        datasets alone (the only current type) this is identical to the Phase-5 value.

        ``simulator`` (keyword-only) is optional: when ``None`` (the canonical Phase-5
        path) each term uses the engine it built at construction. An explicit simulator
        carries its own batch size ``bs`` and is used for the (single) imaging dataset;
        inference algorithms pass one built at the right ``bs``.
        """
        x = self.bij.forward(z)
        params = self.model.to_params(x)

        if simulator is not None:
            if len(self.terms) != 1 or not isinstance(
                    self.terms[0], ImageLikelihoodTerm):
                raise ValueError(
                    "an explicit simulator may only be passed for a single-dataset "
                    "ProbModel; with multiple datasets, omit it and ProbModel uses its "
                    "per-dataset simulators.")
            log_like, chi2 = self.terms[0].log_like(params, simulator=simulator)
            return log_like, chi2 / self._chi2_event_size

        ll_total = None
        chi2_total = None
        for t in self.terms:
            ll, chi2 = t.log_like(params)
            ll_total = ll if ll_total is None else ll_total + ll
            if chi2 is not None:
                chi2_total = chi2 if chi2_total is None else chi2_total + chi2
        red_chi2 = chi2_total / self._chi2_event_size
        return ll_total, red_chi2

    def log_prior(self, z):
        x = self.bij.forward(z)
        # Evaluate each prior at its native dtype: the flat z may have promoted a
        # mixed-dtype model to one wider dtype, which a strict distribution rejects.
        xn = self.model.cast_free_to_native(x)
        return self.prior.log_prob(xn) + self.bij.forward_log_det_jacobian(z)

    def log_prob(self, z, *, simulator=None):
        log_like, red_chi2 = self.log_like(z, simulator=simulator)
        return log_like + self.log_prior(z), red_chi2
