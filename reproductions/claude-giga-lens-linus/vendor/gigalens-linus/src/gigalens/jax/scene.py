"""The scene spec: one source of truth for the prior, bijector, and constants.

See ``design/phase1-model-data-api.md`` §2, §3, §4, §5. A ``LensModel`` is the shared
physical universe (planes of mass and/or light + optional cosmology). From it we
*derive* — never hand-synchronize — the tfp prior over the **unique** free parameters,
the unconstraining+pack bijector, and the constants dict the simulator reads.

Guiding rule (§1): **identity = sharing.** Reuse a ``Component`` → shared profile;
reuse a ``shared(dist)`` handle → shared parameter (one prior term, §4); reuse a number
→ the value is simply equal. Independence is the default; sharing is explicit.

This module is the *parameter/structure* layer (Phase 2). It does not ray-trace or
render — the trace seam and rendering are Phase 3.
"""
from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

from jax import numpy as jnp
from jax import random
from jax import tree_util
from tensorflow_probability.substrates.jax import distributions as tfd, bijectors as tfb


def _z_param_names(example) -> List[str]:
    """Ordered free-parameter names in the SAME order the sampler's flat ``z``
    vector uses — i.e. the order ``pack_sequence_as(example)`` consumes the flat
    list, which is JAX's tree-flatten (sorted-key) order of ``example``.

    This is the ONE correct source for column→name mapping. Do **not** reconstruct
    names from ``_unique`` insertion order or from a profile's ``_params`` list:
    those are NOT the sampler's z-column order (they agree at the component level
    but differ *within* each component), which is the recurring "C-8" mislabel.
    Verified by the perturbation-identity test in ``tests/test_z_param_names.py``.
    """
    paths_leaves = tree_util.tree_flatten_with_path(example)[0]

    def to_str(path) -> str:
        parts = [str(getattr(k, "key", getattr(k, "idx", k))) for k in path]
        return "/".join(parts)

    return [to_str(p) for p, _ in paths_leaves]


# --------------------------------------------------------------------------------
# Flat-z bijector wrapper
# --------------------------------------------------------------------------------
class ZBijector:
    """Maps the flat sampler vector ``z`` (shape ``(..., D)``) to the constrained
    unique-param dict and back — the ONE convention.

    ``forward(z)`` / ``forward_log_det_jacobian(z)`` take a flat array whose last
    axis has length ``num_free_params``; ``inverse(x)`` returns such a flat array.

    Misuse-proofing (the whole point of the wrapper):
      * The retired ``list(z.T)`` / list-of-columns convention is detected and, for
        now, adapted with a loud ``DeprecationWarning`` (shim phase). It will become
        a hard error once all call sites are migrated — see ``_STRICT``.
      * Any other last-axis size is a hard ``ValueError`` immediately, so a silent
        column/dimension mismatch (e.g. a transposed ``z``) can never slip through.

    This is a plain object, not a ``tfb.Bijector`` subclass, because nothing composes
    the model bijector into an outer TFP construct; it only needs these three methods.
    The underlying TFP bijector is available as ``.raw`` for the rare caller that
    genuinely needs it.
    """

    #: Flip to ``True`` (post-migration) to turn the ``list(z.T)`` shim into a hard error.
    _STRICT = False

    def __init__(self, raw: tfb.Bijector, num_free_params: int):
        self.raw = raw
        self.num_free_params = int(num_free_params)

    def _as_flat(self, z: Any, method: str):
        if isinstance(z, (list, tuple)):
            msg = (
                f"ZBijector.{method} received a list/tuple — the retired list(z.T) / "
                f"list-of-columns convention. Pass the flat z array of shape "
                f"(..., {self.num_free_params}) instead."
            )
            if self._STRICT:
                raise TypeError(msg)
            warnings.warn(msg + " Adapting for now; this will become an error.",
                          DeprecationWarning, stacklevel=3)
            z = jnp.stack(z, axis=-1)
        z = jnp.asarray(z)
        if z.ndim == 0 or z.shape[-1] != self.num_free_params:
            raise ValueError(
                f"ZBijector.{method}: expected a flat z whose last axis is "
                f"{self.num_free_params}, got array of shape {tuple(z.shape)}. "
                f"(If you built list(z.T), pass the flat z array; if you transposed "
                f"it, drop the transpose.)")
        return z

    def forward(self, z: Any):
        return self.raw.forward(self._as_flat(z, "forward"))

    def forward_log_det_jacobian(self, z: Any):
        return self.raw.forward_log_det_jacobian(
            self._as_flat(z, "forward_log_det_jacobian"), event_ndims=1)

    def inverse(self, x: Any):
        """Constrained unique-param dict -> flat z, shape ``(..., num_free_params)``.

        The flat z is a single array, so entries are promoted to a common dtype first
        (mirroring the old list convention's ``jnp.stack`` promotion). A model that mixes
        float32/float64 priors — e.g. a float64 grouped prior among float32 scalars, or
        vice versa — would otherwise die in TFP's pack/concat. Promotion keeps such
        models a drop-in; the paired :meth:`LensModel.cast_free_to_native` cast in
        ProbModel.log_prior makes the downstream ``prior.log_prob`` dtype-safe too.
        """
        leaves, treedef = tree_util.tree_flatten(x)
        if leaves:
            dt = jnp.result_type(*[jnp.asarray(v).dtype for v in leaves])
            x = tree_util.tree_unflatten(
                treedef, [jnp.asarray(v).astype(dt) for v in leaves])
        return self.raw.inverse(x)


# --------------------------------------------------------------------------------
# Sharing handle
# --------------------------------------------------------------------------------
class SharedParam:
    """A linked free parameter. Reuse the *same instance* at multiple sites to share.

    Carries one prior distribution and contributes exactly one free parameter to the
    model regardless of how many sites reference it (§4). Created via :func:`shared`.
    """

    __slots__ = ("dist", "uid")
    _counter = 0

    def __init__(self, dist: tfd.Distribution):
        if not isinstance(dist, tfd.Distribution):
            raise TypeError(f"shared(...) expects a tfd.Distribution, got {type(dist)}")
        self.dist = dist
        self.uid = SharedParam._counter
        SharedParam._counter += 1


def shared(dist: tfd.Distribution) -> SharedParam:
    """Wrap a prior so the *same instance*, reused at multiple sites, links them (§4)."""
    return SharedParam(dist)


def _validate_group_dist(dist: tfd.Distribution, names: tuple, where: Any):
    """Require ``dist`` to be a joint distribution with ``event_shape [len(names)]`` — one
    component per grouped/coupled parameter. Shared by the within-component tuple-key path
    (:meth:`LensModel._classify_group`) and the cross-component :func:`coupled` path."""
    es = dist.event_shape
    es = list(es.as_list()) if hasattr(es, "as_list") else list(es)
    if len(es) != 1 or (es[0] is not None and es[0] != len(names)):
        raise ValueError(
            f"grouped/coupled prior for {names} at {where} must be a distribution with "
            f"event_shape [{len(names)}] (one component per grouped parameter); got "
            f"event_shape {es}. Use e.g. tfd.MultivariateNormalTriL, a tfd.Sample-wrapped "
            f"scalar, or a custom joint distribution.")


# --------------------------------------------------------------------------------
# Cross-component coupling (§coupled-priors): one joint prior scattered to sites in
# DIFFERENT components/planes, plus the mass-anchored ``soft_link`` helper.
# --------------------------------------------------------------------------------
class CoupledSlot:
    """One member of a :class:`CoupledGroup`. Placed into a Component's prior dict at a
    scalar (``str``) key, it binds that site to event component ``idx`` of the group's
    joint distribution. Obtained from ``group[name]``; not constructed directly."""

    __slots__ = ("group", "idx")

    def __init__(self, group: "CoupledGroup", idx: int):
        self.group = group
        self.idx = idx


class CoupledGroup:
    """A joint prior over ``k`` scalar sites that may live in DIFFERENT components/planes.

    The coupling is carried entirely by ``dist`` (a ``tfd.Distribution`` with
    ``event_shape [k]``): event component ``i`` is scattered to whichever site holds
    ``group[names[i]]``. Because the constrained<->unconstrained map for the group is
    ``dist.experimental_default_event_space_bijector()``, the sampler explores the
    distribution's OWN latent coordinates while the model receives the constrained
    (physical) values. Build ``dist`` in *offset form* (a broad anchor plus a tight
    offset, pushed through a linear map — see :func:`soft_link`) and the unconstrained
    space carries the coupling (e.g. a centre separation) as its own decorrelated axis,
    while :meth:`LensModel.constrained` / :meth:`LensModel.to_params` yield the absolute
    per-site values. A plain correlated MVN on the absolute coordinates works too, but
    hands the sampler the stiff correlated geometry — prefer the offset form.

    Reference members by NAME (``group["light/cx"]``), never by raw index: ``names`` must
    match ``dist``'s event order, and each name must be placed at exactly one site.
    Created via :func:`coupled`.
    """

    __slots__ = ("dist", "names", "_index", "uid")
    _counter = 0

    def __init__(self, dist: tfd.Distribution, names: Sequence[str]):
        if not isinstance(dist, tfd.Distribution):
            raise TypeError(f"coupled(...) expects a tfd.Distribution; got {type(dist)}.")
        names = tuple(names)
        if len(names) < 2:
            raise ValueError(
                f"coupled(...) needs at least two member names (a one-member coupling is "
                f"just a plain prior); got {names}.")
        if not all(isinstance(n, str) for n in names):
            raise TypeError(f"coupled(...) member names must be strings; got {names}.")
        if len(set(names)) != len(names):
            raise ValueError(f"coupled(...) member names must be unique; got {names}.")
        _validate_group_dist(dist, names, "coupled(...)")
        self.dist = dist
        self.names = names
        self._index = {n: i for i, n in enumerate(names)}
        self.uid = CoupledGroup._counter
        CoupledGroup._counter += 1

    def __getitem__(self, name: str) -> CoupledSlot:
        try:
            idx = self._index[name]
        except (KeyError, TypeError):
            raise KeyError(
                f"coupled group has no member {name!r}; members are {self.names}.")
        return CoupledSlot(self, idx)


def coupled(dist: tfd.Distribution, names: Sequence[str]) -> CoupledGroup:
    """A joint prior spanning sites in *different* components (§coupled-priors).

    ``dist`` has ``event_shape [k]``; ``names`` labels its ``k`` event components, in event
    order. Returns a :class:`CoupledGroup`; drop ``group[name]`` into each site's prior slot
    (each name used exactly once). For the common "tie centres with a tight separation"
    case, prefer :func:`soft_link`, which builds ``dist`` for you.

    Example (mass and light sharing a centre with a tight separation)::

        # base coords (mass_cx, mass_cy, dx, dy): broad anchor + TIGHT offset
        base = tfd.MultivariateNormalDiag([x0, y0, 0.0, 0.0],
                                          [s_pos, s_pos, s_sep, s_sep])
        # lower-triangular map -> (mass_cx, mass_cy, light_cx, light_cy) = anchor(+offset)
        L = jnp.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                       [1., 0., 1., 0.], [0., 1., 0., 1.]])
        dist = tfd.TransformedDistribution(base, tfb.ScaleMatvecTriL(scale_tril=L))
        g = coupled(dist, names=["mass/cx", "mass/cy", "light/cx", "light/cy"])
        mass  = Component(EPL(),    {"center_x": g["mass/cx"],  "center_y": g["mass/cy"], ...})
        light = Component(Sersic(), {"center_x": g["light/cx"], "center_y": g["light/cy"], ...})
    """
    return CoupledGroup(dist, names)


def soft_link(params, anchor, separation, *, n: int = 2, dtype=None,
              name: str = "soft_link"):
    """Tie ``n`` participants' ``params`` to a shared, mass-anchored position with a tight
    *separation* prior (§coupled-priors) — the ergonomic path over :func:`coupled`.

    Participant 0 is the ANCHOR: its value(s) are exactly the ``anchor`` prior. Every other
    participant sits at ``anchor + offset``, the offset an independent zero-mean Gaussian of
    std ``separation``. The coupling is built in offset form, so the sampler's unconstrained
    coordinates are ``(anchor, offset_1, ...)`` — each separation its OWN decorrelated,
    tightly-scaled axis — while :meth:`LensModel.constrained` yields the absolute per-site
    positions.

    Args:
      params: a parameter name, or a sequence of names, that EACH participant contributes
        (e.g. ``("center_x", "center_y")`` to couple 2-D centres).
      anchor: the broad prior on the anchor's absolute value — one ``tfd.Normal`` PER
        parameter, as a sequence matching ``params`` (a bare ``tfd.Normal`` is accepted only
        when ``params`` is a single name). Each coordinate needs its own prior; a single
        distribution is NOT broadcast across multiple params. Must be Normal (the offset form
        needs a Gaussian latent to compose cleanly with the linear map); for a non-Gaussian
        anchor, build the joint yourself with :func:`coupled`.
      separation: std of the offset (participant-minus-anchor) — a scalar (isotropic) or a
        sequence matching ``params``. Must be > 0.
      n: number of participants (anchor + ``n-1`` linked bodies); default 2.
      dtype: float dtype of the coupling. Defaults to the ambient JAX default float
        (float64 when x64 is enabled, else float32) — like ``DiskEllipticity``, so it
        matches a float64 model out of the box (``tfd.Normal(0.0, .)`` alone would pin it
        to float32 and clash with the promoted flat ``z``).

    Returns:
      A tuple of ``n`` dicts ``{param: CoupledSlot}`` — one per participant, participant 0 the
      anchor. Splat each into its Component's priors::

        mass_c, light_c = soft_link(("center_x", "center_y"),
                                    anchor=(tfd.Normal(0.0, 0.5), tfd.Normal(0.0, 0.5)),
                                    separation=0.02)
        Component(EPL(),    {**mass_c,  "theta_E": te, "e1": e1, "e2": e2})
        Component(Sersic(), {**light_c, "R_sersic": rs, "n_sersic": ns})
    """
    params = (params,) if isinstance(params, str) else tuple(params)
    if len(params) < 1:
        raise ValueError("soft_link needs at least one parameter name in `params`.")
    if not all(isinstance(p, str) for p in params):
        raise TypeError(f"soft_link `params` must be parameter-name strings; got {params}.")
    if n < 2:
        raise ValueError(
            f"soft_link needs n >= 2 participants (anchor + >=1 linked); got n={n}.")
    p = len(params)

    if isinstance(anchor, tfd.Distribution):
        # A single distribution is unambiguous only for a single parameter. For a
        # multi-param coupling (e.g. center_x AND center_y) require one prior PER
        # parameter, so distinct physical coordinates are specified explicitly rather
        # than silently sharing one prior (repo-wide "no silent default").
        if p != 1:
            raise TypeError(
                f"soft_link `anchor` must give one prior PER parameter — a sequence of {p} "
                f"tfd.Normal matching params={params} — not a single distribution broadcast "
                f"to all of them (e.g. center_x and center_y are distinct coordinates and "
                f"each needs its own prior). Got a single {type(anchor).__name__}.")
        anchor_list = [anchor]
    else:
        anchor_list = list(anchor)
    if len(anchor_list) != p:
        raise ValueError(
            f"soft_link `anchor` must have one prior per parameter: {p} to match "
            f"params={params}; got {len(anchor_list)}.")
    for a in anchor_list:
        if not isinstance(a, tfd.Normal):
            raise TypeError(
                f"soft_link `anchor` entries must be tfd.Normal (the offset form needs a "
                f"Gaussian anchor to compose cleanly); got {type(a)}. For a non-Gaussian "
                f"anchor, build the joint distribution yourself and use coupled(...).")

    sep_list = ([float(separation)] * p if _is_number(separation)
                else [float(s) for s in separation])
    if len(sep_list) != p:
        raise ValueError(
            f"soft_link `separation` must be a scalar or a sequence matching params "
            f"({p} of them); got {len(sep_list)}.")
    if not all(s > 0 for s in sep_list):
        raise ValueError(
            f"soft_link `separation` must be > 0 (a tight, positive std); got {sep_list}.")

    if dtype is None:
        dtype = jnp.zeros(()).dtype  # ambient default float (float64 iff x64 enabled)
    # base coords: [anchor params (p)] + [offset params (p)] * (n-1). A single-tensor
    # MVNDiag (NOT a Blockwise) so its event-space bijector is the identity and composes
    # with the linear map below — a Blockwise's structured bijector does not compose with a
    # monolithic ScaleMatvec.
    loc = ([jnp.asarray(a.loc, dtype) for a in anchor_list]
           + [jnp.asarray(0.0, dtype)] * ((n - 1) * p))
    scale = ([jnp.asarray(a.scale, dtype) for a in anchor_list]
             + [jnp.asarray(s, dtype) for _ in range(n - 1) for s in sep_list])
    base = tfd.MultivariateNormalDiag(jnp.stack(loc), jnp.stack(scale))

    # lower-triangular map: participant 0 = anchor; participant j = anchor + offset_j.
    k = n * p
    rows = [[0.0] * k for _ in range(k)]
    for i in range(p):
        rows[i][i] = 1.0
    for j in range(1, n):
        for i in range(p):
            out = j * p + i
            rows[out][i] = 1.0        # + anchor
            rows[out][out] = 1.0      # + offset_j (base index j*p + i)
    scale_tril = jnp.asarray(rows, dtype)
    dist = tfd.TransformedDistribution(
        base, tfb.ScaleMatvecTriL(scale_tril=scale_tril), name=name)

    member_names = tuple(f"p{j}/{prm}" for j in range(n) for prm in params)
    group = CoupledGroup(dist, member_names)
    return tuple({prm: group[f"p{j}/{prm}"] for prm in params} for j in range(n))


# --------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------
class Component:
    """A profile bundled with its per-parameter priors (§2.1).

    ``priors`` maps each name in ``profile.params`` to one of: a ``tfd.Distribution``
    (free), a ``shared(dist)`` handle (free + linked), or a number (fixed → constant).
    A missing or unknown parameter raises at derivation time. lstsq-solved amplitudes
    are absent from ``profile.params`` and so are neither free nor fixable here.
    """

    def __init__(self, profile: Any, priors: Dict[str, Any]):
        self.profile = profile
        self.priors = dict(priors)


class Plane:
    """A line-of-sight plane carrying mass and/or light, with a geometry (§2.6).

    Geometry is exactly one of ``deflection_ratio`` (no cosmology) or ``redshift``
    (with cosmology); each may be a number, a ``tfd.Distribution``, a ``shared`` handle,
    or ``None`` (validated by :class:`LensModel` per §3.1).
    """

    def __init__(
        self,
        *,
        redshift: Any = None,
        deflection_ratio: Any = None,
        mass: Sequence[Component] = (),
        light: Sequence[Component] = (),
    ):
        self.redshift = redshift
        self.deflection_ratio = deflection_ratio
        self.mass: List[Component] = list(mass)
        self.light: List[Component] = list(light)

    @property
    def has_mass(self) -> bool:
        return len(self.mass) > 0

    @property
    def has_light(self) -> bool:
        return len(self.light) > 0


def _is_number(x: Any) -> bool:
    if isinstance(x, (SharedParam, tfd.Distribution)):
        return False
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------
class LensModel:
    """The shared physical model; derives prior, bijector, and constants (§2.3).

    Planes are kept in the order given (caller orders observer→source). If a cosmology
    is supplied, every plane must carry a ``redshift``; otherwise lensed-light planes
    carry a ``deflection_ratio`` (single source plane defaults to 1.0; §3.1).

    Derived attributes:
      - ``prior``           : ``tfd.JointDistributionNamed`` over the unique free params.
      - ``bijector``        : maps an unconstrained column-list to the constrained
                              unique-param dict (same convention as the existing
                              ProbModel: call ``bijector.forward(list(z.T))``).
      - ``constants``       : structured (planes/cosmo) dict of fixed values.
      - ``num_free_params`` : sampled dimensionality (shared params counted once).
    """

    _seed = random.PRNGKey(0)

    def __init__(self, planes: Sequence[Plane], cosmo: Optional[Component] = None):
        if len(planes) < 1:
            raise ValueError("LensModel needs at least one plane.")
        self.planes: List[Plane] = list(planes)
        self.cosmo = cosmo

        # Derivation state, filled by _derive().
        self._unique: Dict[str, tfd.Distribution] = {}        # unique_key -> dist
        self._site_to_unique: List[Tuple[tuple, str]] = []    # (path, unique_key)
        self._constants_paths: List[Tuple[tuple, float]] = [] # (path, value)
        self._bare_seen: Dict[int, tuple] = {}                # id(dist) -> first path
        self._coupled_seen: Dict[int, Any] = {}              # group uid -> (group, [idx placed])

        self._validate_geometry()
        # Physicality flag for CONCRETE cosmo constants (misuse_register C3, scene
        # path): the distance calls run jitted, where the cosmology layer's flag is
        # skipped (params are tracers), so fixed-but-unphysical densities are checked
        # here — the last point they are concrete. Sampled cosmo params skip (the
        # prior's support is the guard there).
        if cosmo is not None:
            concrete = {k: float(v) for k, v in cosmo.priors.items()
                        if isinstance(k, str) and _is_number(v)}
            if concrete:
                # H0 only sets the ~1e-4-scale radiation/curvature terms of the check,
                # so when it is sampled, substitute a nominal 70 rather than skip —
                # otherwise a concrete unphysical Om0/k with a sampled H0 (the common
                # inference configuration) would go unflagged (grader round-2, F1).
                cosmo.profile._flag_unphysical_densities(
                    concrete.pop("H0", 70.0), **concrete)
        self._derive()
        # Physicality of profile parameters (misuse_register, physicality layer):
        # fixed values are checked against each profile's code-derived hard
        # domains (violation -> raise), and priors are checked for probability
        # mass inside invalid regions (mass > EPS_MASS -> warn). Runs here — the
        # last point values are concrete — for the same jit-blindness reason as
        # the guards above. The report is attached for model-card persistence.
        from gigalens import physicality as _physicality

        self.physicality_report = _physicality.validate_planes(self.planes)
        _physicality.apply_report(self.physicality_report)

    @property
    def light_components(self) -> List[Component]:
        """All light Components across all planes, in (plane, index) order (§2.5).

        Used by the dataset layer to resolve ``sees`` (identity-matched), to expand
        ``sees="all"``, and to detect dead entities (a light Component no dataset sees,
        §3.6). Mass/cosmo are the always-shared backbone and are not light.
        """
        out: List[Component] = []
        for p in self.planes:
            out.extend(p.light)
        return out

    # -- validation (§3.1, §3.2, §3.3) --------------------------------------------
    def _validate_geometry(self):
        n_lensed_light = sum(
            1 for i, p in enumerate(self.planes)
            if p.has_light and any(self.planes[j].has_mass for j in range(i))
        )
        for i, p in enumerate(self.planes):
            has_dr = p.deflection_ratio is not None
            has_z = p.redshift is not None
            if self.cosmo is not None:
                # Every plane needs a redshift (distances need all plane positions).
                if has_dr:
                    raise ValueError(
                        f"plane {i}: deflection_ratio given but a cosmology is present; "
                        "specify redshift instead (§3.1).")
                if not has_z:
                    raise ValueError(
                        f"plane {i}: redshift is required when a cosmology is present "
                        "(§3.1) — no silent default.")
            else:
                if has_z:
                    raise ValueError(
                        f"plane {i}: redshift given but no cosmology; specify "
                        "deflection_ratio instead (§3.1).")
                lensed = p.has_light and any(self.planes[j].has_mass for j in range(i))
                if lensed and not has_dr:
                    if n_lensed_light == 1:
                        p.deflection_ratio = 1.0  # documented single-source default
                    else:
                        raise ValueError(
                            f"plane {i}: deflection_ratio is required for a lensed-light "
                            f"plane when there are {n_lensed_light} such planes (§3.1) — "
                            "the single-plane 1.0 default applies only to a lone source "
                            "plane.")
        if self.cosmo is not None:
            self._validate_concrete_redshifts()

    def _validate_concrete_redshifts(self):
        """Domain guard for CONCRETE plane redshifts (misuse_register C1, scene path).

        The trace runs jitted, so every ``geom["redshift"]`` is a tracer there — even a
        plain-number redshift the user fixed at construction — and the cosmology layer's
        first-use guard cannot fire. Constants ARE concrete here, so this is the last
        point the misuse is checkable: each concrete redshift must be finite and > 0,
        and concrete redshifts must be non-decreasing in plane order (the documented
        observer→source ordering). Mis-ordering is exactly the silent-negative-
        deflection_ratio misuse: a light plane listed after a mass plane is treated as
        lensed by it, so a front-of-lens redshift there yields ratio < 0 with no signal.
        Free (sampled) redshifts skip the check — bounding those is the prior's job.
        """
        import math
        prev = None   # (index, value) of the last concrete redshift seen
        for i, p in enumerate(self.planes):
            if not _is_number(p.redshift):
                continue
            z = float(p.redshift)
            if not math.isfinite(z) or z <= 0:
                raise ValueError(
                    f"plane {i}: redshift={p.redshift} must be finite and > 0 "
                    "(misuse_register C1).")
            if prev is not None and z < prev[1]:
                raise ValueError(
                    f"planes must be ordered observer→source (non-decreasing redshift): "
                    f"plane {i} has redshift {z} < plane {prev[0]}'s {prev[1]}. A light "
                    "plane listed after a mass plane is lensed by it, so a front-of-lens "
                    "redshift there silently yields a negative deflection_ratio "
                    "(misuse_register C1). Reorder the planes (a foreground light plane "
                    "belongs BEFORE the mass plane, where it is correctly undeflected).")
            prev = (i, z)

    def _classify(self, value: Any, path: tuple):
        """Sort one parameter value into free (bare/shared/coupled) or constant; raise on reuse."""
        if isinstance(value, CoupledSlot):
            # A cross-component coupling member (§coupled-priors): register the group's
            # joint dist once as a single unique entry, and scatter event component
            # ``idx`` to this site — the SAME (path, ukey, idx) triple the tuple-key
            # group uses, so to_params / the bijector / z-naming need no special case.
            grp = value.group
            ukey = f"coupled_{grp.uid}"
            if ukey not in self._unique:
                self._unique[ukey] = grp.dist
                self._coupled_seen[grp.uid] = (grp, [])
            self._coupled_seen[grp.uid][1].append(value.idx)
            self._site_to_unique.append((path, ukey, value.idx))
            return
        if isinstance(value, SharedParam):
            self._site_to_unique.append((path, f"shared_{value.uid}", None))
            self._unique.setdefault(f"shared_{value.uid}", value.dist)
        elif isinstance(value, tfd.Distribution):
            prev = self._bare_seen.get(id(value))
            if prev is not None:
                raise ValueError(
                    f"the same tfd.Distribution object is reused at {prev} and {path}; "
                    "wrap it in shared() to link them, or construct a fresh distribution "
                    "to keep them independent (§3.3).")
            self._bare_seen[id(value)] = path
            ukey = "/".join(map(str, path))
            self._unique[ukey] = value
            self._site_to_unique.append((path, ukey, None))
        elif _is_number(value):
            self._constants_paths.append((path, float(value)))
        else:
            raise TypeError(
                f"parameter at {path} must be a tfd.Distribution, a shared() handle, or "
                f"a number; got {type(value)}.")

    def _classify_component(self, comp: Component, base_path: tuple):
        """Classify a Component's priors. A prior key is either a single parameter
        name (``str`` -> scalar prior) or a ``tuple`` of names (-> one joint prior +
        bijector spanning those params, §grouped-priors). Every parameter in
        ``profile.params`` must be covered exactly once."""
        params = list(comp.profile.params)
        param_set = set(params)
        covered: List[str] = []
        for key in comp.priors:
            names = key if isinstance(key, tuple) else (key,)
            if isinstance(key, tuple) and len(names) < 2:
                raise ValueError(
                    f"grouped prior key {key!r} for {comp.profile} at {base_path} must "
                    f"name at least two parameters; use a plain string for a single "
                    f"parameter.")
            for n in names:
                if not isinstance(n, str):
                    raise TypeError(
                        f"prior key {key!r} for {comp.profile} at {base_path} must be a "
                        f"parameter name (str) or a tuple of names; got a {type(n)}.")
            covered.extend(names)
        unknown = set(covered) - param_set
        if unknown:
            raise ValueError(
                f"unknown parameter(s) {sorted(unknown)} for {comp.profile} at "
                f"{base_path}; valid params are {params}.")
        seen: set = set()
        dup: set = set()
        for n in covered:
            (dup if n in seen else seen).add(n)
        if dup:
            raise ValueError(
                f"parameter(s) {sorted(dup)} are covered by more than one prior key for "
                f"{comp.profile} at {base_path}; each parameter must appear in exactly "
                f"one scalar or grouped prior key.")
        missing = param_set - seen
        if missing:
            raise ValueError(
                f"missing parameter(s) {sorted(missing)} for {comp.profile} at "
                f"{base_path} (§3.2) — no silent default.")
        for key, value in comp.priors.items():
            if isinstance(key, tuple):
                self._classify_group(key, value, base_path)
            else:
                self._classify(value, base_path + (key,))

    def _classify_group(self, names: tuple, value: Any, base_path: tuple):
        """Classify a grouped (tuple-key) prior: one unique entry (a multivariate
        distribution) whose ``k`` components scatter to the ``k`` named sites (§4).
        The distribution's ``experimental_default_event_space_bijector`` is the joint
        unconstraining map (the coupling lives here, not in the profile)."""
        paths = tuple(base_path + (n,) for n in names)
        if isinstance(value, SharedParam):
            _validate_group_dist(value.dist, names, base_path)
            ukey = f"shared_{value.uid}"
            self._unique.setdefault(ukey, value.dist)
        elif isinstance(value, tfd.Distribution):
            prev = self._bare_seen.get(id(value))
            if prev is not None:
                raise ValueError(
                    f"the same tfd.Distribution object is reused at {prev} and "
                    f"{base_path + (names,)}; wrap it in shared() to link them, or "
                    f"construct a fresh distribution to keep them independent (§3.3).")
            self._bare_seen[id(value)] = base_path + (names,)
            _validate_group_dist(value, names, base_path)
            # Key by the pipe-joined member paths (no synthetic prefix) so the group
            # sorts into its first member's z-column slot: a grouped model keeps the
            # same column order as the equivalent scalar model.
            ukey = "|".join("/".join(map(str, p)) for p in paths)
            self._unique[ukey] = value
        else:
            raise TypeError(
                f"a grouped (tuple-key) prior for {names} at {base_path} must be a "
                f"tfd.Distribution or a shared() handle; got {type(value)}. A number "
                f"cannot cover multiple parameters — fix each one with a scalar key.")
        for j, p in enumerate(paths):
            self._site_to_unique.append((p, ukey, j))

    # -- derivation (§4, §5) ------------------------------------------------------
    def _derive(self):
        for i, p in enumerate(self.planes):
            # geometry
            if self.cosmo is not None:
                self._classify(p.redshift, ("planes", i, "geometry", "redshift"))
            elif p.deflection_ratio is not None:
                self._classify(p.deflection_ratio,
                               ("planes", i, "geometry", "deflection_ratio"))
            # mass / light
            for j, comp in enumerate(p.mass):
                self._classify_component(comp, ("planes", i, "mass", j))
            for j, comp in enumerate(p.light):
                self._classify_component(comp, ("planes", i, "light", j))
        if self.cosmo is not None:
            self._classify_component(self.cosmo, ("cosmo",))

        # coupled-group completeness (§coupled-priors): every member of a coupled(...)
        # group must be placed at EXACTLY one site — a missing member leaves a sampled
        # event component wired to nothing; a repeated one silently ties two sites (use
        # shared() if that was intended). The last point the site set is knowable.
        for uid, (grp, idxs) in self._coupled_seen.items():
            k = len(grp.names)
            seen = set(idxs)
            if len(idxs) != len(seen):
                dups = sorted({i for i in idxs if idxs.count(i) > 1})
                raise ValueError(
                    f"coupled member(s) {[grp.names[i] for i in dups]} were placed at more "
                    f"than one site; each coupled(...) member binds exactly one site — use "
                    f"shared() if you meant two sites to be equal (§coupled-priors).")
            missing = set(range(k)) - seen
            if missing:
                raise ValueError(
                    f"coupled member(s) {[grp.names[i] for i in sorted(missing)]} were never "
                    f"placed at a site; every member of a coupled(...) group must appear at "
                    f"exactly one Component prior slot (§coupled-priors).")

        # constants as a structured dict
        self.constants: Dict[str, Any] = {}
        for path, val in self._constants_paths:
            _set_path(self.constants, path, jnp.asarray(val))

        # prior over the unique free params + the flat-z bijector.
        #
        # The bijector maps a flat z of shape (..., D) <-> the constrained unique-param
        # dict, where D = SUM of per-entry UNCONSTRAINED sizes (a scalar prior -> 1
        # column; a grouped/multivariate prior -> as many columns as its unconstrained
        # event size; e.g. a disk-ellipticity pair -> 2, a k-simplex -> k-1). Column
        # order is JAX tree-flatten (sorted-key) order of the unconstrained example.
        if self._unique:
            self.prior = tfd.JointDistributionNamed(dict(self._unique))
            ev_bij = self.prior.experimental_default_event_space_bijector()
            example = self.prior.sample(seed=self._seed)          # constrained dict
            # Native dtype of each free entry, so ProbModel.log_prior can evaluate each
            # prior at its own dtype even when the flat z promoted to a wider one (a
            # float32 prior among float64s, or vice versa). Keyed by unique key.
            self._free_dtypes = {
                "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in p):
                jnp.asarray(v).dtype
                for p, v in tree_util.tree_flatten_with_path(example)[0]}
            unc = ev_bij.inverse(example)                         # unconstrained dict
            leaves = tree_util.tree_flatten_with_path(unc)[0]     # sorted-key order
            sizes = [int(jnp.asarray(v).size) for _, v in leaves]
            eshapes = [list(jnp.asarray(v).shape) for _, v in leaves]
            self.num_free_params = int(sum(sizes))
            # flat (..., D)  ->  unconstrained dict  (a pure, volume-preserving repack:
            # split into per-entry column-chunks, reshape each to its event shape, pack).
            flat_to_named = tfb.Chain([
                tfb.pack_sequence_as(unc),
                tfb.JointMap([tfb.Reshape(event_shape_out=es, event_shape_in=[s])
                              for es, s in zip(eshapes, sizes)]),
                tfb.Split(sizes, axis=-1),
            ])
            self.pack_bij = flat_to_named
            raw = tfb.Chain([ev_bij, flat_to_named])              # flat z -> constrained
            self.bijector = ZBijector(raw, self.num_free_params)
            self.z_param_names = self._build_z_param_names(unc)
        else:
            self.num_free_params = 0
            self.prior = None
            self.pack_bij = None
            self.bijector = None
            self.z_param_names = []
            self._free_dtypes = {}

    def cast_free_to_native(self, unique: Dict[str, Any]) -> Dict[str, Any]:
        """Cast each free-param value to the dtype its prior expects.

        The flat ``z`` promotes a mixed-dtype model to one (wider) dtype, so the
        constrained dict from ``bijector.forward`` is uniform. Before ``prior.log_prob``,
        cast each entry back to its native dtype so a strict distribution (e.g. an
        ``MVNDiag``-backed grouped prior) isn't handed a wider dtype. A no-op for
        dtype-uniform models."""
        out = {}
        for k, v in unique.items():
            dt = self._free_dtypes.get(k)
            out[k] = jnp.asarray(v).astype(dt) if dt is not None else v
        return out

    def _build_z_param_names(self, unc: Dict[str, Any]) -> Optional[List[str]]:
        """Column->name map for the flat z, expanded across grouped entries.

        Scalar entries keep their canonical unique-key name (identical to the old
        ``_z_param_names``, so scalar-only models are byte-for-byte unchanged). A
        grouped entry contributing ``s`` columns emits ``s`` names: its member param
        paths when the group is dimension-preserving (``s`` == #params, the common
        case: disk ellipticity, shear, correlated pairs), else ``<ukey>#j`` for a
        dimension-reducing group (e.g. a k-simplex: k params but k-1 columns).

        Defensive: naming must never break model construction, so any failure here
        degrades to ``None`` rather than raising.
        """
        try:
            group_paths: Dict[str, Dict[int, tuple]] = {}
            for path, ukey, idx in self._site_to_unique:
                if idx is not None:
                    group_paths.setdefault(ukey, {})[idx] = path
            names: List[str] = []
            for keypath, val in tree_util.tree_flatten_with_path(unc)[0]:
                k = keypath[0]
                ukey = str(getattr(k, "key", getattr(k, "idx", k)))
                s = int(jnp.asarray(val).size)
                if s == 1:
                    names.append(ukey)  # scalar entry: canonical unique-key name
                    continue
                members = group_paths.get(ukey)
                if members is not None and len(members) == s:
                    names.extend("/".join(map(str, members[j])) for j in range(s))
                else:
                    names.extend(f"{ukey}#{j}" for j in range(s))
            return names
        except Exception:
            return None

    # -- role helpers + partial-fix (§4; G1 D1) -----------------------------------
    def source_plane_light(self) -> List[Component]:
        """Light Components on lensed (non-primary-deflector) planes (G1 D1).

        A plane's light is "source" light if at least one *earlier* plane carries
        mass (i.e. the light is lensed). The primary deflector's own light (lens
        light, on/before the first mass plane) is excluded. This lets a research
        stage map the role word ``"source"`` onto the actual Components by identity,
        without the core API knowing the old {lens,lens_light,source} vocabulary.
        """
        out: List[Component] = []
        for i, p in enumerate(self.planes):
            lensed = any(self.planes[j].has_mass for j in range(i))
            if lensed:
                out.extend(p.light)
        return out

    def _truth_at(self, truth_scene: Dict[str, Any], path: tuple) -> Any:
        """Look up the value at ``path`` in a structured (planes/cosmo) truth dict."""
        cur = truth_scene
        for key in path:
            try:
                cur = cur[key]
            except (KeyError, TypeError, IndexError) as exc:
                raise KeyError(
                    f"fix_to: truth_scene has no value at site {path} (missing {key!r}); "
                    "the truth must cover every parameter being fixed.") from exc
        return cur

    def fix_to(self, truth_scene: Dict[str, Any],
               free: Sequence[Component] = ()) -> "LensModel":
        """Return a NEW LensModel with every parameter FIXED to its truth value,
        EXCEPT the parameters of the ``free`` Components, which keep their priors (D1).

        ``truth_scene`` is a structured params dict (the §5 ``planes``/``cosmo`` layout,
        e.g. from :meth:`to_params` or the research-side truth adapter); a fixed
        parameter takes ``float(truth_scene[...site...])``. ``free`` Components are
        matched BY OBJECT IDENTITY against this model's Components (like ``sees``); an
        unknown Component raises. lstsq amplitudes are absent from ``profile.params`` and
        so are never fixed or freed (they are solved per evaluation).

        Geometry (redshift / deflection_ratio) is always fixed to its truth value — the
        free set is about light/mass *Components*, not plane geometry.

        Shared-param edge case: if a ``shared()`` handle is referenced by BOTH a free and
        a fixed Component, the link cannot be half-fixed; this build asserts that does not
        occur (raises with the offending unique key) rather than silently picking a side.
        """
        free_ids = {id(c) for c in free}
        all_ids = {id(c) for c in (self.light_components + [c for p in self.planes for c in p.mass])}
        unknown = free_ids - all_ids
        if unknown:
            raise ValueError(
                "fix_to: free references Component(s) not in this model (matched by "
                "object identity); pass the same instances used to build the LensModel.")

        # Detect the shared-param both-free-and-fixed ambiguity (assert it doesn't occur).
        shared_free: Dict[int, bool] = {}
        shared_fixed: Dict[int, bool] = {}

        def _fix_component(comp: Component, base_path: tuple) -> Component:
            """Keep every prior key untouched if the Component is free; otherwise fix
            each covered param to its truth number. A fixed grouped (tuple-key) prior
            becomes individual fixed scalars — a fixed group is just constants. Tracks
            shared usage for the both-free-and-fixed assertion."""
            is_free = id(comp) in free_ids
            new_priors: Dict[Any, Any] = {}
            for key, orig in comp.priors.items():
                names = key if isinstance(key, tuple) else (key,)
                if isinstance(orig, SharedParam):
                    (shared_free if is_free else shared_fixed)[orig.uid] = True
                if is_free:
                    new_priors[key] = orig  # keep dist / shared / number / group as-is
                else:
                    for n in names:
                        new_priors[n] = float(
                            self._truth_at(truth_scene, base_path + (n,)))
            return Component(comp.profile, new_priors)

        new_planes: List[Plane] = []
        for i, p in enumerate(self.planes):
            # geometry: always fixed to truth (if this plane carries one)
            redshift = None
            deflection_ratio = None
            if self.cosmo is not None:
                redshift = float(self._truth_at(
                    truth_scene, ("planes", i, "geometry", "redshift")))
            elif p.deflection_ratio is not None:
                deflection_ratio = float(self._truth_at(
                    truth_scene, ("planes", i, "geometry", "deflection_ratio")))
            mass = [_fix_component(c, ("planes", i, "mass", j))
                    for j, c in enumerate(p.mass)]
            light = [_fix_component(c, ("planes", i, "light", j))
                     for j, c in enumerate(p.light)]
            new_planes.append(Plane(redshift=redshift, deflection_ratio=deflection_ratio,
                                    mass=mass, light=light))

        new_cosmo = None
        if self.cosmo is not None:
            new_cosmo = _fix_component(self.cosmo, ("cosmo",))

        both = set(shared_free) & set(shared_fixed)
        if both:
            raise ValueError(
                f"fix_to: shared() param(s) {sorted(both)} are referenced by BOTH a free "
                "and a fixed Component; the link cannot be half-fixed. Split the shared "
                "handle, or put the linked Components on the same side of `free`.")

        return LensModel(new_planes, cosmo=new_cosmo)

    # -- scatter (§4): unique-param dict -> structured params dict -----------------
    def to_params(self, unique: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble the structured params dict the simulator consumes.

        Starts from the constants and writes each free value to *every* site that
        references it (so a shared parameter's single sampled value is fanned out to
        all its render sites). The returned dict has the §5 layout.
        """
        params = copy.deepcopy(self.constants)
        for path, ukey, idx in self._site_to_unique:
            v = unique[ukey]
            _set_path(params, path, v if idx is None else v[..., idx])
        return params

    def constrained(self, z) -> Dict[str, Any]:
        """Map the sampler's flat unconstrained ``z`` to THE constrained-space params dict
        the simulator consumes — the single, model-agnostic representation of constrained
        space.

        Equivalent to ``to_params(bijector.forward(z))``: the unconstraining+pack bijector
        takes the flat ``z`` to the constrained *unique* dict, then :meth:`to_params`
        scatters it to the structured per-site §5 layout — injecting constants and fanning
        shared params out to every site. Grouped/coupled priors appear here as their
        individual per-site (absolute) values, NOT as the internal k-vectors of
        ``bijector.forward``. A model with no free parameters (``bijector is None``) returns
        the constants alone."""
        if self.bijector is None:
            return self.to_params({})
        return self.to_params(self.bijector.forward(z))

    def to_unique(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Gather a structured constrained params dict back to the *unique* dict — the
        inverse of :meth:`to_params`.

        Reads one value per free unique key from the structured layout: constants are
        dropped, a shared param is read from a single (canonical) site, and a
        grouped/coupled group's members are regrouped from their per-site scalars into the
        k-vector its distribution expects. Leading batch dims are preserved. For a shared
        param whose sites hold *different* values (an out-of-model params dict), the first
        site in derivation order wins — ``to_params`` writes them equal, so a genuine
        constrained point round-trips."""
        scalar: Dict[str, Any] = {}          # ukey -> value          (idx is None)
        grouped: Dict[str, Dict[int, Any]] = {}  # ukey -> {idx: value}
        for path, ukey, idx in self._site_to_unique:
            v = _get_path(params, path)
            if idx is None:
                scalar.setdefault(ukey, v)   # shared: first site wins (all written equal)
            else:
                grouped.setdefault(ukey, {})[idx] = v
        unique: Dict[str, Any] = {k: jnp.asarray(v) for k, v in scalar.items()}
        for ukey, members in grouped.items():
            # members is complete (0..k-1) — enforced at derivation (§coupled-priors).
            unique[ukey] = jnp.stack(
                [jnp.asarray(members[i]) for i in range(len(members))], axis=-1)
        return unique

    def unconstrained(self, params: Dict[str, Any]):
        """Map a structured constrained params dict to the flat unconstrained ``z`` — the
        inverse of :meth:`constrained` (``bijector.inverse(to_unique(params))``).

        Use it to seed a sampler at a physical point: ``unconstrained(constrained(z)) == z``
        for any valid ``z`` (up to bijector round-off). Constants in ``params`` are ignored;
        only the free sites are read. A model with no free parameters returns an empty
        ``(0,)`` vector."""
        if self.bijector is None:
            return jnp.zeros((0,), dtype=jnp.zeros(()).dtype)
        return self.bijector.inverse(self.to_unique(params))


# --------------------------------------------------------------------------------
# Nested-dict path helpers
# --------------------------------------------------------------------------------
def _set_path(d: Dict[str, Any], path: tuple, value: Any):
    """Set ``d[path[0]][path[1]]...= value``, creating intermediate dicts."""
    cur = d
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _get_path(d: Dict[str, Any], path: tuple) -> Any:
    """Read ``d[path[0]][path[1]]...`` — the read dual of :func:`_set_path`."""
    cur = d
    for key in path:
        cur = cur[key]
    return cur
