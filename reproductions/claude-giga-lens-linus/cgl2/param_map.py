"""Bidirectional map: old 46-dim constrained-param labels <-> scene-API params.

The OLD side is the validated claude-giga-lens / foundry-i 46-dim ridge-
marginalized model (cgl.likelihood.build_marg_model, itself a port of
foundry-i _hmc_lib_marg.build_model_marg): EPL(50)+Shear mass, 4 SersicEllipse
lens lights, SersicEllipse + Shapelets(n_max=6, marginalized amps) source.
Its constrained-space labels are the `index_labels` strings of that model:
"mass.*", "shear.*", "LL{0..3}.*", "srcS.*", "srcShp.*".

The NEW side is the vendored gigalens-linus scene API (LensModel of
cgl2.scene_build.build_scene_model): plane 0 = lens plane (mass=[EPL, Shear],
light=[LL0..LL3]), plane 1 = source plane (light=[srcS, srcShp]). Free params
are keyed by the scene's unique keys "planes/{i}/{mass|light}/{j}/{param}" —
exactly the strings in `model.z_param_names`. ALL access is keyed on those
names; z-column order is NEVER assumed (their C-8 mislabel lesson).

CONVENTION RECONCILIATIONS (each verified against both sources, 2026-07-15)
---------------------------------------------------------------------------
1. Ellipticity: BOTH stacks parameterize ellipticity as (e1, e2) with
   phi = arctan2(e2, e1)/2, c = sqrt(e1^2+e2^2), q = (1-c)/(1+c) — verified
   identical in gigalens-sean@58ec9a7 profiles (mass/epl.py, light/sersic.py)
   and gigalens-linus@80916d2 (same files). NO conversion needed. There is no
   position-angle parameter on either side.
2. Centers / orientation: both grids are mean-centered arcsec grids built
   from transform_pix2angle = delta_pix * I with x varying along columns
   (old: lenstronomy PixelGrid via LensSimulatorInterface.get_coords; new:
   LensWCS.pixel_grid). Identity map. NOTE the old grid VALUES are float32
   (get_coords casts), the new are float64 — that is a numerics difference,
   not a labeling one; cgl2.scene_build reconciles it for parity runs.
3. Amplitude units (the conversion_factor reconciliation, mirrors
   cgl.e2.paper_z_to_marg_z's `ie_scale`): the old 46-dim model compares
   M_det = simulate()/conversion_factor to the data and builds design columns
   WITHOUT the conversion factor, so its Ie (and marginal shapelet amps a*)
   are in "no-cf" surface-brightness units. The scene `lstsq_simulate(...,
   return_stacked=True)` basis ALSO carries no conversion factor -> for the
   marg-view scene model the Ie map is the IDENTITY. The scene `simulate()`
   multiplies by conversion_factor (= delta_pix^2), so for a FORWARD-view
   scene model (explicit amplitudes, stock ImageLikelihoodTerm) old-unit
   amplitudes must be divided by cf: Ie_fwd = Ie_old/cf, amp_fwd = a_old/cf.
   Use `ie_scale=1/cf` (values) and remember dL/dIe_old = dL/dIe_fwd / cf.
4. Shapelet amplitude labels: old marg model has NO amp labels (marginalized);
   the forward-view scene model fixes "amp00".."amp27" as constants in
   gigalens _amp_names order (n1,n2 enumeration IDENTICAL in both stacks,
   verified: same loop in both shapelets.py). The ridge Lambda_ii=(i+1)/25
   uses the same ordering.
5. z-space: the two stacks pack z differently (old: JointDistributionSequential
   tree order; new: sorted unique-key order) but the PER-LABEL 1-D default
   event-space bijectors are the same TFP transforms (same scalar prior
   families) — so parity is compared in CONSTRAINED space through this map,
   and each stack's own bijector handles its own z. `scene_z_from_old_labels`
   goes old-labeled values -> scene flat z via model.unconstrained(...).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# canonical label sets
# --------------------------------------------------------------------------- #
_MASS_PARAMS = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y"]
_SHEAR_PARAMS = ["gamma1", "gamma2"]
_SERSIC_PARAMS = ["R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y", "Ie"]
_SHP_PARAMS = ["beta", "center_x", "center_y"]

#: Canonical (documentation) order of the 46 old labels. Storage order for
#: parity-ref vectors; NEVER assumed to be either stack's z order.
OLD_LABELS_46: List[str] = (
    [f"mass.{p}" for p in _MASS_PARAMS]
    + [f"shear.{p}" for p in _SHEAR_PARAMS]
    + [f"LL{i}.{p}" for i in range(4) for p in _SERSIC_PARAMS]
    + [f"srcS.{p}" for p in _SERSIC_PARAMS]
    + [f"srcShp.{p}" for p in _SHP_PARAMS]
)
assert len(OLD_LABELS_46) == 46

#: old component prefix -> scene site prefix (plane/list/index layout of
#: cgl2.scene_build.build_scene_model — the single place that layout is
#: defined; keep in sync with its docstring).
_PREFIX_MAP: Dict[str, str] = {
    "mass": "planes/0/mass/0",
    "shear": "planes/0/mass/1",
    "LL0": "planes/0/light/0",
    "LL1": "planes/0/light/1",
    "LL2": "planes/0/light/2",
    "LL3": "planes/0/light/3",
    "srcS": "planes/1/light/0",
    "srcShp": "planes/1/light/1",
}
_PREFIX_INV: Dict[str, str] = {v: k for k, v in _PREFIX_MAP.items()}


def old_to_scene(label: str) -> str:
    """'LL2.center_x' -> 'planes/0/light/2/center_x' (raises on unknown)."""
    try:
        prefix, param = label.split(".", 1)
        return f"{_PREFIX_MAP[prefix]}/{param}"
    except (ValueError, KeyError) as exc:
        raise KeyError(f"unknown old-model label {label!r}") from exc


def scene_to_old(name: str) -> str:
    """'planes/0/light/2/center_x' -> 'LL2.center_x' (raises on unknown)."""
    site, _, param = name.rpartition("/")
    old = _PREFIX_INV.get(site)
    if old is None:
        raise KeyError(f"scene param name {name!r} is not a mapped 46-dim site")
    return f"{old}.{param}"


SCENE_NAMES_46: List[str] = [old_to_scene(lab) for lab in OLD_LABELS_46]


def audit_roundtrip() -> None:
    """Bijection sanity: old->scene->old is the identity on all 46 labels."""
    for lab in OLD_LABELS_46:
        back = scene_to_old(old_to_scene(lab))
        if back != lab:
            raise AssertionError(f"label roundtrip broke: {lab} -> {back}")
    if len(set(SCENE_NAMES_46)) != 46:
        raise AssertionError("scene names are not unique")


# --------------------------------------------------------------------------- #
# value containers
# --------------------------------------------------------------------------- #
def unique_from_old(labeled: Mapping[str, float], *, ie_scale: float = 1.0,
                    amp_consts: Mapping[str, float] | None = None) -> Dict[str, np.ndarray]:
    """Old-labeled constrained values -> scene unique dict (z_param_names keys).

    Args:
        labeled: {old_label: value} covering all 46 labels.
        ie_scale: multiply every '.Ie' value (reconciliation #3;
            1.0 for the marg view, 1/conversion_factor for the forward view).
        amp_consts: ignored here (amps are scene CONSTANTS, not free params);
            present so call sites can document the forward-view amps flow.

    Returns dict keyed by scene unique keys, float64 scalars.
    """
    out: Dict[str, np.ndarray] = {}
    for lab in OLD_LABELS_46:
        v = float(labeled[lab])
        if lab.endswith(".Ie"):
            v = v * float(ie_scale)
        out[old_to_scene(lab)] = np.float64(v)
    return out


def old_from_unique(unique: Mapping[str, np.ndarray], *,
                    ie_scale: float = 1.0) -> Dict[str, float]:
    """Scene unique dict -> {old_label: value}; ie_scale UNDOES reconciliation
    #3 (pass the same ie_scale used in `unique_from_old`)."""
    out: Dict[str, float] = {}
    for lab in OLD_LABELS_46:
        v = float(np.asarray(unique[old_to_scene(lab)]).reshape(-1)[0])
        if lab.endswith(".Ie"):
            v = v / float(ie_scale)
        out[lab] = v
    return out


def vec46_from_labeled(labeled: Mapping[str, float]) -> np.ndarray:
    """{old_label: value} -> (46,) float64 in OLD_LABELS_46 canonical order."""
    return np.array([float(labeled[lab]) for lab in OLD_LABELS_46], dtype=np.float64)


def labeled_from_vec46(vec: Iterable[float]) -> Dict[str, float]:
    vec = np.asarray(list(vec), dtype=np.float64)
    if vec.shape != (46,):
        raise ValueError(f"expected (46,), got {vec.shape}")
    return {lab: float(v) for lab, v in zip(OLD_LABELS_46, vec)}


def grad_old_from_scene(grad_unique: Mapping[str, np.ndarray], *,
                        ie_scale: float = 1.0) -> np.ndarray:
    """Scene-side gradient dict (d logL / d unique) -> (46,) old-label order.

    Chain rule for reconciliation #3: x_scene = ie_scale * x_old on '.Ie'
    leaves, so dL/dx_old = ie_scale * dL/dx_scene there.
    """
    out = np.empty(46, dtype=np.float64)
    for k, lab in enumerate(OLD_LABELS_46):
        g = float(np.asarray(grad_unique[old_to_scene(lab)]).reshape(-1)[0])
        if lab.endswith(".Ie"):
            g = g * float(ie_scale)
        out[k] = g
    return out


# --------------------------------------------------------------------------- #
# flat-z helpers (scene side; z_param_names-keyed ONLY)
# --------------------------------------------------------------------------- #
def scene_z_from_old_labels(model, labeled: Mapping[str, float], *,
                            ie_scale: float = 1.0) -> np.ndarray:
    """Old-labeled constrained values -> the scene model's flat z.

    Goes through model.unconstrained(to_params(unique)) so the SCENE bijector
    owns its own z packing; we never touch column order.
    """
    unique = unique_from_old(labeled, ie_scale=ie_scale)
    params = model.to_params({k: np.float64(v) for k, v in unique.items()})
    return np.asarray(model.unconstrained(params), dtype=np.float64)


def audit_scene_names(model) -> Tuple[List[str], List[str]]:
    """Check the scene model's z_param_names cover exactly the mapped 46 sites
    (free params). Returns (missing, extra) — both empty on a clean marg-view
    model; the forward view legitimately has NO extras either (amps are
    constants, hence absent from z_param_names)."""
    names = list(getattr(model, "z_param_names", []) or [])
    missing = [n for n in SCENE_NAMES_46 if n not in names]
    extra = [n for n in names if n not in SCENE_NAMES_46]
    return missing, extra
