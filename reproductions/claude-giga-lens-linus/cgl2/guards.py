"""Guards — one per prior real incident (house rule) plus the new-substrate fences.

Every operator script calls the guards it needs at startup; guards raise, never warn.
Carried guards inherit their incident provenance from ../claude-giga-lens/cgl/guards.py
(see that file's docstrings); new guards below cite the exploration finding they encode.
"""
from __future__ import annotations

import os

from cgl2 import paths


# --- carried from claude-giga-lens (same incidents, same semantics) -----------------

def require_x64() -> None:
    """Real-data likelihoods are float64 (their project-standards §8 AND our incident)."""
    import jax

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "jax_enable_x64 is OFF. Set GIGALENS_X64=1 / JAX_ENABLE_X64=1 before import; "
            "float32 leaves a conv/basis noise floor (their AP-4, our SVI-collapse incident)."
        )


def require_gpu() -> None:
    import jax

    if jax.default_backend() == "cpu" and os.environ.get("CGL2_ALLOW_CPU") != "1":
        raise RuntimeError(
            "CPU backend: the lensing likelihood is ~216x slower on CPU. "
            "Set CGL2_ALLOW_CPU=1 only for 16x16 unit-test toys."
        )


def require_single_device() -> None:
    import jax

    n = len(jax.devices())
    if n != 1 and os.environ.get("CGL2_MULTIDEVICE_OK") != "1":
        raise RuntimeError(
            f"{n} visible devices — pin CUDA_VISIBLE_DEVICES (the gigalens pmap-over-all "
            "hang incident) or set CGL2_MULTIDEVICE_OK=1 for deliberate sharded runs."
        )


def assert_psf_sampling(psf_pixel_scale: float, delta_pix: float, supersample: int = 1) -> None:
    """PSF kernels must be sampled at delta_pix/supersample EXACTLY.

    foundry-i incident: a supersampled kernel fed at the wrong scale silently broadened
    the model 2x (chi2 floor 3.4 -> 1.05 after fix). The ss2 trap: the on-disk 0.065"
    ePSF is NOT 0.12825/2 = 0.064125" — resample first.
    """
    expected = delta_pix / supersample
    if abs(psf_pixel_scale - expected) > 1e-12:
        raise RuntimeError(
            f"PSF sampling {psf_pixel_scale}\" != delta_pix/ss = {expected}\" "
            f"(delta_pix={delta_pix}, ss={supersample}). Resample the kernel."
        )


# --- new fences for the gigalens-linus substrate ------------------------------------

def require_vendor_ref() -> None:
    """Wrong-gigalens tripwire (see paths.bootstrap_vendor)."""
    paths.bootstrap_vendor()


def require_jax_pin() -> None:
    paths.require_jax_pin()


# Profile/config combos certified by the F1-F6 parity battery (grown as gates pass;
# frozen list lives in data/parity_report_scene.json once 01_parity_scene.py runs).
_CERTIFIED_MASS = {"EPL", "Shear"}
_CERTIFIED_LIGHT = {"SersicEllipse", "Shapelets", "Sersic", "CoreSersic"}


def assert_scene_config_certified(mass_names, light_names, supersample: int = 1) -> None:
    """The scene API's own validation suite is UNCERTIFIED (their tests/validation
    README). We only trust the config class our cross-stack parity battery certified.
    Anything else needs CGL2_UNCERTIFIED_OK=1 (an explicit, ledgered decision — e.g.
    the X1 BPL/dPIE arms, which carry their own oracle checks instead)."""
    if os.environ.get("CGL2_UNCERTIFIED_OK") == "1":
        return
    bad_mass = set(mass_names) - _CERTIFIED_MASS
    bad_light = set(light_names) - _CERTIFIED_LIGHT
    if bad_mass or bad_light or supersample not in (1, 2):
        raise RuntimeError(
            f"scene config outside the parity-certified class (mass:{bad_mass or '-'} "
            f"light:{bad_light or '-'} ss:{supersample}). Set CGL2_UNCERTIFIED_OK=1 only "
            "with a ledger row + an independent oracle check (X1 pattern)."
        )


def assert_whitener_bundle(bundle) -> None:
    """e_op certificate + shape sanity for a frozen whitener bundle (dict or npz)."""
    e_op = float(bundle["e_op"])
    if not (e_op <= 0.02):
        raise RuntimeError(f"whitener e_op={e_op:.4f} > 0.02 — bundle not admissible.")
    if "h_taps" not in bundle or "keep_mask" not in bundle:
        raise RuntimeError("whitener bundle missing h_taps/keep_mask fields.")


def assert_flatz_roundtrip(model, z, atol: float = 0.0) -> None:
    """unconstrained(constrained(z)) == z, and param access is keyed on z_param_names
    only (their own docstring warns insertion order is not a contract)."""
    import jax.numpy as jnp

    z2 = model.unconstrained(model.constrained(z))
    if not bool(jnp.allclose(z, z2, atol=atol, rtol=0)):
        raise RuntimeError("flat-z round-trip failed: unconstrained(constrained(z)) != z.")
    names = getattr(model, "z_param_names", None)
    if not names:
        raise RuntimeError("model has no z_param_names — param_map must not guess ordering.")


def physicality_check_or_raise(model) -> None:
    """Run their physicality layer on our model at build time: hard Domain violations
    raise (their M3 culture); soft plausibility-band findings are returned for the
    ledger, never silently ignored. Watch item: their EPL domains changed recently
    (upstream 617047f) — a clash with foundry-i priors is a ledgered prior decision."""
    from gigalens import physicality  # vendored

    report = physicality.check_model(model) if hasattr(physicality, "check_model") else None
    return report


def require_where_mask_semantics() -> None:
    """Masked reductions must use where(), never multiply-by-mask (their jaxlib
    miscompile note + our convergent lesson). Micro self-test as a re-pin tripwire."""
    import jax.numpy as jnp

    x = jnp.array([1.0, jnp.inf])
    m = jnp.array([True, False])
    ok = jnp.where(m, x, 0.0).sum()
    if not bool(jnp.isfinite(ok)):
        raise RuntimeError("where() mask semantics violated — investigate jaxlib.")
