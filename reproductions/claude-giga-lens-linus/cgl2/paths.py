"""Canonical paths + vendor bootstrap for the claude-giga-lens-linus campaign.

Import contract: `import cgl2.paths` FIRST in every operator script. The bootstrap
asserts the vendored scene-API ref and refuses to run against the wrong gigalens
(the blessed venv's 2.0 checkout, the live lensing-repos clone, or a drifted vendor).

Copy-vs-import rule (README §conventions): everything referenced here by path is
FROZEN and imported read-only; code we modify is copied into cgl2/ with attribution.
"""
from __future__ import annotations

import os
from pathlib import Path

CGL2_ROOT = Path(__file__).resolve().parent.parent
REPRO_ROOT = CGL2_ROOT.parent

# --- the pinned substrate ---------------------------------------------------------
VENDOR_DIR = CGL2_ROOT / "vendor" / "gigalens-linus"
VENDORED_REF_FILE = VENDOR_DIR / "VENDORED_REF.txt"
# Full SHA of gigalens-linus @ linusu-dev-merge vendored at P0 (2026-07-15).
# Re-pin ONLY via the ledgered procedure in plans/PLAN.md §3.
EXPECTED_VENDOR_REF = "80916d24f3e616edecf9fb66b041c716fa111c29"

# --- the previous campaign (import by path, never copy data) -----------------------
CGL1_ROOT = REPRO_ROOT / "claude-giga-lens"
CGL1_DATA = CGL1_ROOT / "data"                      # whitener bundles, kernels, results
CGL1_PKG = CGL1_ROOT                                # `cgl` package root (old validated stack)
OLD_VENDOR_DIR = CGL1_ROOT / "vendor" / "gigalens-sean"   # 58ec9a7, validated
OLD_VENDOR_REF = "58ec9a7db45c53c49f4994ad4f9659f708346763"

# --- foundry-i real-data products (import by path) ---------------------------------
FOUNDRY_ROOT = REPRO_ROOT / "foundry-i"
FOUNDRY_DATA = FOUNDRY_ROOT / "data"                # cutout_v2d/v3/v3b.npz, ePSFs, warm starts

# --- the team's research repo (READ-ONLY reference; never imported as code) --------
GIGALENS_CODE = Path("/raid/benson/lensing-repos/GIGALens-Code")

# --- campaign artifact dirs ---------------------------------------------------------
DATA_DIR = CGL2_ROOT / "data"                        # gitignored bulk artifacts
FIGS_DIR = CGL2_ROOT / "figs"

_JAX_PIN = "0.6.2"


def bootstrap_vendor() -> None:
    """Assert the vendored scene API is the one being imported, at the pinned ref.

    Raises RuntimeError (never warns) on: missing vendor, ref mismatch, or a
    `gigalens` import that resolves outside the vendor tree.
    """
    if not VENDORED_REF_FILE.exists():
        raise RuntimeError(
            f"vendored gigalens-linus missing at {VENDOR_DIR} — run the P0 vendor step "
            "(git archive @ EXPECTED_VENDOR_REF), see plans/PLAN.md §3."
        )
    ref = VENDORED_REF_FILE.read_text().strip()
    if ref != EXPECTED_VENDOR_REF:
        raise RuntimeError(
            f"VENDORED_REF mismatch: found {ref}, expected {EXPECTED_VENDOR_REF}. "
            "If this is a deliberate re-pin, follow the ledgered re-pin procedure "
            "(plans/PLAN.md §3) and update EXPECTED_VENDOR_REF in the same commit."
        )
    import gigalens  # noqa: deferred import so the check governs first use

    got = Path(gigalens.__file__).resolve()
    if VENDOR_DIR.resolve() not in got.parents:
        raise RuntimeError(
            f"`import gigalens` resolved OUTSIDE the pinned vendor: {got}. "
            "You are probably in the wrong venv (blessed gigalens 2.0, or the live "
            "lensing-repos checkout). Use /raid/benson/.venvs/cgl2."
        )


def require_jax_pin() -> None:
    import jax

    if jax.__version__ != _JAX_PIN:
        raise RuntimeError(
            f"jax=={jax.__version__} but the campaign pin is {_JAX_PIN} "
            "(cross-stack 1e-12 parity + aarch64 XLA workarounds are validated only "
            "there; a nightly-env exception must be a ledgered decision, gate F8)."
        )
