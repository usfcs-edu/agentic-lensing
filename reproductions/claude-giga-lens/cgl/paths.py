"""Canonical paths + vendor bootstrap for the claude-giga-lens campaign.

Rule (README §"copy vs import"): code we modify is COPIED with attribution;
artifacts and the frozen library are IMPORTED BY PATH from their home
reproductions. Bulk data is never duplicated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parent.parent          # reproductions/claude-giga-lens
REPRODUCTIONS = REPRO.parent                             # reproductions/
DATA = REPRO / "data"
FIGS = REPRO / "figs"

# Sibling reproductions we import artifacts from (read-only).
FOUNDRY_I = REPRODUCTIONS / "foundry-i"
FOUNDRY_I_DATA = FOUNDRY_I / "data"
GU2022 = REPRODUCTIONS / "gu-2022"
EUCLID_Q1_DATA = REPRODUCTIONS / "euclid-q1" / "data"

# Vendored gigalens-sean (multinode-2025). Same ref foundry-i vendored; UNPATCHED.
VENDOR = REPRO / "vendor" / "gigalens-sean"
VENDOR_SRC = VENDOR / "src"
EXPECTED_VENDOR_REF = "58ec9a7db45c53c49f4994ad4f9659f708346763"

# foundry-i data products (the real HST system DESI-165.4754-06.0423).
CUTOUT_F140W = FOUNDRY_I_DATA / "cutout_F140W.fits"      # raw SCI/WHT cutout (marg-parity input)
EMPIRICAL_PSF = FOUNDRY_I_DATA / "empirical_psf.npy"     # PSF used by _hmc_lib_marg (legacy convention)
NEARBY_GALAXY_LOC = FOUNDRY_I_DATA / "nearby_galaxy_loc.npz"  # LL2/LL3 prior centers (arcsec)
MAP_REFINED_LSTSQ64 = FOUNDRY_I_DATA / "map_refined_lstsq64.npz"  # 41-dim lstsq start + amps
GU2022_MOCK000 = GU2022 / "data" / "mocks" / "system_000.npz"     # forward-image cross-check
CUTOUT_V2D = FOUNDRY_I_DATA / "cutout_v2d.npz"   # native 0.13", 80x80, ss2 (headline)
CUTOUT_V3 = FOUNDRY_I_DATA / "cutout_v3.npz"     # fine 0.04", 260x260, ss1
CUTOUT_V3B = FOUNDRY_I_DATA / "cutout_v3b.npz"   # binned 0.08", 130x130, ss2
MODEL_MAP_V3COLD = FOUNDRY_I_DATA / "model_map_v3cold.npy"       # for model-subtracted ACF
MODEL_MAP_V3B_COLD = FOUNDRY_I_DATA / "model_map_v3b_cold.npy"
MAP_MARG_PD = FOUNDRY_I_DATA / "map_marg_pd.npz"                 # 46-dim PD mode (parity ref)
HESS_MARG_PD = FOUNDRY_I_DATA / "hess_marg_pd.npz"
HMC_V13_V3B = FOUNDRY_I_DATA / "hmc_v13_v3b.npz"                 # T3 bimodal reference chains

# --- P2a additions (zoo/benchmark; constants only, per the P0 freeze) --------
GU2022_MOCKS = GU2022 / "data" / "mocks"                 # system_XXX.npz (12 mocks)
GU2022_FITS = GU2022 / "data" / "fits"                   # system_XXX_fit.npz (T1 reference)
SVI_V12_V3BR = FOUNDRY_I_DATA / "svi_v12_v3br.npz"       # T3 SVI loc/cov (paper-scale run input)
MAP_V11_V3B_COLD = FOUNDRY_I_DATA / "map_v11_v3b_cold.npz"     # 74-dim MAP, gamma~1.465
MAP_V11_V3B_COLD2D = FOUNDRY_I_DATA / "map_v11_v3b_cold2d.npz"  # 74-dim MAP, gamma~1.369
MAP_V11_V3B_WARM = FOUNDRY_I_DATA / "map_v11_v3b_warm.npz"     # 74-dim MAP, gamma~2.672
LONG_DIAGRAW = tuple(FOUNDRY_I_DATA / f"long_diagraw_s{i}.npz" for i in range(8))
                                                          # T2 long-chain reference (8 seeds)
RESULTS = DATA / "results"                                # per-cell benchmark results (gitignored)
ZOO_FREEZE = DATA / "zoo_freeze.json"                     # THE zoo freeze artifact (20_build_zoo)
ZOO_FREEZE_PARTS = DATA / "zoo_freeze.parts"              # per-target worker outputs
ZOO_VALIDATION = DATA / "zoo_validation.json"             # 21_validate_zoo report
PARITY_REPORT = DATA / "parity_report.json"               # P0 gate evidence (stored T2 logp)


def bootstrap_vendor() -> None:
    """Put the vendored gigalens-sean first on sys.path (shadows any pip install).

    Asserts the vendored ref so a silent vendor change fails loudly.
    Must run before any ``import gigalens.*``.
    """
    ref_file = VENDOR / "VENDORED_REF.txt"
    ref = ref_file.read_text().strip()
    if ref != EXPECTED_VENDOR_REF:
        raise RuntimeError(
            f"vendored gigalens-sean ref mismatch: {ref!r} != {EXPECTED_VENDOR_REF!r} "
            f"({ref_file}); the campaign's parity claims are void on a different ref"
        )
    p = str(VENDOR_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_product(npz_path: Path) -> dict:
    """Load a foundry-i Stage-A data product (img/err_map/keep_mask/psf/meta).

    Same layout as foundry-i `_data_lib.load_v2`, but float64-preserving:
    dtype conversion is the caller's decision (real-data likelihoods are f64).
    """
    z = np.load(npz_path)
    meta = json.loads(str(z["meta"]))
    return dict(
        img=np.asarray(z["img"]),
        err_map=np.asarray(z["err_map"]),
        keep_mask=np.asarray(z["keep_mask"]).astype(bool),
        psf=np.asarray(z["psf"]),
        meta=meta,
    )
