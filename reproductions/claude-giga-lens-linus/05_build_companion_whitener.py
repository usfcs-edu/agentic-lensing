"""05_build_companion_whitener.py — T0.3 companion-mask discriminator input (P1).

Builds the VARIANT whitener bundle for the v3b companion-mask discriminator:
whiten-then-drop. The whitening kernel h (and therefore e_op, logdet_per_pix,
rho_kernel — all kernel properties) is copied VERBATIM from the frozen
production bundle `../claude-giga-lens/data/whitener_v3b.npz`; the ONLY change
is the whitened-domain keep-mask, which is additionally eroded by the
companion-galaxy region so that no whitened pixel whose (2M+1)^2 kernel
support touches the companion disk contributes to the likelihood.

Companion geometry (REUSED from the X1-G0 free-check, cited):
  research/x1_g0_mechanism_check.md + data/x1_g0_effective_radii.json —
  "companion galaxy at (-2.34, -2.86)\" (model LL2/LL3 = lens light, not
  lensed source) excised to 1.2\"". The center is the LL2/LL3 prior center
  from foundry-i/data/nearby_galaxy_loc.npz (arcsec_x=-2.34, arcsec_y=-2.86);
  excision radius 1.2 arcsec (circular), as in X1-G0's arc-map excision.

Grid convention (VALIDATED in-session against cutout_v3b.npz brightness):
  mean-centered gigalens grid, x = (col-(N-1)/2)*delta, y = (row-(N-1)/2)*delta,
  no axis flip; the brightest off-center blob sits at (row 29, col 35) =
  (-2.36, -2.84)" matching the companion center to sub-pixel.

Erosion semantics (cgl.whiten.erode_keep, imported by path from the OLD
validated campaign): keep_w_new = erode_keep(keep_mask & ~disk, M). By
erosion/dilation duality this equals keep_w_old & ~dilate(disk, (2M+1)^2),
which is asserted. e_op is asserted byte-identical (kernel unchanged).

Runs on CPU in the cgl venv (/raid/benson/.venvs/cgl): numpy+scipy only.

Output: data/whitener_v3b_companion_eroded.npz (all original keys; keep_w
replaced; meta JSON extended with a companion_mask provenance section).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "claude-giga-lens"
sys.path.insert(0, str(OLD))                      # import cgl.whiten by path
from cgl.whiten import erode_keep                 # noqa: E402
from scipy.ndimage import binary_dilation         # noqa: E402

SRC_WHITENER = OLD / "data" / "whitener_v3b.npz"
CUTOUT_V3B = HERE.parent / "foundry-i" / "data" / "cutout_v3b.npz"
NEARBY = HERE.parent / "foundry-i" / "data" / "nearby_galaxy_loc.npz"
OUT = HERE / "data" / "whitener_v3b_companion_eroded.npz"

R_EXCISE_ARCSEC = 1.2      # X1-G0 companion excision radius (cited above)


def main():
    wz = np.load(SRC_WHITENER, allow_pickle=True)
    keys = list(wz.keys())
    h = np.asarray(wz["h"], dtype=np.float64)
    keep_w_old = np.asarray(wz["keep_w"]).astype(bool)
    M = int(wz["M"])
    e_op = float(wz["e_op"])

    cz = np.load(CUTOUT_V3B)
    keep_mask = np.asarray(cz["keep_mask"]).astype(bool)
    meta_c = json.loads(str(cz["meta"]))
    N = keep_mask.shape[0]
    delta = float(meta_c["delta_pix"])
    assert (N, delta) == (130, 0.08), (N, delta)

    nb = np.load(NEARBY)
    cx, cy = float(nb["arcsec_x"]), float(nb["arcsec_y"])
    assert (cx, cy) == (-2.34, -2.8600000000000003), (cx, cy)

    # mean-centered grid, no flip (validated in-session; see docstring)
    c0 = (N - 1) / 2.0
    rows, cols = np.mgrid[0:N, 0:N]
    x = (cols - c0) * delta
    y = (rows - c0) * delta
    disk = (x - cx) ** 2 + (y - cy) ** 2 < R_EXCISE_ARCSEC ** 2

    # production keep_w reproduction gate (same check as ledger gate W)
    assert np.array_equal(erode_keep(keep_mask, M), keep_w_old), \
        "on-disk keep_w != erode_keep(keep_mask, M): wrong inputs"

    keep_w_new = erode_keep(keep_mask & ~disk, M)

    # duality cross-check: erode(keep & ~disk) == erode(keep) & ~dilate(disk)
    struct = np.ones((2 * M + 1, 2 * M + 1), dtype=bool)
    alt = keep_w_old & ~binary_dilation(disk, structure=struct)
    assert np.array_equal(keep_w_new, alt), "duality cross-check failed"
    assert not np.any(keep_w_new & ~keep_w_old), "new mask must be a subset"
    n_old, n_new = int(keep_w_old.sum()), int(keep_w_new.sum())
    n_disk = int(disk.sum())

    meta = json.loads(str(wz["meta"]))
    meta["companion_mask"] = dict(
        variant="whiten-then-drop (keep_w eroded by companion region)",
        source_bundle=str(SRC_WHITENER),
        center_arcsec=[cx, cy],
        center_source=str(NEARBY) + " (LL2/LL3 prior center)",
        radius_arcsec=R_EXCISE_ARCSEC,
        geometry_citation=("claude-giga-lens-linus X1-G0: "
                           "research/x1_g0_mechanism_check.md + "
                           "data/x1_g0_effective_radii.json"),
        grid="mean-centered, x=(col-64.5)*0.08, y=(row-64.5)*0.08, no flip",
        erosion="erode_keep(keep_mask & ~disk, M) == keep_w_old & ~dilate(disk,(2M+1)^2)",
        n_disk_px=n_disk, n_keep_w_old=n_old, n_keep_w_new=n_new,
        n_dropped=n_old - n_new,
        kernel_unchanged=True, e_op_unchanged=True,
    )

    out = {}
    for k in keys:
        out[k] = wz[k]
    out["keep_w"] = keep_w_new
    out["meta"] = json.dumps(meta)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)

    # re-load + validate exactly what e2.build_target reads
    rz = np.load(OUT, allow_pickle=True)
    assert np.array_equal(np.asarray(rz["h"], dtype=np.float64), h)
    assert int(rz["M"]) == M
    assert float(rz["e_op"]) == e_op
    assert np.array_equal(np.asarray(rz["keep_w"]).astype(bool), keep_w_new)
    assert np.array_equal(np.asarray(rz["rho_kernel"]),
                          np.asarray(wz["rho_kernel"]))
    assert float(rz["logdet_per_pix"]) == float(wz["logdet_per_pix"])

    print(f"[T0.3] wrote {OUT}")
    print(f"  kernel h {h.shape} + e_op={e_op:.12g} + M={M}: UNCHANGED (asserted)")
    print(f"  companion disk r<{R_EXCISE_ARCSEC}\" @ ({cx},{cy})\": {n_disk} px")
    print(f"  keep_w: {n_old} -> {n_new} (dropped {n_old - n_new}, "
          f"{100.0 * (n_old - n_new) / n_old:.1f}% of whitened dof)")


if __name__ == "__main__":
    main()
