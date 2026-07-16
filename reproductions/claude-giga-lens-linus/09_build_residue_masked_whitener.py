"""09_build_residue_masked_whitener.py — T1.1b residue-masked whitener build (+ STOP gate).

T1.1 was CONFOUNDED by bright-object scene-subtraction residue carried in the
injection datasets' real residual field (research/t11_injection_recovery.md;
G1 decomposition in data/t11_injection_build_report.json + CAMPAIGN.md T1.1
checkpoint: sky chi2_pp 0.952 / faint 0.973 / bright-object 2.71 / center
r<1.2" 5.46). T1.1b fix attempt = FIT-side masking: whiten-then-drop the
residue-dominated regions so the likelihood sees only the sky-dominated field
where the injection is clean. Mirrors 05_build_companion_whitener.py: the
whitening kernel h (and therefore e_op, logdet_per_pix, rho_kernel — all
kernel properties) is copied VERBATIM from the frozen production bundle; ONLY
the whitened-domain keep-mask shrinks, eroded so no whitened pixel whose
(2M+1)^2 kernel support touches the residue region contributes.

Residue regions (definitions PINNED by exact reproduction of the ledgered
decomposition numbers — gate R below):
    faint        = |img| < 5 * median(err_map)          (06_build_injections.py)
    bright-object = keep & ~faint                        (chi2_pp 2.712 -> "2.71")
    center        = keep & (r_arc < 1.2")                (chi2_pp 5.464 -> "5.46")
    drop          = (~faint) | (r_arc < 1.2)             (bright-object U center)
r_arc = mean-centered radius * 0.08"/px (06 convention); img = REAL cutout_v3b
image (the residue lives where the real scene was bright).

PRE-DECLARED STOP GATE (T1.1b tasking, threshold fixed BEFORE this build ran):
if > 40% of the production whitener's kept whitened dof would be lost, STOP —
the experiment would be information-starved; that is itself a finding about
injection methodology. On STOP this script writes the report JSON and NO
whitener bundles, and exits nonzero (06's do-not-stage/submit pattern).

Runs on CPU in the OLD cgl venv (/raid/benson/.venvs/cgl): numpy+scipy only.
  OPENBLAS_NUM_THREADS=4 /raid/benson/.venvs/cgl/bin/python \
  09_build_residue_masked_whitener.py

Outputs:
  data/t11b_residue_mask_report.json            (always)
  data/whitener_v3b_residue_eroded.npz          (only if the STOP gate passes)
  data/whitener_v3b_delta_residue.npz           (only if the STOP gate passes;
      diagonal-control bundle: h=[[1]], M=0, keep_w = keep & ~drop — same
      residue drop through the G4-gated delta-whitener code path)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "claude-giga-lens"
sys.path.insert(0, str(OLD))                      # import cgl.whiten by path
from cgl.whiten import erode_keep                 # noqa: E402
from scipy.ndimage import binary_dilation         # noqa: E402

SRC_WHITENER = OLD / "data" / "whitener_v3b.npz"
CUTOUT_V3B = HERE.parent / "foundry-i" / "data" / "cutout_v3b.npz"
MODEL_MAP = HERE.parent / "foundry-i" / "data" / "model_map_v3b_cold.npy"
BUILD_REPORT = HERE / "data" / "t11_injection_build_report.json"
OUT_CORR = HERE / "data" / "whitener_v3b_residue_eroded.npz"
OUT_DELTA = HERE / "data" / "whitener_v3b_delta_residue.npz"
OUT_REPORT = HERE / "data" / "t11b_residue_mask_report.json"

R_CENTER = 1.2          # arcsec (ARC_BAND[0], 06 convention)
ARC_BAND = (1.2, 4.2)   # arcsec, x1_g0 convention (diagnostics only)
STOP_LOSS_FRAC = 0.40   # PRE-DECLARED in the T1.1b tasking, frozen

# Ledgered decomposition references (CAMPAIGN.md T1.1 checkpoint, G1
# restatement; sky value exact from t11_injection_build_report.json).
REF = dict(sky=0.9522393030717049, faint=0.973, bright=2.71, center=5.46)
REF_TOL = dict(sky=1e-9, faint=0.005, bright=0.005, center=0.005)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    t0 = time.time()
    report = dict(script="09_build_residue_masked_whitener.py",
                  generated=time.strftime("%F %T"), inputs={}, gates={})
    for name, p in dict(whitener_v3b=SRC_WHITENER, cutout_v3b=CUTOUT_V3B,
                        model_map_v3b_cold=MODEL_MAP,
                        t11_injection_build_report=BUILD_REPORT).items():
        report["inputs"][name] = dict(path=str(p), md5=md5(p))

    wz = np.load(SRC_WHITENER, allow_pickle=True)
    keys = list(wz.keys())
    h = np.asarray(wz["h"], dtype=np.float64)
    keep_w_old = np.asarray(wz["keep_w"]).astype(bool)
    M = int(wz["M"])
    e_op = float(wz["e_op"])

    cz = np.load(CUTOUT_V3B)
    img = np.asarray(cz["img"], dtype=np.float64)
    err = np.asarray(cz["err_map"], dtype=np.float64)
    keep = np.asarray(cz["keep_mask"]).astype(bool)
    meta_c = json.loads(str(cz["meta"]))
    N = keep.shape[0]
    delta = float(meta_c["delta_pix"])
    assert (N, delta) == (130, 0.08), (N, delta)
    model_map = np.asarray(np.load(MODEL_MAP), dtype=np.float64)
    residual = img - model_map

    # ---- regions (06_build_injections.py conventions) --------------------- #
    yy, xx = np.indices(img.shape)
    cen = (N - 1) / 2.0
    r_arc = np.hypot(xx - cen, yy - cen) * delta
    faint = np.abs(img) < 5 * np.median(err)
    sky = keep & (r_arc > 4.5) & faint
    bright = keep & ~faint
    center = keep & (r_arc < R_CENTER)
    drop = (~faint) | (r_arc < R_CENTER)          # bright-object U center
    band = (r_arc >= ARC_BAND[0]) & (r_arc <= ARC_BAND[1])

    # ---- gate R: pin region definitions by reproducing ledgered chi2_pp --- #
    w = residual / err
    got = dict(sky=float(np.mean(w[sky] ** 2)),
               faint=float(np.mean(w[keep & faint] ** 2)),
               bright=float(np.mean(w[bright] ** 2)),
               center=float(np.mean(w[center] ** 2)))
    r_ok = all(abs(got[k] - REF[k]) <= REF_TOL[k] for k in REF)
    for k in REF:
        print(f"gate R [{k:6s}] chi2_pp = {got[k]:.6f}  (ledgered {REF[k]}, "
              f"tol {REF_TOL[k]:g}) {'OK' if abs(got[k]-REF[k])<=REF_TOL[k] else 'FAIL'}")
    report["gates"]["R_region_reproduction"] = dict(
        measured=got, ledgered=REF, tol=REF_TOL, ok=bool(r_ok),
        region_defs=dict(
            faint="|img| < 5*median(err_map)  (img = REAL cutout_v3b image)",
            bright_object="keep & ~faint",
            center=f"keep & (r_arc < {R_CENTER})",
            drop="(~faint) | (r_arc < 1.2)   [full grid; bright-object U center]",
            grid="mean-centered, r_arc = hypot(col-64.5, row-64.5)*0.08"))
    assert r_ok, "region definitions do not reproduce the ledgered decomposition"
    # note: center is a subset of bright-object here (all r<1.2 keep pixels
    # are non-faint) — asserted so the union statement is honest
    assert not np.any(center & faint), "center unexpectedly contains faint px"

    # ---- gate P: production keep_w reproduction (same as gate W / 05) ------ #
    p_ok = np.array_equal(erode_keep(keep, M), keep_w_old)
    report["gates"]["P_production_keep_w"] = dict(ok=bool(p_ok))
    assert p_ok, "on-disk keep_w != erode_keep(keep_mask, M): wrong inputs"

    # ---- the residue-eroded whitened keep-mask ---------------------------- #
    keep_w_new = erode_keep(keep & ~drop, M)
    struct = np.ones((2 * M + 1, 2 * M + 1), dtype=bool)
    alt = keep_w_old & ~binary_dilation(drop, structure=struct)
    assert np.array_equal(keep_w_new, alt), "duality cross-check failed"
    assert not np.any(keep_w_new & ~keep_w_old), "new mask must be a subset"

    n_old, n_new = int(keep_w_old.sum()), int(keep_w_new.sum())
    loss_frac = 1.0 - n_new / n_old
    diag_keep = keep & ~drop                       # delta bundle keep (M=0)
    n_diag_old, n_diag_new = int(keep.sum()), int(diag_keep.sum())
    # diagnostics: where would the information come from?
    arc_old, arc_new = int((keep_w_old & band).sum()), int((keep_w_new & band).sum())
    no_halo = int((keep_w_old & ~drop).sum())      # attribution only (invalid
    #                                                for corr whitening)
    rs = r_arc[keep_w_new]
    report["dof"] = dict(
        M=M, kernel_support=f"{2*M+1}x{2*M+1}",
        corr_keep_w_old=n_old, corr_keep_w_new=n_new,
        corr_loss=n_old - n_new, corr_loss_frac=loss_frac,
        diag_keep_old=n_diag_old, diag_keep_new=n_diag_new,
        diag_loss_frac=1.0 - n_diag_new / n_diag_old,
        arc_band_keep_w_old=arc_old, arc_band_keep_w_new=arc_new,
        attribution_region_drop_only_no_halo=dict(
            keep_w=no_halo, loss_frac=1.0 - no_halo / n_old,
            note="drop w/o the (2M+1)^2 kernel-support erosion — NOT a valid "
                 "correlated whitening; shows the region alone already "
                 "exceeds the STOP threshold"),
        survivors_r_arcsec=dict(min=float(rs.min()), median=float(np.median(rs)),
                                max=float(rs.max())),
        center_only_variant_loss_frac=float(
            1.0 - erode_keep(keep & ~(r_arc < R_CENTER), M).sum() / n_old))
    print(f"corr keep_w: {n_old} -> {n_new}  "
          f"(loss {n_old-n_new} = {100*loss_frac:.1f}% of whitened dof)")
    print(f"diag keep  : {n_diag_old} -> {n_diag_new}  "
          f"({100*(1-n_diag_new/n_diag_old):.1f}%)")
    print(f"arc-band (1.2-4.2\") whitened px: {arc_old} -> {arc_new}; "
          f"survivors at r {rs.min():.2f}-{rs.max():.2f}\" (median "
          f"{np.median(rs):.2f}\")")
    print(f"attribution: region drop alone (no {2*M+1}x{2*M+1} halo) already "
          f"loses {100*(1-no_halo/n_old):.1f}%")

    # ---- PRE-DECLARED STOP GATE ------------------------------------------- #
    stopped = loss_frac > STOP_LOSS_FRAC
    report["gates"]["STOP_dof_starvation"] = dict(
        threshold_loss_frac=STOP_LOSS_FRAC,
        threshold_provenance="T1.1b tasking, pre-declared before build",
        measured_loss_frac=loss_frac, stopped=bool(stopped))
    report["verdict"] = ("STOPPED_INFO_STARVED" if stopped else "ADMISSIBLE")
    report["wall_s"] = time.time() - t0

    if not stopped:
        meta = json.loads(str(wz["meta"]))
        meta["residue_mask"] = dict(
            variant="whiten-then-drop (keep_w eroded by residue regions)",
            source_bundle=str(SRC_WHITENER),
            regions=report["gates"]["R_region_reproduction"]["region_defs"],
            n_keep_w_old=n_old, n_keep_w_new=n_new,
            kernel_unchanged=True, e_op_unchanged=True)
        out = {k: wz[k] for k in keys}
        out["keep_w"] = keep_w_new
        out["meta"] = json.dumps(meta)
        np.savez(OUT_CORR, **out)
        rz = np.load(OUT_CORR, allow_pickle=True)
        assert np.array_equal(np.asarray(rz["h"], dtype=np.float64), h)
        assert int(rz["M"]) == M and float(rz["e_op"]) == e_op
        assert np.array_equal(np.asarray(rz["keep_w"]).astype(bool), keep_w_new)
        assert float(rz["logdet_per_pix"]) == float(wz["logdet_per_pix"])
        np.savez(OUT_DELTA, h=np.array([[1.0]]), keep_w=diag_keep, M=0,
                 e_op=0.0, note="T1.1b diagonal-control bundle: delta kernel, "
                                "keep_w = keep_mask & ~residue-drop")
        report["bundles"] = dict(corr=dict(path=str(OUT_CORR), md5=md5(OUT_CORR)),
                                 delta=dict(path=str(OUT_DELTA), md5=md5(OUT_DELTA)))
        print(f"wrote {OUT_CORR} and {OUT_DELTA}")

    OUT_REPORT.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {OUT_REPORT}  verdict={report['verdict']}")
    if stopped:
        raise SystemExit(
            f"T1.1b STOP: residue mask would discard {100*loss_frac:.1f}% of "
            f"whitened dof (> pre-declared {100*STOP_LOSS_FRAC:.0f}%) — "
            "information-starved; NO bundles written, do NOT stage/submit")


if __name__ == "__main__":
    main()
