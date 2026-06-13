#!/usr/bin/env python3
"""101_verify_dr_products.py — Phase 100 data-product audit (runs ON Perlmutter,
CPU-only, login node OK).

Samples bricks from DR9 south / DR9 north / DR10 south coadds on CFS and
verifies, per brick: every image product that EXISTS opens cleanly, carries a
celestial WCS at the expected pixel scale (~0.262"/px), and has the expected
3600x3600 extent. A missing band file = no coverage there (footprint edges;
seen for g/r/z in ~10-30% of random bricks) — recorded, not a failure, because
downstream object manifests filter on sweep NOBS_G/R/Z >= 1. For DR10 south
also records i-band presence (native-griz availability input for Phase 130).

    python 101_verify_dr_products.py --per-tree 8 --out data/v2/dr_products_audit.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

LS = Path("/global/cfs/cdirs/cosmo/data/legacysurvey")
TREES = {
    "dr9_south": (LS / "dr9/south/coadd", ("g", "r", "z")),
    "dr9_north": (LS / "dr9/north/coadd", ("g", "r", "z")),
    "dr10_south": (LS / "dr10/south/coadd", ("g", "r", "z", "i")),
}
PIXSCALE = 0.262  # arcsec/px


def audit_brick(bdir: Path, bands) -> dict:
    brick = bdir.name
    rec = {"brick": brick, "bands": {}, "ok": True}
    for b in bands:
        f = bdir / f"legacysurvey-{brick}-image-{b}.fits.fz"
        e = {"exists": f.exists()}
        if e["exists"]:
            try:
                with fits.open(f, memmap=True) as h:
                    hdu = h[1] if len(h) > 1 else h[0]
                    w = WCS(hdu.header)
                    cd = np.abs(np.diag(w.pixel_scale_matrix)) * 3600.0
                    e.update(shape=list(hdu.data.shape),
                             pixscale=round(float(cd.mean()), 4),
                             wcs_ok=bool(w.has_celestial),
                             shape_ok=hdu.data.shape == (3600, 3600),
                             pixscale_ok=bool(abs(cd.mean() - PIXSCALE) < 0.002))
            except Exception as ex:  # unreadable = audit failure for that band
                e.update(error=str(ex)[:120])
        # missing file = no coverage (allowed; downstream filters on NOBS>=1);
        # an EXISTING file must open with correct WCS/shape/pixscale
        band_ok = (e.get("wcs_ok") and e.get("shape_ok") and e.get("pixscale_ok")) \
            if e["exists"] else True
        e["band_ok"] = bool(band_ok)
        rec["ok"] = rec["ok"] and e["band_ok"]
        rec["bands"][b] = e
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tree", type=int, default=8)
    ap.add_argument("--out", default="data/v2/dr_products_audit.json")
    args = ap.parse_args()
    rng = random.Random(2026)

    report, all_ok = {}, True
    for tree, (root, bands) in TREES.items():
        radirs = sorted(p for p in root.iterdir() if p.is_dir())
        bricks = []
        for ra in rng.sample(radirs, min(args.per_tree, len(radirs))):
            sub = sorted(p for p in ra.iterdir() if p.is_dir())
            if sub:
                bricks.append(rng.choice(sub))
        recs = [audit_brick(b, bands) for b in bricks]
        n_ok = sum(r["ok"] for r in recs)
        i_present = sum(r["bands"].get("i", {}).get("exists", False) for r in recs)
        n_gap = sum(any(not v["exists"] for bb, v in r["bands"].items()
                        if bb in ("g", "r", "z")) for r in recs)
        report[tree] = {"n_bricks": len(recs), "n_ok": n_ok,
                        "n_grz_coverage_gaps": n_gap,
                        "i_band_present": i_present, "bricks": recs}
        all_ok = all_ok and n_ok == len(recs) and len(recs) > 0
        print(f"[audit] {tree}: {n_ok}/{len(recs)} bricks OK, "
              f"{n_gap} with grz coverage gaps"
              + (f", i-band in {i_present}/{len(recs)}" if "i" in bands else ""))

    report["verdict"] = "PASS" if all_ok else "FAIL"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[audit] VERDICT: {report['verdict']} -> {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
