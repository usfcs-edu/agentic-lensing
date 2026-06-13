#!/usr/bin/env python3
"""200_export_737.py — qualification campaign: export the 737 new-and-unseen
DR9-sweep candidates, stage their cutouts as FITS, and write the manifests the
downstream passes (163 crossmatch, lensjudge, the visual workflow) consume.

The 737 set is the FDR<=0.05 group-conformal selection that is genuinely NEW
(unmatched to the 4 local DECaLS catalogs) and unseen (not a training / mined /
NegEval-calibration row). Cutouts already live in data/v2/sweep/vet_topnew.npz
(the top-1500 by p_final; all 737 are in it, ok=True), so no re-extraction.

Writes (under data/v2/campaign/):
  manifest_737.parquet          spine: row_id,RA,DEC,p_final,q_group,brick,members,fits_path
  manifest_737_lensjudge.csv    lensjudge run_batch schema (name,ra,dec,catalog,survey,p_meta)
  stage2_scores_737.parquet     163 input scoped to the 737 (row_id,RA,DEC,footprint,brick,p_stage1,p_final)
  fits/{row_id}.fits            staged grz cube (3,101,101) float32 — byte-identical to 120b

    /home2/benson/.venvs/claudenet/bin/python campaign/200_export_737.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # reproductions/claudenet
DATA = ROOT / "data"
SWEEP = DATA / "v2" / "sweep"
OUT = DATA / "v2" / "campaign"
MEMBERS = ["member_effnet_B", "member_effnet_B3_hard", "member_effnet_S2_hard",
           "member_resnet46_C_hard", "member_zoobot_N"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def select_737(df: pd.DataFrame) -> pd.DataFrame:
    m = ((df["status"] == "NEW") & (~df["is_train_row"]) & (~df["is_mined_row"])
         & (~df["is_negeval_calibration_row"]) & (df["sel_group_a0.05"]))
    return df[m].copy()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fits").mkdir(exist_ok=True)
    XM = _load("cn_120b", ROOT / "120b_unpack_npz_to_fits.py")   # to_bytes

    cand = pd.read_parquet(SWEEP / "candidates_v2.parquet")
    s = select_737(cand).reset_index(drop=True)
    s["row_id"] = s["row_id"].astype(str)
    assert len(s) == 737, f"expected 737, got {len(s)}"
    assert np.isfinite(s.RA).all() and np.isfinite(s.DEC).all(), "non-finite RA/DEC"
    print(f"[200] selected {len(s)} candidates; p_final {s.p_final.min():.3f}-{s.p_final.max():.3f}")

    # cutouts from the vet npz (all 737 present, ok=True)
    z = np.load(SWEEP / "vet_topnew.npz")
    npz_ids = z["row_ids"].astype(str)
    loc = {r: i for i, r in enumerate(npz_ids)}
    okmap = dict(zip(npz_ids, z["ok"]))
    cubes = z["cutouts"]

    n_written = 0
    fits_paths = []
    for r in s.row_id:
        i = loc.get(r)
        assert i is not None and bool(okmap[r]), f"{r} missing/ok=False in vet_topnew.npz"
        cube = np.asarray(cubes[i], dtype=np.float32)
        assert cube.shape == (3, 101, 101), f"{r}: bad cube {cube.shape}"
        fp = OUT / "fits" / f"{r}.fits"
        fp.write_bytes(XM.to_bytes(cube))
        fits_paths.append(str(fp))
        n_written += 1
    print(f"[200] staged {n_written} FITS -> {OUT/'fits'}")

    # spine manifest
    spine = pd.DataFrame({
        "row_id": s.row_id, "RA": s.RA.astype(float), "DEC": s.DEC.astype(float),
        "p_final": s.p_final.astype(float), "q_group": s["q_group"].astype(float),
        "brick": s.brick.astype(str), "footprint": s.footprint.astype(str),
        "p_stage1": s["p_stage1"].astype(float), "fits_path": fits_paths,
    })
    for c in MEMBERS:
        spine[c] = s[c].astype(float)
    spine.to_parquet(OUT / "manifest_737.parquet", index=False)

    # lensjudge run_batch CSV (name=row_id; on-disk catalog/survey key 'claudenet')
    lj = pd.DataFrame({
        "name": s.row_id, "ra": s.RA.astype(float), "dec": s.DEC.astype(float),
        "grade": "?",          # unlabeled candidates (run_batch needs the column)
        "catalog": "claudenet", "survey": "claudenet", "p_meta": s.p_final.astype(float),
    })
    lj.to_csv(OUT / "manifest_737_lensjudge.csv", index=False)

    # 163 input scoped to the 737
    s2 = spine[["row_id", "RA", "DEC", "footprint", "brick", "p_stage1", "p_final"]].copy()
    s2.to_parquet(OUT / "stage2_scores_737.parquet", index=False)

    print(f"[200] wrote manifest_737.parquet, manifest_737_lensjudge.csv, "
          f"stage2_scores_737.parquet -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
