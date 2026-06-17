#!/usr/bin/env python3
"""eval/export_hard_negatives.py — close the active-learning loop (LensJudge -> v3 finder).

Export agent-confirmed non-lenses from the LensJudge survivor-vetting preds as hard negatives in
the ClaudeNet v3 mimic-bank schema. The v2 grader's whole value (the 0.71 lens-vs-mimic AUC) is
telling real lenses from CNN-selected mimics; the objects it confidently rejects with a *named*
contaminant are exactly the hard negatives the next v3 finder retrain wants.

Target schema (reproductions/claudenet/data/v3/mimic_bank.parquet):
    [row_id, RA, DEC, mimic_type, p_final, source]
LensJudge preds carry grade_pred + contaminant + p_lens + p_meta but NO ra/dec, so coordinates are
joined from the run manifests by name. A *confirmed non-lens* = parse_ok AND a named contaminant AND
(grade_pred == 'D' OR p_lens <= --p-max). p_final = the CNN prior p_meta (these are CNN-high mimics).
source = 'lensjudge_v2'. This script does NOT retrain (GPU, separate program); the v3 assembler
(claudenet/312_assemble_mimic_bank.py) folds the new source into the next bank iteration.

  python lensjudge/eval/export_hard_negatives.py
  python lensjudge/eval/export_hard_negatives.py --preds outputs/dr10_vet_top30.parquet --p-max 0.2
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]            # reproductions/lensjudge
OUT = HERE / "outputs"
REPRO = HERE.parent
BANK = REPRO / "claudenet" / "data" / "v3" / "mimic_bank.parquet"   # schema reference
DEFAULT_OUT = REPRO / "claudenet" / "data" / "v3" / "hard_negatives_from_lensjudge.parquet"

# default preds = the survivor-vetting runs (the active-learning signal) + the v2 evidence set
DEFAULT_PREDS = sorted(set(glob.glob(str(OUT / "*vet*.parquet"))) | {str(OUT / "preds_v2.parquet")})
# default coord sources: the frozen benchmark manifests + the per-run manifests (name/ra/dec)
DEFAULT_MANIFESTS = (sorted(glob.glob(str(OUT / "lensbench_*.csv")))
                     + sorted(glob.glob(str(HERE / "manifests_*.csv"))))

# the v3 mimic-bank mimic_type enum (anything else -> 'other')
BANK_TYPES = {"cnn_high_other", "extended_lrg", "compact_rex", "exp_disk", "lrg_companion",
              "merger", "other", "blend", "noise", "star_halo", "unknown", "ring_galaxy",
              "spiral", "satellite_trail"}
BANK_COLS = ["row_id", "RA", "DEC", "mimic_type", "p_final", "source"]


def build_hard_negatives(preds: pd.DataFrame, coords: pd.DataFrame, p_max: float) -> pd.DataFrame:
    """Pure core (unit-tested): preds + a name->(ra,dec[,p_meta_m]) coord table -> bank-schema rows.

    Confirmed non-lens = parse_ok AND a named contaminant AND (grade_pred=='D' OR p_lens<=p_max).
    p_final = the CNN lens-probability in [0,1] (head-logit survivors, p_meta>1, are CNN-high -> 1.0);
    mimic_type = the contaminant clamped to the bank enum ('other' otherwise)."""
    preds = preds.copy()
    preds["name"] = preds["name"].astype(str)
    has_cont = preds["contaminant"].notna() & (preds["contaminant"].astype(str).str.len() > 0) \
        & (preds["contaminant"].astype(str).str.lower() != "none")
    is_d = (preds["grade_pred"].astype(str).str.upper().eq("D")
            if "grade_pred" in preds else pd.Series(False, index=preds.index))
    low_p = pd.to_numeric(preds.get("p_lens"), errors="coerce") <= p_max
    ok = (preds["parse_ok"].fillna(False) if "parse_ok" in preds
          else pd.Series(True, index=preds.index))
    conf = preds[has_cont & (is_d | low_p) & ok].copy()

    conf = conf.merge(coords, on="name", how="left")
    placed = conf.dropna(subset=["ra", "dec"]).drop_duplicates("name", keep="first")

    p_meta = (pd.to_numeric(placed["p_meta"], errors="coerce") if "p_meta" in placed
              else pd.Series(np.nan, index=placed.index))
    if "p_meta_m" in placed:
        p_meta = p_meta.fillna(placed["p_meta_m"])
    p_final = p_meta.where((p_meta >= 0) & (p_meta <= 1), 1.0).fillna(1.0)
    cont_lc = placed["contaminant"].astype(str).str.lower()
    mtype = cont_lc.where(cont_lc.isin(BANK_TYPES), "other")
    return pd.DataFrame({
        "row_id": placed["name"].astype(str),
        "RA": placed["ra"].astype(float),
        "DEC": placed["dec"].astype(float),
        "mimic_type": mtype.values,
        "p_final": p_final.astype(float).values,
        "source": "lensjudge_v2",
    }, columns=BANK_COLS)


def _coord_lookup(manifests) -> pd.DataFrame:
    frames = []
    for m in manifests:
        if not Path(m).exists():
            continue
        d = pd.read_csv(m)
        cols = {c.lower(): c for c in d.columns}
        if not ({"name", "ra", "dec"} <= set(cols)):
            continue
        out = pd.DataFrame({"name": d[cols["name"]].astype(str),
                            "ra": pd.to_numeric(d[cols["ra"]], errors="coerce"),
                            "dec": pd.to_numeric(d[cols["dec"]], errors="coerce")})
        out["p_meta_m"] = (pd.to_numeric(d[cols["p_meta"]], errors="coerce")
                           if "p_meta" in cols else np.nan)
        frames.append(out.dropna(subset=["ra", "dec"]))
    if not frames:
        return pd.DataFrame(columns=["name", "ra", "dec", "p_meta_m"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("name", keep="first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", default=DEFAULT_PREDS)
    ap.add_argument("--manifests", nargs="+", default=DEFAULT_MANIFESTS)
    ap.add_argument("--p-max", type=float, default=0.3,
                    help="confirmed non-lens if grade_pred=='D' OR p_lens<=this")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    # 1) gather preds
    frames = []
    for p in args.preds:
        if not Path(p).exists():
            print(f"  [skip] {p} (missing)"); continue
        d = pd.read_parquet(p)
        keep = [c for c in ["name", "grade_pred", "contaminant", "p_lens", "p_meta",
                            "parse_ok"] if c in d.columns]
        d = d[keep].copy(); d["_src_file"] = Path(p).name
        frames.append(d)
        print(f"  [preds] {Path(p).name}: {len(d)} rows")
    if not frames:
        raise SystemExit("no preds parquets found")
    preds = pd.concat(frames, ignore_index=True)
    preds["name"] = preds["name"].astype(str)

    # 2-4) coord lookup + the pure build (filter -> join -> bank schema)
    coords = _coord_lookup(args.manifests)
    print(f"  coord lookup: {len(coords)} name->(ra,dec) from {len(args.manifests)} manifest(s)")
    bank = build_hard_negatives(preds, coords, args.p_max)
    print(f"  -> {len(bank)} hard negatives after filter + coord join")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bank.to_parquet(out, index=False)
    print(f"\n  wrote {len(bank)} hard negatives -> {out}")
    print(f"  mimic_type: {bank['mimic_type'].value_counts().to_dict()}")
    # schema sanity vs the canonical bank
    if BANK.exists():
        ref = list(pd.read_parquet(BANK, columns=None).columns)
        assert list(bank.columns) == ["row_id", "RA", "DEC", "mimic_type", "p_final", "source"], \
            "schema mismatch"
        print(f"  schema OK (matches mimic_bank.parquet columns: {ref})")
    print("  NOTE: fold into the next bank via claudenet/312_assemble_mimic_bank.py "
          "(source='lensjudge_v2'); not retraining here.")


if __name__ == "__main__":
    main()
