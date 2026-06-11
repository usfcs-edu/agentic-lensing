#!/usr/bin/env python3
"""130_build_griz_manifest.py — Phase 130: extraction manifest for NATIVE
160px griz cutouts of every v1 object (runs LOCALLY — the inputs live only in
local data/: training_split_staged.parquet, eval_*.parquet and the published
candidate catalogs; Perlmutter has none of them).

Replaces the v1 AION member's degraded input (101px grz bilinear-resized to
160px + synthetic i=0.5*(r+z), see 10_build_aion_inputs.py) with native-griz
extraction from the CFS coadds. This script collects (row_id, RA, DEC, source)
for ALL v1 train/val/eval objects:

  * the 66,971 training_split_staged rows (train/val/test; eval_val and
    eval_testneg are verified subsets of these — asserted),
  * the storfer/inchausti held-out candidates: eval_{storfer,inchausti}.parquet
    lack RA/DEC, so coordinates are recovered exactly the way v1's
    10_build_aion_inputs.py keyed them — row_id == `name` in the published
    catalog CSVs (storfer2024/inchausti2025_published_catalog.csv). Any
    candidate row_id that fails to map is REPORTED LOUDLY and dropped — never
    guessed. Candidate row_ids already in the split (curated positives that
    are also published candidates) are deduped after a <1" coordinate check.

DELIBERATE DEVIATION from the original plan wording ("south = DR9 grz +
DR10 i"): south rows take ALL FOUR bands griz from DR10, not DR9-grz + DR10-i,
because (i) the four bands then share one photometric reduction — no per-band
zeropoint/PSF mismatch inside a single cutout — and (ii) AION's
LegacySurveyImage codec was trained on south DECam imagery, so codec
consistency beats reduction-consistency-with-v1 here. The decision gate (132)
compares probes trained on their own inputs (degraded probe on degraded
embeddings, native probe on native), so this remains a fair comparison.

Outputs:
  data/v2/griz_manifest_nobrick.parquet (row_id, RA, DEC, label, source)
      — no brick/footprint yet;
  data/v2/griz_labels.parquet (row_id str, label int, split str) — the
      --labels input 133_lora_finetune_aion.py expects: split = the staged
      split value (train/val/test) for split rows,
      'candidate_storfer'/'candidate_inchausti' for the held-out candidates,
      'testneg' for any eval_testneg row not already in the split (none
      today — the coverage assert below guarantees it). No duplicate row_id
      (asserted).

Downstream (orchestrator-run):

  1. rsync data/v2/griz_manifest_nobrick.parquet to Perlmutter, then assign
     bricks/footprints against the real DR9 survey-bricks table + coadd dirs:
         python 130b_assign_bricks.py        # writes data/v2/griz_manifest_{south,north}.parquet
  2. extract native-griz cutouts with the EXISTING 111 extractor, one run per
     source_release (south carries native i in DR10; DR9 north never has i ->
     111 zero-fills the i plane and records i_ok=False):
         python 111_extract_cutouts_cfs.py --manifest data/v2/griz_manifest_south.parquet \\
             --out-root $SCRATCH/claudenet/cutouts/griz_south --size 160 --bands griz \\
             --release dr10 --workers 32 --shard-size 50000
         python 111_extract_cutouts_cfs.py --manifest data/v2/griz_manifest_north.parquet \\
             --out-root $SCRATCH/claudenet/cutouts/griz_north --size 160 --bands griz \\
             --release dr9 --workers 32 --shard-size 50000
  3. embed with 131_embed_aion_variants.py (base/large/xlarge).

    python 130_build_griz_manifest.py [--out data/v2/griz_manifest_nobrick.parquet]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C

CANDIDATES = (  # eval parquet -> published catalog with the row_id->RA/DEC map
    ("storfer", "eval_storfer.parquet", "storfer2024_published_catalog.csv"),
    ("inchausti", "eval_inchausti.parquet", "inchausti2025_published_catalog.csv"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(C.DATA / "v2" / "griz_manifest_nobrick.parquet"))
    ap.add_argument("--labels-out", default=str(C.DATA / "v2" / "griz_labels.parquet"),
                    help="row_id/label/split table (133_lora_finetune_aion --labels)")
    args = ap.parse_args()
    t0 = time.time()

    # 1. the 66,971 split rows (carry RA/DEC natively) -------------------------
    sp = pd.read_parquet(C.DATA / "training_split_staged.parquet")
    assert sp.row_id.is_unique and not sp[["RA", "DEC"]].isna().any().any()
    frames = [pd.DataFrame({
        "row_id": sp.row_id.astype(str), "RA": sp.RA.astype(float),
        "DEC": sp.DEC.astype(float), "label": sp.label.astype(int),
        "source": "split_" + sp.split.astype(str),
    })]
    print(f"[130] training_split_staged: {len(sp):,} rows "
          f"({dict(sp.split.value_counts())}, {int(sp.label.sum()):,} pos)")

    # eval_val / eval_testneg must already be covered by the split rows
    ids = set(sp.row_id)
    for f in ("eval_val.parquet", "eval_testneg.parquet"):
        ev = pd.read_parquet(C.DATA / f)
        n_in = int(ev.row_id.isin(ids).sum())
        assert n_in == len(ev), f"{f}: {len(ev) - n_in} rows NOT in training_split_staged"
        print(f"[130] {f}: {len(ev):,} rows — all covered by the split rows")

    # 2. candidates: recover RA/DEC via the published catalogs (v1's keying) ---
    n_unmapped = 0
    for name, evf, csvf in CANDIDATES:
        ev = pd.read_parquet(C.DATA / evf)
        cat = pd.read_csv(C.DATA / csvf)
        assert cat.name.is_unique, f"{csvf}: duplicate names"
        m = ev[["row_id"]].merge(cat.rename(columns={"name": "row_id"})[["row_id", "RA", "DEC"]],
                                 on="row_id", how="left")
        bad = m[m.RA.isna() | m.DEC.isna()]
        if len(bad):
            n_unmapped += len(bad)
            print(f"[130] *** WARNING: {len(bad)} {name} candidate row_ids have NO "
                  f"RA/DEC in {csvf} — DROPPED, not guessed: "
                  f"{bad.row_id.head(10).tolist()}{'...' if len(bad) > 10 else ''}")
            m = m.dropna(subset=["RA", "DEC"])
        m["label"], m["source"] = 1, name
        frames.append(m[["row_id", "RA", "DEC", "label", "source"]])
        print(f"[130] {name}: {len(ev):,} candidates -> {len(m):,} with recovered RA/DEC")

    # 3. dedupe row_ids (candidates that are also curated split positives).
    # Split rows come first -> they win. Where the two sources disagree by >1"
    # (2 known Storfer rows with rounded/offset catalog coords) the kept split
    # coordinates are verified against the DESI-RRR.RRRR±DD.DDDD name encoding,
    # which is the authoritative position.
    man = pd.concat(frames, ignore_index=True)
    dup = man[man.row_id.duplicated(keep=False)]
    if len(dup):
        span = dup.groupby("row_id").agg(dra=("RA", lambda s: s.max() - s.min()),
                                         dde=("DEC", lambda s: s.max() - s.min()),
                                         dec=("DEC", "mean"))
        sep = np.hypot(span.dra * np.cos(np.radians(span.dec)), span.dde) * 3600.0
        for rid in span.index[sep >= 1.0]:
            kept = man[man.row_id == rid].iloc[0]            # the split row
            ra_n, dec_n = float(rid[5:13]), float(rid[13:])  # name-encoded coords
            d = np.hypot((kept.RA - ra_n) * np.cos(np.radians(dec_n)),
                         kept.DEC - dec_n) * 3600.0
            assert d < 2.0, f"{rid}: kept coords disagree with the name encoding ({d:.1f}\")"
            print(f"[130] WARNING: {rid}: split/catalog coords {sep[rid]:.1f}\" apart"
                  f" — kept the split coords (match the name encoding to {d:.2f}\")")
        n_dupe = int(man.row_id.duplicated().sum())
        man = man.drop_duplicates("row_id", keep="first").reset_index(drop=True)
        print(f"[130] deduped {n_dupe} candidate row_ids already in the split "
              f"(kept the split rows)")
    assert man.row_id.is_unique

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    man.to_parquet(out, index=False)
    per_src = ", ".join(f"{k}: {v:,}" for k, v in man.source.value_counts().items())
    print(f"[130] wrote {out} — {len(man):,} rows ({per_src}); "
          f"{int(man.label.sum()):,} pos / {int((man.label == 0).sum()):,} neg; "
          f"{n_unmapped} unmapped candidates; {time.time() - t0:.1f}s")

    # 4. griz_labels.parquet — the 133_lora_finetune_aion --labels input -------
    lab = pd.DataFrame({"row_id": man.row_id.astype(str),
                        "label": man.label.astype(int)})
    lab["split"] = np.where(man.source.str.startswith("split_"),
                            man.source.str[len("split_"):],
                            "candidate_" + man.source)
    tn = pd.read_parquet(C.DATA / "eval_testneg.parquet")
    extra = tn[~tn.row_id.astype(str).isin(set(lab.row_id))]
    if len(extra):          # none today: eval_testneg ⊆ split rows (asserted)
        lab = pd.concat([lab, pd.DataFrame({
            "row_id": extra.row_id.astype(str),
            "label": extra.label.astype(int), "split": "testneg"})],
            ignore_index=True)
    assert lab.row_id.is_unique, "griz_labels: duplicate row_id"
    lab_out = Path(args.labels_out)
    lab.to_parquet(lab_out, index=False)
    per_split = ", ".join(f"{k}: {v:,}" for k, v in lab.split.value_counts().items())
    print(f"[130] wrote {lab_out} — {len(lab):,} rows ({per_split})")
    if n_unmapped:
        print(f"[130] *** {n_unmapped} candidate row_ids COULD NOT be mapped to "
              f"RA/DEC — they are missing from the manifest ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
