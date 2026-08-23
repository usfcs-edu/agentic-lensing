#!/usr/bin/env python3
"""golden/build_eval_manifest.py — the golden labels as a LensBench-style eval manifest.

Writes outputs/golden_jwst_manifest.csv with EXACTLY the 11 columns of
eval/build_eval_set.py first (name, ra, dec, survey_key, grade_truth, binary_label, source,
region, tractor_type, p_meta, leak) so imaging/run_batch.py --mode jwst (the ONLY mode that
accepts survey_key=jwst rows), eval/score.py and eval/calibrate.load_preds read it
unchanged, then the golden extras the JWST grader and the analysis need:
unit_id, image_path (the served kit JPEG), render_sha, score_1_4, confidence_lmh, split,
stratum, p_pipeline (the incumbent the E1 arm is read against), pipe_flagged,
binary_label_ge2.

Column semantics (contract "Grade semantics"; endpoints in golden/REGISTRY.md):
  grade_truth      = Xiaosheng's letter (4->A 3->B 2->C 1->D via _util.score_to_letter)
  binary_label     = lens iff score >= 3 (A/B) — the pre-registered PRIMARY binary endpoint,
                     the same cut stats.py, split_halves.py and build_corpus_golden.gate use
  binary_label_ge2 = lens iff score >= 2 (the rubric's "C = possible" view; secondary only)
  source           = "golden_huang"; survey_key = "jwst"; region = JWST proposal id
  tractor_type     = "" and p_meta = NaN (no Tractor / CNN for JWST rows)
  leak             = "desi_train" when the frame flags a 2" overlap with the DESI parity
                     pools (a DESI-trained student was TAUGHT these are non-lenses), else "no"
  p_pipeline       = pipe_inspector_conf / 100 for rows the pipeline flagged; 0.0 for rows it
                     never flagged (it ranked them below every flagged item — dropping them
                     as NaN would delete exactly the incumbent's misses: all 20 N_unflagged
                     and the 26 literature lenses it did not surface); NaN only for a flagged
                     row with no confidence. pipe_flagged says which is which, so the
                     analysis can report the incumbent on flagged rows AND on all rows.

eval/lensbench_gate.py's verdict does NOT apply here: its detection AUC needs rows with
source == "random" and every golden row is source == "golden_huang" (it prints FAIL on a
missing AUC). The golden endpoints are computed by golden/analyze_golden.py.

  python lensjudge/golden/build_eval_manifest.py --labels golden/golden_labels.csv \\
      --frame golden/frame.csv --splits golden/splits.csv [--split validate|align|all] \\
      --out outputs/golden_jwst_manifest.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.golden import _util  # noqa: E402

EVAL_COLS = ["name", "ra", "dec", "survey_key", "grade_truth", "binary_label", "source",
             "region", "tractor_type", "p_meta", "leak"]
EXTRA_COLS = ["unit_id", "image_path", "render_sha", "score_1_4", "confidence_lmh", "split",
              "stratum", "p_pipeline", "pipe_flagged", "binary_label_ge2"]
LENS_MIN_SCORE = 3        # binary_label: lens iff score >= 3 (REGISTRY.md endpoints)
SOURCE = "golden_huang"
SURVEY_KEY = "jwst"
DEFAULT_OUT = _util.LENSJUDGE / "outputs" / "golden_jwst_manifest.csv"


def _read(path: Path) -> pd.DataFrame:
    """Pinned CSVs verify their .sha; unpinned (e.g. a hand-made fixture) fall through."""
    path = Path(path)
    if path.with_suffix(path.suffix + ".sha").exists():
        return _util.read_pinned(path)
    return pd.read_csv(path)


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes")
    try:
        return bool(v) and not pd.isna(v)
    except (TypeError, ValueError):
        return bool(v)


def load_keys(keys_dir: Path) -> pd.DataFrame:
    parts = [_read(p) for p in sorted(Path(keys_dir).glob("*_key.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["kit_id", "item_id", "unit_id", "pass", "render_sha"])


def image_paths(labels: pd.DataFrame, keys: pd.DataFrame, kits_dir: Path) -> pd.Series:
    """unit_id -> served kit JPEG for the canonical (pass-1) render; prefers the key row
    whose render_sha equals the label's render_sha when a unit appears in several kits."""
    if keys.empty:
        return pd.Series([""] * len(labels), index=labels.index, dtype=object)
    k1 = keys[pd.to_numeric(keys["pass"], errors="coerce").fillna(1) == 1].copy()
    k1["unit_id"] = k1["unit_id"].astype(str)
    out = []
    for r in labels.itertuples():
        rows = k1[k1["unit_id"] == str(r.unit_id)]
        if len(rows) > 1 and isinstance(getattr(r, "render_sha", None), str):
            m = rows[rows["render_sha"].astype(str) == r.render_sha]
            rows = m if len(m) else rows
        if rows.empty:
            out.append("")
            continue
        k = rows.iloc[0]
        out.append(str((Path(kits_dir) / str(k["kit_id"]) / "items"
                        / f"{str(k['item_id']).zfill(3)}.jpg").resolve()))
    return pd.Series(out, index=labels.index, dtype=object)


def build(labels: pd.DataFrame, frame: pd.DataFrame, splits: pd.DataFrame, keys: pd.DataFrame,
          kits_dir: Path, split: str = "all") -> pd.DataFrame:
    lab = labels.copy()
    lab["unit_id"] = lab["unit_id"].astype(str)
    fr = frame.copy()
    fr["unit_id"] = fr["unit_id"].astype(str)
    sp = splits[["unit_id", "split"]].copy()
    sp["unit_id"] = sp["unit_id"].astype(str)
    fcols = [c for c in ("ra_deg", "dec_deg", "proposal", "desi_pool_overlap", "pipe_inspector_conf",
                         "pipe_grade_passcount", "stratum", "candidate_id") if c in fr.columns]
    df = lab.merge(fr[["unit_id"] + fcols], on="unit_id", how="left", suffixes=("", "_frame"))
    df = df.merge(sp, on="unit_id", how="left")
    missing = df["split"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} labelled units have no split: {df.loc[missing, 'unit_id'].tolist()[:5]}")
    if split != "all":
        df = df[df["split"] == split].copy()
    # prefer the frame's coordinates (from the run's master CSV); labels carry a copy
    ra = df["ra_deg_frame"] if "ra_deg_frame" in df else df["ra_deg"]
    dec = df["dec_deg_frame"] if "dec_deg_frame" in df else df["dec_deg"]
    letters = df["score_1_4"].map(_util.score_to_letter)
    if "grade_letter" in df and not (df["grade_letter"].astype(str) == letters).all():
        raise ValueError("labels grade_letter disagrees with score_1_4 -> letter map")
    leak = df["desi_pool_overlap"].map(_truthy) if "desi_pool_overlap" in df else pd.Series(False, index=df.index)
    conf = pd.to_numeric(df["pipe_inspector_conf"], errors="coerce") if "pipe_inspector_conf" in df \
        else pd.Series(np.nan, index=df.index)
    flagged = conf.notna()                 # the pipeline scored it
    if "pipe_grade_passcount" in df:       # ... or at least gave it a pass-count
        pc = df["pipe_grade_passcount"].fillna("").astype(str).str.strip()
        flagged = flagged | pc.isin(["A", "B", "C", "D", "U"])
    p_pipe = np.where(flagged, conf / 100.0, 0.0)
    strat = df["stratum_frame"] if "stratum_frame" in df else df.get("stratum", pd.Series("", index=df.index))
    scores = df["score_1_4"].astype(int)
    man = pd.DataFrame({
        "name": df["candidate_id"].astype(str),
        "ra": ra.astype(float), "dec": dec.astype(float),
        "survey_key": SURVEY_KEY,
        "grade_truth": letters,
        "binary_label": np.where(scores >= LENS_MIN_SCORE, "lens", "nonlens"),
        "source": SOURCE,
        "region": df["proposal"].astype(str) if "proposal" in df else "",
        "tractor_type": "",
        "p_meta": np.nan,
        "leak": np.where(leak, "desi_train", "no"),
        "unit_id": df["unit_id"],
        "image_path": image_paths(df, keys, kits_dir).values,
        "render_sha": df["render_sha"].astype(str) if "render_sha" in df else "",
        "score_1_4": df["score_1_4"].astype(int),
        "confidence_lmh": df["confidence_lmh"].astype(str).str.upper(),
        "split": df["split"],
        "stratum": strat.astype(str),
        "p_pipeline": p_pipe,
        "pipe_flagged": flagged.to_numpy(bool),
        "binary_label_ge2": np.where(scores >= 2, "lens", "nonlens"),
    }, index=df.index)
    man = man[EVAL_COLS + EXTRA_COLS].drop_duplicates("name").reset_index(drop=True)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(_util.HERE / "golden_labels.csv"))
    ap.add_argument("--frame", default=str(_util.HERE / "frame.csv"))
    ap.add_argument("--splits", default=str(_util.HERE / "splits.csv"))
    ap.add_argument("--keys-dir", default=str(_util.HERE / "keys"))
    ap.add_argument("--kits-dir", default=str(_util.HERE / "kits"))
    ap.add_argument("--split", choices=("validate", "align", "all"), default="all")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    man = build(_read(args.labels), _read(args.frame), _read(args.splits),
                load_keys(Path(args.keys_dir)), Path(args.kits_dir), args.split)
    sha = _util.pin(man, Path(args.out))
    n_img = int((man["image_path"] != "").sum())
    print(f"[manifest] {len(man)} rows ({args.split}) -> {args.out} (sha {sha}); {n_img} with kit JPEGs")
    print("  by grade_truth:", man["grade_truth"].value_counts().to_dict())
    print("  by binary (score>=3):", man["binary_label"].value_counts().to_dict(),
          " (score>=2):", man["binary_label_ge2"].value_counts().to_dict())
    print("  pipe_flagged:", man["pipe_flagged"].value_counts().to_dict())
    print("  by split:", man["split"].value_counts().to_dict())
    print("  leak:", man["leak"].value_counts().to_dict())


if __name__ == "__main__":
    main()
