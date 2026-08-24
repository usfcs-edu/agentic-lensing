#!/usr/bin/env python3
"""golden/calibrate_thresholds.py — REGISTRY "Deployment rule v2-deploy" item 1: fit the
`opus5_api` letter thresholds on the a2/opus5 DESIGN run and write them to thresholds_v2.json.

Protocol (verbatim from REGISTRY.md, not to be re-interpreted here): using ONLY the
`truth_class == negative`, non-anchor rows of the design parquet, t_A = smallest S with design
FPR ≤ 1 %, t_B = smallest S with FPR ≤ 5 %, tau0 unchanged (0.15). Design-positive recall at
t_A / t_B is REPORTED, never used to fit. The fitted key is written before any holdout or
top-100 letter is computed; the previous file is archived under
outputs/thresholds_v2.pre_opus5.json.

Threshold semantics (`threshold_at_fpr_neg`): candidate thresholds are the observed NEGATIVE
scores; FPR(t) = fraction of negatives with S ≥ t (the `aggregate_v2.assign_letter` test is
`S >= t`); t is the smallest candidate with FPR(t) ≤ target, i.e. the k-th-highest negative
score with k = the largest count whose share is ≤ target (ties push the threshold up; when even
the top negative has FPR > target the threshold is +inf and --write is refused). This is the
REGISTRY wording. It differs from `analyze_truth.threshold_at_fpr` in two ways that matter
for a fit "on negatives only": that function takes its candidate thresholds from
`roc_curve`, i.e. the union of positive AND negative scores (so it can return a positive's
score lying just below the k-th-highest negative — same design FPR, lower threshold, and a
positive used in the fit), and it returns nan unless both classes are present. The achieved
design FPR agrees with it whenever no positive score falls in that gap (asserted in the tests).

  cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/golden/calibrate_thresholds.py \\
      --preds lensjudge/outputs/preds_truth_a2_opus5_design_k1_r1.parquet --model-key opus5_api \\
      --thresholds lensjudge/golden/thresholds_v2.json --fpr-a 0.01 --fpr-b 0.05 --min-neg 50 [--write]

Without --write nothing is written: the calibration record is printed. With --write the run
refuses on any blocker (the replicate's `.meta.json` absent — it is written on completion, so a
parquet still being flushed has none; n_neg < --min-neg; n_neg != --n-neg-expected (200: the
fit is on ALL the negative rows, never a subset); a negative row with S NaN unless
--allow-nan-neg; a non-finite threshold; rows that are not the design half or not arm a2; a
--model-key that disagrees with the parquet's model column; a tau0 that is not the run's; a
thresholds file that is not in the canonical `json.dumps(indent=2) + "\\n"` form —
untouched parts must survive byte-for-byte), archives the current thresholds file to
<archive-dir>/thresholds_v2.pre_<key-without-_api>.json (never overwriting: _2, _3, …), updates
ONLY the given key (other keys and their order preserved; a new key goes right after
`sonnet_api`), and writes <archive-dir>/<model-key>_calibration.json with every number.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from lensjudge.golden import _util, aggregate_v2  # noqa: E402
from lensjudge.golden import run_golden_eval as rge  # noqa: E402
from lensjudge.golden import run_truth_eval as rte  # noqa: E402

OUT = _util.LENSJUDGE / "outputs"
PREDS_DEFAULT = OUT / "preds_truth_a2_opus5_design_k1_r1.parquet"
THRESHOLDS_DEFAULT = _util.HERE / "thresholds_v2.json"
SCORE_COL = "S"
NEG_CLASS = "negative"
DESIGN_HALF = "design"
INSERT_AFTER = "sonnet_api"
ARM_EXPECTED = "a2"
N_NEG_EXPECTED = 200                          # REGISTRY item 1: "the 200 truth_class == negative non-anchor rows"
TAU0 = aggregate_v2.PROVISIONAL["tau0"]       # "tau0 unchanged (0.15)"
_EPS = 1e-12
# columns read from the preds parquet (those absent are simply not reported)
META_COLS = ("run_tag", "arm", "model", "half", "thinking", "effort", "tau0", "thresholds_sha16")


class CalibrationResult(BaseModel):
    """Every number of one calibration run — printed always, written with --write."""
    model_config = ConfigDict(extra="forbid")
    model_key: str
    preds: str
    preds_sha16: str
    run_tag: Optional[str] = None
    arm: Optional[str] = None
    model: Optional[str] = None
    half: Optional[str] = None
    thinking: Optional[str] = None
    effort: Optional[str] = None
    run_thresholds_sha16: Optional[str] = None   # the tuple sha the design run was scored under
    fpr_a: float
    fpr_b: float
    min_neg: int
    n_neg_expected: int       # the protocol's 200 (0 = not checked)
    allow_nan_neg: bool       # --allow-nan-neg: NaN negatives tolerated (never for the registered fit)
    meta_present: bool        # <preds>.meta.json exists = the replicate completed (written on completion)
    n_rows: int
    n_neg: int                # negative, non-anchor, finite S — the rows the fit uses
    n_neg_nan: int            # negative, non-anchor, S NaN (excluded)
    n_anchor_excluded: int    # is_anchor rows dropped from both classes
    n_pos: int                # is_positive, non-anchor, finite S (report only)
    n_pos_nan: int
    tau0: float
    t_A: Optional[float]      # None ⇔ +inf (no negative score reaches the target)
    t_B: Optional[float]
    n_neg_ge_tA: int
    n_neg_ge_tB: int
    fpr_A: float
    fpr_B: float
    n_pos_ge_tA: int
    n_pos_ge_tB: int
    recall_A: Optional[float]  # None when n_pos == 0
    recall_B: Optional[float]
    thresholds: str
    key_present: bool
    existing_entry: Optional[dict] = None
    new_entry: dict
    file_sha16_before: str
    file_sha16_after: str          # sha of the would-be file text (equals the file's after --write)
    tuple_sha16_before: str        # run_truth_eval.thresholds_sha of the key resolved on the OLD table
    tuple_sha16_after: str         # … on the NEW table (this is the thresholds_sha16 later runs carry)
    letter_source_before: str
    letter_source_after: str
    write_blockers: list[str]
    written: bool = False
    written_utc: Optional[str] = None
    archive_path: Optional[str] = None
    calibration_path: Optional[str] = None


# ------------------------------------------------------------------ pure pieces
def threshold_at_fpr_neg(neg_scores, fpr_target: float) -> tuple[float, float, int]:
    """(t, achieved_fpr, n_at_or_above): the smallest observed negative score t with
    FPR(t) = mean(neg >= t) <= fpr_target. NaNs are dropped first; +inf (fpr 0, count 0) when
    no observed score qualifies; raises on an empty input."""
    s = np.asarray(list(neg_scores), dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        raise ValueError("no finite negative scores to calibrate on")
    best: Optional[float] = None
    for v in sorted(set(s.tolist()), reverse=True):       # descending: counts only grow
        c = int((s >= v).sum())
        if c / n <= fpr_target + _EPS:
            best = float(v)
        else:
            break
    if best is None:
        return math.inf, 0.0, 0
    c = int((s >= best).sum())
    return best, c / n, c


def split_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(negatives, positives, n_anchor_excluded) — both frames non-anchor; S left as is."""
    for c in ("truth_class", "is_anchor", "is_positive", SCORE_COL):
        if c not in df.columns:
            raise KeyError(f"preds parquet lacks column {c!r}")
    anchor = df["is_anchor"].fillna(False).astype(bool)
    neg = df[(df["truth_class"].astype(str) == NEG_CLASS) & ~anchor]
    pos = df[df["is_positive"].fillna(False).astype(bool) & ~anchor]
    return neg, pos, int(anchor.sum())


def _finite(s: pd.Series) -> np.ndarray:
    a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return a[np.isfinite(a)]


def _count_ge(a: np.ndarray, t: float) -> int:
    return int((a >= t).sum()) if math.isfinite(t) else 0


def calibrate(df: pd.DataFrame, fpr_a: float, fpr_b: float) -> dict:
    """The fit + report numbers from a preds frame (no file IO)."""
    neg, pos, n_anchor = split_rows(df)
    sn, sp = _finite(neg[SCORE_COL]), _finite(pos[SCORE_COL])
    if len(sn) == 0:
        raise ValueError("no finite-S negative non-anchor rows in the preds parquet")
    t_a, fpr_a_got, n_a = threshold_at_fpr_neg(sn, fpr_a)
    t_b, fpr_b_got, n_b = threshold_at_fpr_neg(sn, fpr_b)
    pa, pb = _count_ge(sp, t_a), _count_ge(sp, t_b)
    return {
        "n_rows": int(len(df)), "n_neg": int(len(sn)), "n_neg_nan": int(len(neg) - len(sn)),
        "n_anchor_excluded": n_anchor, "n_pos": int(len(sp)), "n_pos_nan": int(len(pos) - len(sp)),
        "t_A": None if math.isinf(t_a) else t_a, "t_B": None if math.isinf(t_b) else t_b,
        "n_neg_ge_tA": n_a, "n_neg_ge_tB": n_b, "fpr_A": fpr_a_got, "fpr_B": fpr_b_got,
        "n_pos_ge_tA": pa, "n_pos_ge_tB": pb,
        "recall_A": pa / len(sp) if len(sp) else None, "recall_B": pb / len(sp) if len(sp) else None,
    }


def render_table(table: dict) -> str:
    """The thresholds_v2.json on-disk form: json.dumps(indent=2) + trailing newline."""
    return json.dumps(table, indent=2) + "\n"


def updated_table(table: dict, model_key: str, entry: dict, insert_after: str = INSERT_AFTER) -> dict:
    """A new dict with ONLY `model_key` replaced (in place when present) or inserted right after
    `insert_after` (at the end when that key is absent); every other key untouched, order kept."""
    if model_key in table:
        return {k: (entry if k == model_key else v) for k, v in table.items()}
    out: dict = {}
    for k, v in table.items():
        out[k] = v
        if k == insert_after:
            out[model_key] = entry
    if model_key not in out:
        out[model_key] = entry
    return out


def archive_path(archive_dir: Path, model_key: str) -> Path:
    """<archive-dir>/thresholds_v2.pre_<key-without-_api>.json, suffixed _2, _3, … if taken."""
    stem = model_key[:-4] if model_key.endswith("_api") else model_key
    base = Path(archive_dir) / f"thresholds_v2.pre_{stem}.json"
    if not base.exists():
        return base
    i = 2
    while True:
        p = base.with_name(f"thresholds_v2.pre_{stem}_{i}.json")
        if not p.exists():
            return p
        i += 1


def _uniq(df: pd.DataFrame, col: str) -> Optional[str]:
    if col not in df.columns:
        return None
    vals = sorted(set(str(v) for v in df[col].dropna().tolist()))
    return "|".join(vals) if vals else None


def tuple_sha(table: dict, model_key: str) -> tuple[str, str]:
    """(thresholds_sha16, letter_source) exactly as run_truth_eval computes them for a run:
    aggregate_v2.resolve_thresholds on the table, then run_truth_eval.thresholds_sha."""
    thr = dict(aggregate_v2.resolve_thresholds(table, model_key))
    thr["model_key"] = model_key
    return rte.thresholds_sha(thr), str(thr["letter_source"])


# ------------------------------------------------------------------ the run
def run(preds: Path, model_key: str, thresholds: Path, fpr_a: float = 0.01, fpr_b: float = 0.05,
        min_neg: int = 50, write: bool = False, archive_dir: Path = OUT,
        out: Optional[Path] = None, n_neg_expected: int = N_NEG_EXPECTED,
        allow_nan_neg: bool = False) -> CalibrationResult:
    """Fit, check, and (with write=True and no blocker) archive + update the thresholds file.
    Raises SystemExit with the blockers when write is requested but refused. The fit is
    complete only when the replicate is: its `.meta.json` exists (written on completion —
    a parquet still being flushed has none), every negative non-anchor row has a finite S
    (`allow_nan_neg` lifts that for a dev run), and their count is exactly
    `n_neg_expected` (the protocol's 200; 0 disables the count check)."""
    preds, thresholds, archive_dir = Path(preds), Path(thresholds), Path(archive_dir)
    df = pd.read_parquet(preds)
    fit = calibrate(df, fpr_a, fpr_b)
    meta_present = rge.meta_path(preds).exists()

    text_before = thresholds.read_text()
    table = json.loads(text_before)
    if not isinstance(table, dict):
        raise SystemExit(f"[calib] {thresholds} is not a JSON object")
    existing = table.get(model_key)
    entry = {"tau0": TAU0, "t_A": fit["t_A"], "t_B": fit["t_B"]}
    new_table = updated_table(table, model_key, entry)
    text_after = render_table(new_table)
    sha_before, sha_after = _util.sha_text(text_before), _util.sha_text(text_after)
    tsha_before, src_before = tuple_sha(table, model_key)
    tsha_after, src_after = tuple_sha(new_table, model_key)

    blockers: list[str] = []
    if not meta_present:
        blockers.append(f"{rge.meta_path(preds).name} is absent: the replicate has not completed (the meta is "
                        "written on completion; a parquet still being flushed must not be fit on)")
    if fit["n_neg"] < min_neg:
        blockers.append(f"n_neg={fit['n_neg']} < --min-neg {min_neg}")
    if n_neg_expected and fit["n_neg"] != n_neg_expected:
        blockers.append(f"n_neg={fit['n_neg']} != --n-neg-expected {n_neg_expected} (the protocol fits on ALL "
                        f"{n_neg_expected} negative non-anchor rows, never a subset)")
    if fit["n_neg_nan"] and not allow_nan_neg:
        blockers.append(f"{fit['n_neg_nan']} negative non-anchor row(s) have S NaN: top them up (--only-nan) "
                        "before fitting, or --allow-nan-neg for a dev fit")
    if fit["t_A"] is None or fit["t_B"] is None:
        blockers.append("a threshold is +inf (no negative score reaches the FPR target)")
    half = _uniq(df, "half")
    if half is not None and half != DESIGN_HALF:
        blockers.append(f"preds half={half!r} is not {DESIGN_HALF!r} (calibration is design-only)")
    model = _uniq(df, "model")
    if model is not None and rte.model_key(model) != model_key:
        blockers.append(f"--model-key {model_key} disagrees with the parquet's model={model!r} "
                        f"(-> {rte.model_key(model)})")
    tau0_run = _uniq(df, "tau0")
    if tau0_run is not None and (len(tau0_run.split("|")) != 1 or abs(float(tau0_run) - TAU0) > _EPS):
        blockers.append(f"parquet tau0={tau0_run} != {TAU0} (tau0 must be unchanged)")
    if render_table(table) != text_before:
        blockers.append(f"{thresholds} is not in canonical json.dumps(indent=2)+newline form; "
                        "untouched parts could not be preserved byte-for-byte")
    arm = _uniq(df, "arm")
    if arm is not None and arm != ARM_EXPECTED:
        blockers.append(f"preds arm={arm!r} is not the registered calibration arm {ARM_EXPECTED!r}")

    res = CalibrationResult(
        model_key=model_key, preds=str(preds), preds_sha16=_util.sha_file(preds),
        run_tag=_uniq(df, "run_tag"), arm=arm, model=model, half=half,
        thinking=_uniq(df, "thinking"), effort=_uniq(df, "effort"),
        run_thresholds_sha16=_uniq(df, "thresholds_sha16"),
        fpr_a=fpr_a, fpr_b=fpr_b, min_neg=min_neg, n_neg_expected=int(n_neg_expected),
        allow_nan_neg=bool(allow_nan_neg), meta_present=meta_present, tau0=TAU0, **fit,
        thresholds=str(thresholds), key_present=model_key in table,
        existing_entry=existing if isinstance(existing, dict) else None, new_entry=entry,
        file_sha16_before=sha_before, file_sha16_after=sha_after,
        tuple_sha16_before=tsha_before, tuple_sha16_after=tsha_after,
        letter_source_before=src_before, letter_source_after=src_after,
        write_blockers=blockers)
    if not write:
        return res
    if blockers:
        raise SystemExit("[calib] REFUSED --write: " + "; ".join(blockers))

    archive_dir.mkdir(parents=True, exist_ok=True)
    arch = archive_path(archive_dir, model_key)
    shutil.copyfile(thresholds, arch)
    assert _util.sha_file(arch) == _util.sha_file(thresholds), "archive copy differs from source"
    thresholds.write_text(text_after)
    assert _util.sha_file(thresholds) == sha_after, "written thresholds file differs from the rendered text"
    res.written = True
    res.written_utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res.archive_path = str(arch)
    cal = Path(out) if out else archive_dir / f"{model_key}_calibration.json"
    cal.parent.mkdir(parents=True, exist_ok=True)
    res.calibration_path = str(cal)
    cal.write_text(json.dumps(res.model_dump(), indent=2) + "\n")
    return res


def summary(res: CalibrationResult) -> str:
    fmt = lambda v: "inf" if v is None else f"{v:.6g}"          # noqa: E731
    rec = lambda v: "n/a" if v is None else f"{v:.3f}"          # noqa: E731
    lines = [
        f"[calib] {res.model_key} on {Path(res.preds).name} (sha {res.preds_sha16}; arm={res.arm} "
        f"model={res.model} half={res.half} thinking={res.thinking} effort={res.effort})",
        f"[calib] rows {res.n_rows}: n_neg {res.n_neg} (+{res.n_neg_nan} NaN S), n_pos {res.n_pos} "
        f"(+{res.n_pos_nan} NaN S), anchors excluded {res.n_anchor_excluded}",
        f"[calib] t_A = {fmt(res.t_A)}  design FPR {res.fpr_A:.4f} ({res.n_neg_ge_tA}/{res.n_neg} >= t_A)  "
        f"design-positive recall {rec(res.recall_A)} ({res.n_pos_ge_tA}/{res.n_pos}) [report only]",
        f"[calib] t_B = {fmt(res.t_B)}  design FPR {res.fpr_B:.4f} ({res.n_neg_ge_tB}/{res.n_neg} >= t_B)  "
        f"design-positive recall {rec(res.recall_B)} ({res.n_pos_ge_tB}/{res.n_pos}) [report only]",
        f"[calib] tau0 {res.tau0} (unchanged); existing {res.model_key} entry: "
        f"{'absent' if not res.key_present else json.dumps(res.existing_entry)}",
        f"[calib] thresholds file sha16 {res.file_sha16_before} -> {res.file_sha16_after}; "
        f"thresholds_sha16 (tuple) {res.tuple_sha16_before} [{res.letter_source_before}] -> "
        f"{res.tuple_sha16_after} [{res.letter_source_after}]",
    ]
    if res.write_blockers:
        lines.append("[calib] write blockers: " + "; ".join(res.write_blockers))
    if res.written:
        lines.append(f"[calib] WROTE {res.thresholds}; archived previous to {res.archive_path}; "
                     f"record {res.calibration_path}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, default=PREDS_DEFAULT, help="design-half preds parquet")
    ap.add_argument("--model-key", default="opus5_api", help="thresholds_v2.json key to fit")
    ap.add_argument("--thresholds", type=Path, default=THRESHOLDS_DEFAULT)
    ap.add_argument("--fpr-a", type=float, default=0.01)
    ap.add_argument("--fpr-b", type=float, default=0.05)
    ap.add_argument("--min-neg", type=int, default=50, help="refuse --write below this many negatives")
    ap.add_argument("--n-neg-expected", type=int, default=N_NEG_EXPECTED,
                    help=f"refuse --write unless exactly this many finite-S negative non-anchor rows are present "
                         f"(the protocol's {N_NEG_EXPECTED}; 0 disables)")
    ap.add_argument("--allow-nan-neg", action="store_true",
                    help="tolerate negative rows with S NaN (dev fits only; the registered fit needs 0)")
    ap.add_argument("--write", action="store_true", help="archive + update the thresholds file")
    ap.add_argument("--archive-dir", type=Path, default=OUT,
                    help="where thresholds_v2.pre_<key>.json and <key>_calibration.json go")
    ap.add_argument("--out", type=Path, default=None,
                    help="calibration record path (default <archive-dir>/<model-key>_calibration.json)")
    a = ap.parse_args(argv)
    res = run(a.preds, a.model_key, a.thresholds, a.fpr_a, a.fpr_b, a.min_neg, a.write,
              a.archive_dir, a.out, n_neg_expected=a.n_neg_expected, allow_nan_neg=a.allow_nan_neg)
    print(summary(res), flush=True)
    print(json.dumps(res.model_dump(), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
