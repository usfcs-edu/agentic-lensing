#!/usr/bin/env python3
"""golden/analyze_truth.py — the pre-registered truth-eval endpoints (P1–P3 + secondaries).

Inputs: the runner's `outputs/preds_truth_{arm}_{model}_{split}_k{K}_r{k}.parquet` replicates
(one row per item, `to_row` columns: name, p_lens = S, grade_pred = letter, S_arb, p_evidence,
no_opinion_<role>, parse_fail_roles, cost_usd, escalate, ...), their `_votes.parquet` siblings
(one row per persona call with the raw JSON — for reason_audit) and `.meta.json` (the
completion marker with the tuple), `golden/truth_manifest.csv` (truth_class, is_positive,
is_anchor, is_stress, cowls_band, cowls_theta_E, layout, field_class, known_type,
centre_is_deflector) and `golden/truth_splits.csv` (half). The manifest is authoritative for
every label: the parquet's join columns are not trusted. A k=1 tuple is analysed as its arm
(`a1`); a k=3 replicate tuple of the same arm as `a1k3` (its replicates pooled, the flip
rates from them) — two registered tuples are never pooled into one number. `discover`
requires every parquet's `.meta.json` (on the holdout: REFUSES one without — an interrupted
or subset replicate is not a score-once record), asserts one tuple per (arm, K) group, that
each parquet holds exactly `meta.n` rows and, on the holdout, that `meta.n` equals the
number of manifest rows on that half; the tuple of every group is printed.

Endpoints (plan PART 2 "Evaluation without a human campaign"; every number with a CI):
  P1  recall of the split's positives at 5 % (and 10 %) FPR on the split's N1 negatives, per
      arm (`eval/score.recovery_at_fpr`: roc_curve convention, threshold = the last ROC point
      with FPR <= target, i.e. the 10th-highest negative score at 5 %/200; a0's ROC has 4
      points and every point is written out); Clopper–Pearson 95 % CI on the positives;
      exact McNemar between the arm and the baseline on the positives at each arm's OWN
      5 %-FPR threshold (one-sided binomial on the discordant pairs: 5 clean promotions and 0
      demotions -> p = 0.03125; the two-sided p is reported beside it).
  P2  the letters frozen on design hold their FPR on this split: FPR at t_A / t_B (the values
      the parquet rows were lettered with — the runner writes `t_A`/`t_B` on every row; the
      thresholds file via aggregate_v2.MODEL_KEYS is the fallback) and, independently, the
      share of negatives lettered A and A/B — Clopper–Pearson. The registered claim
      (REGISTRY.md, restated 2026-08-23): the holdout FPR is not significantly ABOVE its
      target, i.e. the CP 95 % LOWER bound at t_A <= 1 % and at t_B <= 5 % (`P2_holds_*`
      rows); with 200 negatives that admits <= 5/200 at t_A and <= 17/200 at t_B. The plan's
      original "upper CI <= 2.5 % / 7.5 %" is reported beside it (`P2_upper_ci_ok_*`) but is
      stricter than the rule it tests: an upper bound <= 2.5 % needs 0/200 although t_A is set
      at <= 2/200 on design.
  P3  fraction of positives lettered A/B (old scheme: 0/31 COWLS), Clopper–Pearson.
  Secondary: paired dAUC (`parity/phase_d_analysis.paired_boot`, 2,000, + `delong_p`) for
  arm − baseline, a2 − a0, a3 − a2 (the render effect), a1arb − a1 (S_arb vs S); the module
  RNG is RESEEDED `np.random.default_rng(2026)` before EVERY bootstrap so each endpoint is
  reproducible on its own (critique M14); per-stratum recall at the global 5 %-FPR threshold
  (cowls_band; theta_E <= 1 / 1–2"; galaxy vs cluster; layout; field_class) as counts and
  rates; Spearman(S, cowls_theta_E) on positives with a permutation p (registered: rho >=
  −0.2); forbidden-ground and locates-feature rates and per-role no_opinion rates from the
  votes (`reason_audit`); D-rate on stress_D (reported, not gating); letter / lens-call flip
  rate across replicate pairs (`eval/lensbench_gate.grade_flip_rate`); parse-failure rate
  (S is NaN -> excluded from every rate, counted here); cost per item; the anchors table
  (prediction vs observed letter / S / scale_class / contaminant per arm; anchors never count
  toward any endpoint above).

Outputs: `outputs/truth_results.csv` (statistic, arm, split, value, ci_lo, ci_hi, n),
`outputs/truth_anchors.csv`, `outputs/truth_summary.md`.

  python lensjudge/golden/analyze_truth.py --split holdout --model sonnet [--baseline a0]
      [--outputs-dir outputs] [--manifest golden/truth_manifest.csv]
      [--splits golden/truth_splits.csv] [--thresholds golden/thresholds_v2.json] [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import beta, binomtest, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.eval import lensbench_gate  # noqa: E402
from lensjudge.golden import _util, aggregate_v2, reason_audit  # noqa: E402
from lensjudge.parity import phase_d_analysis  # noqa: E402

OUT = config.OUT
MANIFEST = _util.HERE / "truth_manifest.csv"
SPLITS = _util.HERE / "truth_splits.csv"
THRESHOLDS = _util.HERE / "thresholds_v2.json"
ARMS = ("a0", "a1", "a2", "a3", "attr")
CRITIC_ROLES = ("artifact", "geometry", "morphology")
FPR_TARGETS = (0.05, 0.10)
P2_TARGETS = {"t_A": 0.01, "t_B": 0.05}          # design-FPR targets the letters were frozen at
P2_UPPER_PLAN = {"t_A": 0.025, "t_B": 0.075}     # the plan's original upper-CI wording (reported)
N_BOOT = 2000
SEED = _util.SEED
N_PERM = 2000
LENS_LETTERS = ("A", "B")
# The design anchors (PI-derived, design-only, never scored as truth) and the predictions
# written down before the first call (plan PART 2; REGISTRY.md › anchors overrides these
# when that table exists — see load_anchor_predictions).
ANCHORS = {
    "J20954380-1094330": {"rank": 15, "prediction": "letter A or B (theta_E 3.6\", group scale)"},
    "J18030075+2309921": {"rank": 7, "prediction": "|letter(7) - letter(14)| <= 1 on the same SW-only field, re-centred 1.17\" apart"},
    "J18030108+2309932": {"rank": 14, "prediction": "|letter(7) - letter(14)| <= 1 on the same SW-only field, re-centred 1.17\" apart"},
    "J5186648-1343587": {"rank": 16, "prediction": "scale_class cluster, deflector_is_centre false, not D"},
    "J18805344+1121596": {"rank": 13, "prediction": "letter D with spiral_arm upheld"},
}


# ------------------------------------------------------------------ small statistics
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI; (nan, nan) when n == 0."""
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def mcnemar_exact(b: int, c: int) -> tuple[float, float]:
    """(one-sided p that the arm promotes more than it demotes, two-sided p) on the discordant
    pairs b = arm detects & baseline misses, c = the reverse. b = 5, c = 0 -> 0.03125."""
    n = b + c
    if n == 0:
        return float("nan"), float("nan")
    return (float(binomtest(b, n, 0.5, alternative="greater").pvalue),
            float(binomtest(b, n, 0.5, alternative="two-sided").pvalue))


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, fpr_target: float) -> tuple[float, float, float]:
    """(recall, threshold, achieved fpr) at the last ROC point with FPR <= target — the
    `eval/score.recovery_at_fpr` convention (same recall), but on the FULL curve
    (drop_intermediate=False) so the threshold is exactly the design's "10th-highest negative
    score" and the achieved FPR is the real one (sklearn's default drops the corner of a
    horizontal run and would report the run's start)."""
    m = ~pd.isna(s)
    y, s = np.asarray(y)[m].astype(int), np.asarray(s)[m].astype(float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx]), float(thr[idx]), float(fpr[idx])


def roc_points(y: np.ndarray, s: np.ndarray) -> list[tuple[float, float, float]]:
    """Every (fpr, tpr, threshold) of the ROC — a0's 4-point curve is reported in full."""
    m = ~pd.isna(s)
    y, s = np.asarray(y)[m].astype(int), np.asarray(s)[m].astype(float)
    if len(np.unique(y)) < 2:
        return []
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    return [(float(f), float(t), float(h)) for f, t, h in zip(fpr, tpr, thr)]


def spearman_perm(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> tuple[float, float, int]:
    """Spearman rho with a permutation p (two-sided), own RNG."""
    m = ~(pd.isna(x) | pd.isna(y))
    x, y = np.asarray(x)[m].astype(float), np.asarray(y)[m].astype(float)
    n = len(x)
    if n < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), n
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        r = float(spearmanr(x, rng.permutation(y))[0])
        if abs(r) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, (cnt + 1) / (n_perm + 1), n


def reseed(seed: int = SEED) -> None:
    """phase_d_analysis.paired_boot draws from a MODULE-GLOBAL generator seeded at import;
    reseed before every endpoint so each one is reproducible on its own."""
    phase_d_analysis.RNG = np.random.default_rng(seed)


# ------------------------------------------------------------------ loading
def _read_csv(path: Path, **kw) -> pd.DataFrame:
    path = Path(path)
    if path.with_suffix(path.suffix + ".sha").exists():
        return _util.read_pinned(path, **kw)
    return pd.read_csv(path, **kw)


def _bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() in ("true", "1", "yes")).astype(bool)


def load_manifest(manifest: Path = MANIFEST, splits: Optional[Path] = SPLITS) -> pd.DataFrame:
    """Manifest indexed by name with normalised bools and the split half (from truth_splits
    when given, else the manifest's own `half`). `y` = is_positive & ~is_anchor (anchors never
    count); `is_negative` = truth_class == negative."""
    man = _read_csv(manifest, dtype={"name": str, "candidate_id": str})
    if "name" not in man.columns:
        man["name"] = man["candidate_id"].astype(str)
    for c in ("is_positive", "is_anchor", "is_stress", "centre_is_deflector", "in_frame"):
        man[c] = _bool(man[c]) if c in man.columns else False
    if splits is not None and Path(splits).exists():
        sp = _read_csv(splits, dtype={"candidate_id": str})
        man["half"] = man["name"].map(sp.set_index("candidate_id")["half"]).fillna(man.get("half", ""))
    man["half"] = man["half"].fillna("").astype(str)
    man["truth_class"] = man.get("truth_class", pd.Series("", index=man.index)).fillna("").astype(str)
    man["is_negative"] = man["truth_class"].eq("negative")
    man["y"] = (man["is_positive"] & ~man["is_anchor"]).astype(int)
    for c in ("cowls_band", "layout", "field_class", "known_type"):
        man[c] = man[c].fillna("").astype(str) if c in man.columns else ""
    man["cowls_theta_E"] = pd.to_numeric(man.get("cowls_theta_E", np.nan), errors="coerce")
    # the COWLS catalogue writes 0.0 where no Einstein radius was measured: a placeholder,
    # not a radius — it would enter the Spearman endpoint as the rank-lowest point
    man.loc[man["cowls_theta_E"] <= 0, "cowls_theta_E"] = np.nan
    return man.set_index("name")


def scale_group(row: pd.Series) -> str:
    """galaxy vs cluster scale from the truth class / known type (cowls and gLS/gLe/LeQ ->
    galaxy; lit_cluster or LeG/LeI -> cluster)."""
    tc, kt = str(row.get("truth_class", "")), str(row.get("known_type", ""))
    if tc == "cowls" or tc == "lit_galaxy" or kt in ("gLS", "gLe", "LeQ"):
        return "galaxy"
    if tc == "lit_cluster" or kt in ("LeG", "LeI"):
        return "cluster"
    return ""


def theta_bin(t) -> str:
    if pd.isna(t):
        return ""
    return "theta_E<=1" if t <= 1.0 else "theta_E_1-2" if t <= 2.0 else "theta_E>2"


_NAME_RE = re.compile(r"^preds_truth_(?P<arm>[a-z0-9]+)_(?P<model>[^_]+)_(?P<split>design|holdout)"
                      r"(?:_k(?P<K>\d+))?_r(?P<k>\d+)\.parquet$")


def group_label(arm: str, K: int) -> str:
    """The analysis name of a (arm, replicate-count) tuple group: `a1` for K=1, `a1k3` for K=3."""
    return arm if int(K) == 1 else f"{arm}k{int(K)}"


def read_meta(pred_path: Path) -> Optional[dict]:
    mp = pred_path.with_name(pred_path.stem + ".meta.json")
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except Exception:  # noqa: BLE001
        return None


def discover(outputs_dir: Path, model: str, split: str, arms=ARMS, strict: Optional[bool] = None,
             n_expected: Optional[int] = None) -> dict[str, list[Path]]:
    """{group label: [replicate parquet paths sorted by r]} for
    preds_truth_{arm}_{model}_{split}_k{K}_r{k}.parquet (the pre-k name `..._r{k}` is read as
    K=1). Every parquet's `.meta.json` is read; `strict` (default: True on the holdout) REFUSES
    a parquet without one, a group whose metas disagree on the tuple, a parquet whose row
    count differs from `meta.n`, and — with `n_expected` — a meta whose `n` is not the whole
    half. Off the holdout the same problems are printed as warnings (design smoke runs are
    partial on purpose)."""
    strict = (split == "holdout") if strict is None else strict
    groups: dict[tuple, list[Path]] = {}
    for p in sorted(Path(outputs_dir).glob(f"preds_truth_*_{_util.safe_name(model)}_{split}_*.parquet")):
        if p.name.endswith("_votes.parquet"):
            continue
        m = _NAME_RE.match(p.name)
        if not m or m.group("arm") not in arms or m.group("model") != _util.safe_name(model):
            continue
        K = int(m.group("K") or 1)
        groups.setdefault((m.group("arm"), K), []).append(p)
    found: dict[str, list[Path]] = {}
    problems: list[str] = []
    for (arm, K), paths in sorted(groups.items()):
        paths = sorted(paths, key=lambda p: int(_NAME_RE.match(p.name).group("k")))
        tuples = []
        for p in paths:
            meta = read_meta(p)
            if meta is None:
                problems.append(f"{p.name}: no .meta.json (interrupted or subset replicate)")
                continue
            tup = dict(meta.get("tuple") or {})
            tuples.append(json.dumps(tup, sort_keys=True))
            n_rows = len(pd.read_parquet(p, columns=["name"]))
            if meta.get("n") is not None and n_rows != int(meta["n"]):
                problems.append(f"{p.name}: {n_rows} rows but meta.n = {meta['n']}")
            if n_expected is not None and meta.get("n") is not None and int(meta["n"]) != int(n_expected):
                problems.append(f"{p.name}: meta.n = {meta['n']} but the {split} half has {n_expected} rows"
                                + (" (--limit/--ids-file subset)" if meta.get("limit") or meta.get("ids_file") else ""))
            if meta.get("limit") or meta.get("ids_file"):
                problems.append(f"{p.name}: scored a subset (limit={meta.get('limit')}, ids_file={meta.get('ids_file')})")
        if len(set(tuples)) > 1:
            problems.append(f"{arm} k={K}: replicates carry {len(set(tuples))} different tuples")
        label = group_label(arm, K)
        found[label] = paths
        if tuples:
            print(f"[analyze] {label}: {len(paths)} replicate(s), tuple {tuples[0]}")
    if problems:
        msg = "\n  ".join(problems)
        if strict:
            raise SystemExit(f"[analyze] REFUSED on the {split}:\n  {msg}")
        print(f"[analyze] WARNING ({split}, not enforced):\n  {msg}")
    return found


def votes_for(pred_path: Path) -> Optional[pd.DataFrame]:
    v = pred_path.with_name(pred_path.stem + "_votes.parquet")
    return pd.read_parquet(v) if v.exists() else None


def load_arms(found: dict[str, list[Path]]) -> tuple[dict[str, list[pd.DataFrame]], dict[str, list[pd.DataFrame]]]:
    preds = {arm: [pd.read_parquet(p) for p in paths] for arm, paths in found.items()}
    votes = {arm: [v for v in (votes_for(p) for p in paths) if v is not None] for arm, paths in found.items()}
    return preds, votes


def thresholds_from_preds(reps: list[pd.DataFrame]) -> Optional[dict]:
    """The t_A / t_B / tau0 the rows were actually lettered with (the runner writes them on
    every row); None when the columns are absent; raises when replicates disagree."""
    vals = {}
    for key in ("t_A", "t_B", "tau0"):
        seen = set()
        for r in reps:
            if key in r.columns:
                seen |= set(pd.to_numeric(r[key], errors="coerce").dropna().round(6).unique().tolist())
        if len(seen) > 1:
            raise SystemExit(f"[analyze] replicates carry different {key} values: {sorted(seen)}")
        vals[key] = seen.pop() if seen else None
    if vals["t_A"] is None or vals["t_B"] is None:
        return None
    src = set()
    for r in reps:
        if "letter_source" in r.columns:
            src |= set(r["letter_source"].dropna().astype(str))
    return {**vals, "source": "+".join(sorted(src)) or "parquet"}


def derive_arms(preds: dict[str, list[pd.DataFrame]]) -> dict[str, list[pd.DataFrame]]:
    """Add `a1arb` (a1 scored by S_arb, lettered by its own column when present) so the
    arbitrator arm is analysed like any other arm."""
    out = dict(preds)
    if "a1" in preds and all("S_arb" in r.columns for r in preds["a1"]):
        reps = []
        for r in preds["a1"]:
            d = r.copy()
            d["p_lens"] = pd.to_numeric(d["S_arb"], errors="coerce")
            if "letter_arb" in d.columns:
                d["grade_pred"] = d["letter_arb"]
            reps.append(d)
        out["a1arb"] = reps
    return out


def pooled(reps: list[pd.DataFrame]) -> pd.DataFrame:
    """One row per name: mean score over replicates (NaN ignored; all-NaN stays NaN), the
    modal letter (first replicate breaks ties), mean cost, any-parse-failure flag."""
    cat = pd.concat([r.assign(_k=i) for i, r in enumerate(reps)], ignore_index=True)
    cat["p_lens"] = pd.to_numeric(cat["p_lens"], errors="coerce")
    g = cat.groupby("name", sort=False)
    out = pd.DataFrame({"score": g["p_lens"].mean()})
    out["letter"] = g["grade_pred"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else "")
    if "cost_usd" in cat.columns:
        out["cost_usd"] = g["cost_usd"].mean()
    if "parse_fail_roles" in cat.columns:
        out["parse_fail"] = g["parse_fail_roles"].agg(lambda s: bool(s.fillna("").astype(str).str.strip().ne("").any()))
    else:
        out["parse_fail"] = out["score"].isna()
    for c in ("scale_class", "contaminant", "escalate", "S", "S_arb", "p_evidence", "letter_llm", "rationale"):
        if c in cat.columns:
            out[c] = g[c].first()
    for role in CRITIC_ROLES:
        c = f"no_opinion_{role}"
        if c in cat.columns:
            out[c] = g[c].mean()
    return out


# ------------------------------------------------------------------ endpoints
def _row(stat: str, arm: str, split: str, value, lo=np.nan, hi=np.nan, n=np.nan) -> dict:
    f = lambda v: float(v) if v is not None and not (isinstance(v, str)) else v  # noqa: E731
    return {"statistic": stat, "arm": arm, "split": split, "value": f(value),
            "ci_lo": f(lo), "ci_hi": f(hi), "n": (int(n) if not pd.isna(n) else np.nan)}


def join_arm(man: pd.DataFrame, pool: pd.DataFrame, split: str) -> pd.DataFrame:
    """Manifest rows of this split that the arm scored (anchors kept but y = 0; the
    endpoints select on the flags)."""
    j = man[man["half"] == split].join(pool, how="inner")
    j["scale_group"] = j.apply(scale_group, axis=1)
    j["theta_bin"] = j["cowls_theta_E"].map(theta_bin)
    return j


def _endpoint_frame(j: pd.DataFrame) -> pd.DataFrame:
    """Positives (non-anchor) + N1 negatives with a finite score — the P1 population."""
    sel = ((j["y"] == 1) | j["is_negative"]) & ~j["is_anchor"] & j["score"].notna()
    return j[sel]


def recall_rows(arm: str, split: str, j: pd.DataFrame) -> tuple[list[dict], dict]:
    """P1 per arm: recall at each FPR target with CP CI, the thresholds, the full ROC for a
    4-point arm, and the per-positive detection vector at 5 % (for McNemar)."""
    rows, det = [], {}
    e = _endpoint_frame(j)
    y, s = e["y"].to_numpy(int), e["score"].to_numpy(float)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    rows.append(_row("n_positives_scored", arm, split, n_pos, n=n_pos))
    rows.append(_row("n_negatives_scored", arm, split, n_neg, n=n_neg))
    for ft in FPR_TARGETS:
        rec, thr, fpr_ach = threshold_at_fpr(y, s, ft)
        k = int(round(rec * n_pos)) if not np.isnan(rec) else 0
        lo, hi = clopper_pearson(k, n_pos)
        tag = f"{int(ft * 100)}pct"
        rows.append(_row(f"P1_recall_at_fpr{tag}", arm, split, rec, lo, hi, n_pos))
        rows.append(_row(f"P1_threshold_at_fpr{tag}", arm, split, thr, n=n_neg))
        rows.append(_row(f"P1_achieved_fpr{tag}", arm, split, fpr_ach, n=n_neg))
        if ft == 0.05 and not np.isnan(thr):
            det = {"thr": thr, "detected": (e.loc[e["y"] == 1, "score"] >= thr)}
    pts = roc_points(y, s)
    if 0 < len(pts) <= 8:           # a0's pass-count ROC: report every point
        for i, (f, t, h) in enumerate(pts):
            rows.append(_row(f"P1_roc_point{i}_fpr", arm, split, f, n=n_neg))
            rows.append(_row(f"P1_roc_point{i}_tpr", arm, split, t, n=n_pos))
            rows.append(_row(f"P1_roc_point{i}_thr", arm, split, h))
    if n_pos and n_neg:
        rows.append(_row("AUC", arm, split, float(roc_auc_score(y, s)), n=n_pos + n_neg))
    return rows, det


def mcnemar_rows(arm: str, base: str, split: str, det_arm: dict, det_base: dict) -> list[dict]:
    """Exact McNemar on the shared positives at each arm's own 5 %-FPR threshold."""
    if not det_arm or not det_base:
        return []
    a, b = det_arm["detected"], det_base["detected"]
    shared = a.index.intersection(b.index)
    a, b = a.loc[shared].astype(bool), b.loc[shared].astype(bool)
    promo, demo = int((a & ~b).sum()), int((~a & b).sum())
    p1, p2 = mcnemar_exact(promo, demo)
    tag = f"{arm}_vs_{base}"
    return [_row("P1_mcnemar_promotions", tag, split, promo, n=len(shared)),
            _row("P1_mcnemar_demotions", tag, split, demo, n=len(shared)),
            _row("P1_mcnemar_p_onesided", tag, split, p1, n=promo + demo),
            _row("P1_mcnemar_p_twosided", tag, split, p2, n=promo + demo)]


def letter_rows(arm: str, split: str, j: pd.DataFrame, thresholds: Optional[dict]) -> list[dict]:
    """P2 (FPR of the frozen letters / thresholds on the negatives) and P3 (positives at A/B)."""
    rows = []
    neg = j[j["is_negative"] & ~j["is_anchor"] & j["score"].notna()]
    pos = j[(j["y"] == 1) & j["score"].notna()]
    n_neg, n_pos = len(neg), len(pos)
    for name, letters in (("A", ("A",)), ("AB", ("A", "B"))):
        k = int(neg["letter"].isin(letters).sum())
        lo, hi = clopper_pearson(k, n_neg)
        rows.append(_row(f"P2_fpr_letter_{name}", arm, split, k / n_neg if n_neg else np.nan, lo, hi, n_neg))
    if thresholds and not arm.startswith("a0"):      # a0's score is a pass-count, not S
        for key in ("t_A", "t_B"):
            t = thresholds.get(key)
            if t is None:
                continue
            k = int((neg["score"] >= float(t)).sum())
            lo, hi = clopper_pearson(k, n_neg)
            rows.append(_row(f"P2_fpr_at_{key}", arm, split, k / n_neg if n_neg else np.nan, lo, hi, n_neg))
            rows.append(_row(f"P2_{key}", arm, split, float(t)))
            # the registered P2 test: not significantly above the design target (CP lower
            # bound <= target); the plan's upper-CI wording reported beside it
            if n_neg:
                rows.append(_row(f"P2_holds_{key}", arm, split, float(lo <= P2_TARGETS[key]), n=n_neg))
                rows.append(_row(f"P2_upper_ci_ok_{key}", arm, split, float(hi <= P2_UPPER_PLAN[key]), n=n_neg))
    k = int(pos["letter"].isin(LENS_LETTERS).sum())
    lo, hi = clopper_pearson(k, n_pos)
    rows.append(_row("P3_positives_at_AB", arm, split, k / n_pos if n_pos else np.nan, lo, hi, n_pos))
    for L in "ABCD":
        rows.append(_row(f"letter_share_positives_{L}", arm, split,
                         float(pos["letter"].eq(L).mean()) if n_pos else np.nan, n=n_pos))
        rows.append(_row(f"letter_share_negatives_{L}", arm, split,
                         float(neg["letter"].eq(L).mean()) if n_neg else np.nan, n=n_neg))
    return rows


def dauc_rows(tag: str, split: str, ja: pd.DataFrame, jb: pd.DataFrame, n_boot: int, seed: int) -> list[dict]:
    """Paired dAUC(a − b) on the shared P1 population; the module RNG reseeded first."""
    ea, eb = _endpoint_frame(ja), _endpoint_frame(jb)
    shared = ea.index.intersection(eb.index)
    if len(shared) == 0:
        return []
    y = ea.loc[shared, "y"].to_numpy(int)
    a, b = ea.loc[shared, "score"].to_numpy(float), eb.loc[shared, "score"].to_numpy(float)
    if len(np.unique(y)) < 2:
        return []
    reseed(seed)
    d, lo, hi, n = phase_d_analysis.paired_boot(y, a, b, n=n_boot)
    p = phase_d_analysis.delong_p(y, a, b)
    return [_row("dAUC", tag, split, d, lo, hi, n), _row("dAUC_delong_p", tag, split, p, n=n)]


def stratum_rows(arm: str, split: str, j: pd.DataFrame, thr: float) -> list[dict]:
    """Per-stratum recall at the arm's global 5 %-FPR threshold (counts and CP CIs)."""
    rows = []
    if thr is None or np.isnan(thr):
        return rows
    pos = j[(j["y"] == 1) & j["score"].notna()]
    strata = {"cowls_band": pos["cowls_band"], "theta_bin": pos["theta_bin"],
              "scale_group": pos["scale_group"], "layout": pos["layout"],
              "field_class": pos["field_class"], "truth_class": pos["truth_class"],
              "centre_is_deflector": pos["centre_is_deflector"].map({True: "centre", False: "offcentre"})}
    for name, col in strata.items():
        for val, sub in pos.groupby(col.fillna("").astype(str)):
            if val == "":
                continue
            k, n = int((sub["score"] >= thr).sum()), len(sub)
            lo, hi = clopper_pearson(k, n)
            rows.append(_row(f"recall_at_fpr5pct[{name}={val}]", arm, split, k / n, lo, hi, n))
            rows.append(_row(f"n_detected[{name}={val}]", arm, split, k, n=n))
    return rows


def theta_rows(arm: str, split: str, j: pd.DataFrame, n_perm: int, seed: int) -> list[dict]:
    pos = j[(j["y"] == 1) & j["score"].notna() & j["cowls_theta_E"].notna()]
    rho, p, n = spearman_perm(pos["score"].to_numpy(float), pos["cowls_theta_E"].to_numpy(float), n_perm, seed)
    return [_row("spearman_S_vs_theta_E", arm, split, rho, n=n), _row("spearman_S_vs_theta_E_perm_p", arm, split, p, n=n)]


def quality_rows(arm: str, split: str, j: pd.DataFrame, reps: list[pd.DataFrame],
                 votes: list[pd.DataFrame]) -> list[dict]:
    """Parse failures, cost, no_opinion per role, D-rate on stress_D, escalation, flip rates,
    forbidden-ground / locates-feature rates from the votes."""
    rows = []
    n_all = len(j)
    if "parse_fail" in j.columns:
        k = int(j["parse_fail"].astype(bool).sum())
        lo, hi = clopper_pearson(k, n_all)
        rows.append(_row("parse_failure_rate", arm, split, k / n_all if n_all else np.nan, lo, hi, n_all))
    if "cost_usd" in j.columns:
        rows.append(_row("cost_usd_per_item", arm, split, float(pd.to_numeric(j["cost_usd"], errors="coerce").mean()), n=n_all))
    for role in CRITIC_ROLES:
        c = f"no_opinion_{role}"
        if c in j.columns and j[c].notna().any():
            rows.append(_row(f"no_opinion_rate[{role}]", arm, split, float(pd.to_numeric(j[c], errors="coerce").mean()),
                             n=int(j[c].notna().sum())))
    if "escalate" in j.columns:
        e = _bool(j["escalate"].fillna(False))
        rows.append(_row("escalation_rate", arm, split, float(e.mean()) if n_all else np.nan, n=n_all))
    sd = j[j["truth_class"].eq("stress_D") & j["score"].notna()]
    if len(sd):
        k = int(sd["letter"].eq("D").sum())
        lo, hi = clopper_pearson(k, len(sd))
        rows.append(_row("D_rate_stress_D", arm, split, k / len(sd), lo, hi, len(sd)))
    for tc in ("stress_U", "anomalymatch"):
        st = j[j["truth_class"].eq(tc) & j["score"].notna()]
        if len(st):
            rows.append(_row(f"AB_rate_{tc}", arm, split, float(st["letter"].isin(LENS_LETTERS).mean()), n=len(st)))
            rows.append(_row(f"mean_S_{tc}", arm, split, float(st["score"].mean()), n=len(st)))
    # replicate stability: every pair of replicates through the repo's flip-rate helper
    if len(reps) >= 2:
        gf, lf, dp = [], [], []
        for i in range(len(reps)):
            for k2 in range(i + 1, len(reps)):
                m = lensbench_gate.grade_flip_rate(reps[i], reps[k2])
                if m.get("n_shared"):
                    gf.append(m["grade_flip_rate"]); lf.append(m["lens_call_flip_rate"]); dp.append(m["mean_abs_p_lens_delta"])
        if gf:
            rows.append(_row("letter_flip_rate[replicate pairs]", arm, split, float(np.mean(gf)), n=len(gf)))
            rows.append(_row("lens_call_flip_rate[replicate pairs]", arm, split, float(np.mean(lf)), n=len(gf)))
            rows.append(_row("mean_abs_S_delta[replicate pairs]", arm, split, float(np.mean(dp)), n=len(gf)))
    # refutation grounds from the votes (new scheme) — restricted to this split's items
    if votes:
        crit = pd.concat([reason_audit.votes_to_critics(v) for v in votes], ignore_index=True)
        crit = crit[crit["id"].isin(j.index)]
        if len(crit):
            t = reason_audit.audit_table(crit)
            n_ref = int(t.loc["n_refutations", "all"])
            for cat in (reason_audit.FORBIDDEN + reason_audit.FORBIDDEN_FLAGS + reason_audit.USAGE_FLAGS
                        + ("forbidden_only", "locates_feature", "any_ground")):
                v = float(t.loc[cat, "all"])
                k = int(round(v * n_ref)) if n_ref else 0
                lo, hi = clopper_pearson(k, n_ref)
                rows.append(_row(f"refutation_{cat}_rate", arm, split, v, lo, hi, n_ref))
            for role in CRITIC_ROLES:
                if role in t.columns:
                    rows.append(_row(f"refutation_forbidden_only_rate[{role}]", arm, split,
                                     float(t.loc["forbidden_only", role]), n=int(t.loc["n_refutations", role])))
            rows.append(_row("n_refutations", arm, split, n_ref, n=n_ref))
            for role in CRITIC_ROLES:
                sub = crit[crit["persona"] == role]
                if len(sub):
                    rows.append(_row(f"no_opinion_rate_votes[{role}]", arm, split, float(sub["no_opinion"].mean()), n=len(sub)))
    return rows


# ------------------------------------------------------------------ anchors
def load_anchor_predictions(registry_md: Optional[Path] = None) -> dict:
    """ANCHORS, overridden by a `## Truth-eval anchors` table in REGISTRY.md when one exists
    (columns candidate_id, rank, prediction)."""
    preds = {k: dict(v) for k, v in ANCHORS.items()}
    p = Path(registry_md) if registry_md else _util.HERE / "REGISTRY.md"
    if p.exists():
        try:
            from lensjudge.golden.run_golden_eval import _tables
            for r in _tables(p.read_text()).get("Truth-eval anchors", []):
                cid = r.get("candidate_id") or r.get("id")
                if cid:
                    preds[cid] = {"rank": r.get("rank", preds.get(cid, {}).get("rank", "")),
                                  "prediction": r.get("prediction", preds.get(cid, {}).get("prediction", ""))}
        except Exception:  # noqa: BLE001
            pass
    return preds


def anchors_table(man: pd.DataFrame, pools: dict[str, pd.DataFrame], split: str,
                  predictions: Optional[dict] = None) -> pd.DataFrame:
    """One row per (anchor, arm): predicted vs observed letter / S / scale_class / contaminant."""
    predictions = predictions or load_anchor_predictions()
    rows = []
    anchors = man[man["is_anchor"]] if man["is_anchor"].any() else man[man.index.isin(predictions)]
    for cid, r in anchors.iterrows():
        pr = predictions.get(cid, {})
        for arm, pool in pools.items():
            if cid not in pool.index:
                continue
            p = pool.loc[cid]
            rows.append({"candidate_id": cid, "rank": pr.get("rank", ""), "split": split, "arm": arm,
                         "prediction": pr.get("prediction", ""), "letter": p.get("letter", ""),
                         "S": p.get("score", np.nan), "S_arb": p.get("S_arb", np.nan),
                         "p_evidence": p.get("p_evidence", np.nan), "scale_class": p.get("scale_class", ""),
                         "contaminant": p.get("contaminant", ""), "letter_llm": p.get("letter_llm", ""),
                         "rationale": str(p.get("rationale", ""))[:300]})
    cols = ["candidate_id", "rank", "split", "arm", "prediction", "letter", "S", "S_arb", "p_evidence",
            "scale_class", "contaminant", "letter_llm", "rationale"]
    return pd.DataFrame(rows, columns=cols)


# ------------------------------------------------------------------ driver
def analyze(man: pd.DataFrame, arms: dict[str, list[pd.DataFrame]], split: str, baseline: str = "a0",
            thresholds: Optional[dict] = None, votes: Optional[dict[str, list[pd.DataFrame]]] = None,
            n_boot: int = N_BOOT, seed: int = SEED, n_perm: int = N_PERM,
            predictions: Optional[dict] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All endpoints for one split -> (results long table, anchors table)."""
    votes = votes or {}
    arms = derive_arms(arms)
    pools = {arm: pooled(reps) for arm, reps in arms.items() if reps}
    joined = {arm: join_arm(man, pool, split) for arm, pool in pools.items()}
    rows: list[dict] = []
    det: dict[str, dict] = {}
    for arm, j in joined.items():
        r, d = recall_rows(arm, split, j)
        rows += r
        det[arm] = d
        # the thresholds the rows were lettered with take precedence over the file's
        thr_arm = thresholds_from_preds(arms[arm]) or thresholds
        rows += letter_rows(arm, split, j, thr_arm)
        rows += stratum_rows(arm, split, j, d.get("thr", np.nan) if d else np.nan)
        rows += theta_rows(arm, split, j, n_perm, seed)
        # derived arms (a1arb) share a1's calls: their votes-based audits would only repeat a1's
        rows += quality_rows(arm, split, j, arms[arm], votes.get(arm, []))
    for arm in joined:
        if arm != baseline and baseline in joined:
            rows += mcnemar_rows(arm, baseline, split, det.get(arm, {}), det.get(baseline, {}))
            rows += dauc_rows(f"{arm}_minus_{baseline}", split, joined[arm], joined[baseline], n_boot, seed)
    for a, b in (("a3", "a2"), ("a1arb", "a1"), ("attr", "a1")):
        if a in joined and b in joined and b != baseline:
            rows += dauc_rows(f"{a}_minus_{b}", split, joined[a], joined[b], n_boot, seed)
            rows += mcnemar_rows(a, b, split, det.get(a, {}), det.get(b, {}))
    res = pd.DataFrame(rows, columns=["statistic", "arm", "split", "value", "ci_lo", "ci_hi", "n"])
    anchors = anchors_table(man, pools, split, predictions)
    return res, anchors


def summary_md(res: pd.DataFrame, anchors: pd.DataFrame, split: str, baseline: str) -> str:
    """A short markdown digest of the headline rows (the CSV is the record)."""
    L = [f"# Truth-eval endpoints — split `{split}` (baseline `{baseline}`)", ""]

    def get(stat, arm):
        r = res[(res["statistic"] == stat) & (res["arm"] == arm)]
        return None if r.empty else r.iloc[0]

    arms = [a for a in res["arm"].unique() if "_vs_" not in a and "_minus_" not in a]
    L += ["| arm | n_pos | n_neg | recall@5%FPR [CI] | recall@10%FPR | AUC | P2 FPR(A) [CI] | P2 FPR(A/B) [CI] | P3 pos@A/B [CI] | parse-fail | $/item |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        def fmt(stat, ci=True, pct=True):
            r = get(stat, a)
            if r is None or pd.isna(r["value"]):
                return "—"
            v = r["value"]
            s = f"{v:.0%}" if pct else f"{v:.3f}"
            if ci and not pd.isna(r["ci_lo"]):
                s += f" [{r['ci_lo']:.0%}, {r['ci_hi']:.0%}]" if pct else f" [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            return s
        npos = get("n_positives_scored", a); nneg = get("n_negatives_scored", a)
        L.append(f"| {a} | {int(npos['value']) if npos is not None else '—'} | {int(nneg['value']) if nneg is not None else '—'} | "
                 f"{fmt('P1_recall_at_fpr5pct')} | {fmt('P1_recall_at_fpr10pct', ci=False)} | {fmt('AUC', ci=False, pct=False)} | "
                 f"{fmt('P2_fpr_letter_A')} | {fmt('P2_fpr_letter_AB')} | {fmt('P3_positives_at_AB')} | "
                 f"{fmt('parse_failure_rate', ci=False)} | {fmt('cost_usd_per_item', ci=False, pct=False)} |")
    tags = sorted(set(res.loc[res["arm"].str.contains("_vs_"), "arm"]) |
                  {t.replace("_minus_", "_vs_") for t in res.loc[res["arm"].str.contains("_minus_"), "arm"]})
    if tags:
        L += ["", "## Paired comparisons", "", "| comparison | promotions | demotions | McNemar p (1-sided) | dAUC [CI] | DeLong p |",
              "|---|---|---|---|---|---|"]
        for t in tags:
            pr, de, p = get("P1_mcnemar_promotions", t), get("P1_mcnemar_demotions", t), get("P1_mcnemar_p_onesided", t)
            d, dp = get("dAUC", t.replace("_vs_", "_minus_")), get("dAUC_delong_p", t.replace("_vs_", "_minus_"))
            cells = [t.replace("_vs_", " vs "),
                     "—" if pr is None else str(int(pr["value"])),
                     "—" if de is None else str(int(de["value"])),
                     "—" if p is None or pd.isna(p["value"]) else f"{p['value']:.4f}",
                     "—" if d is None else f"{d['value']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]",
                     "—" if dp is None or pd.isna(dp["value"]) else f"{dp['value']:.3g}"]
            L.append("| " + " | ".join(cells) + " |")
    sec = res[res["statistic"].str.startswith(("spearman", "refutation_", "no_opinion", "D_rate", "letter_flip", "lens_call_flip", "escalation"))]
    if len(sec):
        L += ["", "## Secondary monitors", "", "| statistic | arm | value | CI | n |", "|---|---|---|---|---|"]
        for _, r in sec.iterrows():
            ci = "" if pd.isna(r["ci_lo"]) else f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            L.append(f"| {r['statistic']} | {r['arm']} | {r['value']:.3f} | {ci} | {'' if pd.isna(r['n']) else int(r['n'])} |")
    if len(anchors):
        L += ["", "## Design anchors (never scored as truth)", "", "| rank | id | arm | prediction | letter | S | scale_class | contaminant |", "|---|---|---|---|---|---|---|---|"]
        for _, r in anchors.iterrows():
            L.append(f"| {r['rank']} | {r['candidate_id']} | {r['arm']} | {r['prediction']} | {r['letter']} | "
                     f"{'' if pd.isna(r['S']) else f'{r['S']:.3f}'} | {r['scale_class']} | {r['contaminant']} |")
    return "\n".join(L) + "\n"


def write_outputs(res: pd.DataFrame, anchors: pd.DataFrame, split: str, baseline: str,
                  prefix: Path) -> tuple[Path, Path, Path]:
    """<prefix>_results.csv / <prefix>_anchors.csv / <prefix>_summary_<split>.md. The two CSVs
    are MERGED by split (rows of this split replaced, other splits kept) so design and holdout
    live in one record; the summary is per split."""
    prefix = Path(prefix)
    res_path = prefix.with_name(prefix.name + "_results.csv")
    anc_path = prefix.with_name(prefix.name + "_anchors.csv")
    md_path = prefix.with_name(prefix.name + f"_summary_{split}.md")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    for path, df in ((res_path, res), (anc_path, anchors)):
        if path.exists():
            old = pd.read_csv(path)
            if "split" in old.columns:
                old = old[old["split"] != split]
                df = pd.concat([old, df], ignore_index=True)
        df.to_csv(path, index=False)
    md_path.write_text(summary_md(res, anchors, split, baseline))
    return res_path, anc_path, md_path


def load_thresholds(path: Path, model: str) -> Optional[dict]:
    """thresholds_v2.json -> the model's {t_A, t_B, tau0} via aggregate_v2.MODEL_KEYS (the
    runner's key; provisional when null/absent). The FALLBACK only: `analyze` prefers the
    t_A/t_B written on the parquet rows."""
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    key = aggregate_v2.MODEL_KEYS.get(model, f"{model}_api")
    t = d.get(key)
    if not t or t.get("t_A") is None or t.get("t_B") is None:
        t = dict(d.get("provisional", {}) or {})
        t["source"] = "provisional"
    else:
        t = dict(t); t["source"] = f"{key}_calibrated"
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("design", "holdout"), default="holdout")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--baseline", default="a0")
    ap.add_argument("--outputs-dir", type=Path, default=OUT)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--splits", type=Path, default=SPLITS)
    ap.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-prefix", type=Path, help="default <outputs-dir>/truth -> truth_results.csv, truth_anchors.csv, truth_summary_<split>.md")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="holdout: analyse parquets without .meta.json / with a subset (NOT a score-once record)")
    a = ap.parse_args(argv)
    man = load_manifest(a.manifest, a.splits)
    n_half = int((man["half"] == a.split).sum())
    found = discover(a.outputs_dir, a.model, a.split, strict=(a.split == "holdout" and not a.allow_incomplete),
                     n_expected=(n_half if a.split == "holdout" else None))
    if not found:
        raise SystemExit(f"no preds_truth_*_{a.model}_{a.split}_*.parquet under {a.outputs_dir}")
    print("arms:", {k: len(v) for k, v in found.items()})
    preds, votes = load_arms(found)
    thr = load_thresholds(a.thresholds, a.model)
    res, anchors = analyze(man, preds, a.split, a.baseline, thr, votes, a.n_boot, a.seed)
    res_path, anc_path, md_path = write_outputs(res, anchors, a.split, a.baseline, a.out_prefix or (a.outputs_dir / "truth"))
    md = summary_md(res, anchors, a.split, a.baseline)
    print(md)
    print(f"wrote {res_path} ({len(res)} rows), {anc_path} ({len(anchors)} rows), {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
