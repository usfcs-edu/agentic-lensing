#!/usr/bin/env python3
"""golden/analyze_golden.py — the pre-registered endpoints of the golden model arms (E1/E2/E3).

run_golden_eval.py writes one parquet per (arm, model, split, replicate) and run_batch.summarize
prints the repo's historical A/B/C-vs-D confusion after each — that line is NOT the golden
endpoint. The endpoint definitions live in golden/REGISTRY.md ("Endpoints"); this script is
their only implementation, so the number that gets quoted is computed once, the same way,
from the manifest's `binary_label` (XH score >= 3) and never from a per-run printout:

  primary    E2 - E1 paired dAUC on the validate half (parity/phase_d_analysis.paired_boot:
             class-stratified paired bootstrap, 2,000 resamples, seed 2026; + delong_p), per
             replicate k and POOLED (mean score per unit over the replicates of each arm);
             and dPurity at recall 0.8 (purity = precision of the smallest top-ranked set that
             recovers 80% of the score >= 3 units), same paired bootstrap.
  secondary  absolute AUC + CI per arm (pooled and per replicate), by stratum (n >= 8 per
             class); E1 vs the incumbent `p_pipeline` BOTH on the pipeline-flagged rows and
             on all rows (unflagged = 0, the rank below everything it surfaced); raw QWK of
             the predicted letter vs XH's letter; machine self-consistency QWK across the
             replicates of one arm; recall of K_cowls / L_known per arm; E3 - E1 by the same
             dAUC when E3 parquets are given.

Scores: `--score p_lens` (Anthropic path, default) or `s_exp` / `p_lens_logprob` (open
backend; `s_exp` is llm_client.ORDINAL_W everywhere). A parse failure is scored 0 (no lens
evidence; eval/score.py convention) and counted.

  python lensjudge/golden/analyze_golden.py --split validate \\
      --e1 "outputs/preds_golden_e1_sonnet_validate_r*.parquet" \\
      --e2 "outputs/preds_golden_e2_sonnet_validate_r*.parquet" \\
      [--e3 "outputs/preds_golden_e3_sonnet_validate_r*.parquet"] \\
      --manifest outputs/golden_jwst_manifest.csv --out outputs/golden_results.csv

Writes `statistic,value,ci_lo,ci_hi` rows (+ a .json twin with n's and p-values). Every row
is labelled "pilot" when the manifest has the lite frame's 250 units (plan: lite numbers are
pilot numbers; anything promoted re-verifies on the next draw).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score, roc_auc_score  # noqa: E402

from lensjudge.golden import _util  # noqa: E402
from lensjudge.parity import phase_d_analysis  # noqa: E402

OUT = _util.LENSJUDGE / "outputs"
N_BOOT = 2000
RECALL_TARGET = 0.8
LENS_MIN_SCORE = 3
MIN_STRATUM_CLASS = 8          # a per-stratum AUC needs >= this many of each class
SCORES = ("p_lens", "s_exp", "p_lens_logprob")
LETTERS = ("A", "B", "C", "D")
LETTER_ORD = _util.LETTER_TO_ORD


# ------------------------------------------------------------------ inputs
def _read(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.with_suffix(path.suffix + ".sha").exists():
        return _util.read_pinned(path)
    return pd.read_csv(path)


def load_manifest(path: Path, split: str | None) -> pd.DataFrame:
    """The eval manifest (build_eval_manifest.py) restricted to one split; the truth columns."""
    m = _read(path)
    if split is not None and "split" in m.columns:
        m = m[m["split"].astype(str) == split]
    if m.empty:
        raise SystemExit(f"no rows for split={split!r} in {path}")
    m = m.copy()
    m["name"] = m["name"].astype(str)
    m["y"] = (pd.to_numeric(m["score_1_4"]).astype(int) >= LENS_MIN_SCORE).astype(int)
    if "binary_label" in m.columns:       # the manifest's own column must agree with the cut
        assert (m["y"] == (m["binary_label"].astype(str) == "lens").astype(int)).all(), \
            "manifest binary_label is not the score>=3 cut (rebuild it with build_eval_manifest.py)"
    m["letter"] = pd.to_numeric(m["score_1_4"]).astype(int).map(_util.score_to_letter)
    return m.set_index("name")


def load_replicates(patterns: list[str], score: str) -> list[pd.DataFrame]:
    """One DataFrame per parquet (name, score, grade_pred, parse_ok), in sorted path order."""
    paths: list[Path] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        paths += [Path(h) for h in hits] if hits else [Path(pat)]
    reps = []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"predictions parquet not found: {p}")
        df = pd.read_parquet(p)
        if score not in df.columns:
            raise SystemExit(f"{p}: no column {score!r} (have {sorted(df.columns)[:12]}...)")
        s = pd.to_numeric(df[score], errors="coerce")
        ok = df["parse_ok"].astype(bool) if "parse_ok" in df.columns else s.notna()
        reps.append(pd.DataFrame({"name": df["name"].astype(str), "score": s.where(ok, 0.0).fillna(0.0),
                                  "grade_pred": df.get("grade_pred", pd.Series([None] * len(df))),
                                  "parse_ok": ok.to_numpy(bool), "path": str(p)}).drop_duplicates("name"))
    return reps


def pooled(reps: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean score per name over the replicates (+ modal predicted letter)."""
    allr = pd.concat(reps, ignore_index=True)
    g = allr.groupby("name")
    out = g["score"].mean().to_frame("score")
    out["parse_ok"] = g["parse_ok"].mean()
    out["grade_pred"] = g["grade_pred"].agg(lambda s: s.dropna().mode().iloc[0] if s.dropna().size else None)
    return out


# ------------------------------------------------------------------ statistics
def purity_at_recall(y: np.ndarray, s: np.ndarray, recall: float = RECALL_TARGET) -> float:
    """Precision of the smallest score-ranked top set that recovers `recall` of the positives
    (ties broken by rank order after a stable sort; NaN when there is no positive)."""
    y = np.asarray(y, int); s = np.asarray(s, float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-s, kind="stable")
    hits = np.cumsum(y[order])
    k = int(np.searchsorted(hits, np.ceil(recall * n_pos) - 1e-9)) + 1
    return float(hits[k - 1] / k)


def paired_purity_boot(y, a, b, n=N_BOOT, recall=RECALL_TARGET):
    """Paired class-stratified bootstrap of purity@recall(a) - purity@recall(b); same RNG
    discipline as phase_d_analysis.paired_boot (module RNG, reseeded by the caller)."""
    y, a, b = np.asarray(y, int), np.asarray(a, float), np.asarray(b, float)
    d0 = purity_at_recall(y, a, recall) - purity_at_recall(y, b, recall)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        bi = np.concatenate([phase_d_analysis.RNG.choice(pos, len(pos)),
                             phase_d_analysis.RNG.choice(neg, len(neg))])
        if len(np.unique(y[bi])) == 2:
            v.append(purity_at_recall(y[bi], a[bi], recall) - purity_at_recall(y[bi], b[bi], recall))
    return d0, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(y))


def _row(stat: str, value, lo=np.nan, hi=np.nan) -> dict:
    f = lambda v: (round(float(v), 4) if v is not None and pd.notna(v) else np.nan)  # noqa: E731
    return {"statistic": stat, "value": f(value), "ci_lo": f(lo), "ci_hi": f(hi)}


def _qwk_letters(pred, truth) -> float:
    p = pd.Series(pred).map(LETTER_ORD); t = pd.Series(truth).map(LETTER_ORD)
    ok = p.notna() & t.notna()
    if ok.sum() < 2 or (p[ok].nunique() < 2 and t[ok].nunique() < 2):
        return np.nan
    return float(cohen_kappa_score(t[ok].astype(int), p[ok].astype(int), weights="quadratic"))


def arm_rows(tag: str, man: pd.DataFrame, pool: pd.DataFrame, reps: list[pd.DataFrame],
             n_boot: int, min_class: int = MIN_STRATUM_CLASS) -> tuple[list[dict], dict]:
    """Absolute AUC (pooled + per replicate), per-stratum AUC, raw QWK, self-consistency
    QWK, literature recall, parse rate for one arm."""
    rows, extra = [], {}
    j = man.join(pool, how="inner")
    y, s = j["y"].to_numpy(int), j["score"].to_numpy(float)
    rows.append(_row(f"{tag}/n", len(j)))
    rows.append(_row(f"{tag}/parse_rate", float(j["parse_ok"].mean())))
    if len(np.unique(y)) == 2:
        a, lo, hi, n = phase_d_analysis.auc_ci(y, s, n=n_boot)
        rows.append(_row(f"{tag}/auc_ge3[pooled]", a, lo, hi))
        rows.append(_row(f"{tag}/purity_at_recall{RECALL_TARGET:.1f}[pooled]", purity_at_recall(y, s)))
    for k, r in enumerate(reps, start=1):
        jk = man.join(r.set_index("name"), how="inner")
        yk, sk = jk["y"].to_numpy(int), jk["score"].to_numpy(float)
        if len(np.unique(yk)) == 2:
            rows.append(_row(f"{tag}/auc_ge3[r{k}]", roc_auc_score(yk, sk)))
            rows.append(_row(f"{tag}/purity_at_recall{RECALL_TARGET:.1f}[r{k}]", purity_at_recall(yk, sk)))
    if "stratum" in j.columns:
        for st, sub in j.groupby("stratum", sort=True):
            ys = sub["y"].to_numpy(int)
            if (ys == 1).sum() >= min_class and (ys == 0).sum() >= min_class:
                a, lo, hi, n = phase_d_analysis.auc_ci(ys, sub["score"].to_numpy(float), n=n_boot)
                rows.append(_row(f"{tag}/auc_ge3[stratum={st}]", a, lo, hi))
            rows.append(_row(f"{tag}/n[stratum={st}]", len(sub)))
    # letters: raw QWK vs XH, and the arm's own replicate-to-replicate consistency
    rows.append(_row(f"{tag}/qwk_letter_vs_xh[pooled]", _qwk_letters(j["grade_pred"], j["letter"])))
    if len(reps) >= 2:
        qs = []
        for i in range(len(reps)):
            for k in range(i + 1, len(reps)):
                a_ = reps[i].set_index("name")["grade_pred"]; b_ = reps[k].set_index("name")["grade_pred"]
                both = a_.index.intersection(b_.index)
                qs.append(_qwk_letters(a_.loc[both], b_.loc[both]))
        rows.append(_row(f"{tag}/qwk_self_consistency[mean over replicate pairs]", float(np.nanmean(qs))))
    for st in ("K_cowls", "L_known"):
        if "stratum" in j.columns and (j["stratum"] == st).any():
            sub = j[j["stratum"] == st]
            thr = float(np.quantile(s, 1 - y.mean())) if y.mean() > 0 else np.inf   # top-|pos| cut
            rows.append(_row(f"{tag}/recall_top_set[stratum={st}]", float((sub["score"] >= thr).mean())))
    extra[tag] = {"n": int(len(j)), "n_pos": int(y.sum()), "n_replicates": len(reps)}
    return rows, extra


def paired_rows(tag: str, man: pd.DataFrame, a_pool: pd.DataFrame, b_pool: pd.DataFrame,
                a_reps: list[pd.DataFrame], b_reps: list[pd.DataFrame], n_boot: int) -> tuple[list[dict], dict]:
    """dAUC and dPurity of arm a minus arm b: pooled and per replicate k (paired by k)."""
    rows, extra = [], {}
    j = man.join(a_pool["score"].rename("a"), how="inner").join(b_pool["score"].rename("b"), how="inner")
    y = j["y"].to_numpy(int)
    if len(np.unique(y)) == 2:
        d, lo, hi, n = phase_d_analysis.paired_boot(y, j["a"].to_numpy(float), j["b"].to_numpy(float), n=n_boot)
        p = phase_d_analysis.delong_p(y, j["a"].to_numpy(float), j["b"].to_numpy(float))
        rows.append(_row(f"{tag}/dAUC_ge3[pooled]", d, lo, hi))
        rows.append(_row(f"{tag}/dAUC_ge3_delong_p[pooled]", p))
        dp, plo, phi, _ = paired_purity_boot(y, j["a"].to_numpy(float), j["b"].to_numpy(float), n=n_boot)
        rows.append(_row(f"{tag}/dPurity_at_recall{RECALL_TARGET:.1f}[pooled]", dp, plo, phi))
        rows.append(_row(f"{tag}/n[pooled]", n))
        extra[tag] = {"dAUC": d, "ci": [lo, hi], "delong_p": p, "dPurity": dp, "ci_purity": [plo, phi], "n": n}
    for k, (ra, rb) in enumerate(zip(a_reps, b_reps), start=1):
        jk = (man.join(ra.set_index("name")["score"].rename("a"), how="inner")
                 .join(rb.set_index("name")["score"].rename("b"), how="inner"))
        yk = jk["y"].to_numpy(int)
        if len(np.unique(yk)) == 2:
            d, lo, hi, n = phase_d_analysis.paired_boot(yk, jk["a"].to_numpy(float), jk["b"].to_numpy(float), n=n_boot)
            rows.append(_row(f"{tag}/dAUC_ge3[r{k}]", d, lo, hi))
            dp, plo, phi, _ = paired_purity_boot(yk, jk["a"].to_numpy(float), jk["b"].to_numpy(float), n=n_boot)
            rows.append(_row(f"{tag}/dPurity_at_recall{RECALL_TARGET:.1f}[r{k}]", dp, plo, phi))
    return rows, extra


def incumbent_rows(man: pd.DataFrame, e1_pool: pd.DataFrame, n_boot: int) -> list[dict]:
    """E1 vs p_pipeline on the flagged rows and on all rows (unflagged = 0)."""
    rows = []
    if "p_pipeline" not in man.columns:
        return rows
    j = man.join(e1_pool["score"], how="inner")
    flagged = j["pipe_flagged"].map(lambda v: str(v).lower() in ("true", "1")) if "pipe_flagged" in j.columns \
        else pd.to_numeric(j["p_pipeline"], errors="coerce").notna()
    for pop, sub in (("flagged_only", j[flagged]), ("all", j)):
        pp = pd.to_numeric(sub["p_pipeline"], errors="coerce")
        if pop == "all":
            pp = pp.where(flagged.loc[sub.index], 0.0)
        ok = pp.notna().to_numpy()
        y, a, b = sub["y"].to_numpy(int)[ok], sub["score"].to_numpy(float)[ok], pp.to_numpy(float)[ok]
        if len(np.unique(y)) == 2:
            d, lo, hi, n = phase_d_analysis.paired_boot(y, a, b, n=n_boot)
            rows.append(_row(f"e1_minus_incumbent/dAUC_ge3[{pop}]", d, lo, hi))
            rows.append(_row(f"e1_minus_incumbent/delong_p[{pop}]", phase_d_analysis.delong_p(y, a, b)))
            ai, lo_i, hi_i, _ = phase_d_analysis.auc_ci(y, b, n=n_boot)
            rows.append(_row(f"incumbent/auc_ge3[{pop}]", ai, lo_i, hi_i))
            rows.append(_row(f"incumbent/n[{pop}]", n))
    return rows


# ------------------------------------------------------------------ driver
def analyze(man: pd.DataFrame, arms: dict[str, list[pd.DataFrame]], n_boot: int = N_BOOT,
            seed: int = _util.SEED) -> tuple[pd.DataFrame, dict]:
    """arms: {"e1": [replicate DataFrames], "e2": [...], "e3": [...] (optional)}."""
    phase_d_analysis.RNG = np.random.default_rng(seed)
    pools = {k: pooled(v) for k, v in arms.items() if v}
    rows: list[dict] = [_row("n_units_in_split", len(man)), _row("n_pos_ge3", int(man["y"].sum()))]
    extra: dict = {"n_boot": n_boot, "seed": seed, "arms": {}}
    for tag in ("e1", "e2", "e3"):
        if tag in pools:
            r, x = arm_rows(tag, man, pools[tag], arms[tag], n_boot)
            rows += r; extra["arms"].update(x)
    if "e1" in pools and "e2" in pools:
        r, x = paired_rows("e2_minus_e1", man, pools["e2"], pools["e1"], arms["e2"], arms["e1"], n_boot)
        rows += r; extra.update(x)
    if "e1" in pools and "e3" in pools:
        r, x = paired_rows("e3_minus_e1", man, pools["e3"], pools["e1"], arms["e3"], arms["e1"], n_boot)
        rows += r; extra.update(x)
    if "e1" in pools:
        rows += incumbent_rows(man, pools["e1"], n_boot)
    return pd.DataFrame(rows, columns=["statistic", "value", "ci_lo", "ci_hi"]), extra


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default=str(OUT / "golden_jwst_manifest.csv"))
    ap.add_argument("--split", default="validate", choices=("validate", "align", "all"))
    ap.add_argument("--e1", nargs="+", default=None, help="E1 parquets (globs ok)")
    ap.add_argument("--e2", nargs="+", default=None)
    ap.add_argument("--e3", nargs="+", default=None)
    ap.add_argument("--score", default="p_lens", choices=SCORES)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=_util.SEED)
    ap.add_argument("--out", default=str(OUT / "golden_results.csv"))
    a = ap.parse_args(argv)
    if not a.e1:
        ap.error("--e1 is required (the zero-shot reference arm)")
    man = load_manifest(Path(a.manifest), None if a.split == "all" else a.split)
    arms = {k: load_replicates(getattr(a, k), a.score) for k in ("e1", "e2", "e3") if getattr(a, k)}
    res, extra = analyze(man, arms, n_boot=a.n_boot, seed=a.seed)
    label = "pilot (lite frame)" if len(man) <= 250 else "full"
    pd.set_option("display.width", 160, "display.max_rows", 500, "display.max_colwidth", 90)
    print(f"\n=== Golden endpoints, split={a.split}, score={a.score}, n={len(man)} units "
          f"({int(man['y'].sum())} with XH score >= 3) — {label} ===\n")
    print(res.to_string(index=False))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    extra.update({"split": a.split, "score": a.score, "label": label, "manifest": a.manifest,
                  "inputs": {k: sorted({str(p) for r in v for p in r["path"].unique()}) for k, v in arms.items()}})
    out.with_suffix(".json").write_text(json.dumps(extra, indent=2, default=float))
    print(f"\nsaved {out} and {out.with_suffix('.json')}")
    print("Read every QWK against the ladder {E0 ceiling, 0.42, 0.29, ~0.00}; falsifier of the resolution")
    print("thesis: E1 AUC <= 0.66 with CI excluding 0.80 and no E2/E3 movement (REGISTRY.md).")


if __name__ == "__main__":
    main()
