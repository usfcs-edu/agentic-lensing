#!/usr/bin/env python3
"""golden/stats.py — the intra-rater human ceiling (E0) and the XH-vs-pipeline agreement tables.

Why this exists: every LensJudge generation was scored against labels whose own reliability
was never measured on a per-rater basis (eval/score.py prints "CONSENSUS-REFERENCED, NO HUMAN
CEILING" on every run). The golden campaign gives one named expert (Xiaosheng Huang, XH) a blind
shuffled queue with 40 byte-identical repeats; this script turns the repeat pairs into the
program's first measured intra-rater ceiling, in the SAME statistics the repo already reports
for the published inter-rater numbers, so the two can be read on one ladder:

  reused verbatim   eval/human_ceiling.pair        -> QWK / linear kappa / exact / within-1 /
                                                     Spearman on LETTERS (the cross-team rows
                                                     in outputs/human_ceiling.csv use this)
  reused verbatim   parity/intergrader_stats._stats -> exact / one-step / two-step / symmetrised
                                                     QWK / Krippendorff alpha / within-letter
                                                     dispersion (the Paper II 0.554 / 0.42 rows)
  added here        unsymmetrised Cohen QWK(pass1, pass2); drift mean(s2 - s1) with an exact
                    sign test (the criterion-shift measurement); confidence L/M/H agreement;
                    binary (score >= 3) self-consistency; subgroup rows by prior_exposure and
                    stratum (n >= 8 else skipped; `layout` and `lit_known` are NOT subgroups —
                    build_kit excludes gray layouts and literature-known units from the
                    repeats, so every pair is colour / not-literature and the contrast is
                    empty by construction).

The repeat set is NOT a random sample of the queue: build_kit.pick_repeat_units draws it
stratified on the OBSERVED pass-1 score (balanced across levels so the QWK CI is narrow).
Any statistic that conditions on s1 therefore regresses to the mean — a rater with NO drift
shows a negative mean(s2 - s1) whenever the repeats over-represent the high end (simulated:
-0.08 to -0.12 at the lite marginals). So two versions of every drift / agreement number are
reported: the raw rows ("score-balanced repeat set") and `ipw_*` rows that re-weight each
pair by (pass-1 marginal share of its s1 level) / (repeat-set share of that level) — the
inverse-probability weights that recover the population value; plus per-level rows
`drift_mean_s2_minus_s1[s1=k]`, which condition explicitly and need no weights. The Rojas
73% anchor is a population number: read it against the ipw rows.

`_stats` assumes an UNORDERED pair (s_lo <= s_hi: its one_step/two_step are `hi - lo == 1/2`),
so it is fed min/max of the two passes — exactly "comparable in kind" to the identity-lost
Paper II pairs. Everything that needs pass ORDER (unsymmetrised QWK, drift) is in the added
block. Q (the within-letter masks) is the pass-1 letter: "given he said A the first time, how
often does he say A again". `_stats`' qwk_grader_vs_consensus is dropped (degenerate when the
'consensus' is pass 1 itself) and its `p_one_grader_reject` is renamed
`p_either_pass_score1` (here it is P(min(s1, s2) == 1), not a second grader's rejection).

CIs: 2,000-resample percentile bootstrap over repeat UNITS, seed 2026 (intergrader_stats
convention). Output schema is the repo's `statistic,value,ci_lo,ci_hi` so
parity/human_baseline_summary.py can lift rows by name. When golden_grades.csv holds more
than one kit (Campaign 2), pass --kit-id: pass-1 rows must be unique per unit.

  python lensjudge/golden/stats.py --grades golden/golden_grades.csv \
      --labels golden/golden_labels.csv --frame golden/frame.csv \
      --out outputs/golden_intrarater.csv
  python lensjudge/golden/stats.py --agreement ... --out outputs/golden_agreement.csv
  python lensjudge/golden/stats.py --simulate      # expected CI width at n=40 (design check)

`--agreement` is the XH-vs-pipeline table: AUC of the run's `pipe_inspector_conf` against XH
score >= 3 (class-stratified bootstrap, phase_d_analysis.auc_ci) reported TWICE — on the rows
the pipeline flagged (`[flagged_only]`, the incumbent's own ranking) and on all rows (`[all]`,
unflagged = ranked below every flagged item, confidence 0: the version that charges the
incumbent for the lenses it never surfaced) — the XH x `pipe_grade_passcount` cross-tab +
Spearman on VERIFIED rows only (a persona pass-count is not a co-rater's grade — "different
semantics, not a QWK headline"), and XH recall on the K_cowls / L_known strata.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest, spearmanr  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from lensjudge.eval import human_ceiling  # noqa: E402
from lensjudge.golden import _util  # noqa: E402
from lensjudge.parity import intergrader_stats, phase_d_analysis  # noqa: E402

GOLDEN = _util.HERE
OUT = _util.LENSJUDGE / "outputs"
N_BOOT = 2000
PAIR_LABEL = "INTRA: Huang pass1 vs pass2 (JWST v1)"
SUBGROUPS = ("prior_exposure", "stratum")     # layout / lit_known are constant on repeats
MIN_SUBGROUP_N = 8
CONF_ORD = {"L": 0, "M": 1, "H": 2}
LENS_MIN_SCORE = 3          # "LENS iff score >= 3" (contract; few-shot + primary endpoint)
# the anchors every golden number is read against (parity/intergrader_stats.py, eval/human_ceiling.py)
ANCHORS = [
    ("Rojas+23 intra-rater repeat consistency (55 experts, DES-quality)", "73% identical / QWK ~0.8"),
    ("Paper II two-grader exact score agreement (n=1,310, truncated at acceptance)", "0.554"),
    ("Paper II symmetrised QWK / Krippendorff alpha (same)", "0.42 [0.37, 0.47]"),
    ("Paper II individual grader vs published consensus QWK (UPPER bound)", "0.776"),
    ("cross-team QWK: DESI vs SuGOHI / DESI vs Euclid", "0.29 / 0.17"),
]


def _reseed_reused(seed: int) -> None:
    """The reused modules draw from module-level RNGs; reseed them so this table is
    reproducible regardless of what else ran in the process."""
    human_ceiling._RNG = np.random.default_rng(seed)
    intergrader_stats._RNG = np.random.default_rng(seed)
    phase_d_analysis.RNG = np.random.default_rng(seed)


# --------------------------------------------------------------------------- pairs
def pivot_pairs(grades: pd.DataFrame, frame: pd.DataFrame | None = None,
                kit_id: str | None = None) -> pd.DataFrame:
    """golden_grades.csv (one row per unit x pass) -> one row per repeated unit:
    s1/s2 (score), c1/c2 (L/M/H), l1/l2 (letter), idx1/idx2 (presentation index),
    session1/session2, w_ipw (inverse-probability weight to the pass-1 marginal, see module
    doc), plus the subgroup columns from frame.csv when given. `kit_id` selects one kit when
    the grades file holds several (required then: a unit may have a pass-1 row per kit)."""
    g = grades.copy()
    g["pass"] = g["pass"].astype(int)
    if kit_id is not None:
        g = g[g["kit_id"].astype(str) == str(kit_id)]
    elif "kit_id" in g.columns and g["kit_id"].nunique() > 1:
        raise ValueError(f"golden_grades holds {g['kit_id'].nunique()} kits "
                         f"({sorted(g['kit_id'].astype(str).unique())}); pass kit_id to pick one")
    p1 = g[g["pass"] == 1].set_index("unit_id")
    p2 = g[g["pass"] == 2].set_index("unit_id")
    assert p1.index.is_unique and p2.index.is_unique, "golden_grades must be one row per (unit, pass)"
    both = p1.index.intersection(p2.index)
    pairs = pd.DataFrame({
        "unit_id": both,
        "s1": p1.loc[both, "score_1_4"].astype(int).to_numpy(),
        "s2": p2.loc[both, "score_1_4"].astype(int).to_numpy(),
        "c1": p1.loc[both, "confidence_lmh"].astype(str).str.upper().to_numpy(),
        "c2": p2.loc[both, "confidence_lmh"].astype(str).str.upper().to_numpy(),
        "idx1": p1.loc[both, "presentation_index"].to_numpy(),
        "idx2": p2.loc[both, "presentation_index"].to_numpy(),
        "session1": p1.loc[both, "session_id"].astype(str).to_numpy(),
        "session2": p2.loc[both, "session_id"].astype(str).to_numpy(),
    })
    pairs["l1"] = pairs["s1"].map(_util.score_to_letter)
    pairs["l2"] = pairs["s2"].map(_util.score_to_letter)
    pairs["w_ipw"] = ipw_weights(p1["score_1_4"].astype(int).to_numpy(), pairs["s1"].to_numpy())
    if frame is not None:
        cols = [c for c in SUBGROUPS if c in frame.columns]
        pairs = pairs.merge(frame[["unit_id", *cols]], on="unit_id", how="left")
    return pairs.sort_values("unit_id").reset_index(drop=True)


def ipw_weights(pass1_scores: np.ndarray, s1: np.ndarray) -> np.ndarray:
    """Per-pair weight (pass-1 marginal share of the pair's s1 level) / (repeat-set share of
    that level), normalised to mean 1. Undoes the score-stratified draw of the repeats so a
    weighted mean is an estimate for the queue, not for the balanced design."""
    s1 = np.asarray(s1, int)
    if len(s1) == 0:
        return np.zeros(0)
    pop = pd.Series(np.asarray(pass1_scores, int)).value_counts(normalize=True)
    rep = pd.Series(s1).value_counts(normalize=True)
    w = np.array([float(pop.get(s, 0.0)) / float(rep[s]) for s in s1])
    return w / w.mean() if w.sum() > 0 else w


def _wmean(x, w) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    return float((x * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def _as_intergrader_frame(pairs: pd.DataFrame) -> pd.DataFrame:
    """The DataFrame intergrader_stats._stats expects: s_lo <= s_hi (unordered pair), Q = the
    pass-1 letter, pair_ok True (load_pairs filters on it; _stats itself does not read it)."""
    s1, s2 = pairs["s1"].to_numpy(), pairs["s2"].to_numpy()
    return pd.DataFrame({"s_lo": np.minimum(s1, s2), "s_hi": np.maximum(s1, s2),
                         "Q": pairs["l1"].to_numpy(), "pair_ok": True})


def _qwk(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(np.unique(a)) < 2 and len(np.unique(b)) < 2:
        return np.nan
    return float(cohen_kappa_score(a, b, weights="quadratic"))


def _rho(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan
    return float(spearmanr(a, b)[0])


def _added_stats(pairs: pd.DataFrame) -> dict:
    """The statistics neither reused module computes (need pass ORDER or confidence)."""
    s1, s2 = pairs["s1"].to_numpy(), pairs["s2"].to_numpy()
    c1 = pairs["c1"].map(CONF_ORD).to_numpy(float)
    c2 = pairs["c2"].map(CONF_ORD).to_numpy(float)
    b1, b2 = (s1 >= LENS_MIN_SCORE).astype(int), (s2 >= LENS_MIN_SCORE).astype(int)
    d = s2 - s1
    w = pairs["w_ipw"].to_numpy(float) if "w_ipw" in pairs.columns else np.ones(len(pairs))
    ipw = {                                                # re-weighted to the pass-1 marginal
        "ipw_exact_score_agree": _wmean(d == 0, w),
        "ipw_within1_step": _wmean(np.abs(d) <= 1, w),
        "ipw_drift_mean_s2_minus_s1": _wmean(d, w),
        "ipw_drift_frac_up": _wmean(d > 0, w),
        "ipw_drift_frac_down": _wmean(d < 0, w),
        "ipw_conf_exact_agree": _wmean(c1 == c2, w),
        "ipw_binary_ge3_exact_agree": _wmean(b1 == b2, w),
    }
    return {
        "qwk_pass1_vs_pass2": _qwk(s1, s2),                 # unsymmetrised Cohen QWK
        "linear_kappa_pass1_vs_pass2": (np.nan if len(np.unique(np.r_[s1, s2])) < 2
                                        else float(cohen_kappa_score(s1, s2, weights="linear"))),
        "spearman_pass1_vs_pass2": _rho(s1, s2),
        "within1_step": float((np.abs(d) <= 1).mean()),
        "drift_mean_s2_minus_s1": float(d.mean()),
        "drift_frac_up": float((d > 0).mean()),
        "drift_frac_down": float((d < 0).mean()),
        "conf_exact_agree": float((c1 == c2).mean()),
        "conf_spearman": _rho(c1, c2),
        "conf_mean_shift_c2_minus_c1": float((c2 - c1).mean()),
        "binary_ge3_exact_agree": float((b1 == b2).mean()),
        "binary_ge3_kappa": _qwk(b1, b2),                    # 2 classes: QWK == plain kappa
        "p_ge3_pass1": float(b1.mean()),
        "p_ge3_pass2": float(b2.mean()),
        **ipw,
    }


def _full_stats(pairs: pd.DataFrame) -> dict:
    """Everything with a bootstrap CI: reused `_stats` keys (+ within_D, - the degenerate
    grader-vs-consensus) followed by the added block."""
    ig = _as_intergrader_frame(pairs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # empty within-letter masks -> nan
        st = intergrader_stats._stats(ig)
        d_mask = ig["Q"].to_numpy() == "D"
        st["exact_agree_within_D"] = float(((ig["s_hi"] - ig["s_lo"])[d_mask] == 0).mean())
    st.pop("qwk_grader_vs_consensus", None)
    st["p_either_pass_score1"] = st.pop("p_one_grader_reject")    # honest name for a self-pair
    st = {k: float(v) for k, v in st.items()}
    st.update(_added_stats(pairs))
    return st


def _subgroup_stats(pairs: pd.DataFrame) -> dict:
    s1, s2 = pairs["s1"].to_numpy(), pairs["s2"].to_numpy()
    c1 = pairs["c1"].map(CONF_ORD).to_numpy(float)
    c2 = pairs["c2"].map(CONF_ORD).to_numpy(float)
    return {
        "exact_score_agree": float((s1 == s2).mean()),
        "within1_step": float((np.abs(s2 - s1) <= 1).mean()),
        "qwk_pass1_vs_pass2": _qwk(s1, s2),
        "drift_mean_s2_minus_s1": float((s2 - s1).mean()),
        "conf_exact_agree": float((c1 == c2).mean()),
    }


def _boot_rows(df: pd.DataFrame, fn, n_boot: int, rng, prefix: str = "") -> list[dict]:
    """Point estimate + percentile bootstrap (over rows of df) for every key fn returns."""
    point = fn(df)
    boots: dict[str, list[float]] = {k: [] for k in point}
    n = len(df)
    for _ in range(n_boot):
        s = df.iloc[rng.integers(0, n, n)]
        for k, v in fn(s).items():
            if v is not None and not np.isnan(v):
                boots[k].append(v)
    rows = []
    for k, v in point.items():
        if boots[k]:
            lo, hi = np.percentile(boots[k], [2.5, 97.5])
        else:
            lo = hi = np.nan
        rows.append({"statistic": prefix + k, "value": round(float(v), 3),
                     "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3)})
    return rows


def _row(stat: str, value, lo=np.nan, hi=np.nan) -> dict:
    return {"statistic": stat, "value": (round(float(value), 3) if pd.notna(value) else np.nan),
            "ci_lo": (round(float(lo), 3) if pd.notna(lo) else np.nan),
            "ci_hi": (round(float(hi), 3) if pd.notna(hi) else np.nan)}


def sign_test_p(d: np.ndarray) -> float:
    """Exact two-sided sign test on the pass-2 minus pass-1 differences (ties dropped)."""
    up, down = int((d > 0).sum()), int((d < 0).sum())
    if up + down == 0:
        return 1.0
    return float(binomtest(up, up + down, 0.5).pvalue)


def intrarater_table(pairs: pd.DataFrame, n_boot: int = N_BOOT, seed: int = _util.SEED,
                     subgroups=SUBGROUPS, min_n: int = MIN_SUBGROUP_N) -> pd.DataFrame:
    """pairs (pivot_pairs output) -> `statistic,value,ci_lo,ci_hi` rows."""
    _reseed_reused(seed)
    rng = np.random.default_rng(seed)
    rows = [_row("n_pairs", len(pairs))]

    # 1. eval/human_ceiling.pair on letters — the cross-team rows' own function, verbatim
    hc = human_ceiling.pair(PAIR_LABEL, pairs["l1"], pairs["l2"])
    if hc is not None:
        rows.append(_row(f"{PAIR_LABEL} | qwk", hc["qwk"], hc["qwk_lo"], hc["qwk_hi"]))
        for k in ("linear_kappa", "exact_agree", "within1", "spearman"):
            rows.append(_row(f"{PAIR_LABEL} | {k}", hc[k]))

    # 2. intergrader_stats._stats (unordered pair) + the added ordered/confidence block
    rows += _boot_rows(pairs, _full_stats, n_boot, rng)
    rows.append(_row("drift_sign_test_p", sign_test_p(pairs["s2"].to_numpy() - pairs["s1"].to_numpy())))
    rows.append(_row("drift_n_up", int((pairs["s2"] > pairs["s1"]).sum())))
    rows.append(_row("drift_n_down", int((pairs["s2"] < pairs["s1"]).sum())))
    # per pass-1 level: conditioning made explicit (a level at the top can only drift down)
    for s, sub in pairs.groupby("s1", sort=True):
        pre = f"s1={int(s)}/"
        rows.append(_row(pre + "n_pairs", len(sub)))
        rows.append(_row(pre + "w_ipw", float(sub["w_ipw"].iloc[0]) if "w_ipw" in sub else 1.0))
        rows += _boot_rows(sub.reset_index(drop=True), lambda d: {
            "drift_mean_s2_minus_s1": float((d["s2"] - d["s1"]).mean()),
            "exact_score_agree": float((d["s1"] == d["s2"]).mean()),
        }, n_boot, rng, pre)

    # 3. subgroups (n >= min_n else skipped): prefix "<col>=<value>/<stat>"
    for col in subgroups:
        if col not in pairs.columns:
            continue
        for val, sub in pairs.groupby(col, dropna=False, sort=True):
            if len(sub) < min_n:
                continue
            pre = f"{col}={val}/"
            rows.append(_row(pre + "n_pairs", len(sub)))
            rows += _boot_rows(sub.reset_index(drop=True), _subgroup_stats, n_boot, rng, pre)
    return pd.DataFrame(rows, columns=["statistic", "value", "ci_lo", "ci_hi"])


# ----------------------------------------------------------------------- agreement
def _passcount_levels(pc: pd.Series) -> pd.Series:
    """pipe_grade_passcount as clean upper-case levels; NaN/'' (unflagged) -> 'none'. U stays U."""
    return pc.fillna("").astype(str).str.strip().str.upper().replace("", "none")


def _verified_mask(pc: pd.Series) -> pd.Series:
    """pipe_grade_passcount in A/B/C/D = the run's personas actually ruled (U / '' did not)."""
    return _passcount_levels(pc).isin(list("ABCD"))


def agreement_table(labels: pd.DataFrame, frame: pd.DataFrame, n_boot: int = N_BOOT,
                    seed: int = _util.SEED, min_n: int = MIN_SUBGROUP_N) -> pd.DataFrame:
    """XH (pass-1 canonical label) vs the JWST run's own signals, `statistic,value,ci_lo,ci_hi`."""
    _reseed_reused(seed)
    rng = np.random.default_rng(seed)
    # the run's signals live in frame.csv; stratum is on both (labels wins when present)
    keep = [c for c in ("pipe_inspector_conf", "pipe_grade_passcount", "stratum")
            if c in frame.columns and c not in labels.columns]
    df = labels.merge(frame[["unit_id", *keep]], on="unit_id", how="left")
    assert "stratum" in df.columns, "stratum must be on labels or frame"
    df["score"] = df["score_1_4"].astype(int)
    df["ge3"] = (df["score"] >= LENS_MIN_SCORE).astype(int)
    df["ge2"] = (df["score"] >= 2).astype(int)
    rows = [_row("n_units", len(df))]

    # AUC of the incumbent against XH's binary verdict, class-stratified bootstrap, in two
    # populations: [flagged_only] = rows with an inspector confidence (the pipeline's own
    # ranking of what it surfaced); [all] = every row, unflagged ones at confidence 0 (ranked
    # below everything it flagged — otherwise its misses, the 20 N_unflagged and the
    # literature lenses it never surfaced, would be dropped from its own score)
    conf_all = pd.to_numeric(df.get("pipe_inspector_conf"), errors="coerce")
    flagged = np.array(conf_all.notna().to_numpy(), dtype=bool, copy=True)
    if "pipe_grade_passcount" in df.columns:      # a pass-count without a confidence still counts
        flagged = flagged | _passcount_levels(df["pipe_grade_passcount"]).isin(list("ABCDU")).to_numpy()
    df["pipe_flagged"] = flagged

    def _auc_rows(sub: pd.DataFrame, tag: str, pop: str):
        """AUC rows; the overall ones (stratum tag '') are always emitted (NaN when a class
        is missing) so a skipped headline is visible; stratum rows are skipped when n < min_n."""
        out = []
        for ycol in ("ge3", "ge2"):
            s = pd.to_numeric(sub["pipe_inspector_conf"], errors="coerce").to_numpy(float)
            if pop == "all":
                s = np.where(sub["pipe_flagged"].to_numpy(bool), s, 0.0)
            y = sub[ycol].to_numpy(int)
            ok = ~np.isnan(s)
            computable = ok.sum() >= min_n and len(np.unique(y[ok])) == 2
            if not computable and tag:
                continue
            a, lo, hi, n = (phase_d_analysis.auc_ci(y, s, n=n_boot) if computable
                            else (np.nan, np.nan, np.nan, int(ok.sum())))
            out.append(_row(f"auc_pipe_inspector_conf_vs_xh_{ycol}[{pop}]{tag}", a, lo, hi))
            out.append(_row(f"n_auc_pipe_inspector_conf_vs_xh_{ycol}[{pop}]{tag}", n))
        return out
    if "pipe_inspector_conf" in df.columns:
        for pop in ("flagged_only", "all"):
            rows += _auc_rows(df, "", pop)
        for st, sub in df.groupby("stratum", sort=True):
            for pop in ("flagged_only", "all"):
                rows += _auc_rows(sub, f"[stratum={st}]", pop)

    # pass-count: verified rows only; Spearman + cross-tab. A persona pass-count is the run's
    # vote tally, not a co-rater's grade, so this is never a QWK headline.
    if "pipe_grade_passcount" in df.columns:
        pc = _passcount_levels(df["pipe_grade_passcount"])
        ver = df[_verified_mask(pc)].copy()
        ver["pc_ord"] = pc[ver.index].map(_util.LETTER_TO_ORD)
        tag = "[verified rows only; different semantics - not a QWK headline]"
        rows.append(_row("n_verified_passcount" + tag, len(ver)))
        if len(ver) >= min_n:
            rows += _boot_rows(ver.reset_index(drop=True),
                               lambda d: {"spearman_xh_score_vs_pipe_passcount" + tag:
                                          _rho(d["score"], d["pc_ord"])}, n_boot, rng)
            xt = pd.crosstab(ver["pc_ord"].map({v: k for k, v in _util.LETTER_TO_ORD.items()}),
                             ver["score"])
            for pcl in xt.index:
                for sc in xt.columns:
                    rows.append(_row(f"xtab_verified[pipe={pcl},xh={sc}]", int(xt.loc[pcl, sc])))
        # XH's verdict distribution under every pass-count level incl. U / unflagged
        for lvl, sub in df.groupby(pc, sort=True):
            if len(sub) < min_n:
                continue
            rows.append(_row(f"n[pipe_grade_passcount={lvl}]", len(sub)))
            rows += _boot_rows(sub.reset_index(drop=True),
                               lambda d, lvl=lvl: {f"p_xh_ge3[pipe_grade_passcount={lvl}]": float(d["ge3"].mean()),
                                                   f"xh_score_mean[pipe_grade_passcount={lvl}]": float(d["score"].mean())},
                               n_boot, rng)

    # recall on the literature strata: does he call the COWLS / known lenses lenses?
    for st in ("K_cowls", "L_known"):
        sub = df[df["stratum"] == st]
        rows.append(_row(f"n[stratum={st}]", len(sub)))
        if len(sub) == 0:
            continue
        rows += _boot_rows(sub.reset_index(drop=True),
                           lambda d, st=st: {f"recall_xh_ge3[stratum={st}]": float(d["ge3"].mean()),
                                             f"recall_xh_ge2[stratum={st}]": float(d["ge2"].mean()),
                                             f"xh_score_mean[stratum={st}]": float(d["score"].mean())},
                           n_boot, rng)
    return pd.DataFrame(rows, columns=["statistic", "value", "ci_lo", "ci_hi"])


# ----------------------------------------------------------------------- design check
def simulate_ci_width(n: int, marginal, p_exact: float = 0.73, p_adjacent: float = 0.24,
                      n_rep: int = 40, n_boot: int = 300, seed: int = _util.SEED) -> dict:
    """Expected 95% percentile-bootstrap CI width of the unsymmetrised QWK (and of exact
    agreement) for n repeat pairs whose pass-1 marginal is `marginal` (4 probabilities for
    scores 1..4). Pass 2 equals pass 1 w.p. p_exact, moves one step w.p. p_adjacent, else
    jumps to a random other level. Median over n_rep replicate datasets. This is the sizing
    argument for STRATIFYING the repeats: QWK collapses when one level dominates."""
    rng = np.random.default_rng(seed)
    marginal = np.asarray(marginal, float) / np.sum(marginal)
    widths_qwk, widths_exact, qwks = [], [], []
    for _ in range(n_rep):
        s1 = rng.choice([1, 2, 3, 4], size=n, p=marginal)
        u = rng.random(n)
        step = np.where(rng.random(n) < 0.5, -1, 1)
        s2 = np.where(u < p_exact, s1,
                      np.where(u < p_exact + p_adjacent, np.clip(s1 + step, 1, 4),
                               np.clip(s1 + 2 * step, 1, 4)))
        qwks.append(_qwk(s1, s2))
        bq, be = [], []
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            q = _qwk(s1[i], s2[i])
            if not np.isnan(q):
                bq.append(q)
            be.append(float((s1[i] == s2[i]).mean()))
        widths_qwk.append(np.diff(np.percentile(bq, [2.5, 97.5]))[0] if bq else np.nan)
        widths_exact.append(np.diff(np.percentile(be, [2.5, 97.5]))[0])
    return {"n": n, "marginal": [round(float(m), 2) for m in marginal], "p_exact": p_exact,
            "qwk_median": round(float(np.nanmedian(qwks)), 3),
            "qwk_ci_width": round(float(np.nanmedian(widths_qwk)), 3),
            "exact_ci_width": round(float(np.median(widths_exact)), 3)}


# ------------------------------------------------------------------------------ CLI
def print_anchors() -> None:
    print("\nAnchors (read every golden number against these):")
    for what, val in ANCHORS:
        print(f"  {val:<22s} {what}")
    print("  Paper II numbers are truncated at acceptance (Score >= 2): an UPPER bound on inter-rater")
    print("  agreement, compared here with an untruncated intra-rater set -- say so when quoting.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grades", default=str(GOLDEN / "golden_grades.csv"))
    ap.add_argument("--labels", default=str(GOLDEN / "golden_labels.csv"))
    ap.add_argument("--frame", default=str(GOLDEN / "frame.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--agreement", action="store_true", help="XH vs pipeline table instead of the ceiling")
    ap.add_argument("--simulate", action="store_true", help="print the expected CI width at n=40 and exit")
    ap.add_argument("--kit-id", default=None, help="select one kit when golden_grades.csv holds several")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=_util.SEED)
    a = ap.parse_args(argv)
    pd.set_option("display.width", 160, "display.max_rows", 500, "display.max_colwidth", 90)

    if a.simulate:
        for marg in ((0.25, 0.25, 0.25, 0.25), (0.70, 0.10, 0.10, 0.10)):
            print(simulate_ci_width(40, marg))
        return

    frame = _util.read_pinned(a.frame)
    if a.agreement:
        labels = _util.read_pinned(a.labels)
        res = agreement_table(labels, frame, n_boot=a.n_boot, seed=a.seed)
        out = Path(a.out or OUT / "golden_agreement.csv")
        print(f"\n=== XH (pass-1 label) vs JWST-run signals, n={len(labels)} units ===")
        print("(pass-count rows are VERIFIED items only; different semantics - not a QWK headline)\n")
    else:
        grades = _util.read_pinned(a.grades)
        pairs = pivot_pairs(grades, frame, kit_id=a.kit_id)
        if pairs.empty:   # between session 1 and the repeats there is nothing to pair yet
            raise SystemExit("no (pass 1, pass 2) repeat pairs in the grades file yet — run "
                             "build_kit.py --add-repeats, grade them, collect, then re-run")
        res = intrarater_table(pairs, n_boot=a.n_boot, seed=a.seed)
        out = Path(a.out or OUT / "golden_intrarater.csv")
        print(f"\n=== Intra-rater reliability, XH pass1 vs pass2, JWST v1 render, n={len(pairs)} repeat pairs ===")
        print(f"score-pair table (rows pass 1, cols pass 2):\n{pd.crosstab(pairs['s1'], pairs['s2'])}\n")
        print("(raw rows = the score-BALANCED repeat set; ipw_* rows re-weight to the pass-1 marginal;\n"
              " drift conditioned on s1 regresses to the mean - read drift off ipw_* / s1=k rows)\n")
    print(res.to_string(index=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nsaved {out}")
    print_anchors()


if __name__ == "__main__":
    main()
