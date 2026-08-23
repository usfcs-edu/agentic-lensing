#!/usr/bin/env python3
"""golden/drift_report.py — ordering / fatigue diagnostics and the session-1 pilot timing verdict.

Two questions the campaign design pre-registered (plan Phases 2-3):

  1. Does the grader's criterion move along the queue? One blind shuffle means stratum is
     independent of position, so a trend of score mean / P(score=1) / seconds with
     `presentation_index` (campaign drift) or with position inside a session (fatigue) is a
     rater effect, not a content effect. Reported as Spearman rho with a 2,000-resample
     bootstrap CI (seed 2026), per-quartile means, and per-session summaries.

  2. The pilot rule, fixed before the campaign so it cannot be improvised: the first 30
     exposures are timed; if the MEAN seconds per exposure exceeds 75 s the `U_tail` /
     `N_unflagged` strata are cut from 30/20 to 15/10 — executed by `build_kit.py
     --drop-units` (retires UNGRADED items of those strata as a new manifest version; the
     key and every graded item stay; the survivors are the graded ones plus a deterministic
     hash core). The verdict is printed, together with the projected hours for the rest of
     the queue at the observed pace.

Score drift uses PASS-1 rows only: the 40 repeats were drawn stratified on observed pass-1
scores, so their score distribution is not the queue's. Timing uses every exposure.

  python lensjudge/golden/drift_report.py --grades golden/golden_grades.csv \
      --frame golden/frame.csv --out outputs/golden_drift.csv [--pilot-n 30] [--queue-total N]

Output: `statistic,value,ci_lo,ci_hi` rows (the intergrader_stats schema) + printed verdict.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.golden import _util  # noqa: E402
from lensjudge.golden.stats import _boot_rows, _rho, _row  # noqa: E402

GOLDEN = _util.HERE
OUT = _util.LENSJUDGE / "outputs"
N_BOOT = 2000
PILOT_N = 30                 # first 30 exposures of session 1 are the timing pilot
PILOT_MEAN_LIMIT_S = 75.0    # mean > 75 s -> cut U_tail/N_unflagged to 15/10 (plan, pre-specified)
N_REPEATS_PLANNED = 40       # lite frame: 250 items + 40 repeats
SECONDS_CLIP = 300.0         # a second, rest-robust mean (informational; the rule uses the raw mean)


def _position_in_session(g: pd.DataFrame) -> pd.Series:
    """0-based rank of presentation_index within each session (fatigue axis)."""
    return g.groupby("session_id")["presentation_index"].rank(method="first").astype(int) - 1


def pilot_verdict(mean_seconds: float) -> str:
    if np.isnan(mean_seconds):
        return "PILOT: no timed exposures yet - no verdict"
    if mean_seconds > PILOT_MEAN_LIMIT_S:
        return (f"PILOT VERDICT: mean {mean_seconds:.1f} s > {PILOT_MEAN_LIMIT_S:.0f} s -> "
                "DROP U_tail/N_unflagged to 15/10 (pre-specified rule): run "
                "`build_kit.py --kit-id <kit> --drop-units` BEFORE --add-repeats, never --force")
    return (f"PILOT VERDICT: mean {mean_seconds:.1f} s <= {PILOT_MEAN_LIMIT_S:.0f} s -> "
            "KEEP the lite frame as drawn (U_tail 30 / N_unflagged 20)")


def pilot_summary(grades: pd.DataFrame, pilot_n: int = PILOT_N, queue_total: int | None = None) -> dict:
    """Timing of the first `pilot_n` exposures (by presentation_index, all passes) and the
    projection for the remaining queue at that pace."""
    g = grades.sort_values("presentation_index")
    sec = pd.to_numeric(g["seconds"], errors="coerce")
    pilot = sec.iloc[:pilot_n].dropna()
    n_done = len(g)
    total = queue_total if queue_total is not None else n_done
    remaining = max(0, total - n_done)
    mean = float(pilot.mean()) if len(pilot) else np.nan
    return {
        "pilot_n": int(len(pilot)),
        "pilot_seconds_median": float(pilot.median()) if len(pilot) else np.nan,
        "pilot_seconds_mean": mean,
        "pilot_seconds_mean_clipped300": float(pilot.clip(upper=SECONDS_CLIP).mean()) if len(pilot) else np.nan,
        "pilot_seconds_p90": float(pilot.quantile(0.9)) if len(pilot) else np.nan,
        "queue_total": int(total),
        "queue_done": int(n_done),
        "queue_remaining": int(remaining),
        "pilot_projected_hours_remaining": (remaining * mean / 3600.0) if not np.isnan(mean) else np.nan,
        "verdict": pilot_verdict(mean),
    }


def drift_table(grades: pd.DataFrame, n_boot: int = N_BOOT, seed: int = _util.SEED,
                pilot_n: int = PILOT_N, queue_total: int | None = None) -> pd.DataFrame:
    """golden_grades.csv rows -> `statistic,value,ci_lo,ci_hi` drift / fatigue / pilot rows."""
    rng = np.random.default_rng(seed)
    g = grades.copy()
    g["pass"] = g["pass"].astype(int)
    g["score"] = g["score_1_4"].astype(int)
    g["is1"] = (g["score"] == 1).astype(int)
    g["seconds"] = pd.to_numeric(g["seconds"], errors="coerce")
    g["session_id"] = g["session_id"].astype(str)
    g["presentation_index"] = g["presentation_index"].astype(int)
    g = g.sort_values("presentation_index").reset_index(drop=True)
    g["pos_in_session"] = _position_in_session(g)
    p1 = g[g["pass"] == 1].reset_index(drop=True)

    rows = [_row("n_exposures", len(g)), _row("n_pass1", len(p1)),
            _row("n_sessions", g["session_id"].nunique())]

    # campaign drift (queue position) and fatigue (position inside the session), pass-1 scores
    if len(p1) >= 3:
        rows += _boot_rows(p1, lambda d: {
            "spearman_score_vs_presentation_index[pass1]": _rho(d["score"], d["presentation_index"]),
            "spearman_score1_vs_presentation_index[pass1]": _rho(d["is1"], d["presentation_index"]),
            "spearman_score_vs_pos_in_session[pass1]": _rho(d["score"], d["pos_in_session"]),
            "spearman_score1_vs_pos_in_session[pass1]": _rho(d["is1"], d["pos_in_session"]),
        }, n_boot, rng)
    # timing: every exposure (repeats are real exposures)
    if g["seconds"].notna().sum() >= 3:
        gs = g.dropna(subset=["seconds"]).reset_index(drop=True)
        rows += _boot_rows(gs, lambda d: {
            "spearman_seconds_vs_presentation_index[all]": _rho(d["seconds"], d["presentation_index"]),
            "spearman_seconds_vs_pos_in_session[all]": _rho(d["seconds"], d["pos_in_session"]),
            "spearman_seconds_vs_score[all]": _rho(d["seconds"], d["score"]),
            "seconds_median[all]": float(d["seconds"].median()),
            "seconds_mean[all]": float(d["seconds"].mean()),
        }, n_boot, rng)

    # per quartile of queue position (pass 1): the coarse picture behind the rho
    if len(p1) >= 8:
        q = pd.qcut(p1["presentation_index"].rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"])
        for lab, sub in p1.groupby(q, observed=True, sort=True):
            rows.append(_row(f"{lab}/n[pass1]", len(sub)))
            rows += _boot_rows(sub.reset_index(drop=True), lambda d, lab=lab: {
                f"{lab}/score_mean[pass1]": float(d["score"].mean()),
                f"{lab}/p_score1[pass1]": float(d["is1"].mean()),
                f"{lab}/seconds_median[pass1]": float(d["seconds"].median()),
            }, n_boot, rng)

    # per session
    for sid, sub in g.groupby("session_id", sort=True):
        s1 = sub[sub["pass"] == 1].reset_index(drop=True)
        pre = f"session={sid}/"
        rows.append(_row(pre + "n_exposures", len(sub)))
        rows.append(_row(pre + "n_pass1", len(s1)))
        rows.append(_row(pre + "first_presentation_index", int(sub["presentation_index"].min())))
        if len(s1) >= 3:
            rows += _boot_rows(s1, lambda d, pre=pre: {
                pre + "score_mean[pass1]": float(d["score"].mean()),
                pre + "p_score1[pass1]": float(d["is1"].mean()),
                pre + "spearman_score_vs_pos_in_session[pass1]": _rho(d["score"], d["pos_in_session"]),
            }, n_boot, rng)
        ss = sub.dropna(subset=["seconds"]).reset_index(drop=True)
        if len(ss) >= 3:
            rows += _boot_rows(ss, lambda d, pre=pre: {
                pre + "seconds_median": float(d["seconds"].median()),
                pre + "seconds_mean": float(d["seconds"].mean()),
            }, n_boot, rng)

    # the pilot (first pilot_n exposures overall == session 1's first pilot_n)
    ps = pilot_summary(g, pilot_n=pilot_n, queue_total=queue_total)
    for k, v in ps.items():
        if k != "verdict":
            rows.append(_row(k, v))
    return pd.DataFrame(rows, columns=["statistic", "value", "ci_lo", "ci_hi"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grades", default=str(GOLDEN / "golden_grades.csv"))
    ap.add_argument("--frame", default=str(GOLDEN / "frame.csv"),
                    help="only used to size the queue (n_frame + 40 repeats) when --queue-total is absent")
    ap.add_argument("--out", default=str(OUT / "golden_drift.csv"))
    ap.add_argument("--pilot-n", type=int, default=PILOT_N)
    ap.add_argument("--queue-total", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=_util.SEED)
    a = ap.parse_args(argv)

    grades = _util.read_pinned(a.grades)
    total = a.queue_total
    if total is None and Path(a.frame).exists():
        total = len(_util.read_pinned(a.frame)) + N_REPEATS_PLANNED
    res = drift_table(grades, n_boot=a.n_boot, seed=a.seed, pilot_n=a.pilot_n, queue_total=total)
    pd.set_option("display.width", 160, "display.max_rows", 500, "display.max_colwidth", 80)
    print(f"\n=== Drift / fatigue / pilot timing, {len(grades)} exposures ===\n")
    print(res.to_string(index=False))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nsaved {out}\n")
    ps = pilot_summary(grades, pilot_n=a.pilot_n, queue_total=total)
    print(f"pilot: n={ps['pilot_n']}  median {ps['pilot_seconds_median']:.1f} s  mean {ps['pilot_seconds_mean']:.1f} s "
          f"(clipped@{SECONDS_CLIP:.0f}s {ps['pilot_seconds_mean_clipped300']:.1f} s)  "
          f"remaining {ps['queue_remaining']}/{ps['queue_total']} -> "
          f"{ps['pilot_projected_hours_remaining']:.2f} h at the pilot pace")
    print(ps["verdict"])


if __name__ == "__main__":
    main()
