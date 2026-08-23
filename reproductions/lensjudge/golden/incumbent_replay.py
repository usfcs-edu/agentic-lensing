#!/usr/bin/env python3
"""golden/incumbent_replay.py — zero-API re-aggregation of the JWST run's recorded verdicts.

The incumbent verifier graded a candidate by a PASS-COUNT over three adversarial personas
(artifact / geometry / morphology; `J/scripts/09_rank_report.py:grade()`: 3 passes = A, 2 = B,
1 = C, 0 = D, never examined = U). This module replays that aggregation from the raw verdict
files so the truth evaluation has a $0 baseline that is provably the published one:

  1. load EVERY `J/results/verdicts/verify_*.jsonl` — including the six `*_ctl*` files that
     `09_rank_report.py`'s glob matches but whose verdicts never reached the committed
     `results/verifications.csv` (a stale export: diag_forensics §C2, critique M6). The loader
     is the rank report's rule line for line: `line.strip().rstrip(",")`, skip blanks, skip
     unparsable/ non-dict/ id-less lines, persona `lower()[:20]`, verdict lower/strip with
     anything outside {pass, fail, uncertain} coerced to "uncertain", KEEP-FIRST per
     (id, persona) in sorted-glob then line order. The rank report also truncates
     `alternative[:200]` / `notes[:300]` into the CSV; the replay keeps the text VERBATIM
     (`truncate=False`) because reason_audit.py regexes run on it — 121/1050 notes in the CSV
     are exactly 300 chars, i.e. cut mid-sentence.
  2. aggregate with `passcount_incumbent` (golden/aggregate_v2 when present, else the local
     table-identical copy below) and assert the replay reproduces `results.csv`
     n_pass / n_fail / n_uncertain / grade for all 350 ids the CSV verified;
  3. list the 12 recovered `*_ctl*` ids (COWLS controls, graded U in results.csv although
     three verdicts exist for each) with their COWLS codes from `control_recovery.csv`;
  4. write `outputs/incumbent_replay.csv` (id, n_pass, n_fail, n_uncertain, letter_incumbent,
     recovered_ctl = absent from verifications.csv, per-persona verdict / alternative / notes
     verbatim, + n_votes, any_ctl_file (a kept row came from a *_ctl* file — true for the 12
     AND for 3 D ids whose ctl row sorts before their f-batch row, exactly as the rank report
     kept them), the published grade and the COWLS code / ranking / theta_E for the diff) and
     print the per-persona pass table, the pairwise Cohen
     kappa between personas (diag_forensics §a: 17/12/8 passes; kappa 0.63/0.49/0.46) and a
     counterfactual table — what the letters would have been under two alternative readings
     of the same verdicts (uncertain = half a pass; only a fail that NAMES an alternative
     vetoes). The counterfactuals are the cheapest possible answer to "can re-weighting the
     old verdicts restore recall?" (the design's expectation: no — new calls are needed).

Everything here is read-only on J and needs no API. Run:

  python lensjudge/golden/incumbent_replay.py [--out outputs/incumbent_replay.csv]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.golden import _util  # noqa: E402

PERSONAS = ("artifact", "geometry", "morphology")
VERDICTS = ("pass", "fail", "uncertain")
VERDICT_DIR = _util.JWST_REPO / "results" / "verdicts"
RESULTS_CSV = _util.JWST_REPO / "results" / "results.csv"
VERIFICATIONS_CSV = _util.JWST_REPO / "results" / "verifications.csv"
CONTROL_RECOVERY_CSV = _util.JWST_REPO / "results" / "control_recovery.csv"
OUT_CSV = _util.LENSJUDGE / "outputs" / "incumbent_replay.csv"
# the rank report's CSV truncation (09_rank_report.py:44-45) — applied only on request
ALT_TRUNC, NOTES_TRUNC = 200, 300
# diag_forensics §a targets the replay must reproduce on the 350 CSV-verified ids
EXPECT_PASSES = {"artifact": 17, "morphology": 12, "geometry": 8}
EXPECT_KAPPA = {("artifact", "geometry"): 0.628, ("geometry", "morphology"): 0.486,
                ("artifact", "morphology"): 0.461}


# ------------------------------------------------------------------ the pass-count letter
def _passcount_local(verdicts) -> tuple[int, int, int, str]:
    """Table-identical to `J/scripts/09_rank_report.py:grade()` for an examined candidate:
    (n_pass, n_fail, n_uncertain, letter) with A/B/C/D by n_pass = 3/2/1/0 and "U" when
    there are no verdicts at all (never examined). `verdicts` is any iterable of verdict
    strings or of dicts/records carrying a `verdict` field; unknown strings count as
    "uncertain" exactly as the loader coerces them."""
    vs = []
    for v in verdicts:
        if isinstance(v, dict):
            v = v.get("verdict", "")
        elif not isinstance(v, str):
            v = getattr(v, "verdict", "")
        v = str(v).lower().strip()
        vs.append(v if v in VERDICTS else "uncertain")
    if not vs:
        return 0, 0, 0, "U"
    n_pass = sum(v == "pass" for v in vs)
    n_fail = sum(v == "fail" for v in vs)
    n_unc = sum(v == "uncertain" for v in vs)
    letter = {3: "A", 2: "B", 1: "C"}.get(n_pass, "D")
    return n_pass, n_fail, n_unc, letter


try:  # the shared aggregator (WP-2) owns the canonical copy once it lands
    from lensjudge.golden.aggregate_v2 import passcount_incumbent  # type: ignore
    PASSCOUNT_SOURCE = "golden.aggregate_v2"
except Exception:  # noqa: BLE001
    passcount_incumbent = _passcount_local
    PASSCOUNT_SOURCE = "incumbent_replay (local copy; aggregate_v2 not importable)"


# ------------------------------------------------------------------ loader
def load_verdicts(verdict_dir: Path = VERDICT_DIR, truncate: bool = False,
                  keep: str = "first") -> pd.DataFrame:
    """All verdict lines from every verify_*.jsonl under verdict_dir (incl. *_ctl*), parsed
    and deduplicated by 09_rank_report.py's rule. Columns: id, persona, verdict, alternative,
    notes, file, line_no, dup_rank (0 = the kept row). keep="first" is the rank report's
    rule; keep="all" returns every line (the 33 re-verified pairs stay visible)."""
    rows, bad = [], 0
    for f in sorted(glob.glob(str(Path(verdict_dir) / "verify_*.jsonl"))):
        with open(f) as fh:
            for i, line in enumerate(fh):
                line = line.strip().rstrip(",")
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    bad += 1
                    continue
                if not isinstance(d, dict) or "id" not in d:
                    bad += 1
                    continue
                v = str(d.get("verdict", "")).lower().strip()
                alt, notes = str(d.get("alternative", "")), str(d.get("notes", ""))
                if truncate:
                    alt, notes = alt[:ALT_TRUNC], notes[:NOTES_TRUNC]
                rows.append({"id": str(d["id"]),
                             "persona": str(d.get("persona", "")).lower()[:20],
                             "verdict": v if v in VERDICTS else "uncertain",
                             "alternative": alt, "notes": notes,
                             "file": Path(f).name, "line_no": i})
    ver = pd.DataFrame(rows, columns=["id", "persona", "verdict", "alternative", "notes",
                                      "file", "line_no"])
    ver.attrs["malformed"] = bad
    if ver.empty:
        return ver
    ver["dup_rank"] = ver.groupby(["id", "persona"]).cumcount()
    if keep == "first":
        ver = ver[ver["dup_rank"] == 0].reset_index(drop=True)
    return ver


def aggregate(ver: pd.DataFrame) -> pd.DataFrame:
    """One row per id: pass-count letter + the three personas' verdict/alternative/notes."""
    recs = []
    for cid, g in ver.groupby("id", sort=True):
        n_pass, n_fail, n_unc, letter = passcount_incumbent(g["verdict"].tolist())
        row = {"id": cid, "n_pass": n_pass, "n_fail": n_fail, "n_uncertain": n_unc,
               "n_votes": int(len(g)), "letter_incumbent": letter,
               # any kept verdict from a *_ctl* file (3 ids have a ctl row kept because
               # "verify_<p>_ctl0" sorts before "verify_<p>_f0"); recovered_ctl (absent from
               # verifications.csv) is set by reproduce_results, which knows the CSV
               "any_ctl_file": bool(g["file"].str.contains("_ctl").any()),
               "recovered_ctl": False}
        by = g.set_index("persona")
        for p in PERSONAS:
            if p in by.index:
                r = by.loc[p]
                row[f"{p}_verdict"] = r["verdict"]
                row[f"{p}_alternative"] = r["alternative"]
                row[f"{p}_notes"] = r["notes"]
            else:
                row[f"{p}_verdict"] = row[f"{p}_alternative"] = row[f"{p}_notes"] = ""
        recs.append(row)
    return pd.DataFrame(recs)


# ------------------------------------------------------------------ checks against results.csv
def reproduce_results(agg: pd.DataFrame, results_csv: Path = RESULTS_CSV,
                      verifications_csv: Path = VERIFICATIONS_CSV) -> tuple[pd.DataFrame, list[str]]:
    """Assert the replay reproduces results.csv (n_pass, n_fail, n_uncertain, grade) for
    every id verifications.csv verified, and return (agg joined with the published grade,
    the ids present in the raw jsonl but absent from verifications.csv = the recovered ctl
    verdicts, each graded U in results.csv)."""
    res = pd.read_csv(results_csv, dtype={"id": str}).set_index("id")
    csv_ids = set(pd.read_csv(verifications_csv, dtype={"id": str})["id"])
    agg = agg.copy()
    agg["grade_published"] = agg["id"].map(res["grade"]).fillna("")
    verified = agg[agg["id"].isin(csv_ids)]
    if len(verified) != len(csv_ids):
        raise AssertionError(f"replay covers {len(verified)} of the {len(csv_ids)} CSV-verified ids")
    pub = res.loc[verified["id"]]
    for col, mine in (("n_pass", "n_pass"), ("n_fail", "n_fail"), ("n_uncertain", "n_uncertain")):
        bad = verified["id"][pub[col].to_numpy(int) != verified[mine].to_numpy(int)].tolist()
        if bad:
            raise AssertionError(f"{col} differs from results.csv for {len(bad)} ids, e.g. {bad[:5]}")
    bad = verified["id"][pub["grade"].astype(str).to_numpy() != verified["letter_incumbent"].to_numpy()].tolist()
    if bad:
        raise AssertionError(f"grade differs from results.csv for {len(bad)} ids, e.g. {bad[:5]}")
    hidden = sorted(set(agg["id"]) - csv_ids)
    agg["recovered_ctl"] = agg["id"].isin(hidden)
    for cid in hidden:                       # the stale export graded every one of them U
        if cid in res.index and str(res.loc[cid, "grade"]) != "U":
            raise AssertionError(f"{cid} is absent from verifications.csv but graded {res.loc[cid, 'grade']}")
    return agg, hidden


def cowls_codes(ids, control_csv: Path = CONTROL_RECOVERY_CSV) -> pd.DataFrame:
    """COWLS code / ranking / theta_E for the given ids (empty columns when not a control)."""
    cols = ["id", "cowls_code", "ranking", "nA", "einstein_radius"]
    if not Path(control_csv).exists():
        return pd.DataFrame({"id": list(ids)}).assign(cowls_code="", ranking="", nA=np.nan, einstein_radius=np.nan)
    cr = pd.read_csv(control_csv, dtype={"id": str})[cols].drop_duplicates("id")
    out = pd.DataFrame({"id": list(ids)}).merge(cr, on="id", how="left")
    out["cowls_code"] = out["cowls_code"].fillna("")
    out["ranking"] = out["ranking"].fillna("")
    return out


# ------------------------------------------------------------------ diag_forensics §a tables
def persona_table(ver: pd.DataFrame) -> pd.DataFrame:
    """pass / fail / uncertain counts and pass rate per persona."""
    t = ver.groupby("persona")["verdict"].value_counts().unstack(fill_value=0)
    for v in VERDICTS:
        if v not in t.columns:
            t[v] = 0
    t = t[list(VERDICTS)].reindex(list(PERSONAS)).fillna(0).astype(int)
    t["n"] = t.sum(axis=1)
    t["pass_rate"] = (t["pass"] / t["n"].where(t["n"] > 0)).round(4)
    return t


def _kappa(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's kappa for two binary vectors (no sklearn dependency for a 2x2 table)."""
    n = len(x)
    if n == 0:
        return float("nan")
    po = float((x == y).mean())
    pe = float((x.mean() * y.mean()) + ((1 - x.mean()) * (1 - y.mean())))
    return float("nan") if pe == 1.0 else (po - pe) / (1 - pe)


def pairwise_kappa(ver: pd.DataFrame) -> pd.DataFrame:
    """Cohen kappa of pass-vs-not between each persona pair (diag_forensics §a)."""
    w = ver.pivot(index="id", columns="persona", values="verdict")
    rows = []
    for a, b in (("artifact", "geometry"), ("geometry", "morphology"), ("artifact", "morphology")):
        if a not in w.columns or b not in w.columns:
            continue
        both = w[[a, b]].dropna()
        x = (both[a] == "pass").to_numpy(int)
        y = (both[b] == "pass").to_numpy(int)
        rows.append({"pair": f"{a}-{b}", "n": int(len(both)), "agreement": round(float((x == y).mean()), 4),
                     "kappa": round(_kappa(x, y), 3), "n_both_pass": int((x & y).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ counterfactual letters
def _letter_from_score(s: float) -> str:
    return "A" if s >= 3 else "B" if s >= 2 else "C" if s >= 1 else "D"


def counterfactual_letters(agg: pd.DataFrame) -> pd.DataFrame:
    """Three readings of the SAME verdicts, one column each:
       published            A/B/C/D by n_pass (the incumbent rule);
       uncertain_half       n_pass + 0.5 * n_uncertain, floored to the same 3/2/1 ladder
                            (2 pass + 1 uncertain stays B; 1 pass + 2 uncertain -> B);
       named_alt_veto_only  a fail vetoes ONLY when the persona named an alternative
                            (non-blank `alternative`); letter = 3 - n_named_fails on the
                            same ladder (an unexplained fail or an uncertain no longer costs).
    """
    out = agg[["id"]].copy()
    out["published"] = agg["letter_incumbent"]
    out["uncertain_half"] = (agg["n_pass"] + 0.5 * agg["n_uncertain"]).map(_letter_from_score)
    named = np.zeros(len(agg), int)
    for p in PERSONAS:
        named += ((agg[f"{p}_verdict"] == "fail") & (agg[f"{p}_alternative"].str.strip() != "")).to_numpy(int)
    out["named_alt_veto_only"] = [_letter_from_score(3 - k) for k in named]
    return out


def counterfactual_table(agg: pd.DataFrame, cowls_ids=()) -> pd.DataFrame:
    """Letter census per rule over all replayed ids, plus how many COWLS controls reach
    A/B/C under each rule (the recall question the design says re-weighting cannot fix)."""
    cf = counterfactual_letters(agg)
    rows = []
    for rule in ("published", "uncertain_half", "named_alt_veto_only"):
        vc = cf[rule].value_counts()
        row = {"rule": rule, **{L: int(vc.get(L, 0)) for L in "ABCD"}, "n": int(len(cf))}
        if len(cowls_ids):
            sub = cf[cf["id"].isin(set(cowls_ids))]
            row["cowls_n"] = int(len(sub))
            row["cowls_ABC"] = int(sub[rule].isin(["A", "B", "C"]).sum())
            row["cowls_AB"] = int(sub[rule].isin(["A", "B"]).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ driver
def replay(verdict_dir: Path = VERDICT_DIR, results_csv: Path = RESULTS_CSV,
           verifications_csv: Path = VERIFICATIONS_CSV, control_csv: Path = CONTROL_RECOVERY_CSV,
           out_csv: Optional[Path] = OUT_CSV, quiet: bool = False) -> dict:
    """Load, aggregate, assert, write. Returns {"table": agg, "hidden": [...], "persona": ...,
    "kappa": ..., "counterfactual": ...}."""
    ver = load_verdicts(verdict_dir)
    agg = aggregate(ver)
    agg, hidden = reproduce_results(agg, results_csv, verifications_csv)
    codes = cowls_codes(agg["id"], control_csv).set_index("id")
    agg["cowls_code"] = agg["id"].map(codes["cowls_code"]).fillna("")
    agg["cowls_ranking"] = agg["id"].map(codes["ranking"]).fillna("")
    agg["cowls_theta_E"] = agg["id"].map(codes["einstein_radius"])
    cols = (["id", "n_pass", "n_fail", "n_uncertain", "letter_incumbent", "recovered_ctl"]
            + [f"{p}_{f}" for p in PERSONAS for f in ("verdict", "alternative", "notes")]
            + ["n_votes", "any_ctl_file", "grade_published", "cowls_code", "cowls_ranking", "cowls_theta_E"])
    agg = agg[cols]
    csv_ids = set(pd.read_csv(verifications_csv, dtype={"id": str})["id"])
    ver350 = ver[ver["id"].isin(csv_ids)]
    persona = persona_table(ver350)
    kappa = pairwise_kappa(ver350)
    cowls_ids = codes.index[codes["cowls_code"] != ""].tolist()
    cf = counterfactual_table(agg, cowls_ids)
    if out_csv is not None:
        sha = _util.pin(agg, Path(out_csv))
    else:
        sha = ""
    if not quiet:
        print(f"verdict lines kept: {len(ver)} (malformed {ver.attrs.get('malformed', 0)}); "
              f"ids: {agg['id'].nunique()}; passcount from {PASSCOUNT_SOURCE}")
        print(f"reproduces results.csv n_pass/n_fail/n_uncertain/grade for {len(csv_ids)}/{len(csv_ids)} verified ids")
        print(f"recovered {len(hidden)} *_ctl* ids graded U in results.csv although 3 verdicts exist:")
        rec = agg[agg["id"].isin(hidden)]
        for _, r in rec.iterrows():
            print(f"  {r['id']:20s} {r['cowls_code']:20s} {str(r['cowls_ranking']):8s} "
                  f"theta_E={r['cowls_theta_E'] if pd.notna(r['cowls_theta_E']) else 'nan'!s:>6} "
                  f"-> {r['letter_incumbent']} ({r['n_pass']}/{r['n_fail']}/{r['n_uncertain']})")
        print("\nper-persona verdicts (350 CSV-verified ids; diag_forensics §a):")
        print(persona.to_string())
        print("\npairwise Cohen kappa (pass vs not):")
        print(kappa.to_string(index=False))
        print("\ncounterfactual letters over the replayed ids:")
        print(cf.to_string(index=False))
        if out_csv is not None:
            print(f"\nwrote {out_csv} (sha {sha}, {len(agg)} rows)")
    return {"table": agg, "hidden": hidden, "persona": persona, "kappa": kappa,
            "counterfactual": cf, "verdicts": ver}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdict-dir", type=Path, default=VERDICT_DIR)
    ap.add_argument("--results", type=Path, default=RESULTS_CSV)
    ap.add_argument("--verifications", type=Path, default=VERIFICATIONS_CSV)
    ap.add_argument("--controls", type=Path, default=CONTROL_RECOVERY_CSV)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    a = ap.parse_args(argv)
    if not Path(a.verdict_dir).is_dir():
        raise SystemExit(f"{a.verdict_dir} not found (set LENSJUDGE_JWST_REPO)")
    replay(a.verdict_dir, a.results, a.verifications, a.controls, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
