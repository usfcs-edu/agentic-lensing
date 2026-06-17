#!/usr/bin/env python3
"""92_make_v3_tables.py — auto-generate the v3 LaTeX tables for the ClaudeNet report
from the tracked artifacts (so every number is reproducible, not hand-transcribed).
Writes papers/tables/*.tex; each .tex section \\input's the fragment it needs.
Robust: skips a table whose source artifact is absent.

  tab_progression  — D6: orig/v2-lean/v3blend8/v3-head on mimic-FPR + Euclid AUC.
  tab_a2           — A2: v2->v3 mimic-recovery gain + the random-FPR no-regression trade.
  tab_cvet         — C-vet/D-vet A/B yield (v2 vs v3-head vs v3blend8 top-30).
  tab_dr1011       — DR10 vs DR11 EDF-F/S recall + cross-validated counts.
  tab_catalog      — appendix: the cross-validated candidate catalog (D4 ∪ D5).
  tab_qualified    — appendix: the panel-qualified NEW v3 grade-A.

    /home2/benson/.venvs/claudenet/bin/python 92_make_v3_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C

V3 = C.DATA / "v3"            # JSON summaries / verdicts
V3FIND = C.ROOT / "v3"        # tracked findings CSVs (confirmed catalogs, qualified)
LJ = C.ROOT.parent / "lensjudge" / "outputs"
TAB = C.ROOT / "papers" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
EUCLID = C.ROOT.parent / "euclid-q1" / "data" / "raw" / "q1_discovery_engine_lens_catalog.csv"


def jload(p):
    return json.load(open(p)) if p.exists() else None


def write(name, body):
    (TAB / f"{name}.tex").write_text(body)
    print(f"[92t] wrote tables/{name}.tex")


def esc(s):
    return str(s).replace("_", "\\_")


def tab_progression():
    mp = jload(V3 / "model_progression.json")
    if not mp:
        return
    order = ["orig (effnet_B, random-neg)", "v2lean (hard-mined, 5)", "v3blend8 (8)", "v3head (learned)"]
    lab = {"orig (effnet_B, random-neg)": "orig (random-neg CNN)", "v2lean (hard-mined, 5)": "v2-lean",
           "v3blend8 (8)": "v3blend8", "v3head (learned)": "v3-head"}
    rows = []
    for m in order:
        d = mp[m]
        rows.append(f"{lab[m]} & {d['seed_storfer']:.3f} & {d['dr10_storfer']:.3f} & "
                    f"{d['dr10_inchausti']:.3f} & {d['euclid_AvsC_auc']:.3f}\\\\")
    body = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{Candidate quality across model generations (D6). recovery@matched-mimic-FPR(0.05) "
        "and the \\emph{independent} Euclid grade-A-vs-C AUC. The seed column is selection-biased "
        "(the seed bank is what the v2/v3 ensemble scored high), so orig-vs-v3 there is invalid; "
        "the trustworthy comparison is the Euclid AUC, where all models sit within $\\sim$1\\,SE "
        "($\\approx$0.06, $n_A{=}66$).}\n\\label{tab:progression}\\small\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "model & seed mimic@0.05 & DR10-eval@0.05 & DR10-eval@0.05 & Euclid A-vs-C\\\\\n"
        " & (Storfer) & (Storfer) & (Inchausti) & AUC (indep.)\\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write("tab_progression", body)


def tab_a2():
    d = jload(V3 / "a2_v3_verdict.json")
    if not d:
        return
    e = d["ensemble"]; nr = d["no_regression"]
    def cell(sp, ver, phi):
        return f"{e[sp][ver][phi]:.3f}"
    body = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{A2 contaminant-aware retraining (mimic-blend $f{=}0.5$), v2-lean$\\to$v3blend8, on the "
        "held-out DR10 mimic-eval. Mimic-recovery rises at every operating point; the random-galaxy "
        "recall (testneg null) drops only slightly (the staf327 trade).}\n\\label{tab:a2}\\small\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{recovery@mimic-FPR (Storfer)} & "
        "\\multicolumn{2}{c}{recovery@random-FPR(0.01)}\\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        " & @0.05 & @0.01 & Storfer & Inchausti\\\\\n\\midrule\n"
        f"v2-lean & {cell('storfer','v2','0.05')} & {cell('storfer','v2','0.01')} & "
        f"{nr['storfer']['v2']:.3f} & {nr['inchausti']['v2']:.3f}\\\\\n"
        f"v3blend8 & \\textbf{{{cell('storfer','v3','0.05')}}} & \\textbf{{{cell('storfer','v3','0.01')}}} & "
        f"{nr['storfer']['v3']:.3f} & {nr['inchausti']['v3']:.3f}\\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write("tab_a2", body)


def _grade_counts(parquet):
    p = LJ / parquet
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    vc = d["grade_pred"].value_counts().to_dict()
    return {g: int(vc.get(g, 0)) for g in ("A", "B", "C", "D")}


def tab_cvet():
    rows = [("v2 p\\_final (DR10)", "dr10_vet_top30.parquet"),
            ("v3-head (DR10)", "cv3_vet_top30.parquet"),
            ("v3blend8 (DR10)", "cv3b8_vet_top30.parquet")]
    body_rows = []
    for lab, pq in rows:
        c = _grade_counts(pq)
        if c is None:
            continue
        ab = c["A"] + c["B"]
        body_rows.append(f"{lab} & {c['A']} & {c['B']} & {c['C']} & {c['D']} & \\textbf{{{ab}}}\\\\")
    if not body_rows:
        return
    body = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{C-vet agentic grading of the top-30 NEW candidates (same harness, \\code{rubric\\_imaging\\_v2}). "
        "v3's mimic-aware ranking does not raise the top-of-list A/B yield over v2 at DECaLS resolution "
        "(the discovery frontier is resolution-bounded); the D's stay lrg\\_companion-dominated.}\n"
        "\\label{tab:cvet}\\small\n\\begin{tabular}{lccccc}\n\\toprule\n"
        "ranking & A & B & C & D & A/B\\\\\n\\midrule\n"
        + "\n".join(body_rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write("tab_cvet", body)


def tab_dr1011():
    d10 = jload(V3 / "cv3_edf_select_summary.json")
    d11 = jload(V3 / "cv3_dr11edffs_select_summary.json")
    if not (d10 and d11):
        return
    def rec(s, g, k):
        return s.get("recall", {}).get(g, {}).get(k, "--")
    # confirmed counts + cross-release overlap
    c10 = pd.read_csv(V3FIND / "cv3_edf_confirmed.csv") if (V3FIND / "cv3_edf_confirmed.csv").exists() else None
    c11 = pd.read_csv(V3FIND / "cv3_dr11edffs_confirmed.csv") if (V3FIND / "cv3_dr11edffs_confirmed.csv").exists() else None
    n10 = len(c10) if c10 is not None else "--"
    n11 = len(c11) if c11 is not None else "--"
    ov = uni = "--"
    if c10 is not None and c11 is not None:
        e10 = set(c10.euclid_id.astype(str)); e11 = set(c11.euclid_id.astype(str))
        ov = len(e10 & e11); uni = len(e10 | e11)
    body = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{DR10 vs DR11 on the same EDF-F/S DECam sky: v3blend8 region-local recovery of "
        "Euclid grade-A lenses into the top-$K$, and the cross-validated NEW-confirmed counts. DR11's "
        "deeper reductions give a small, consistent edge; v3 transfers cleanly across DECam releases "
        "(contrast the DR11-north BASS/MzLS failure, D3).}\n\\label{tab:dr1011}\\small\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "release & Euclid-A in top-100 & in top-1000 & NEW-confirmed A/B & \\\\\n\\midrule\n"
        f"DR10 (D4) & {rec(d10,'A','top100')} & {rec(d10,'A','top1000')} & {n10} & \\\\\n"
        f"DR11 (D5) & {rec(d11,'A','top100')} & {rec(d11,'A','top1000')} & {n11} & \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
        f"\\\\[2pt]{{\\small Cross-release (by Euclid id): {ov} shared, {uni} unique cross-validated lenses.}}\n"
        "\\end{table}\n")
    write("tab_dr1011", body)


def tab_catalog():
    fs = [(V3FIND / "cv3_edf_confirmed.csv", "DR10"), (V3FIND / "cv3_dr11edffs_confirmed.csv", "DR11")]
    parts = []
    for p, rel in fs:
        if p.exists():
            d = pd.read_csv(p); d["release"] = rel; parts.append(d)
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True).astype({"euclid_id": str, "name": str})
    # coords from the Euclid catalog by id
    if EUCLID.exists():
        q = pd.read_csv(EUCLID).astype({"id_str": str})[["id_str", "right_ascension", "declination"]]
        df = df.merge(q, left_on="euclid_id", right_on="id_str", how="left")
    else:
        df["right_ascension"] = np.nan; df["declination"] = np.nan
    # one row per unique Euclid lens; mark release(s)
    g = df.groupby("euclid_id")
    rows = []
    for eid, sub in g:
        rel = "both" if sub.release.nunique() > 1 else sub.release.iloc[0]
        r = sub.iloc[0]
        rows.append((esc(r["name"]), r["right_ascension"], r["declination"],
                     r["grade_truth"], r["grade_pred"], r["p_lens_tier1"], r["p_lens_tier2"], rel))
    rows.sort(key=lambda t: (t[3], -float(t[6]) if pd.notna(t[6]) else 0))
    lines = [f"{n} & {ra:.4f} & {dec:.4f} & {gt} & {gp} & {t1:.2f}$\\to${t2:.2f} & {rel}\\\\"
             for (n, ra, dec, gt, gp, t1, t2, rel) in rows]
    body = (
        "{\\small\n\\begin{longtable}{lcccccc}\n"
        "\\caption{Cross-validated strong-lens candidates: DESI v3 survivors that match a Euclid Q1 "
        "discovery-engine lens, NEW to the DESI lens-finder catalogues (Storfer/Inchausti/Huang), "
        "graded at Euclid 0.1$''$. These are independently recovered Euclid catalogue lenses, "
        "\\emph{not} net-new discoveries. $p_{\\rm lens}$ shows the DESI$\\to$Euclid tier flip.}\\\\\n"
        "\\label{tab:catalog}\\\\\n\\toprule\n"
        "DESI id & RA & Dec & Euclid & LensJudge & $p_{\\rm lens}$ & release\\\\\n"
        " & (deg) & (deg) & grade & @0.1$''$ & DESI$\\to$Euclid & \\\\\n\\midrule\n\\endfirsthead\n"
        "\\toprule DESI id & RA & Dec & Euclid & LensJudge & $p_{\\rm lens}$ & release\\\\\\midrule\\endhead\n"
        + "\n".join(lines) +
        "\n\\bottomrule\n\\end{longtable}\n}\n")
    write("tab_catalog", body)
    print(f"[92t]   catalog: {len(rows)} unique cross-validated lenses")


def tab_qualified():
    p = V3FIND / "cv3_new_gradeA.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).astype({"name": str})
    # join panel verdict if present
    panel = LJ / "d2_newA_panel.parquet"
    pg = {}
    if panel.exists():
        pp = pd.read_parquet(panel).astype({"name": str})
        pg = dict(zip(pp["name"], pp["grade_pred"]))
    rows = []
    for _, r in df.iterrows():
        rows.append(f"{esc(r['name'])} & {r.get('p_lens','--')} & {r.get('list','--')} & "
                    f"{pg.get(str(r['name']),'--')}\\\\")
    body = (
        "\\begin{table}[t]\\centering\n"
        "\\caption{Three NEW v3 grade-A candidates from C-vet and their qualification panel verdict (D2; "
        "advocate/skeptic/morphology/contaminant, 2-of-$N$ A-rule). DESI-resolution only (no Euclid "
        "coverage at these positions): DESI-qualified candidates, not confirmations. A fourth C-vet "
        "survivor (\\texttt{s\\_310364\\_6649}) was found to be the already-published Huang+2020 grade-A "
        "lens \\texttt{DESI-038.2078-03.3906} ($0.2''$ match) and is excluded here as a recovery, not a "
        "new candidate.}\n"
        "\\label{tab:qualified}\\small\n\\begin{tabular}{lccc}\n\\toprule\n"
        "DESI id & C-vet $p_{\\rm lens}$ & ranking & panel grade\\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write("tab_qualified", body)


def main():
    for f in (tab_progression, tab_a2, tab_cvet, tab_dr1011, tab_catalog, tab_qualified):
        try:
            f()
        except Exception as e:
            print(f"[92t] {f.__name__} skipped: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
