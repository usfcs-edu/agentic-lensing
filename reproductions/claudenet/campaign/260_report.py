#!/usr/bin/env python3
"""260_report.py — write the qualification-campaign report (Markdown + LaTeX),
embedding the actual RGB gallery images + identifiers of the qualified (and
escalation) candidates.

Tracked outputs (not under the gitignored data/):
  campaign/report/REPORT.md            Markdown, renders images inline on GitHub
  campaign/report/figs/*.png           the gallery composites the report embeds
  campaign/report/candidates_qualified.csv   the shortlist (also tracked)
  papers/campaign_section.tex          LaTeX section (\\input into a papers build)

    /home2/benson/.venvs/claudenet/bin/python campaign/260_report.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "campaign"
GAL = OUT / "gallery"
REP = ROOT / "campaign" / "report"
FIGS = REP / "figs"
PAPERS = ROOT / "papers"


def _vc(s):
    return s.value_counts().reindex(["A", "B", "C", "D"]).fillna(0).astype(int).to_dict()


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main() -> int:
    REP.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    full = pd.read_parquet(OUT / "consensus_full_737.parquet")
    full["row_id"] = full["row_id"].astype(str)
    qual = pd.read_parquet(OUT / "candidates_qualified.parquet")
    vis = pd.read_parquet(OUT / "visual_grades_verified.parquet")
    recall = {}
    rp = OUT / "crossmatch_recall_737.json"
    if rp.exists():
        recall = json.load(open(rp))

    n = len(full)
    status_vc = full.status.value_counts().to_dict()
    n_new = int((full.status == "NEW").sum())
    my_first = _vc(vis.first_grade)
    my_final = _vc(vis.my_grade)
    lj = _vc(full.lensjudge_grade.dropna()) if "lensjudge_grade" in full else {}
    n_qual = len(qual)
    tiers = full[full.qualified].tier.value_counts().to_dict() if "qualified" in full else {}
    n_esc = int(full.get("escalation", pd.Series([], dtype=bool)).sum())

    fullx = full.set_index("row_id")

    def copy_set(glob_pat):
        figs = []
        for src in sorted(GAL.glob(glob_pat)):
            dst = FIGS / src.name
            shutil.copy(src, dst)
            rid = "_".join(src.stem.split("_")[2:])
            figs.append((rid, dst.name))
        return figs

    qfigs = []
    for _, r in qual.iterrows():
        src = GAL / f"{r['tier']}_{int(r['rank']):03d}_{r['row_id']}.png"
        if src.exists():
            dst = FIGS / src.name
            shutil.copy(src, dst)
            qfigs.append((r, dst.name))
    recfigs = copy_set("recovered_*.png")     # >=B by either grader (known lenses)
    newfigs = copy_set("topnew_*.png")        # best still-NEW (the new population)
    n_known = int((full.status != "NEW").sum())

    # ---- Markdown ----
    L = []
    L.append("# Qualifying the 737 ClaudeNet v2 DR9-sweep lens candidates\n")
    L.append("A campaign to test whether any of the 737 new-and-unseen candidates from the "
             "ClaudeNet v2 DR9 sky sweep (group-conformal FDR ≤ 0.05) could be genuinely "
             "**undiscovered** strong lenses — by (1) crossmatching against external sources "
             "beyond the 4 local DECaLS catalogs, and (2) judging every candidate visually "
             "two independent ways.\n")

    L.append("## Methods\n")
    L.append("**The 737 set.** From `candidates_v2.parquet`: `status==NEW` (unmatched to "
             "storfer2024/inchausti2025/huang2021/curated at 5″) and not a training / mined / "
             "calibration row, selected at per-group conformal FDR ≤ 0.05. All 737 are DECaLS "
             "south, with finite RA/DEC and the exact CNN-seen grz cutouts in hand.\n")
    L.append("**Expanded crossmatch.** Each candidate was re-checked against external sources: "
             "a **SIMBAD** cone search (5″, lens otypes gLe/gLS/LeG/…) and the **VizieR** lens "
             "catalogs reachable from this host — DES strong-lens candidates (Jacobs+2019, "
             "`J/ApJS/243/17`) and KiDS LinKS (Petrillo+2019, `J/MNRAS/484/3879`). NED and "
             "SuGOHI were not reachable from this host (documented, not silently dropped; "
             "SuGOHI is HSC-footprint-heavy with little DECaLS-south overlap).\n")
    L.append("**Two independent visual passes.** (a) A *structured* pass: subagents view the "
             "four rendered views (full / 2.5×-zoom / lens-light-residual / high-contrast) per "
             "candidate and grade against the Huang-VI five-criterion A/B/C/D rubric; every A/B "
             "is then adversarially re-checked by a **skeptic** that must actively refute the "
             "lens hypothesis (the grade holds only if it survives at ≥B). (b) The independent "
             "**lensjudge** harness (separate prompts, separate orchestration, Claude-opus), "
             "grading the same on-disk pixels. The two graders share no code path — the point "
             "of the consensus.\n")
    L.append("**Qualified definition.** A candidate is *qualified* iff it is **still NEW** after "
             "the expanded crossmatch **and** graded **A or B by BOTH** passes. Tiers: gold "
             "(A&A), silver (≥B, ≥1 A), bronze (B&B). A candidate graded ≥B by exactly one pass "
             "is an *escalation* target (listed, not qualified).\n")

    L.append("## Findings\n")
    L.append(f"**Crossmatch.** Of {n} candidates, status after the expanded crossmatch: "
             f"`{status_vc}`. **{n_new} remain NEW** (not in any local *or* queried external "
             f"catalog, no SIMBAD lens-type).\n")
    L.append("**Visual grading (distributions).**\n")
    L.append(md_table(
        [["my structured pass (first)", my_first.get("A", 0), my_first.get("B", 0),
          my_first.get("C", 0), my_first.get("D", 0)],
         ["my structured pass (post-skeptic)", my_final.get("A", 0), my_final.get("B", 0),
          my_final.get("C", 0), my_final.get("D", 0)],
         ["lensjudge (opus)", lj.get("A", 0), lj.get("B", 0), lj.get("C", 0), lj.get("D", 0)]],
        ["grader", "A", "B", "C", "D"]) + "\n")
    L.append(f"The structured pass's skeptic is deliberately harsh: of "
             f"{my_first.get('A',0)+my_first.get('B',0)} first-pass A/B it confirmed only "
             f"{my_final.get('A',0)+my_final.get('B',0)} at ≥B. This is the honest cost of "
             f"demanding the lens evidence survive active refutation at DECaLS resolution "
             f"(θ_E ≈ 1–2″ = 4–8 px).\n")
    L.append(f"**Consensus.** Qualified (NEW & both passes ≥B): **{n_qual}** "
             f"(gold {tiers.get('gold',0)}, silver {tiers.get('silver',0)}, "
             f"bronze {tiers.get('bronze',0)}). Escalation (one pass ≥B, still NEW): "
             f"**{n_esc}**.\n")

    # the candidates either final grader rates >=B (the validation set)
    rec = full[(full.my_grade.isin(["A", "B"])) | (full.lensjudge_grade.isin(["A", "B"]))].copy()
    n_rec_known = int((rec.status != "NEW").sum())

    L.append("## Can any of the 737 be undiscovered lenses?\n")
    L.append(f"**On this evidence, no.** Of the 737, **{n_new} remain genuinely NEW** after the "
             f"expanded crossmatch, but **none of them is rated A or B by either independent "
             f"grader's strongest pass** (the skeptic-verified structured pass, or the "
             f"factored multiagent lensjudge pass). Qualified (still-NEW AND ≥B by both): "
             f"**{n_qual}**.\n")
    L.append(f"Crucially, this is *not* the graders rejecting everything. Together they rated "
             f"**{len(rec)} candidates ≥B** — and **{n_rec_known} of those {len(rec)} are "
             f"already-catalogued lenses** ({int((rec.nearest_catalog=='des_jacobs2019').sum())} "
             f"matching DES at sub-arcsecond separation, the rest SIMBAD lens-types). The dual "
             f"grader + skeptic consensus therefore **re-discovered real, known lenses** — a "
             f"clean internal validation that the vetting identifies genuine lenses — and found "
             f"nothing among the {n_new} new candidates that rises to the same confidence. The "
             f"737 are dominated by the LRG+companion/blend false-positive population, exactly "
             f"the lens-vs-non-lens distinction the conformal step (null = *random galaxy*, not "
             f"*non-lens*) could not make.\n")

    if recfigs:
        L.append("## Lenses the campaign re-discovered (the validation set)\n")
        L.append("Every candidate either grader graded ≥B. All are already catalogued — shown "
                 "here as proof the two independent visual passes recognise genuine lens "
                 "morphology. Each panel: full | zoom | lens-light residual.\n")
        for rid, fn in recfigs:
            r = fullx.loc[rid]
            L.append(f"**{rid}** — RA={r['RA']:.6f} DEC={r['DEC']:.6f}, p_final={r['p_final']:.3f}, "
                     f"status **{r['status']}** (nearest {r.get('nearest_catalog','-')} at "
                     f"{r.get('nearest_sep_arcsec', float('nan')):.2f}″); visual **{r['my_grade']}**, "
                     f"lensjudge **{r['lensjudge_grade']}**.")
            L.append(f"![{rid}](figs/{fn})\n")

    if newfigs:
        L.append("## The best of the genuinely-NEW candidates\n")
        L.append("The top still-NEW candidates by mean grader probability — the strongest of the "
                 f"{n_new}. None reached ≥B from either final grader (all C/D); shown for "
                 "transparency. These are the natural targets if higher-resolution follow-up "
                 "(HSC/Euclid/HST) or spectroscopy is ever pursued, but on DECaLS grz alone "
                 "they are not confident lenses.\n")
        for rid, fn in newfigs:
            r = fullx.loc[rid]
            L.append(f"**{rid}** — RA={r['RA']:.6f} DEC={r['DEC']:.6f}, p_final={r['p_final']:.3f}; "
                     f"visual **{r['my_grade']}**, lensjudge **{r['lensjudge_grade']}**. "
                     f"_{r.get('rationale_visual','')}_")
            L.append(f"![{rid}](figs/{fn})\n")

    L.append("## Honest limits\n")
    L.append("- *Qualified ≠ confirmed.* Every candidate here is a follow-up target, not a "
             "lens; confirmation needs higher-resolution imaging or spectroscopy.\n")
    L.append("- *Crossmatch is partial.* SIMBAD + DES + KiDS were queried; NED and SuGOHI were "
             "unreachable from this host. \"Still NEW\" means *not in the queried sources*, a "
             "stronger statement than the original 4-catalog NEW but not exhaustive.\n")
    L.append("- *Both graders are Claude.* Independence here means different harness / prompt / "
             "orchestration, not statistical independence; the skeptic pass is the adversarial "
             "counterweight. Agreement is reported, not assumed.\n")
    L.append("- *Conformal null.* The FDR≤0.05 that defined the 737 controls against the "
             "*random-galaxy* population, so lens-mimicking false positives are expected in it; "
             "this campaign is exactly the lens-vs-non-lens filter the conformal step could not "
             "provide.\n")

    (REP / "REPORT.md").write_text("\n".join(L) + "\n")
    shutil.copy(OUT / "candidates_qualified.csv", REP / "candidates_qualified.csv")
    print(f"[260] wrote {REP/'REPORT.md'} ({n_qual} qualified, {len(recfigs)} recovered, "
          f"{len(newfigs)} top-new figs)")

    # ---- LaTeX section ----
    def texesc(s):
        return str(s).replace("_", r"\_").replace("&", r"\&")

    tex = [r"\section{Qualifying the 737 ClaudeNet v2 sweep candidates}",
           r"A campaign to test whether any of the 737 new-and-unseen DR9-sweep candidates "
           r"(group-conformal FDR$\le$0.05) could be genuinely undiscovered strong lenses, via "
           r"an expanded crossmatch (SIMBAD + DES/KiDS VizieR; NED/SuGOHI unreachable) and two "
           r"methodologically-independent visual passes: a structured Huang-VI rubric pass with "
           r"an adversarial skeptic re-check, and the independent \texttt{lensjudge} harness "
           r"(factored multiagent, Claude-opus). \emph{Qualified} = still NEW after the "
           r"crossmatch AND graded A/B by BOTH passes.",
           r"\par\medskip\noindent\textbf{Result.} Of " + str(n) + r" candidates, " +
           str(n_known) + r" (" + f"{100*n_known/n:.0f}" + r"\%) coincide with a known lens or "
           r"lens-candidate (" + str(int((full.nearest_catalog == 'des_jacobs2019').sum())) +
           r" DES, plus KiDS and SIMBAD lens-types); \textbf{" + str(n_new) + r"} remain NEW. "
           r"\emph{None} of the " + str(n_new) + r" new candidates is graded A/B by either "
           r"grader's strongest pass, so \textbf{qualified $=" + str(n_qual) + r"$}. This is not "
           r"the graders rejecting everything: together they graded " + str(len(rec)) +
           r" candidates $\ge$B, and \textbf{all " + str(len(rec)) + r" are already-catalogued "
           r"lenses} (" + str(int((rec.nearest_catalog == 'des_jacobs2019').sum())) +
           r" sub-arcsecond DES matches) --- the consensus re-discovered real lenses and found "
           r"nothing comparable among the new ones.",
           ]
    tex.append(r"\par\medskip\noindent\emph{Lenses the campaign re-discovered "
               r"(already catalogued; the validation):}")
    for rid, fn in recfigs:
        r = fullx.loc[rid]
        tex.append(r"\begin{figure}[t]\centering")
        tex.append(rf"\includegraphics[width=0.92\textwidth]{{../campaign/report/figs/{fn}}}")
        cap = (rf"\textbf{{{texesc(rid)}}}: RA={r['RA']:.5f}, DEC={r['DEC']:.5f}, "
               rf"$p_{{\rm final}}$={r['p_final']:.3f}; status {texesc(r['status'])} "
               rf"(nearest {texesc(r.get('nearest_catalog','-'))} at "
               rf"{r.get('nearest_sep_arcsec', float('nan')):.2f}$''$); "
               rf"visual {r['my_grade']}, lensjudge {r['lensjudge_grade']}. "
               rf"Full $|$ zoom $|$ residual.")
        tex.append(rf"\caption{{{cap}}}")
        tex.append(r"\end{figure}")
    (PAPERS / "campaign_section.tex").write_text("\n".join(tex) + "\n")
    print(f"[260] wrote {PAPERS/'campaign_section.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
