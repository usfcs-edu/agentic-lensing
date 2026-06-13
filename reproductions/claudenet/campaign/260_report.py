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

    # copy gallery composites the report embeds
    qfigs = []
    for _, r in qual.iterrows():
        src = GAL / f"{r['tier']}_{int(r['rank']):03d}_{r['row_id']}.png"
        if src.exists():
            dst = FIGS / src.name
            shutil.copy(src, dst)
            qfigs.append((r, dst.name))
    efigs = []
    for src in sorted(GAL.glob("escalation_*.png")):
        dst = FIGS / src.name
        shutil.copy(src, dst)
        efigs.append((src.stem.split("_", 2)[-1], dst.name))

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

    L.append("## Can any of the 737 be undiscovered lenses?\n")
    if n_qual > 0:
        L.append(f"**Yes — {n_qual} candidate(s) survive every filter**: still unmatched to any "
                 "queried catalog, and independently graded ≥B by two methodologically distinct "
                 "Claude graders (one after adversarial skeptic review). These are the strongest "
                 "follow-up targets. **This is not a confirmation** — DECaLS grz cannot confirm a "
                 "small-θ_E arc; qualification means *worth higher-resolution imaging "
                 "(HSC/Euclid/HST) or spectroscopy*, the only thing that turns candidate into "
                 "lens.\n")
    else:
        L.append("**No candidate passes the strict double-≥B bar after the adversarial skeptic** "
                 "— so on this evidence the 737 are dominated by the LRG+companion/blend "
                 "false-positive population, consistent with the conformal null being *random "
                 "galaxy* (not *non-lens*). The honest answer to *could any be undiscovered "
                 "lenses* is: a small **escalation set** (one grader ≥B) remains as the only "
                 "plausible follow-up shortlist; none is supported strongly enough to call a "
                 "qualified candidate.\n")

    if qfigs:
        L.append("## Qualified candidates (gallery)\n")
        L.append("Each panel: full | zoom | lens-light residual; caption carries the identifier, "
                 "coordinates, scores and both grades.\n")
        for r, fn in qfigs:
            L.append(f"### {r['row_id']}  (tier **{r['tier']}**, rank {int(r['rank'])})\n")
            L.append(f"- RA={r['RA']:.6f}  DEC={r['DEC']:.6f}  |  p_final={r['p_final']:.3f}  "
                     f"q_group={r.get('q_group', float('nan')):.2e}  |  status={r['status']}")
            L.append(f"- visual grade **{r['my_grade']}**, lensjudge **{r['lensjudge_grade']}**")
            L.append(f"- visual rationale: {r.get('rationale_visual','')}")
            L.append(f"- lensjudge rationale: {r.get('lensjudge_rationale','')}\n")
            L.append(f"![{r['row_id']}](figs/{fn})\n")

    if efigs:
        L.append("## Escalation set (one grader ≥B — follow-up shortlist)\n")
        em = full.set_index("row_id")
        for rid, fn in efigs:
            r = em.loc[rid] if rid in em.index else None
            cap = ("" if r is None else
                   f"RA={r['RA']:.6f} DEC={r['DEC']:.6f} p_final={r['p_final']:.3f} "
                   f"visual={r['my_grade']} lensjudge={r['lensjudge_grade']}")
            L.append(f"**{rid}** — {cap}\n\n![{rid}](figs/{fn})\n")

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
    print(f"[260] wrote {REP/'REPORT.md'} ({n_qual} qualified, {len(efigs)} escalation figs)")

    # ---- LaTeX section ----
    tex = [r"\section{Qualifying the 737 ClaudeNet v2 sweep candidates}",
           r"A campaign to test whether any of the 737 new-and-unseen DR9-sweep candidates "
           r"(group-conformal FDR$\le$0.05) could be genuinely undiscovered strong lenses, via "
           r"an expanded crossmatch (SIMBAD + DES/KiDS VizieR; NED/SuGOHI unreachable) and two "
           r"independent visual passes (a structured Huang-VI rubric pass with an adversarial "
           r"skeptic re-check, and the independent \texttt{lensjudge} harness on Claude-opus). "
           r"\emph{Qualified} = still NEW after crossmatch AND graded A/B by BOTH passes.",
           r"\par\medskip\noindent Of " + str(n) + r" candidates, " + str(n_new) +
           r" remain NEW after the expanded crossmatch. The structured skeptic confirmed only " +
           str(my_final.get("A", 0) + my_final.get("B", 0)) + r" of " +
           str(my_first.get("A", 0) + my_first.get("B", 0)) + r" first-pass A/B. "
           r"Qualified: \textbf{" + str(n_qual) + r"} (gold " + str(tiers.get("gold", 0)) +
           r", silver " + str(tiers.get("silver", 0)) + r", bronze " + str(tiers.get("bronze", 0)) +
           r"); escalation set " + str(n_esc) + r".",
           ]
    for r, fn in qfigs:
        tex.append(r"\begin{figure}[t]\centering")
        tex.append(rf"\includegraphics[width=0.92\textwidth]{{../campaign/report/figs/{fn}}}")
        cap = (rf"\textbf{{{r['row_id']}}} (tier {r['tier']}): RA={r['RA']:.5f}, "
               rf"DEC={r['DEC']:.5f}, $p_{{\rm final}}$={r['p_final']:.3f}; "
               rf"visual {r['my_grade']}, lensjudge {r['lensjudge_grade']}. "
               rf"Full $|$ zoom $|$ residual.")
        tex.append(rf"\caption{{{cap}}}")
        tex.append(r"\end{figure}")
    (PAPERS / "campaign_section.tex").write_text("\n".join(tex) + "\n")
    print(f"[260] wrote {PAPERS/'campaign_section.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
