"""Aggregate perspective-diverse judge votes into one grade (Option C).

Rules (the §9.1 evidence discipline, applied to a panel):
  * ordinal consensus = median grade across judges;
  * SKEPTIC-VETO — if the Skeptic grades D and names a decisive contaminant, cap the
    grade at C unless >=2 judges independently grade A (genuine arc evidence overrides);
  * A-grade requires >=2 judges at A (no lone-advocate A);
  * confidence = fraction of judges within one grade of the consensus (quorum signal);
  * escalate to a human when the panel disagrees (spread >=2) or the grade is A
    (follow-up-bound), per the human-in-loop gate.
"""
from __future__ import annotations

import numpy as np

from lensjudge.common.schemas import CriteriaScores, ImageGrade, JudgeVote

_ORD = {"A": 0, "B": 1, "C": 2, "D": 3}
_INV = {v: k for k, v in _ORD.items()}


def aggregate(votes: list[JudgeVote]) -> ImageGrade:
    if not votes:
        return ImageGrade(grade="D", p_lens=0.0, confidence=0.0,
                          escalate_to_human=True, rationale="panel: no votes")
    g = [_ORD[v.grade] for v in votes]
    median_ord = int(round(float(np.median(g))))
    n_A = sum(1 for v in votes if v.grade == "A")
    skeptic = next((v for v in votes if v.role == "skeptic"), None)

    final = median_ord
    contaminant = None
    if skeptic is not None and skeptic.grade == "D" and skeptic.contaminant and n_A < 2:
        final = max(final, _ORD["C"])           # skeptic-veto: cannot be A/B
        contaminant = skeptic.contaminant
    if final == _ORD["A"] and n_A < 2:           # no lone-advocate A
        final = _ORD["B"]
    grade = _INV[min(3, max(0, final))]

    spread = max(g) - min(g)
    within_one = sum(1 for x in g if abs(x - median_ord) <= 1)
    confidence = round(within_one / len(votes), 3)
    escalate = (spread >= 2) or (grade == "A")

    crit = {k: int(round(float(np.mean([getattr(v.criteria, k) for v in votes]))))
            for k in CriteriaScores.model_fields}
    summary = " | ".join(f"{v.role}:{v.grade}(p={v.p_lens:.2f}"
                         f"{',cont='+v.contaminant if v.contaminant else ''})" for v in votes)
    return ImageGrade(
        grade=grade, criteria=crit, p_lens=round(float(np.mean([v.p_lens for v in votes])), 3),
        confidence=confidence, contaminant=contaminant, escalate_to_human=escalate,
        rationale=f"panel[{len(votes)}] -> {grade} (spread {spread}). " + summary,
    )
