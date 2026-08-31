"""Critique submission: one reviewer's verdicts on one proposal run.

``submit_critique`` writes ``critiques/<image_id>.<reviewer>.<run_id>.json`` (immutable critique
document), merges each per-item verdict into the item's ``review`` block of the LensMark file,
computes ``delta_arcsec`` for edited items from ``edit_of`` and records the critique path in
``provenance.critiques``. The file is saved through the campaign (atomic write + diff log) and one
extra ``op:"critique"`` log line is appended so the history shows when the review happened.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from . import coords
from .model import Critique, CritiqueItem, ItemBase, LensMarkFile, ProposalRun, Review
from .store import Campaign, atomic_write_text

GEOMETRY_ANCHORS = ("head", "center", "pos")   # the per-type anchor used for delta_arcsec
COUNT_KEYS = ("proposed", "accepted", "edited", "rejected", "invalid", "unreviewed", "added_by_human")


def critique_path(campaign: Campaign, critique: Critique) -> Path:
    return campaign.critiques_dir / f"{critique.image_id}.{critique.reviewer}.{critique.run_id}.json"


def _relative(campaign: Campaign, path: Path) -> str:
    return path.relative_to(campaign.root).as_posix()


def delta_arcsec(item: ItemBase, file: LensMarkFile) -> Optional[float]:
    """Angular distance between the pre-edit anchor stored in ``item.edit_of`` and the current one
    (arrow head, circle centre, text position); None when ``edit_of`` holds no prior anchor."""
    if not item.edit_of:
        return None
    for key in GEOMETRY_ANCHORS:
        prior = item.edit_of.get(key)
        cur = getattr(item, key, None)
        if prior is None or cur is None:
            continue
        try:
            return coords.dist_arcsec(prior, cur, file.image.width, file.image.height, file.image.cutout_arcsec)
        except (TypeError, IndexError, ValueError):
            return None
    return None


def run_counts(file: LensMarkFile, run_id: str) -> dict[str, int]:
    """Status tally of the items a run proposed, plus the recall signal (human items flagged missed_by_model)."""
    counts = {k: 0 for k in COUNT_KEYS}
    for it in file.items:
        if it.created_by.kind == "claude" and it.created_by.run_id == run_id:
            counts["proposed"] += 1
            if it.status == "proposed":
                counts["unreviewed"] += 1
            elif it.status in counts:
                counts[it.status] += 1
        elif it.created_by.kind != "claude" and it.review is not None and it.review.verdict == "missed_by_model":
            counts["added_by_human"] += 1
    return counts


def _status_from_verdict(item: ItemBase, verdict: str) -> str:
    """Resolve a still-``proposed`` item from its verdict (the UI normally sets status itself; this
    keeps the CLI / API path coherent): correct -> accepted; spurious/redundant -> rejected;
    wrong_* -> edited when the item was edited (edit_of present), else rejected."""
    if verdict == "correct":
        return "accepted"
    if verdict in ("spurious", "redundant"):
        return "rejected"
    if verdict.startswith("wrong_"):
        return "edited" if item.edit_of else "rejected"
    return item.status


def merge_reviews(file: LensMarkFile, critique: Critique) -> list[str]:
    """Copy every CritiqueItem into the matching item's ``review``; returns the unknown item ids."""
    unknown: list[str] = []
    for ci in critique.items:
        it = file.item(ci.item_id)
        if it is None:
            unknown.append(ci.item_id)
            continue
        d = delta_arcsec(it, file)
        if d is None:
            d = ci.delta_arcsec
        ci.delta_arcsec = d
        it.review = Review(verdict=ci.verdict, severity=ci.severity, comment=ci.comment,
                           reviewer=critique.reviewer, reviewed_at=critique.reviewed_at, delta_arcsec=d)
        if it.status == "proposed":
            it.status = _status_from_verdict(it, ci.verdict)  # type: ignore[assignment]
    return unknown


def find_run(file: Optional[LensMarkFile], run_id: str) -> Optional[ProposalRun]:
    if file is None:
        return None
    for r in file.provenance.proposal_runs:
        if r.run_id == run_id:
            return r
    return None


def submit_critique(campaign: Campaign, critique: Critique) -> Path:
    """Persist the critique, merge its verdicts into the LensMark file and save. Returns the critique path."""
    file = campaign.load(critique.image_id)
    if file is None:
        raise FileNotFoundError(f"no annotation file for {critique.image_id!r}; save the reviewed file before critiquing")
    run = find_run(file, critique.run_id)
    if run is not None:
        critique.model = critique.model or run.model
        critique.effort = critique.effort or run.effort
    unknown = merge_reviews(file, critique)
    if not critique.counts:
        critique.counts = run_counts(file, critique.run_id)
    path = critique_path(campaign, critique)
    campaign.critiques_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(critique.model_dump(mode="json", exclude_none=True), indent=2,
                                       ensure_ascii=False) + "\n")
    rel = _relative(campaign, path)
    if rel not in file.provenance.critiques:
        file.provenance.critiques.append(rel)
    campaign.save(critique.image_id, file, actor=critique.reviewer, source="critique")
    campaign.append_log(critique.image_id, actor=critique.reviewer, source="critique", op="critique",
                        item_id="$critique", run_id=critique.run_id, file=rel, counts=critique.counts,
                        unknown_items=unknown or None)
    return path


def load_critique(path: Path) -> Critique:
    with open(path, encoding="utf-8") as f:
        return Critique.model_validate(json.load(f))


def list_critiques(campaign: Campaign, image_id: Optional[str] = None) -> list[Critique]:
    """Every parseable critique document, ordered by (image_id, reviewed_at, run_id, reviewer)."""
    out: list[Critique] = []
    if not campaign.critiques_dir.is_dir():
        return out
    for p in sorted(campaign.critiques_dir.glob("*.json")):
        try:
            c = load_critique(p)
        except (ValueError, OSError):
            continue
        if image_id is None or c.image_id == image_id:
            out.append(c)
    out.sort(key=lambda c: (c.image_id, c.reviewed_at, c.run_id, c.reviewer))
    return out


def latest_critique(campaign: Campaign, image_id: str) -> Optional[Critique]:
    cs = list_critiques(campaign, image_id)
    return cs[-1] if cs else None


def critique_summary(critique: Critique) -> dict[str, Any]:
    """Compact row used by the API listing."""
    return {"image_id": critique.image_id, "run_id": critique.run_id, "reviewer": critique.reviewer,
            "reviewed_at": critique.reviewed_at, "model": critique.model, "effort": critique.effort,
            "n_items": len(critique.items), "counts": dict(critique.counts),
            "would_use_as_fewshot": critique.panel.would_use_as_fewshot}
