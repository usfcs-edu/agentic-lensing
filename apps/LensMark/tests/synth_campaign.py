"""Synthetic campaign data shared by test_critique_eval / test_export / test_patch.

Builds LensMark files on a throw-away ``nine`` copy: Claude-proposed items carrying
``created_by.run_id`` + a ``ProposalRun`` in provenance (what the propose pipeline writes), human items,
and critiques. Nothing here calls Claude.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from lensmark.model import (Arrow, CreatedBy, EinsteinRing, ItemBase, LensMarkFile, MaskCircle, ProposalRun,
                            TextNote, parse_item)
from lensmark.store import Campaign


def claude_by(run_id: str, model: str = "claude-opus-5", effort: str = "xhigh") -> CreatedBy:
    return CreatedBy(kind="claude", model=model, effort=effort, run_id=run_id)


def human_by(reviewer: str = "xhuang") -> CreatedBy:
    return CreatedBy(kind="human", reviewer=reviewer)


def arrow(id: str, head, tail, label: str, color: str = "cyan", **kw) -> Arrow:
    return Arrow(id=id, head=list(head), tail=list(tail), label=label, color=color, **kw)


def mask(id: str, center, r: float, kind: str = "galaxy", **kw) -> MaskCircle:
    return MaskCircle(id=id, center=list(center), radius_arcsec=r, kind=kind, **kw)


def ring(id: str, center, theta: float, **kw) -> EinsteinRing:
    return EinsteinRing(id=id, center=list(center), theta_e_arcsec=theta, **kw)


def note(id: str, pos, text: str, **kw) -> TextNote:
    return TextNote(id=id, pos=list(pos), text=text, **kw)


def seed_run(campaign: Campaign, image_id: str, run_id: str, items: Iterable[ItemBase], *,
             model: str = "claude-opus-5", effort: str = "xhigh", cost_usd: float = 0.1, n_invalid: int = 0,
             parse_ok: bool = True, duration_s: float = 10.0, proposed_theta_e: Optional[float] = None,
             human_items: Iterable[ItemBase] = ()) -> LensMarkFile:
    """Append a Claude run (items status=proposed, stamped with the run) + optional human items, then save."""
    f = campaign.load_or_new(image_id)
    items = list(items)
    for it in items:
        it.created_by = claude_by(run_id, model, effort)
        it.status = "proposed"
        f.items.append(it)
    for it in human_items:
        it.created_by = human_by()
        it.status = "accepted"
        f.items.append(it)
    f.provenance.proposal_runs.append(ProposalRun(
        run_id=run_id, model=model, effort=effort, engine="fixture", cost_usd=cost_usd, duration_s=duration_s,
        n_items_proposed=len(items), n_invalid=n_invalid, parse_ok=parse_ok,
        proposal_file=f"proposals/{image_id}.{run_id}.json",
        proposed_system=({"verdict": "likely_lens", "theta_e": {"value_arcsec": proposed_theta_e, "method": "geometric"}}
                         if proposed_theta_e is not None else None)))
    return campaign.save(image_id, f, actor="test", source="cli")


def set_fields(campaign: Campaign, image_id: str, changes: dict[str, dict[str, Any]]) -> LensMarkFile:
    """Emulate the UI's PUT: ``{item_id: {field: value}}`` applied to the saved file (re-validated)."""
    f = campaign.load(image_id)
    assert f is not None
    for iid, fields in changes.items():
        it = f.item(iid)
        assert it is not None, iid
        d = it.model_dump(mode="json", exclude_none=True)
        d.update(fields)
        f.items[f.items.index(it)] = parse_item(d)
    f = LensMarkFile.model_validate(f.to_dict())
    return campaign.save(image_id, f, actor="ui", source="ui")


def accepted_file(campaign: Campaign, image_id: str, items: Iterable[ItemBase], *, verdict: Optional[str] = None,
                  rank: Optional[int] = None, description: str = "", theta_e: Optional[float] = None,
                  native_scale: Optional[float] = None) -> LensMarkFile:
    """A human-authored file: every item keeps the status it was built with (default accepted)."""
    f = campaign.load_or_new(image_id)
    for it in items:
        if it.created_by.kind == "human" and it.created_by.reviewer is None:
            it.created_by = human_by()
        f.items.append(it)
    if verdict is not None:
        f.system.verdict = verdict
    if rank is not None:
        f.system.rank = rank
    if theta_e is not None:
        f.system.theta_e.value_arcsec = theta_e
        f.system.theta_e.method = "human"
    f.system.description = description
    if native_scale is not None:
        f.image.native_pixel_scale_arcsec = native_scale
    return campaign.save(image_id, f, actor="test", source="cli")
