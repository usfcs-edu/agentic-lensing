"""Proposal pipeline: prompt -> engine -> ``Proposal`` -> strict items -> immutable run file -> merge + save.

A run never touches reviewed items: before appending its own it prunes only the still-``proposed``
(never reviewed) items of earlier runs, plus Claude-created ``invalid`` ones. The immutable
``proposals/<id>.<run_id>.json`` is the dataset row (request, prompt sha, raw output, repairs, usage,
cost); ``provenance.proposal_runs`` in the file carries its summary (``ProposalRun``).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from .. import config, imaging
from ..model import CreatedBy, ItemBase, LensMarkFile, Proposal, ProposalRun, now_iso
from ..store import Campaign, atomic_write_text
from ..validate import validate_proposal
from . import parse
from .engine import get_engine
from .engine_base import Engine, EngineRequest, EngineResult, EventCallback

PROMPT_PATH = config.PROMPT_DIR / "propose_v1.md"


def engine_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The schema as the CLI accepts it: without ``$schema`` (claude 2.1.251 rejects a draft-2020-12
    ``$schema`` with 'no schema with key or ref ...'); ``title``/``description`` are harmless."""
    return {k: v for k, v in schema.items() if k not in ("$schema", "$id")}


PROPOSAL_SCHEMA: dict[str, Any] = engine_schema(json.loads(
    (config.SCHEMA_DIR / "lensmark-proposal-1.0.schema.json").read_text(encoding="utf-8")))
RUN_SCHEMA_VERSION = "lensmark-proposal-run/1.0"
FEWSHOT_SCHEMA_VERSION = "lensmark-fewshot/1.0"
REPAIR_TEXT = ("Your previous reply was not a valid JSON object for the required schema. Re-emit ONLY the JSON "
               "object - keys `system` {verdict, description, theta_e, p_lens, grade, tags} and `items` [...] - "
               "with no prose, no code fence and nothing before or after it. Your previous reply:\n\n")


class ProposeRequest(BaseModel):
    model: Optional[str] = None          # alias or full id; default = campaign config
    effort: Optional[str] = None         # default = campaign config; ignored for models without effort
    budget: Optional[float] = None       # USD per call; default = LENSMARK_MAX_BUDGET_USD
    fewshot: Optional[str] = None        # few-shot bundle directory (exports/fewshot)
    engine: Optional[str] = None         # sdk | fixture; default = LENSMARK_ENGINE
    include_grid: bool = True


# ----------------------------------------------------------------------------- prompt
def prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def prompt_sha256() -> str:
    """sha256 of the frozen doctrine (``propose_v1.md``) - the prompt version recorded per run."""
    return hashlib.sha256(prompt_template().encode("utf-8")).hexdigest()


def _orientation(file: LensMarkFile) -> str:
    ns = "North is UP (towards smaller v)" if file.image.north_up else "North is DOWN (towards larger v)"
    ew = "East is LEFT (towards smaller u)" if file.image.east_left else "East is RIGHT (towards larger u)"
    return f"{ns}; {ew}."


def render_system(file: LensMarkFile, mask_cap: int = config.MASK_CAP) -> str:
    """The doctrine with this image's geometry substituted (placeholders replaced verbatim)."""
    W, H = file.image.width, file.image.height
    ps = file.image.pixel_scale_arcsec
    subs = {
        "{cutout_arcsec}": f"{file.image.cutout_arcsec:g}",
        "{pixel_scale_arcsec}": f"{ps:.4f}",
        "{W}": str(W), "{H}": str(H),
        "{px_per_arcsec}": f"{1 / ps:.1f}",
        "{frac_per_arcsec}": f"{1 / file.image.cutout_arcsec:.4f}",
        "{orientation}": _orientation(file),
        "{mask_cap}": str(mask_cap),
    }
    text = prompt_template()
    for k, v in subs.items():
        text = text.replace(k, v)
    return text


def _text(s: str) -> dict[str, Any]:
    return {"type": "text", "text": s}


def _r3(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_r3(x) for x in v]
    if isinstance(v, float):
        return round(v, 3)
    return v


def compact_item(it: dict[str, Any]) -> dict[str, Any]:
    """{type,label,color,geometry} - the shape shown to the model for existing / few-shot items."""
    t = it.get("type")
    out: dict[str, Any] = {"type": t, "label": it.get("label"), "color": it.get("color")}
    if t == "arrow":
        out.update(tail=it.get("tail"), head=it.get("head"))
    elif t == "mask_circle":
        out.update(center=it.get("center"), radius_arcsec=it.get("radius_arcsec"), kind=it.get("kind"))
    elif t == "einstein_ring":
        out.update(center=it.get("center"), theta_e_arcsec=it.get("theta_e_arcsec"))
    elif t == "text":
        out.update(pos=it.get("pos"), text=it.get("text"))
    return {k: _r3(v) for k, v in out.items() if v is not None}


def _reviewed_items(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [compact_item(it) for it in doc.get("items", []) if it.get("status", "accepted") in ("accepted", "edited")]


def _resolve_bundle(campaign: Campaign, fewshot: str) -> Path:
    p = Path(fewshot).expanduser()
    if not p.is_absolute() and not p.is_dir() and (campaign.root / p).is_dir():
        p = campaign.root / p
    if not (p / "manifest.json").is_file():
        raise FileNotFoundError(f"few-shot bundle has no manifest.json: {p}")
    return p


def fewshot_blocks(bundle: Path) -> tuple[list[dict[str, Any]], str]:
    """Content blocks for a few-shot bundle (stable prefix, so it caches) and the bundle's sha."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    sha_file = bundle / "prompt.sha256"
    if sha_file.is_file():
        sha = sha_file.read_text(encoding="utf-8").split()[0]
    else:
        sha = manifest.get("prompt_sha256") or hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    examples = manifest.get("examples", [])
    blocks: list[dict[str, Any]] = [_text(
        f"Worked examples ({len(examples)}) reviewed by an expert, in the exact style and coordinate "
        "convention expected. For each: the original cutout, the annotated rendering, then its items and description.")]
    for k, ex in enumerate(examples, 1):
        orig = imaging.upsample(imaging.load_rgb(bundle / ex["png"]))
        annot = imaging.upsample(imaging.load_rgb(bundle / ex["annot"]))
        doc = json.loads((bundle / ex["json"]).read_text(encoding="utf-8"))
        md = bundle / ex["md"] if ex.get("md") else None
        desc = md.read_text(encoding="utf-8").strip() if md and md.is_file() else doc.get("system", {}).get("description", "")
        sysb = doc.get("system", {})
        te = (sysb.get("theta_e") or {}).get("value_arcsec")
        blocks.append(_text(f"### Example {k} ({ex['id']}) - original cutout"))
        blocks.append(imaging.image_block(orig))
        blocks.append(_text(f"### Example {k} ({ex['id']}) - annotated"))
        blocks.append(imaging.image_block(annot))
        blocks.append(_text(
            f"### Example {k} ({ex['id']}) - items and description\n"
            f"items: {json.dumps(_reviewed_items(doc), ensure_ascii=False)}\n"
            f"verdict: {sysb.get('verdict')}; theta_e_arcsec: {te}\n"
            f"description: {desc}"))
    return blocks, sha


def _task_text(image_id: str, file: LensMarkFile, up_size: tuple[int, int], include_grid: bool) -> str:
    W, H = file.image.width, file.image.height
    existing = _reviewed_items(file.to_dict())
    if existing:
        note = ("Already-accepted annotations on this image (do NOT duplicate them; propose what is missing "
                "and you may still propose the Einstein ring and masks):\n" + json.dumps(existing, ensure_ascii=False))
    else:
        note = "There are no existing annotations on this image."
    imgs = ("The two images below are (1) the cutout, upsampled to {uw}x{uh} px by nearest-neighbour (no new "
            "information), and (2) the same with the labelled 0.1-step u/v grid." if include_grid else
            "The image below is the cutout, upsampled to {uw}x{uh} px by nearest-neighbour (no new information).")
    return (f"Annotate strong-lens candidate `{image_id}`.\n"
            f"Image facts: native {W}x{H} px, {file.image.cutout_arcsec:g}\" across -> "
            f"{file.image.pixel_scale_arcsec:.4f}\"/px (pixel-scale source: {file.image.scale_source or 'config'}). "
            + imgs.format(uw=up_size[0], uh=up_size[1]) +
            " Coordinates are FRACTIONS of the image, so the upsampling does not change them.\n"
            f"{note}\n"
            "Return ONE JSON object matching the schema.")


def build_prompt(campaign: Campaign, image_id: str, file: LensMarkFile, req: ProposeRequest,
                 ) -> tuple[str, list[dict[str, Any]], str, Optional[str]]:
    """(system prompt, user content blocks, prompt_sha256, fewshot_sha256 or None)."""
    system = render_system(file)
    content: list[dict[str, Any]] = []
    fsha: Optional[str] = None
    if req.fewshot:
        blocks, fsha = fewshot_blocks(_resolve_bundle(campaign, req.fewshot))
        content.extend(blocks)
    im = imaging.upsample(imaging.load_rgb(campaign.image_path(image_id)))
    content.append(_text(_task_text(image_id, file, im.size, req.include_grid)))
    content.append(imaging.image_block(im))
    if req.include_grid:
        content.append(imaging.image_block(imaging.grid_overlay(im)))
    return system, content, prompt_sha256(), fsha


# ----------------------------------------------------------------------------- run
def new_run_id() -> str:
    return f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def _parse(res: EngineResult) -> tuple[Optional[Proposal], Optional[str]]:
    obj = res.structured if isinstance(res.structured, dict) else parse.extract_json_block(res.text or "")
    if obj is None:
        return None, "no JSON object in the reply"
    if not isinstance(obj.get("items"), list):
        return None, "JSON object has no `items` list"
    try:
        return Proposal.model_validate(obj), None
    except ValidationError as e:
        return None, str(e)[:400]


def _wrap_events(cb: Optional[EventCallback]) -> Optional[EventCallback]:
    """Forward engine events; the engine's own terminal phases become ``partial`` so that only the
    pipeline's final ``done``/``error`` (which carries the ProposalRun) closes an SSE stream."""
    if cb is None:
        return None

    def inner(ev: dict[str, Any]) -> None:
        ph = ev.get("phase")
        if ph in ("done", "error"):
            ev = {**ev, "phase": "partial", "detail": f"engine {ph}: {ev.get('detail', '')}".strip()}
        cb(ev)
    return inner


def _stale_proposal(it: ItemBase) -> bool:
    return it.status == "proposed" or (it.status == "invalid" and it.created_by.kind == "claude")


def _sum_usage(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, (int, float)) and isinstance(out.get(k), (int, float)):
            out[k] = out[k] + v
        elif k not in out:
            out[k] = v
    return out


def _append_trace(target: Path, source: Optional[str]) -> None:
    if source and Path(source).is_file():
        with open(target, "a", encoding="utf-8") as out, open(source, encoding="utf-8") as src:
            shutil.copyfileobj(src, out)


async def run_propose(campaign: Campaign, image_id: str, req: ProposeRequest, *,
                      engine: Optional[Engine] = None, on_event: Optional[EventCallback] = None) -> ProposalRun:
    """One Claude proposal for ``image_id``: call, validate, persist the run, merge into the file."""
    model = config.resolve_model(req.model or campaign.config.get("default_model") or config.DEFAULT_MODEL)
    effort_req = req.effort or campaign.config.get("default_effort") or config.DEFAULT_EFFORT
    effort = effort_req if config.model_supports_effort(model) else None
    engine = engine or get_engine(req.engine)
    run_id = new_run_id()
    started_at = now_iso()
    t0 = time.monotonic()
    emit = _wrap_events(on_event)

    file = campaign.load_or_new(image_id)
    system, content, psha, fsha = build_prompt(campaign, image_id, file, req)
    budget = req.budget if req.budget is not None else config.max_budget_usd()
    ereq = EngineRequest(system=system, content=content, schema=PROPOSAL_SCHEMA, model=model, effort=effort,
                         max_budget_usd=budget, max_turns=config.MAX_TURNS, cwd=campaign.root,
                         fixture_key=image_id, purpose="propose")
    res = await engine.run(ereq, on_event=emit)
    cost = float(res.cost_usd or 0.0)
    usage = dict(res.usage or {})
    turns = int(res.num_turns or 0)
    raw_structured, raw_text, thinking = res.structured, res.text, res.thinking
    engine_raw = dict(res.raw)
    trace_sources = [res.raw.get("trace_path")]
    proposal: Optional[Proposal] = None
    parse_error: Optional[str] = None
    parse_ok = True
    repair_turns = 0

    if res.error is None:
        proposal, parse_error = _parse(res)
        if proposal is None:
            parse_ok = False
            repair_turns = 1
            prev = res.text or (json.dumps(res.structured) if res.structured is not None else "(empty reply)")
            rreq = EngineRequest(system=system, content=[_text(REPAIR_TEXT + prev)], schema=PROPOSAL_SCHEMA,
                                 model=model, effort=effort, max_budget_usd=budget, max_turns=1,
                                 cwd=campaign.root, fixture_key=image_id, purpose="propose")
            res2 = await engine.run(rreq, on_event=emit)
            cost += float(res2.cost_usd or 0.0)
            usage = _sum_usage(usage, res2.usage or {})
            turns += int(res2.num_turns or 0)
            trace_sources.append(res2.raw.get("trace_path"))
            engine_raw["repair"] = dict(res2.raw)
            if res2.error is None:
                p2, err2 = _parse(res2)
                if p2 is not None:
                    proposal, parse_error = p2, None
                    raw_structured, raw_text, thinking = res2.structured, res2.text, res2.thinking
                else:
                    parse_error = err2
            else:
                parse_error = res2.error
    error = res.error or (None if proposal is not None else f"parse_failed: {parse_error}")

    created_by = CreatedBy(kind="claude", model=model, effort=effort, run_id=run_id)
    if proposal is not None:
        file.items = [it for it in file.items if not _stale_proposal(it)]     # ids are minted against the pruned file
    val = validate_proposal(proposal or Proposal(), file, created_by, status="proposed", mask_cap=config.MASK_CAP)
    duration = round(time.monotonic() - t0, 3)
    proposal_name = f"{image_id}.{run_id}.json"
    run = ProposalRun(
        run_id=run_id, model=model, effort=effort, engine=engine.name, prompt_sha256=psha, fewshot_sha256=fsha,
        started_at=started_at, duration_s=duration, usage=usage or None, cost_usd=cost, num_turns=turns,
        n_items_proposed=len(proposal.items) if proposal else 0, n_invalid=val.n_invalid,
        n_repaired=val.n_repaired, parse_ok=parse_ok, proposal_file=proposal_name, error=error,
        proposed_system=proposal.system.model_dump(mode="json") if proposal else None)

    campaign.proposals_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": RUN_SCHEMA_VERSION, "run_id": run_id, "image_id": image_id, "started_at": started_at,
        "duration_s": duration, "engine": engine.name, "model": model, "effort": effort,
        "request": {**req.model_dump(), "model_resolved": model, "effort_resolved": effort, "budget_usd": budget,
                    "max_turns": config.MAX_TURNS, "mask_cap": config.MASK_CAP},
        "prompt_sha256": psha, "fewshot_sha256": fsha,
        "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "n_content_blocks": len(content), "n_image_blocks": sum(1 for b in content if b.get("type") == "image"),
        "parse_ok": parse_ok, "parse_error": parse_error, "repair_turns": repair_turns, "error": error,
        "raw": {"structured": raw_structured, "text": raw_text, "thinking": thinking, "engine": engine_raw},
        "repairs": val.repairs, "invalid": val.invalid,
        "usage": usage, "cost_usd": cost, "num_turns": turns,
        "proposed_system": run.proposed_system,
        "items": [it.model_dump(mode="json", exclude_none=True) for it in val.items],
    }
    atomic_write_text(campaign.proposals_dir / proposal_name, json.dumps(doc, indent=2, ensure_ascii=False, default=str) + "\n")
    if any(trace_sources):
        tpath = campaign.proposals_dir / f"{image_id}.{run_id}.trace.jsonl"
        for src in trace_sources:
            _append_trace(tpath, src)
        with open(tpath, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "event": "validated", "run_id": run_id, "n_items": len(val.items),
                                "n_invalid": val.n_invalid, "n_repaired": val.n_repaired, "error": error}) + "\n")

    if proposal is not None:
        file.items.extend(val.items)
    file.provenance.proposal_runs.append(run)
    campaign.save(image_id, file, actor="claude", source="claude")

    if error:
        if on_event:
            on_event({"phase": "error", "detail": error, "cost_usd": cost, "run": run.model_dump(mode="json")})
    else:
        if on_event:
            on_event({"phase": "validated", "detail": f"{len(val.items)} items ({val.n_invalid} invalid, {val.n_repaired} repaired)",
                      "n_items": len(val.items), "n_invalid": val.n_invalid, "n_repaired": val.n_repaired, "cost_usd": cost})
            on_event({"phase": "done", "detail": f"run {run_id} cost ${cost:.4f}", "cost_usd": cost,
                      "n_items": len(val.items), "run": run.model_dump(mode="json")})
    return run


# ----------------------------------------------------------------------------- listing / CLI
def _run_from_file(path: Path, run_id: str) -> ProposalRun:
    d = json.loads(path.read_text(encoding="utf-8"))
    return ProposalRun(
        run_id=run_id, model=d.get("model") or "?", effort=d.get("effort"), engine=d.get("engine", "sdk"),
        prompt_sha256=d.get("prompt_sha256"), fewshot_sha256=d.get("fewshot_sha256"), started_at=d.get("started_at"),
        duration_s=d.get("duration_s"), usage=d.get("usage") or None, cost_usd=d.get("cost_usd"),
        num_turns=d.get("num_turns"), n_items_proposed=len(d.get("items", [])), n_invalid=len(d.get("invalid", [])),
        n_repaired=len({r.get("item_id") for r in d.get("repairs", [])}), parse_ok=bool(d.get("parse_ok", True)),
        proposal_file=path.name, error=d.get("error"), proposed_system=d.get("proposed_system"))


def list_runs(campaign: Campaign, image_id: str) -> list[ProposalRun]:
    """Runs in ``provenance.proposal_runs`` order (append order), then any ``proposals/<id>.<run>.json``
    not listed there (sorted by run id) - run ids have 1 s resolution, so file order is the tie-break."""
    file = campaign.load(image_id)
    runs = {r.run_id: r for r in (file.provenance.proposal_runs if file else [])}
    prefix = f"{image_id}."
    orphans: list[ProposalRun] = []
    if campaign.proposals_dir.is_dir():
        for p in sorted(campaign.proposals_dir.glob(f"{image_id}.run-*.json")):
            rid = p.name[len(prefix):-len(".json")]
            if rid not in runs and not rid.endswith(".trace"):
                try:
                    orphans.append(_run_from_file(p, rid))
                except Exception:
                    continue
    return list(runs.values()) + orphans


def load_run(campaign: Campaign, image_id: str, run_id: str) -> dict[str, Any]:
    p = campaign.proposals_dir / f"{image_id}.{run_id}.json"
    if not p.is_file():
        raise FileNotFoundError(f"no proposal file {p.name} in {campaign.proposals_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def cli_propose(dir: str, *, image_id: Optional[str] = None, model: Optional[str] = None,
                effort: Optional[str] = None, budget: Optional[float] = None, fewshot: Optional[str] = None,
                engine: Optional[str] = None, concurrency: int = 2) -> int:
    """``lensmark propose DIR``: one run per image without a run (or just ``--id``); one line each."""
    campaign = Campaign(dir)
    ids = [image_id] if image_id else [i for i in campaign.list_ids() if not list_runs(campaign, i)]
    if not ids:
        print("nothing to do: every image already has a proposal run (use --id to re-run one)")
        return 0
    req = ProposeRequest(model=model, effort=effort, budget=budget, fewshot=fewshot, engine=engine)
    eng = get_engine(engine)
    sem = asyncio.Semaphore(max(1, concurrency))
    failures = 0

    async def one(i: str) -> None:
        nonlocal failures
        async with sem:
            try:
                run = await run_propose(campaign, i, req, engine=eng)
            except Exception as e:
                failures += 1
                print(f"{i:<20} ERROR {type(e).__name__}: {e}")
                return
            status = f"ERROR {run.error}" if run.error else "ok"
            if run.error:
                failures += 1
            print(f"{i:<20} {run.run_id} {run.model}/{run.effort or '-'} items={run.n_items_proposed} "
                  f"invalid={run.n_invalid} repaired={run.n_repaired} cost=${run.cost_usd or 0:.4f} "
                  f"{run.duration_s or 0:.1f}s {status}", flush=True)

    async def main() -> None:
        await asyncio.gather(*(one(i) for i in ids))

    asyncio.run(main())
    return 1 if failures else 0
