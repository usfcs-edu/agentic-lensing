"""golden/panel.py — the evidence-first JWST panel: advocate → critics → arbitrator, one item.

`grade_panel(cand, ...)` is the per-item fan-out the truth-eval runner (golden/run_truth_eval.py)
and the attribution / incumbent arms call. It turns one candidate into ≤ 5 model calls, every
one of them through `grader_jwst.grade_candidate` (so every call writes a `golden_content_audit`
event, validates into its own pydantic record — never ImageGrade — and costs what grader_direct
meters), then hands the records to `schemas_panel.assemble` / `aggregate_v2` for the score:

  ADVOCATE   full footer-cropped composite; schema AdvocateRecord; system = advocate.md + note
    │  p_evidence < tau0  →  no critics, no arbitrator (S = p_evidence, letter by rule)
  CRITICS    artifact | geometry | morphology in PARALLEL (asyncio.gather); schema CriticRecord;
             system = critic_common.md + "\\n" + critic_<role>.md + note; each with its OWN views
             (golden/views.py: the artifact critic sees the whole composite, geometry (b),(d),(e)
             ×2 + the 20" context pair when a stamp dir is given, morphology (c),(d),(e) ×2 —
             layout-conditional, never a subtraction panel); user text = the VIEW paragraph +
             the advocate's numbered items (and its scale class) — no p_evidence, no inspector
             claim, nothing about the object
    │  no critic named an alternative  →  no arbitrator
  ARBITRATOR full composite + every record as JSON; schema ArbitratorRecord
  aggregate  schemas_panel.assemble → PanelResult (S, S_arb, letter, letter_source, a, r, ...)

Modes: `full` (the above), `advocate_only` (A2: one call, S = p_evidence), `attr` (the
attribution arm: the same prompts but the SAME full composite to every role, so A1 − attr
isolates the per-role views), `incumbent` (A0: the three incumbent briefs byte-equal to
verify_workflow.js inside prompts/personas/incumbent/wrapper.md, schema IncumbentVerdict,
letter = aggregate_v2.passcount_incumbent; `claim` adds the inspector's claim fields to the
USER message — the incumbent read them from its job file, and keeping them out of the system
prompt keeps each role's system sha16 item-agnostic and stable for the registry tuple. The
wrapper has TWO item-agnostic variants — "given with the image below" / "not available" —
and the one sent must match whether a claim rides in the user message: `persona_set` is the
claim variant, `persona_set_noclaim` the other; an item without a claim under a prompt that
promises one, or the reverse, is refused).

Render: `render="v2r"` serves `image_path_v2r` (golden/render_v2.py: slot (f) = signed-chi
SW | LW montage) AND the matching VIEW text — panel_gloss.json's `renders["v2r"]` view sets
with golden/render_v2_desc.md appended (`render_desc`; read from disk when not given). The
image and its description ship as one unit or not at all (views.view_text refuses).

Invariants the tests pin: role order advocate → critics → arbitrator; the note is the LAST
thing in every persona system prompt (note="" for the incumbent, whose wrapper is
self-contained — the v1 note's "bluer"/"default C or D" must never be auto-appended); panel
(f) (and (c) in a gray layout) never reaches geometry or morphology (image count + gloss);
critics are skipped below tau0; the arbitrator is skipped without a named alternative; any
parse failure ⇒ S NaN with the failed roles listed (row kept). Per-role traces go to
`trace_dir/<name>_<role>.jsonl` (+ `<name>_panel.jsonl`, one summary event).

Item-agnostic by construction: the model sees the VIEW text, the records of earlier roles and
pixels — never the candidate id, coordinates, pipeline grade or inspector confidence. Every
text block the panel writes is recorded IN FULL in the audit event (audit_full_text=True) so
golden/audit_traces.py can lexicon-check the located-items JSON too.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lensjudge.common import hooks  # noqa: E402
from lensjudge.golden import _util, aggregate_v2, grader_jwst, schemas_panel, views  # noqa: E402

PROMPTS = _util.LENSJUDGE / "prompts"
PERSONA_DIR_V1 = PROMPTS / "personas" / "jwst_v1"
PERSONA_DIR_INCUMBENT = PROMPTS / "personas" / "incumbent"
NOTE_V2_PATH = PROMPTS / "jwst_note_v2.md"
RENDER_V2_DESC_PATH = _util.HERE / "render_v2_desc.md"   # = render_v2.DESC_PATH (no import: numpy-free load)
MODES = ("full", "advocate_only", "attr", "incumbent")
CRITIC_ROLES = aggregate_v2.CRITIC_ROLES            # ("artifact", "geometry", "morphology")
ROLES = aggregate_v2.ROLES                          # advocate, critics, arbitrator
INCUMBENT_ROLES = ("artifact", "morphology", "geometry")   # verify_workflow.js order
CRITIC_JOIN = "\n"                                  # critic_common + "\n" + critic_<role> (tests/test_golden_prompts.py)
RESPOND = "Respond with ONLY the JSON object."
# wrapper.md placeholders (token replacement, never str.format: the text carries JSON braces)
WRAPPER_PLACEHOLDERS = ("brief", "persona", "item_id", "claim_center", "claim_quadrant", "claimed_evidence")
CLAIM_ABSENT = "not available"                      # wrapper text when no inspector claim is passed
CLAIM_IN_USER = "given with the image below"        # wrapper text when the claim rides in the user message
CLAIM_EVIDENCE_CHARS = 400                          # inspections.csv evidence[:400] (contract)
ITEM_ID = "item"                                    # the opaque id every record echoes (no candidate id)


# ------------------------------------------------------------------ persona sets
def read_note(path: Path = NOTE_V2_PATH) -> str:
    return Path(path).read_text()


def read_render_desc(render: str, path: Path = RENDER_V2_DESC_PATH) -> Optional[str]:
    """The description shipped with a non-v1 render (None for v1). Raises when the file is
    absent: the v2r image is never sent without golden/render_v2_desc.md."""
    if render == "v1":
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"render {render!r} needs its description {p} (golden/render_v2.py writes it)")
    return p.read_text()


def load_persona_set(persona_dir: Path = PERSONA_DIR_V1, note_text: Optional[str] = None) -> dict:
    """role -> system prompt text (WITHOUT the note; grade_candidate appends `note=`) for the
    five panel roles. Critics = critic_common + "\\n" + critic_<role>. Raises when a file is
    missing — a half-written persona set must never run."""
    d = Path(persona_dir)
    texts = {}
    for f in ("advocate", "critic_common", "critic_artifact", "critic_geometry",
              "critic_morphology", "arbitrator"):
        p = d / f"{f}.md"
        if not p.exists():
            raise FileNotFoundError(f"persona set {d} lacks {f}.md")
        texts[f] = p.read_text()
    out = {"advocate": texts["advocate"], "arbitrator": texts["arbitrator"]}
    for role in CRITIC_ROLES:
        out[role] = texts["critic_common"] + CRITIC_JOIN + texts[f"critic_{role}"]
    if note_text:   # sanity: the note must not already be inside a brief (it would be doubled)
        for role, t in out.items():
            if note_text.strip() and note_text.strip() in t:
                raise ValueError(f"{role} brief already contains the note text")
    return out


def fill_wrapper(wrapper: str, persona: str, brief: str, item_id: str = ITEM_ID,
                 claim_center: str = CLAIM_ABSENT, claim_quadrant: str = CLAIM_ABSENT,
                 claimed_evidence: str = CLAIM_ABSENT) -> str:
    """The incumbent wrapper by token replacement (the contract of tests/test_golden_prompts)."""
    out = wrapper
    for k, v in (("brief", brief), ("persona", persona), ("item_id", item_id),
                 ("claim_center", claim_center), ("claim_quadrant", claim_quadrant),
                 ("claimed_evidence", claimed_evidence)):
        out = out.replace("{" + k + "}", v)
    return out


def load_incumbent_set(persona_dir: Path = PERSONA_DIR_INCUMBENT, claim_in_user: bool = False) -> dict:
    """role -> filled wrapper (system prompt) for the three incumbent personas. With
    `claim_in_user` the claim placeholders say the claim rides in the user message; without,
    they say it is not available — either way the system prompt is item-agnostic."""
    d = Path(persona_dir)
    w = d / "wrapper.md"
    if not w.exists():
        raise FileNotFoundError(f"incumbent set {d} lacks wrapper.md")
    wrapper = w.read_text()
    fill = CLAIM_IN_USER if claim_in_user else CLAIM_ABSENT
    out = {}
    for role in INCUMBENT_ROLES:
        p = d / f"{role}.md"
        if not p.exists():
            raise FileNotFoundError(f"incumbent set {d} lacks {role}.md")
        out[role] = fill_wrapper(wrapper, role, p.read_text(), claim_center=fill,
                                 claim_quadrant=fill, claimed_evidence=fill)
        left = sorted(set(re.findall(r"\{[a-z_]+\}", out[role])))   # {id, persona, ...} / JSON braces never match
        if left:
            raise ValueError(f"wrapper placeholders left unfilled for {role}: {left}")
    return out


def persona_set_sha16(persona_dir: Path) -> str:
    """sha16 over every *.md and *.json in the persona dir (name -> file sha16): the
    `persona_set_sha16` ingredient of the truth-eval tuple."""
    d = Path(persona_dir)
    files = sorted(list(d.glob("*.md")) + list(d.glob("*.json")))
    return _util.sha_json({p.name: _util.sha_file(p) for p in files})


# ------------------------------------------------------------------ claims (incumbent)
def claim_fields(claim) -> Optional[dict]:
    """Normalise an inspector claim to {claim_center, claim_quadrant, claimed_evidence}. Accepts
    the inspections.csv names (lens_at_center, quadrant_lens, evidence) or the job-file names,
    a pre-formatted string (→ claimed_evidence), or None. Evidence is cut at 400 chars."""
    if claim is None:
        return None
    if isinstance(claim, str):
        return {"claim_center": CLAIM_ABSENT, "claim_quadrant": CLAIM_ABSENT,
                "claimed_evidence": claim[:CLAIM_EVIDENCE_CHARS]}
    def pick(*keys):
        for k in keys:
            v = claim.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)) and str(v).strip():
                return str(v)
        return CLAIM_ABSENT
    return {"claim_center": pick("claim_center", "lens_at_center"),
            "claim_quadrant": pick("claim_quadrant", "quadrant_lens"),
            "claimed_evidence": pick("claimed_evidence", "evidence")[:CLAIM_EVIDENCE_CHARS]}


# ------------------------------------------------------------------ text blocks
def items_text(advocate, gloss: dict) -> str:
    """The critics' only candidate-specific text: the advocate's numbered items as JSON
    (plus its scale class). Never p_evidence, never the criteria scores, never the notes."""
    h = gloss["headers"]
    items = [it.model_dump() if hasattr(it, "model_dump") else dict(it) for it in (advocate.items or [])]
    if items:
        body = h["items"] + "\n" + json.dumps(items, separators=(",", ":"))
    else:
        body = h["items_none"]
    sc = getattr(advocate, "scale_class", None)
    if sc:
        body += "\n" + h["scale_class"] + " " + str(sc)
    return body


def records_text(advocate, critics: dict, gloss: dict) -> str:
    """The arbitrator's texts: the advocate record, then each parsed critic record, as JSON."""
    parts = [gloss["headers"]["arbitrator_texts"], advocate.model_dump_json()]
    for role in CRITIC_ROLES:
        rec = critics.get(role)
        if rec is not None:
            parts.append(rec.model_dump_json())
    return "\n".join(parts)


def lead_text(view_text: str, body: str = "") -> str:
    """VIEW paragraph [+ blank line + body] + the JSON instruction."""
    return view_text + ("\n\n" + body if body else "") + "\n\n" + RESPOND


# ------------------------------------------------------------------ results
@dataclass
class IncumbentResult:
    """One candidate through the incumbent baseline (A0): three verdicts, pass-count letter.
    Parse failure in any persona ⇒ p_lens NaN, letter None (the pre-registered policy), with
    the failed roles listed; the counts are over the parsed verdicts."""
    verdicts: dict                    # role -> IncumbentVerdict | None
    n_pass: int
    n_fail: int
    n_uncertain: int
    letter: Optional[str]
    parse_failures: list
    cost_usd: float
    calls: int
    system_sha16s: dict
    raw: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def parse_ok(self) -> bool:
        return not self.parse_failures

    @property
    def p_lens(self) -> float:
        return float("nan") if self.parse_failures else self.n_pass / 3.0


INCUMBENT_EXTRA_COLS = ("n_pass", "n_fail", "n_uncertain",
                        *(f"verdict_{r}" for r in INCUMBENT_ROLES))


def incumbent_row(result: IncumbentResult, cand: dict) -> dict:
    """The schemas_panel.ROW_COLS row shape (so every arm shares one reader) + the
    INCUMBENT_EXTRA_COLS: p_lens = n_pass/3, grade_pred = the pass-count letter,
    contaminant = the failing personas' alternatives, rationale = their notes."""
    nan = float("nan")
    vs = result.verdicts
    fails = [v for v in vs.values() if v is not None and v.verdict == "fail"]
    row: dict[str, Any] = {k: None for k in schemas_panel.ROW_COLS}
    row.update({
        "name": cand["name"], "grade_truth": cand.get("grade"), "catalog": cand.get("catalog"),
        "region": cand.get("region"), "p_meta": cand.get("p_meta"), "parse_ok": result.parse_ok,
        "turns": result.calls,
        "error": ("parse_fail:" + "+".join(result.parse_failures)) if result.parse_failures else None,
        "grade_pred": result.letter, "p_lens": result.p_lens, "confidence": None,
        "contaminant": "; ".join(v.alternative for v in fails if v.alternative) or None,
        "escalate": any(v.verdict == "uncertain" for v in vs.values() if v is not None),
        "rationale": " | ".join(f"{r}: {v.notes}" for r, v in vs.items() if v is not None and v.notes),
        "S": nan, "S_arb": nan, "letter_source": "passcount_incumbent",
        "parse_fail_roles": "+".join(result.parse_failures), "calls": result.calls,
        "cost_usd": result.cost_usd, "n_pass": result.n_pass, "n_fail": result.n_fail,
        "n_uncertain": result.n_uncertain,
    })
    for r in INCUMBENT_ROLES:
        v = vs.get(r)
        row[f"verdict_{r}"] = v.verdict if v is not None else None
        row[f"alt_{r}"] = (v.alternative or None) if v is not None else None
        row[f"system_sha16_{r}"] = result.system_sha16s.get(r)
    return row


# ------------------------------------------------------------------ the fan-out
def _cand_for(cand: dict, render: str) -> dict:
    """The cand dict a role call sees: name + the composite path for this render (+ its sha
    when the manifest carries one). Nothing else leaves this function — the role content is
    built here, not by grader_direct's DESI text."""
    if render == "v1":
        path, sha = cand.get("image_path"), cand.get("render_sha")
    elif render == "v2r":
        path, sha = cand.get("image_path_v2r"), cand.get("render_sha_v2r")
    else:
        raise ValueError(f"render must be v1 or v2r, got {render!r}")
    if not isinstance(path, str) or not path.strip():
        raise FileNotFoundError(f"{cand.get('name')}: no image_path for render {render}")
    out = {"name": cand.get("name"), "image_path": path}
    if cand.get("unit_id"):
        out["unit_id"] = cand["unit_id"]
    if isinstance(sha, str) and sha.strip():
        out["render_sha"] = sha
    return out


def _open_composite(path: str):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if img.width != 752 or img.height not in (540, 562):
        raise ValueError(f"{path}: composite is {img.size}, expected 752x540 or 752x562")
    return img


def _trace(trace_dir, name: str, role: str) -> Optional[str]:
    if trace_dir is None:
        return None
    return str(Path(trace_dir) / f"{_util.safe_name(name)}_{role}.jsonl")


async def _call(role: str, cand_role: dict, *, model, system_prompt, note, schema, content,
                extra_views, trace_path) -> dict:
    """One role call through grader_jwst; returns a small dict the fan-out aggregates."""
    res = await grader_jwst.grade_candidate(
        cand_role, model=model, system_prompt=system_prompt, note=note, schema=schema,
        content=content, extra_views=extra_views, trace_path=trace_path, audit_full_text=True)
    return {"role": role, "record": res.grade if res.parse_ok else None, "parse_ok": bool(res.parse_ok),
            "raw": res.raw, "cost": float(res.cost_usd or 0.0), "error": res.error,
            "system_sha16": res.meta.get("system_sha16"), "n_images": res.meta.get("n_images"),
            "n_extra_views": res.meta.get("n_extra_views"), "render_sha": res.meta.get("render_sha")}


def _role_content(cand_role: dict, role_views: list, body: str, view_text: str):
    """(content, extra_views): the first view is the candidate image — the kit JPEG bytes
    themselves for a composite view (jwst_content checks render_sha), a PNG crop otherwise;
    later views ride as extra_views ([label, image] pairs) so the audit counts them."""
    label0, img0 = role_views[0]
    lead = lead_text(view_text, body)
    if label0 == view_text:            # composite view set: the label IS the VIEW paragraph
        content = grader_jwst.jwst_content(cand_role, gloss=lead)
        if content is None:
            raise FileNotFoundError(f"{cand_role.get('name')}: composite {cand_role.get('image_path')} missing")
    else:
        content = [{"type": "text", "text": lead}, {"type": "text", "text": f"[view 1] {label0}"},
                   views.image_block(img0)]
    extra = [(f"[view {k}] {label}", img) for k, (label, img) in enumerate(role_views[1:], 2)]
    return content, extra


async def grade_panel(cand: dict, *, model: Optional[str], persona_dir=None, note_text: Optional[str] = None,
                      thresholds: Optional[dict] = None, mode: str = "full", claim=None,
                      trace_dir=None, stamp_dir=None, render: str = "v1", gloss: Optional[dict] = None,
                      persona_set: Optional[dict] = None, persona_set_noclaim: Optional[dict] = None,
                      render_desc: Optional[str] = None):
    """One candidate through the panel (modes full / advocate_only / attr) or the incumbent
    (mode incumbent). Returns schemas_panel.PanelResult, or IncumbentResult for the incumbent.

    cand: a truth_manifest / frame row dict with name, image_path (image_path_v2r for
    render="v2r"), layout, render_sha. `thresholds`: an aggregate_v2.resolve_thresholds dict
    (tau0 read from it; default provisional). `persona_set`: a pre-loaded
    load_persona_set / load_incumbent_set dict (else loaded from persona_dir);
    `persona_set_noclaim`: the incumbent set with the "not available" claim wording, used
    for an item that has no claim when `persona_set` is the claim variant. `stamp_dir`:
    golden/stamps/<id>/ to render the 20" context pair for the geometry critic (ctx20 arm).
    `render_desc`: the text shipped with a non-v1 render (default: golden/render_v2_desc.md).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if render not in views.RENDERS:
        raise ValueError(f"render must be one of {views.RENDERS}, got {render!r}")
    thresholds = dict(thresholds or aggregate_v2.resolve_thresholds({}, "provisional"))
    tau0 = float(thresholds.get("tau0", aggregate_v2.PROVISIONAL["tau0"]))
    g = gloss or views.load_gloss()
    name = str(cand.get("name"))
    layout = cand.get("layout")
    if not isinstance(layout, str) or not layout.strip():
        raise ValueError(f"{name}: cand['layout'] is required (color | gray_sw_only | gray_lw_only)")
    cand_role = _cand_for(cand, render)
    summary = {"name": name, "mode": mode, "render": render, "layout": layout, "tau0": tau0,
               "claim_mode": "inspector" if claim is not None else "none"}

    # ---------------------------------------------------------------- incumbent (A0)
    if mode == "incumbent":
        cf = claim_fields(claim)
        if persona_set is None:
            sysp = load_incumbent_set(persona_dir or PERSONA_DIR_INCUMBENT, claim_in_user=cf is not None)
        elif cf is None and persona_set_noclaim is not None:
            sysp = persona_set_noclaim
        else:
            sysp = persona_set
        # the wrapper must say what the user message does: a prompt that promises a claim
        # "given with the image below" with none attached (or the reverse) is refused
        for r in INCUMBENT_ROLES:
            promises = CLAIM_IN_USER in sysp[r]
            if promises != (cf is not None):
                raise ValueError(f"{name}/{r}: incumbent wrapper {'promises' if promises else 'denies'} a claim "
                                 f"in the user message but {'none' if cf is None else 'one'} is attached "
                                 f"(pass persona_set_noclaim for claim-less items)")
        body = ""
        if cf is not None:
            body = "\n".join(f'"{k}": {cf[k]}' for k in ("claim_center", "claim_quadrant", "claimed_evidence"))
        # the wrapper carries its own panel-layout text; the user text is a bare tag (+ claim)
        lead = "[composite]" + ("\n" + body if body else "")
        content = grader_jwst.jwst_content(cand_role, gloss=lead)
        if content is None:
            raise FileNotFoundError(f"{name}: composite {cand_role.get('image_path')} missing")
        calls = await asyncio.gather(*(
            _call(r, cand_role, model=model, system_prompt=sysp[r], note="",
                  schema=schemas_panel.IncumbentVerdict, content=content, extra_views=None,
                  trace_path=_trace(trace_dir, name, f"incumbent_{r}"))
            for r in INCUMBENT_ROLES))
        verdicts = {c["role"]: c["record"] for c in calls}
        fails = [c["role"] for c in calls if not c["parse_ok"]]
        n_pass, n_fail, n_unc, letter = aggregate_v2.passcount_incumbent(
            [v for v in verdicts.values() if v is not None])
        if fails:
            letter = None
        res = IncumbentResult(verdicts=verdicts, n_pass=n_pass, n_fail=n_fail, n_uncertain=n_unc,
                              letter=letter, parse_failures=fails,
                              cost_usd=sum(c["cost"] for c in calls), calls=len(calls),
                              system_sha16s={c["role"]: c["system_sha16"] for c in calls},
                              raw={c["role"]: c["raw"] for c in calls},
                              meta={**summary, "cost_by_role": {c["role"]: c["cost"] for c in calls},
                                    "errors": {c["role"]: c["error"] for c in calls if c["error"]},
                                    "render_sha": calls[0]["render_sha"]})
        if trace_dir:
            hooks.Trace(_trace(trace_dir, name, "panel")).write(
                "golden_panel", **summary, roles=list(INCUMBENT_ROLES), n_pass=n_pass, n_fail=n_fail,
                n_uncertain=n_unc, letter=letter, parse_failures=fails, cost_usd=res.cost_usd)
        return res

    # ---------------------------------------------------------------- the panel
    note = read_note() if note_text is None else note_text
    sysp = persona_set or load_persona_set(persona_dir or PERSONA_DIR_V1, note)
    img = _open_composite(cand_role["image_path"])
    ctx20 = views.ctx20_image(stamp_dir) if (stamp_dir is not None and mode != "attr") else None
    desc = read_render_desc(render) if render_desc is None else render_desc

    def role_views_for(role):
        # the attribution arm: the same full composite (and composite VIEW text) to every role;
        # the render picks the composite view set (v2r: the chi (f) text + its description)
        vr = "advocate" if mode == "attr" else role
        rv = views.role_views(img, layout, vr, ctx20=ctx20, gloss=g, render=render, render_desc=desc)
        vs = views.view_set(vr, layout, g, render)
        vt = views.view_text(vr, layout, g, with_ctx20=(ctx20 is not None and len(rv) > 1
                                                        and rv[-1][0] == vs.get("ctx20_text")),
                             render=render, render_desc=desc)
        return rv, vt

    calls: list[dict] = []
    cost_by_role: dict[str, float] = {}
    sha_by_role: dict[str, str] = {}
    raw_by_role: dict[str, str] = {}
    views_by_role: dict[str, list] = {}

    def note_call(c):
        calls.append(c)
        cost_by_role[c["role"]] = c["cost"]
        sha_by_role[c["role"]] = c["system_sha16"]
        raw_by_role[c["role"]] = c["raw"]

    # advocate
    rv, vt = role_views_for("advocate")
    views_by_role["advocate"] = [lbl for lbl, _ in rv]
    content, extra = _role_content(cand_role, rv, "", vt)
    adv_call = await _call("advocate", cand_role, model=model, system_prompt=sysp["advocate"], note=note,
                           schema=schemas_panel.AdvocateRecord, content=content, extra_views=extra,
                           trace_path=_trace(trace_dir, name, "advocate"))
    note_call(adv_call)
    advocate = adv_call["record"]
    critics: dict = {}
    arbitrator = None
    arbitrator_called = False
    fails = [] if adv_call["parse_ok"] else ["advocate"]
    skipped = []

    # critics: only with a parsed advocate above tau0, never in advocate_only
    if advocate is not None and mode != "advocate_only" and advocate.p_evidence >= tau0:
        body = items_text(advocate, g)
        jobs = []
        for role in CRITIC_ROLES:
            rv, vt = role_views_for(role)
            views_by_role[role] = [lbl for lbl, _ in rv]
            content, extra = _role_content(cand_role, rv, body, vt)
            jobs.append(_call(role, cand_role, model=model, system_prompt=sysp[role], note=note,
                              schema=schemas_panel.CriticRecord, content=content, extra_views=extra,
                              trace_path=_trace(trace_dir, name, role)))
        for c in await asyncio.gather(*jobs):
            note_call(c)
            critics[c["role"]] = c["record"]
            if not c["parse_ok"]:
                fails.append(c["role"])
        # arbitrator: only when a parsed critic named an alternative and no critic failed
        # (a failed critic already makes the row a parse failure — no call can rescue it)
        named = any(aggregate_v2.critic_named(critics.get(r)) for r in CRITIC_ROLES)
        if named and not fails:
            rv, vt = role_views_for("arbitrator")
            views_by_role["arbitrator"] = [lbl for lbl, _ in rv]
            content, extra = _role_content(cand_role, rv, records_text(advocate, critics, g), vt)
            arb_call = await _call("arbitrator", cand_role, model=model, system_prompt=sysp["arbitrator"],
                                   note=note, schema=schemas_panel.ArbitratorRecord, content=content,
                                   extra_views=extra, trace_path=_trace(trace_dir, name, "arbitrator"))
            note_call(arb_call)
            arbitrator_called = True
            arbitrator = arb_call["record"]
            if not arb_call["parse_ok"]:
                fails.append("arbitrator")
        else:
            skipped.append("arbitrator:" + ("parse_fail" if fails else "no_alternative"))
    elif advocate is not None and mode != "advocate_only":
        skipped.append(f"critics:p_evidence<{tau0}")
        skipped.append("arbitrator:no_critics")

    meta = {**summary, "cost_by_role": cost_by_role, "views_by_role": views_by_role,
            "n_images_by_role": {c["role"]: c["n_images"] for c in calls},
            "errors": {c["role"]: c["error"] for c in calls if c["error"]},
            "skipped": skipped, "ctx20_used": ctx20 is not None, "render_sha": adv_call["render_sha"],
            "note_sha16": _util.sha_text(note) if note else None,
            "render_desc_sha16": _util.sha_text(desc) if desc else None}
    result = schemas_panel.assemble(advocate, critics, arbitrator, thresholds, parse_failures=fails,
                                    cost_usd=sum(cost_by_role.values()), calls=len(calls),
                                    system_sha16s=sha_by_role, raw=raw_by_role, meta=meta,
                                    arbitrator_called=arbitrator_called)
    if trace_dir:
        hooks.Trace(_trace(trace_dir, name, "panel")).write(
            "golden_panel", **summary, roles=[c["role"] for c in calls], skipped=skipped,
            p_evidence=result.p_evidence, S=result.S, S_arb=result.S_arb, letter=result.letter,
            letter_source=result.letter_source, parse_failures=result.parse_failures,
            cost_usd=result.cost_usd, system_sha16s=sha_by_role)
    return result


def to_row(result, cand: dict) -> dict:
    """One predictions row for either result type (PanelResult → schemas_panel.to_row)."""
    if isinstance(result, IncumbentResult):
        return incumbent_row(result, cand)
    return schemas_panel.to_row(result, cand)


# ------------------------------------------------------------------ CLI (dry inspection)
def main(argv=None) -> int:
    """Print what each role WOULD receive for one candidate — system prompt shas, view labels,
    image counts — without any model call (the plumbing check before a paid run)."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", required=True, help="footer-cropped kit JPEG or a stamps/<id>_v1.jpg")
    ap.add_argument("--layout", default="color", choices=views.LAYOUTS)
    ap.add_argument("--mode", default="full", choices=MODES)
    ap.add_argument("--persona-dir", default=None)
    ap.add_argument("--note", default=str(NOTE_V2_PATH))
    ap.add_argument("--stamp-dir", default=None)
    ap.add_argument("--render", default="v1", choices=views.RENDERS, help="v2r: the chi-(f) VIEW text + description")
    args = ap.parse_args(argv)
    note = read_note(Path(args.note))
    img = _open_composite(args.image)
    ctx = views.ctx20_image(args.stamp_dir) if args.stamp_dir else None
    desc = read_render_desc(args.render)
    if args.mode == "incumbent":
        sysp = load_incumbent_set(Path(args.persona_dir) if args.persona_dir else PERSONA_DIR_INCUMBENT)
        for r, t in sysp.items():
            print(f"{r:11s} system sha16 {_util.sha_text(t)}  ({len(t)} chars, note: none)")
        return 0
    sysp = load_persona_set(Path(args.persona_dir) if args.persona_dir else PERSONA_DIR_V1, note)
    print(f"persona set sha16 {persona_set_sha16(Path(args.persona_dir) if args.persona_dir else PERSONA_DIR_V1)}; "
          f"note sha16 {_util.sha_text(note)}; gloss sha16 {views.gloss_sha16()}")
    for role in ROLES:
        full = grader_jwst.with_note(sysp[role], note)
        vr = "advocate" if args.mode == "attr" else role
        rv = views.role_views(img, args.layout, vr, ctx20=ctx, render=args.render, render_desc=desc)
        print(f"{role:11s} system sha16 {_util.sha_text(full)}  note last: {full.endswith(note)}  "
              f"images: {len(rv)}")
        for k, (lbl, im) in enumerate(rv, 1):
            print(f"    view {k}: {im.size}  {lbl[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
