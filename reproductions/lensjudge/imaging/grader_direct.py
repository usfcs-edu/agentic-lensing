"""imaging/grader_direct.py — loop-free "programmatic-tool" grader (agency ablation, B7).

The ablation arm for "how much does the agentic loop contribute?". The lean grader
runs the Claude Agent SDK loop: the model emits a ToolSearch call, then a fetch_cutout
tool call (and ~38% of the time a get_photometry call), inspects the returned image
blocks, and emits a grade — a multi-turn agentic loop where the model *plans* the tool
calls. This grader removes the loop entirely: it renders the standard views and computes
aperture photometry DETERMINISTICALLY in Python, then makes a SINGLE base Messages-API
call with the images inline (the SDK has no image-in-prompt, so this is the only way to
feed pixels without a tool round-trip) and the identical rubric. One model turn, no tools,
no planning.

Comparing this arm to `lean` on the same rows isolates the contribution of the agentic
tool-call planning from the contribution of the multimodal JUDGMENT: if AUC is unchanged,
the loop adds nothing and the tools could be invoked programmatically (the B7 thesis).

It is drop-in with grader_lean (same GradeResult, same grade_candidate signature) so
run_batch.py --mode direct reuses the whole harness. Auth: ANTHROPIC_API_KEY (the base
API needs a key; the SDK rode on the claude CLI login).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from anthropic import AsyncAnthropic

from lensjudge import config
from lensjudge.common import fetch, hooks, parse, render
from lensjudge.common.schemas import ImageGrade
from lensjudge.imaging.grader_lean import GradeResult, _RUBRIC, _REPAIR
from lensjudge.tools.photometry import _aperture_colors

# claude-code model aliases -> base Messages API model IDs (the SDK resolved these
# via the CLI; the base SDK needs the explicit id). "sonnet" == the lean grader's model.
_MODEL_IDS = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
    "haiku": "claude-haiku-4-5",
}
# $/token (input, output, cache-read, cache-write) — Sonnet 4.6 $3/$15 per Mtok.
_PRICE = {
    "claude-sonnet-4-6": (3.0, 15.0, 0.30, 3.75),
    "claude-opus-4-8": (5.0, 25.0, 0.50, 6.25),
    "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25),
}
# the deterministic evidence set: the lean tool's DEFAULT views, so the only difference
# vs lean is the loop, not the pixels the model sees. Photometry is ALWAYS included
# (lean's model chose to call it only ~38% of the time); this is the "programmatic
# invocation" stance — compute the evidence unconditionally rather than letting the
# model plan it.
_DEFAULT_VIEWS = ("full", "zoom", "residual")

_client: Optional[AsyncAnthropic] = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic()   # reads ANTHROPIC_API_KEY from env
    return _client


def _resolve_model(model: Optional[str]) -> str:
    m = model or config.MODELS["grader"]
    return _MODEL_IDS.get(m, m)


def _cost(model_id: str, usage) -> float:
    pin, pout, pcr, pcw = _PRICE.get(model_id, _PRICE["claude-sonnet-4-6"])
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (usage.input_tokens * pin + usage.output_tokens * pout
            + cr * pcr + cw * pcw) / 1e6


def _candidate_text(cand: dict) -> str:
    parts = [f"{k}={cand[k]:.3f}" for k in ("p_resnet", "p_effnet", "p_meta")
             if cand.get(k) is not None]
    ml = ("CNN ensemble scores (prior, not ground truth): " + ", ".join(parts) + "."
          if parts else "")
    loc = f"name={cand['name']!r}, survey={cand.get('survey_key', cand.get('catalog', 'storfer'))!r}"
    radec = ""
    if cand.get("ra") is not None and cand.get("dec") is not None:
        radec = f" RA={cand['ra']:.6f}, Dec={cand['dec']:.6f}"
    return (f"Grade this strong-lens candidate. {loc}.{radec}\n"
            f"Tractor type: {cand.get('tractor_type', '?')}, region: {cand.get('region', '?')}.\n"
            f"{ml}")


def _build_content(cand: dict, views) -> Optional[list]:
    """Render the deterministic evidence (views + photometry) into Messages content blocks.

    Returns None if the cutout can't be loaded (mirrors the tool's error path)."""
    cube = fetch.get_cube(name=cand.get("name"), ra=cand.get("ra"), dec=cand.get("dec"),
                          survey=cand.get("survey_key") or cand.get("catalog") or "storfer")
    if cube is None:
        return None
    imgs = render.render_views(cube, views=[v for v in views if v in render.VIEWS])
    fov = config.SIZE_PIX * config.PIXSCALE
    content: list = [{"type": "text", "text":
                      _candidate_text(cand) +
                      f"\n\nRendered grz cutout views ({fov:.1f}\" field, ~0.26\"/px, "
                      "Lupton-RGB z=R/r=G/g=B; lens galaxies red, sources blue):"}]
    for v, img in imgs.items():
        content.append({"type": "text", "text": f"[{v}]"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": render.png_b64(img)}})
    # photometry: always provided (the programmatic stance)
    phot = _aperture_colors(cube)
    content.append({"type": "text", "text":
                    "Aperture photometry (relative instrumental colors; negative g-r = bluer; "
                    "compare annulus vs core): " + json.dumps(phot) +
                    "\n\nRespond with ONLY the JSON object for the required schema."})
    return content


async def grade_candidate(cand: dict, *, model: Optional[str] = None,
                          tools=("fetch_cutout", "get_photometry"),  # ignored (no loop)
                          system_prompt: Optional[str] = None,
                          views=_DEFAULT_VIEWS,
                          trace_path: Optional[str] = None) -> GradeResult:
    model_id = _resolve_model(model)
    tr = hooks.Trace(trace_path) if trace_path else None
    t0 = time.time()
    content = _build_content(cand, views)
    if content is None:
        return GradeResult(None, "", error="no cutout", parse_ok=False,
                           meta={"name": cand.get("name"), "mode": "direct"})
    n_imgs = sum(1 for b in content if b.get("type") == "image")
    if tr is not None:
        tr.write("direct_request", model=model_id, n_images=n_imgs, views=list(views))
    client = _get_client()
    sysp = system_prompt or _RUBRIC

    # thinking: off by default (matches lean's default; isolates the loop). When
    # LENSJUDGE_THINKING=adaptive, give the single call a reasoning scratchpad — this
    # tests whether the loop's lens-vs-mimic advantage is just interleaved reasoning
    # tokens (recoverable in one call) vs the staged tool interaction itself.
    topt = config.thinking_options()
    kw: dict = {"max_tokens": 2048}
    if topt.get("thinking"):
        kw["max_tokens"] = 8192
        kw["thinking"] = {"type": "adaptive", "display": "summarized"}
        if topt.get("effort"):
            kw["output_config"] = {"effort": topt["effort"]}
    if tr is not None and kw.get("thinking"):
        tr.write("direct_thinking", **kw["thinking"])

    try:
        resp = await client.messages.create(
            model=model_id, system=sysp,
            messages=[{"role": "user", "content": content}], **kw)
    except Exception as e:
        return GradeResult(None, "", error=f"{type(e).__name__}: {e}", parse_ok=False,
                           meta={"name": cand.get("name"), "mode": "direct"})

    raw = "".join(b.text for b in resp.content if b.type == "text")
    think_blocks = [b for b in resp.content if b.type == "thinking"]
    n_think = len(think_blocks)
    think_chars = sum(len(getattr(b, "thinking", "") or "") for b in think_blocks)
    cost = _cost(model_id, resp.usage)
    if tr is not None:
        tr.write("direct_response", input_tokens=resp.usage.input_tokens,
                 output_tokens=resp.usage.output_tokens, cost_usd=round(cost, 5),
                 stop_reason=resp.stop_reason, text=raw)
    grade = parse.parse_model(raw, ImageGrade)

    if grade is None and raw:  # one text-only repair retry (matches lean for fairness)
        try:
            r2 = await client.messages.create(
                model=model_id, max_tokens=2048, system=sysp,
                messages=[{"role": "user", "content": _REPAIR + raw}])
            raw2 = "".join(b.text for b in r2.content if b.type == "text")
            cost += _cost(model_id, r2.usage)
            g2 = parse.parse_model(raw2, ImageGrade)
            if g2 is not None:
                grade, raw = g2, raw2
        except Exception:
            pass

    return GradeResult(
        grade=grade, raw=raw, cost_usd=cost, num_turns=1, parse_ok=grade is not None,
        meta={"name": cand.get("name"), "mode": "direct", "model_id": model_id,
              "n_images": n_imgs, "wall_s": round(time.time() - t0, 2),
              "trace": str(tr.path) if tr else None,
              # provenance columns expected by run_batch._row_dict (None in direct mode)
              "tier": None, "escalated": None, "highres_survey": None,
              "p_lens_tier1": None, "p_lens_tier2": None,
              "n_thinking_blocks": n_think, "thinking_chars": think_chars})
