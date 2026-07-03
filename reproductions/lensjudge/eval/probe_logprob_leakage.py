#!/usr/bin/env python3
"""eval/probe_logprob_leakage.py — one-time LIVE probe: are grade-token logprobs trustworthy
under structured/guided decoding on this server?

v5 scores p_lens from the token distribution at the ImageGrade ``"grade":"X"`` position
(llm_client.extract_grade_probs). Risk: some structured-output backends report POST-mask
logprobs — the grammar mask can leave a single legal token with ~all the mass, which would fake
certainty and destroy calibration. This probe asks the live server to grade a deliberately
AMBIGUOUS described object twice (guided ON vs OFF) and prints the top tokens at the grade
position for both. Interpretation:

  healthy   : guided ON shows several A/B/C/D alternatives with spread similar to guided OFF
  LEAKAGE   : guided ON collapses to one token at ~prob 1.0 while OFF shows spread
              -> score logprobs with guided OFF for the grade call, or pin --logprobs-mode

Usage (server already running):
  LENSJUDGE_BACKEND=openai LENSJUDGE_BASE_URL=http://localhost:8000/v1 \
    python -m lensjudge.eval.probe_logprob_leakage --model <served-id>
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lensjudge.common import llm_client  # noqa: E402
from lensjudge.common.schemas import ImageGrade  # noqa: E402

_PROMPT = (
    "You are grading a strong-lens candidate from a TEXT description only (no image). The "
    "description is deliberately ambiguous: 'a red elliptical galaxy with a faint bluish smudge "
    "~1.5 arcsec to one side; unclear curvature; no counter-image visible; moderate seeing.' "
    "Respond with ONLY an ImageGrade JSON object: keys grade (A/B/C/D), criteria{blue_source,"
    "low_surface_brightness,curvature,counter_images,arc_morphology}, p_lens, confidence, "
    "contaminant, escalate_to_human, rationale."
)


async def _one(model: str, guided: bool):
    client = llm_client._get_client()
    kw: dict = {"logprobs": True, "top_logprobs": 10}
    if guided:
        schema = ImageGrade.model_json_schema()
        if llm_client._structured_mode() == "new":
            kw["extra_body"] = {"structured_outputs": {"json": schema}}
        else:
            kw["extra_body"] = {"guided_json": schema}
    resp = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": _PROMPT}],
        max_tokens=1024, temperature=0.0, **kw)
    choice = resp.choices[0]
    gp = llm_client._grade_probs_from_choice(choice)
    # also dump the raw top tokens at the grade position for eyeballing
    lp = getattr(choice, "logprobs", None)
    content = llm_client._f(lp, "content") if lp else None
    raw_top = None
    if content:
        texts = [str(llm_client._f(t, "token") or "") for t in content]
        m = llm_client._GRADE_KEY_RE.search("".join(texts))
        if m:
            pos, off = m.end(), 0
            for i, s in enumerate(texts):
                if off <= pos < off + len(s):
                    raw_top = [(str(llm_client._f(c, "token")), round(float(llm_client._f(c, "logprob")), 3))
                               for c in (llm_client._f(content[i], "top_logprobs") or [])]
                    break
                off += len(s)
    return gp, raw_top, (choice.message.content or "")[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("LENSJUDGE_MODEL_GRADER"))
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("need --model or LENSJUDGE_MODEL_GRADER")
    for guided in (False, True):
        gp, raw_top, head = asyncio.run(_one(args.model, guided))
        tag = "GUIDED ON " if guided else "GUIDED OFF"
        print(f"\n=== {tag} ===")
        print(f"  reply head : {head!r}")
        print(f"  grade_probs: {gp}")
        print(f"  raw top@grade: {raw_top}")
    print("\nInterpretation: LEAKAGE if guided-ON collapses to a single ~1.0 token while "
          "guided-OFF shows spread. Healthy if both show comparable A/B/C/D spread.")


if __name__ == "__main__":
    main()
