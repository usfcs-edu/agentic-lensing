You are adversarially verifying candidate galaxy-scale strong gravitational
lenses found in JWST NIRCam imaging. A previous pass flagged these; most
flagged candidates are NOT real lenses, and your role is to find the ones that
are not.

{brief}

The image is attached below. Respond with exactly one JSON object: {id, persona, verdict, alternative, notes}
The previous pass's "claim_center": {claim_center}; "claim_quadrant": {claim_quadrant};
"claimed_evidence": {claimed_evidence}.

PANEL LAYOUT (6 panels, 2 rows):
  Top row    = full 10" field:      [normal] [deep] [colour]
  Bottom row = central 3.5" zoom:   [deep] [colour] [deflector-subtracted residual]
The candidate galaxy is at the exact centre, marked by four yellow ticks.
North up, East left; NE=top-left, NW=top-right, SE=bottom-left, SW=bottom-right.

STEP 2. Judge each IMAGE on its own merits. The previous pass ran at
deliberately high recall, so many claims are vague, low-confidence "worth a
second look" flags -- treat the claim as context, not as something you must
accept or literally refute. If the claimed evidence is weak but you can see a
genuinely lens-like configuration yourself, that still counts as "pass"; say so
in your notes. Conversely a confident-sounding claim with an obvious mundane
explanation is a "fail".

Decide whether a strong-lens interpretation SURVIVES your specific line of
attack. Reserve "uncertain" for cases your particular lens cannot settle;
prefer "fail" when you can name a convincing mundane explanation.

The JSON object, exactly:
{"id":"{item_id}","persona":"{persona}","verdict":"pass|fail|uncertain","alternative":"...","notes":"..."}

  verdict      "pass" = survives your attack and looks like a genuine lens
               "fail" = you found a convincing alternative explanation
               "uncertain" = your lens cannot decide
  alternative  the non-lens explanation you favour, or "" if none
  notes        one short sentence of specific reasoning

Respond with ONLY the JSON object.
