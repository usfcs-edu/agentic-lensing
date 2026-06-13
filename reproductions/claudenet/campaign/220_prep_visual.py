#!/usr/bin/env python3
"""220_prep_visual.py — emit the per-candidate input for the visual-judging
Workflow: each candidate's row_id, p_final, and the 4 rendered PNG paths. The
Workflow (launched separately) chunks these into batches, spawns grading
subagents that Read the PNGs, and returns structured grades.

Writes data/v2/campaign/visual_input.json (a JSON list the Workflow reads as args).

    /home2/benson/.venvs/claudenet/bin/python campaign/220_prep_visual.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "campaign"


def main() -> int:
    man = pd.read_parquet(OUT / "manifest_737.parquet")
    man["row_id"] = man["row_id"].astype(str)
    items = []
    for r in man.itertuples():
        items.append({
            "row_id": r.row_id,
            "p_final": round(float(r.p_final), 4),
            "full": r.png_full, "zoom": r.png_zoom,
            "residual": r.png_residual, "highcontrast": r.png_highcontrast,
        })
    # deterministic order (by row_id) so batch composition is reproducible
    items.sort(key=lambda d: d["row_id"])
    (OUT / "visual_input.json").write_text(json.dumps(items))
    print(f"[220] wrote {len(items)} candidates -> {OUT/'visual_input.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
