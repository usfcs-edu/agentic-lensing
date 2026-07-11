#!/usr/bin/env python3
"""Phase C3 smoke: the matched-inputs grader on 6 seed-2026 train-pool candidates.

Draw exactly 6 candidates from outputs/parity_train_pool.csv split=train (2 graded-A,
2 graded-C, 2 graded-D; seed 2026 — the pool CSV, never the parity bench arms) and:

- LIVE (ANTHROPIC_API_KEY set): grade each with imaging/grader_matched.py on the
  anthropic backend, <=3 concurrent (also keeps the wide fetches polite), and save
  results.csv + traces under outputs/matched_smoke/.
- DRY RUN (no key): render every view, assemble the FULL prompt for each candidate,
  save the pieces under outputs/matched_smoke/<name>/ (view PNGs + prompt.md with
  image placeholders + request-shape stats) plus the rubric once, and validate the
  pipeline minus the API call (content builds, image counts, metadata block, JSON
  serializability of the would-be request, mag sanity).

  python lensjudge/parity/smoke_matched.py [--live|--dry]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.imaging import grader_matched  # noqa: E402

SEED = 2026
POOL = config.OUT / "parity_train_pool.csv"
OUTDIR = config.OUT / "matched_smoke"


def pick_candidates() -> pd.DataFrame:
    pool = pd.read_csv(POOL, low_memory=False)
    tr = pool[pool["split"] == "train"]
    picks = pd.concat([tr[tr["grade"] == g].sample(2, random_state=SEED)
                       for g in ("A", "C", "D")], ignore_index=True)
    return picks


def _save_content(cand: dict, content: list, info: dict, cdir: Path) -> dict:
    """Write the assembled prompt: PNGs for image blocks, prompt.md interleaving the
    text blocks with image placeholders, and per-candidate stats."""
    cdir.mkdir(parents=True, exist_ok=True)
    lines, n_img, b64_bytes = [], 0, 0
    view_names = [v for v in info.get("views", []) if v != "wide"] + (
        ["wide"] if info.get("wide_ok") else [])
    for blk in content:
        if blk["type"] == "text":
            lines.append(blk["text"] + "\n")
        else:
            vname = view_names[n_img] if n_img < len(view_names) else f"image{n_img}"
            png = cdir / f"view_{n_img:02d}_{vname}.png"
            data = blk["source"]["data"]
            png.write_bytes(base64.b64decode(data))
            b64_bytes += len(data)
            lines.append(f"![{vname}]({png.name})\n")
            n_img += 1
    (cdir / "prompt.md").write_text("\n".join(lines))
    # the exact would-be request body (images elided) — proves serializability
    body = {"model": "<resolved at call time>", "system": "<rubric_matched.md>",
            "messages": [{"role": "user", "content": [
                b if b["type"] == "text" else
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": f"<{len(b['source']['data'])} b64 chars>"}}
                for b in content]}]}
    (cdir / "request_shape.json").write_text(json.dumps(body, indent=2))
    return {"n_images": n_img, "b64_chars": b64_bytes}


def dry_run(picks: pd.DataFrame) -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "rubric_matched.md").write_text(grader_matched._RUBRIC_MATCHED)
    rows, failures = [], 0
    for cand in picks.to_dict("records"):
        name = cand["name"]
        content, info = grader_matched._build_content(cand)
        checks = {}
        if content is None:
            print(f"[FAIL] {name}: no cutout")
            failures += 1
            rows.append({"name": name, "grade_truth": cand["grade"], "content_ok": False})
            continue
        stats = _save_content(cand, content, info, OUTDIR / name)
        text_all = "\n".join(b["text"] for b in content if b["type"] == "text")
        # sanity: aperture mags in the metadata block within a plausible range
        import re
        mags = [float(m) for m in re.findall(r"[grz]=(\d+\.\d+)", text_all)]
        checks = {
            "content_ok": True,
            "n_images": stats["n_images"],
            "images_ok": stats["n_images"] == (4 if info["wide_ok"] else 3),
            "wide_ok": info["wide_ok"],
            "metadata_block": "CANDIDATE METADATA" in text_all,
            "cnn_score_present": "CNN recommendation score:" in text_all
                                 and "not available" not in
                                 text_all.split("CNN recommendation score:")[1].split("\n")[0],
            "json_contract_tail": "ONLY the JSON object" in content[-1]["text"],
            "mags_sane": all(12.0 < m < 26.0 for m in mags) and len(mags) >= 1,
            "metadata_sources": ",".join(info["metadata_sources"]) or "cand-row-only",
            "b64_chars": stats["b64_chars"],
        }
        hard = ("images_ok", "metadata_block", "json_contract_tail", "mags_sane")
        ok = all(checks[k] for k in hard)
        failures += (not ok)
        print(f"[{'ok' if ok else 'FAIL'}] {name} (graded {cand['grade']}, "
              f"{cand['survey_key']}): {stats['n_images']} images, wide={info['wide_ok']}, "
              f"meta={checks['metadata_sources']}, ~{stats['b64_chars']//1000}k b64 chars")
        rows.append({"name": name, "grade_truth": cand["grade"],
                     "survey_key": cand["survey_key"], **checks})
    pd.DataFrame(rows).to_csv(OUTDIR / "dry_run_summary.csv", index=False)
    print(f"\n[dry-run] {len(picks) - failures}/{len(picks)} candidates validated "
          f"-> {OUTDIR}")
    return failures


async def live_run(picks: pd.DataFrame) -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(3)   # polite: also bounds wide fetches vs legacysurvey.org
    async def one(cand):
        async with sem:
            return cand, await grader_matched.grade_candidate(
                cand, trace_path=str(OUTDIR / "traces" / f"{cand['name']}.jsonl"))
    results = await asyncio.gather(*(one(c) for c in picks.to_dict("records")))
    rows, failures = [], 0
    for cand, g in results:
        ok = g.parse_ok
        failures += (not ok)
        gr = g.grade
        print(f"[{'ok' if ok else 'FAIL'}] {cand['name']} (graded {cand['grade']}): "
              + (f"grade={gr.grade} p_lens={gr.p_lens:.2f} cost=${g.cost_usd:.4f} "
                 f"wide={g.meta.get('wide_ok')}" if gr else f"error={g.error}"))
        rows.append({"name": cand["name"], "grade_truth": cand["grade"],
                     "survey_key": cand["survey_key"], "parse_ok": g.parse_ok,
                     "grade_pred": gr.grade if gr else None,
                     "p_lens": gr.p_lens if gr else None,
                     "confidence": gr.confidence if gr else None,
                     "escalate": gr.escalate_to_human if gr else None,
                     "contaminant": gr.contaminant if gr else None,
                     "rationale": (gr.rationale if gr else (g.raw or ""))[:500],
                     "cost_usd": g.cost_usd, "wall_s": g.meta.get("wall_s"),
                     "wide_ok": g.meta.get("wide_ok"),
                     "metadata_sources": g.meta.get("metadata_sources"),
                     "error": g.error})
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "results.csv", index=False)
    print(f"\n[live] {len(df) - failures}/{len(df)} parsed, total cost "
          f"${df['cost_usd'].sum():.3f} -> {OUTDIR / 'results.csv'}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--live", action="store_true", help="force the live API run")
    g.add_argument("--dry", action="store_true", help="force the dry run")
    args = ap.parse_args()
    live = args.live or (not args.dry and bool(os.environ.get("ANTHROPIC_API_KEY")))
    picks = pick_candidates()
    print(f"[smoke] seed={SEED} picks:")
    cnn_col = [c for c in ("p_pub", "p_meta") if c in picks.columns][:1]
    print(picks[["name", "survey_key", "grade"] + cnn_col].to_string(index=False))
    print(f"[smoke] mode: {'LIVE (anthropic backend)' if live else 'DRY RUN (no API call)'}\n")
    failures = asyncio.run(live_run(picks)) if live else dry_run(picks)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
